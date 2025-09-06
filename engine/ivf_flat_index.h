#pragma once

#include <vector>
#include <memory>
#include <atomic>
#include <unordered_map>
#include <unordered_set>
#include <chrono>
#include "kernels.cuh"
#include "transfer_manager.h"

namespace vdb {

class IVFFlatIndex {
public:
    struct Config {
        uint32_t dimension;
        uint32_t nlist;
        kernels::Metric metric;
        bool use_gpu = true;
        size_t max_gpu_memory = 8ULL << 30;
    };
    
    struct InvertedList {
        std::vector<float> vectors;
        std::vector<uint64_t> ids;
        size_t count = 0;
        
        void* gpu_vectors = nullptr;
        void* gpu_ids = nullptr;
        size_t gpu_capacity = 0;
        bool on_gpu = false;
        
        std::atomic<uint64_t> access_count{0};
        std::chrono::steady_clock::time_point last_access;
    };
    
    struct SearchParams {
        uint32_t nprobe = 10;
        uint32_t k = 10;
        bool use_exact_rerank = false;
    };
    
    IVFFlatIndex(const Config& config, TransferManager* tm);
    ~IVFFlatIndex();
    
    void train(const float* vectors, uint64_t n_vectors);
    
    void add(const float* vectors, const uint64_t* ids, uint64_t n_vectors);
    
    void search(const float* queries, uint32_t n_queries,
                const SearchParams& params,
                float* distances, uint64_t* indices);
    
    void search_batch(const std::vector<float*>& queries,
                     const std::vector<SearchParams>& params,
                     std::vector<float*>& distances,
                     std::vector<uint64_t*>& indices);
    
    void warmup_lists(const std::vector<uint32_t>& list_ids);
    void evict_list(uint32_t list_id);
    
    size_t get_gpu_memory_usage() const;
    size_t get_total_vectors() const { return total_vectors_; }
    
    void save(const std::string& path) const;
    void load(const std::string& path);
    
private:
    Config config_;
    TransferManager* tm_;
    
    std::vector<float> centroids_;
    std::vector<std::unique_ptr<InvertedList>> lists_;
    
    size_t total_vectors_ = 0;
    size_t gpu_memory_used_ = 0;
    
    std::unordered_map<uint32_t, cudaStream_t> list_streams_;
    
    void assign_to_lists(const float* vectors, uint64_t n_vectors,
                         std::vector<uint32_t>& assignments);
    
    void search_list(uint32_t list_id, const float* query,
                    uint32_t k, float* distances, uint64_t* indices,
                    cudaStream_t stream);
    
    void search_list_cpu(uint32_t list_id, const float* query, uint32_t k,
                        float* distances, uint64_t* indices);
    
    void search_list_gpu(uint32_t list_id, const float* query, uint32_t k,
                        float* distances, uint64_t* indices, cudaStream_t stream);
    
    void assign_to_lists_gpu(const float* vectors, uint64_t n_vectors,
                            std::vector<uint32_t>& assignments);
    
    void load_list_to_gpu(uint32_t list_id);
    void evict_list_from_gpu(uint32_t list_id);
    
    std::vector<uint32_t> select_nprobe_lists(const float* query, uint32_t nprobe);
    
    void merge_results(const std::vector<std::vector<float>>& list_distances,
                      const std::vector<std::vector<uint64_t>>& list_indices,
                      uint32_t k, float* distances, uint64_t* indices);
};

class IVFPQIndex {
public:
    struct Config {
        uint32_t dimension;
        uint32_t nlist;
        uint32_t m;
        uint32_t nbits = 8;
        kernels::Metric metric;
        bool use_gpu = true;
        size_t max_gpu_memory = 8ULL << 30;
    };
    
    struct PQInvertedList {
        std::vector<uint8_t> codes;
        std::vector<uint64_t> ids;
        size_t count = 0;
        
        void* gpu_codes = nullptr;
        void* gpu_ids = nullptr;
        size_t gpu_capacity = 0;
        bool on_gpu = false;
        
        std::atomic<uint64_t> access_count{0};
        std::chrono::steady_clock::time_point last_access;
    };
    
    struct SearchParams {
        uint32_t nprobe = 10;
        uint32_t k = 10;
        bool use_exact_rerank = false;
        uint32_t rerank_k = 0;
    };
    
    IVFPQIndex(const Config& config, TransferManager* tm);
    ~IVFPQIndex();
    
    void train(const float* vectors, uint64_t n_vectors);
    
    void add(const float* vectors, const uint64_t* ids, uint64_t n_vectors);
    
    void search(const float* queries, uint32_t n_queries,
                const SearchParams& params,
                float* distances, uint64_t* indices);
    
    void warmup_lists(const std::vector<uint32_t>& list_ids);
    
    size_t get_gpu_memory_usage() const;
    size_t get_total_vectors() const { return total_vectors_; }
    
    void save(const std::string& path) const;
    void load(const std::string& path);
    
private:
    Config config_;
    TransferManager* tm_;
    
    std::vector<float> centroids_;
    std::vector<float> pq_centroids_;
    std::vector<std::unique_ptr<PQInvertedList>> lists_;
    
    size_t total_vectors_ = 0;
    size_t gpu_memory_used_ = 0;
    uint32_t ks_;
    uint32_t dsub_;
    
    void* gpu_pq_centroids_ = nullptr;
    void* gpu_dist_tables_ = nullptr;
    
    void train_pq(const float* vectors, uint64_t n_vectors);
    
    void encode_vectors(const float* vectors, uint64_t n_vectors,
                       uint8_t* codes);
    
    void compute_distance_tables(const float* queries, uint32_t n_queries,
                                float* tables, cudaStream_t stream);
    
    void search_list_pq(uint32_t list_id, const float* dist_table,
                       uint32_t k, float* distances, uint64_t* indices,
                       cudaStream_t stream);
    
    void load_list_to_gpu(uint32_t list_id);
    void evict_list_from_gpu(uint32_t list_id);
};

class GpuCache {
public:
    struct Entry {
        uint32_t list_id;
        size_t size;
        void* data;
        std::chrono::steady_clock::time_point last_access;
        uint64_t access_count;
    };
    
    GpuCache(size_t max_size, TransferManager* tm);
    ~GpuCache();
    
    void* get(uint32_t list_id);
    void put(uint32_t list_id, const void* data, size_t size);
    void evict(uint32_t list_id);
    void evict_lru();
    
    size_t get_used_memory() const { return used_memory_; }
    float get_hit_rate() const;
    
private:
    size_t max_size_;
    size_t used_memory_ = 0;
    TransferManager* tm_;
    
    std::unordered_map<uint32_t, Entry> cache_;
    std::mutex mutex_;
    
    std::atomic<uint64_t> hits_{0};
    std::atomic<uint64_t> misses_{0};
};

}