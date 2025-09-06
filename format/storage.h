#pragma once

#include <arrow/api.h>
#include <arrow/io/api.h>
#include <arrow/ipc/api.h>
#include <arrow/compute/api.h>
#include <memory>
#include <string>
#include <vector>
#include <json/json.h>

namespace vdb {
namespace storage {

struct IndexManifest {
    std::string index_name;
    std::string epoch;
    uint32_t dimension;
    uint32_t nlist;
    std::string metric;
    
    struct PQParams {
        uint32_t m = 0;
        uint32_t nbits = 8;
    } pq_params;
    
    struct ShardInfo {
        uint32_t list_id;
        std::string path;
        size_t num_vectors;
        size_t file_size;
    };
    
    std::vector<ShardInfo> shards;
    std::chrono::system_clock::time_point created_at;
    
    Json::Value to_json() const;
    static IndexManifest from_json(const Json::Value& json);
    
    void save(const std::string& path) const;
    static IndexManifest load(const std::string& path);
};

class ArrowStorage {
public:
    ArrowStorage();
    ~ArrowStorage();
    
    arrow::Result<std::shared_ptr<arrow::Table>> read_vectors(
        const std::string& path,
        int64_t offset = 0,
        int64_t length = -1);
    
    arrow::Status write_vectors(
        const std::string& path,
        const float* vectors,
        const uint64_t* ids,
        size_t n_vectors,
        uint32_t dimension);
    
    arrow::Result<std::shared_ptr<arrow::Table>> read_centroids(
        const std::string& path);
    
    arrow::Status write_centroids(
        const std::string& path,
        const float* centroids,
        uint32_t nlist,
        uint32_t dimension);
    
    arrow::Result<std::shared_ptr<arrow::Table>> read_pq_codebooks(
        const std::string& path);
    
    arrow::Status write_pq_codebooks(
        const std::string& path,
        const float* codebooks,
        uint32_t m,
        uint32_t ks,
        uint32_t dsub);
    
    arrow::Result<std::shared_ptr<arrow::Buffer>> mmap_file(
        const std::string& path);
    
private:
    std::shared_ptr<arrow::MemoryPool> pool_;
    
    std::shared_ptr<arrow::Schema> create_vector_schema(uint32_t dimension);
    std::shared_ptr<arrow::Schema> create_centroid_schema(uint32_t dimension);
    std::shared_ptr<arrow::Schema> create_pq_schema(uint32_t dsub);
};

class NVMeOptimizedReader {
public:
    struct Config {
        size_t read_ahead_size = 4 << 20;
        size_t alignment = 4096;
        int io_depth = 32;
        bool use_direct_io = true;
    };
    
    explicit NVMeOptimizedReader(const Config& config);
    ~NVMeOptimizedReader();
    
    void* read_aligned(const std::string& path, size_t offset, 
                      size_t size, void* buffer = nullptr);
    
    void read_async(const std::string& path, size_t offset,
                   size_t size, void* buffer,
                   std::function<void(int)> callback);
    
    void prefetch(const std::string& path, size_t offset, size_t size);
    
    void wait_all();
    
private:
    Config config_;
    
    struct IOContext;
    std::unique_ptr<IOContext> ctx_;
    
    void* allocate_aligned_buffer(size_t size);
    void free_aligned_buffer(void* buffer);
};

class ShardManager {
public:
    struct Shard {
        uint32_t list_id;
        std::string base_path;
        size_t num_vectors;
        
        std::shared_ptr<arrow::Buffer> vectors_mmap;
        std::shared_ptr<arrow::Buffer> ids_mmap;
        std::shared_ptr<arrow::Buffer> codes_mmap;
        
        bool is_loaded = false;
        std::chrono::steady_clock::time_point last_access;
    };
    
    ShardManager(const std::string& base_path, 
                NVMeOptimizedReader* reader);
    ~ShardManager();
    
    void create_shard(uint32_t list_id);
    
    void append_to_shard(uint32_t list_id,
                        const float* vectors,
                        const uint64_t* ids,
                        size_t n_vectors,
                        uint32_t dimension);
    
    void append_codes_to_shard(uint32_t list_id,
                              const uint8_t* codes,
                              const uint64_t* ids,
                              size_t n_codes,
                              uint32_t code_size);
    
    std::shared_ptr<Shard> load_shard(uint32_t list_id);
    void unload_shard(uint32_t list_id);
    
    void compact_shard(uint32_t list_id);
    
    std::vector<uint32_t> list_shards() const;
    
private:
    std::string base_path_;
    NVMeOptimizedReader* reader_;
    ArrowStorage arrow_storage_;
    
    std::unordered_map<uint32_t, std::shared_ptr<Shard>> shards_;
    std::mutex mutex_;
    
    std::string get_shard_path(uint32_t list_id) const;
};

class EpochManager {
public:
    struct Epoch {
        std::string id;
        IndexManifest manifest;
        std::string base_path;
        bool active = false;
        std::chrono::system_clock::time_point created_at;
    };
    
    EpochManager(const std::string& base_path);
    ~EpochManager();
    
    std::string create_epoch(const IndexManifest& manifest);
    
    void activate_epoch(const std::string& epoch_id);
    void deactivate_epoch(const std::string& epoch_id);
    
    std::shared_ptr<Epoch> get_active_epoch() const;
    std::shared_ptr<Epoch> get_epoch(const std::string& epoch_id) const;
    
    std::vector<std::string> list_epochs() const;
    
    void cleanup_old_epochs(int keep_n = 3);
    
private:
    std::string base_path_;
    std::map<std::string, std::shared_ptr<Epoch>> epochs_;
    std::string active_epoch_id_;
    mutable std::shared_mutex mutex_;
    
    std::string generate_epoch_id() const;
    void persist_epoch_state();
    void load_epoch_state();
};

}
}