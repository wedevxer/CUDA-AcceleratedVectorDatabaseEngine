#pragma once

#include <grpcpp/grpcpp.h>
#include <grpcpp/health_check_service_interface.h>
#include "../proto/vdb.grpc.pb.h"
#include "../engine/ivf_flat_index.h"
#include "../engine/prefetcher.h"
#include "../format/storage.h"
#include <memory>
#include <unordered_map>
#include <shared_mutex>
#include <queue>
#include <condition_variable>
#include <thread>
#include <atomic>
#include <future>
#include <chrono>
#include <any>

namespace vdb {
namespace server {

class QueryServiceImpl final : public vdb::QueryService::Service {
public:
    struct Config {
        size_t max_batch_size = 64;
        size_t coalesce_window_ms = 2;
        size_t max_concurrent_searches = 16;
        size_t gpu_memory_limit = 8ULL << 30;
        std::string data_path = "/mnt/nvme/vdb";
    };
    
    explicit QueryServiceImpl(const Config& config);
    ~QueryServiceImpl() override;
    
    grpc::Status Search(grpc::ServerContext* context,
                       const vdb::SearchRequest* request,
                       vdb::SearchResponse* response) override;
    
    grpc::Status Warmup(grpc::ServerContext* context,
                       const vdb::WarmupRequest* request,
                       google::protobuf::Empty* response) override;
    
    grpc::Status LoadIndex(grpc::ServerContext* context,
                          const vdb::LoadIndexRequest* request,
                          google::protobuf::Empty* response) override;
    
private:
    Config config_;
    
    struct IndexState {
        std::unique_ptr<IVFFlatIndex> ivf_flat;
        std::unique_ptr<IVFPQIndex> ivf_pq;
        std::shared_ptr<storage::EpochManager::Epoch> epoch;
        std::chrono::steady_clock::time_point loaded_at;
    };
    
    std::unordered_map<std::string, std::unique_ptr<IndexState>> indices_;
    std::shared_mutex indices_mutex_;
    
    std::unique_ptr<TransferManager> transfer_manager_;
    std::unique_ptr<IOUringPrefetcher> io_prefetcher_;
    std::unique_ptr<AdaptivePrefetcher> adaptive_prefetcher_;
    std::unique_ptr<ListPrefetcher> list_prefetcher_;
    std::unique_ptr<storage::ShardManager> shard_manager_;
    std::unique_ptr<storage::EpochManager> epoch_manager_;
    
    struct BatchedSearch {
        std::vector<SearchRequest> requests;
        std::vector<grpc::ServerContext*> contexts;
        std::vector<SearchResponse*> responses;
        std::vector<std::promise<grpc::Status>> promises;
        std::chrono::steady_clock::time_point enqueue_time;
    };
    
    std::queue<BatchedSearch> search_queue_;
    std::mutex queue_mutex_;
    std::condition_variable queue_cv_;
    
    std::thread batch_processor_;
    std::atomic<bool> stop_{false};
    
    void batch_processor_loop();
    void process_batch(BatchedSearch& batch);
    
    IndexState* get_index(const std::string& name);
    grpc::Status load_index_internal(const std::string& name, 
                                    const std::string& epoch);
};

class AdminServiceImpl final : public vdb::AdminService::Service {
public:
    explicit AdminServiceImpl(QueryServiceImpl* query_service);
    ~AdminServiceImpl() override;
    
    grpc::Status CreateIndex(grpc::ServerContext* context,
                            const vdb::CreateIndexRequest* request,
                            google::protobuf::Empty* response) override;
    
    grpc::Status BuildEpoch(grpc::ServerContext* context,
                           const vdb::BuildEpochRequest* request,
                           google::protobuf::Empty* response) override;
    
    grpc::Status ActivateEpoch(grpc::ServerContext* context,
                              const vdb::ActivateEpochRequest* request,
                              google::protobuf::Empty* response) override;
    
    grpc::Status GetStats(grpc::ServerContext* context,
                         const vdb::StatsRequest* request,
                         vdb::StatsResponse* response) override;
    
private:
    QueryServiceImpl* query_service_;
    
    struct BuildJob {
        std::string index_name;
        std::string source_path;
        std::string epoch_id;
        std::thread worker;
        std::atomic<bool> running{true};
        std::atomic<float> progress{0.0f};
    };
    
    std::unordered_map<std::string, std::unique_ptr<BuildJob>> build_jobs_;
    std::mutex jobs_mutex_;
    
    void build_index_worker(BuildJob* job);
};

class RequestCoalescer {
public:
    struct Config {
        size_t max_batch_size = 64;
        std::chrono::milliseconds window_duration{2};
        size_t max_queue_size = 1024;
    };
    
    explicit RequestCoalescer(const Config& config);
    ~RequestCoalescer();
    
    template<typename Request, typename Response>
    void enqueue(const Request& req, Response* resp,
                std::function<void(std::vector<Request>&, 
                                  std::vector<Response*>&)> batch_func);
    
    void start();
    void stop();
    
private:
    Config config_;
    
    struct PendingRequest {
        std::any request;
        std::any response;
        std::function<void()> process_func;
        std::chrono::steady_clock::time_point enqueue_time;
    };
    
    std::queue<PendingRequest> queue_;
    std::mutex queue_mutex_;
    std::condition_variable queue_cv_;
    
    std::thread processor_;
    std::atomic<bool> running_{false};
    
    void processor_loop();
};

class RateLimiter {
public:
    struct Config {
        size_t requests_per_second = 10000;
        size_t burst_size = 100;
    };
    
    explicit RateLimiter(const Config& config);
    
    bool try_acquire(size_t tokens = 1);
    void acquire(size_t tokens = 1);
    
    void update_rate(size_t requests_per_second);
    
private:
    std::mutex mutex_;
    size_t max_tokens_;
    size_t tokens_;
    std::chrono::steady_clock::time_point last_refill_;
    std::chrono::nanoseconds refill_period_;
    
    void refill();
};

class MetricsCollector {
public:
    MetricsCollector();
    ~MetricsCollector();
    
    void record_search_latency(const std::string& index, 
                              double latency_ms);
    void record_search_recall(const std::string& index, 
                             float recall);
    void increment_search_count(const std::string& index);
    void record_gpu_memory(size_t bytes);
    void record_nvme_bandwidth(double gbps);
    
    struct Metrics {
        struct IndexMetrics {
            uint64_t search_count = 0;
            double avg_latency_ms = 0;
            double p50_latency_ms = 0;
            double p95_latency_ms = 0;
            double p99_latency_ms = 0;
            float avg_recall = 0;
        };
        
        std::unordered_map<std::string, IndexMetrics> indices;
        size_t gpu_memory_bytes = 0;
        double nvme_bandwidth_gbps = 0;
        double qps = 0;
    };
    
    Metrics get_metrics() const;
    std::string prometheus_format() const;
    
private:
    mutable std::mutex mutex_;
    
    struct LatencyHistogram {
        std::vector<double> values;
        void add(double value);
        double percentile(double p) const;
    };
    
    std::unordered_map<std::string, LatencyHistogram> latencies_;
    std::unordered_map<std::string, std::vector<float>> recalls_;
    std::unordered_map<std::string, uint64_t> search_counts_;
    
    std::atomic<size_t> gpu_memory_bytes_{0};
    std::atomic<double> nvme_bandwidth_gbps_{0};
    
    std::chrono::steady_clock::time_point start_time_;
};

}
}