#include "query_service.h"
#include <grpcpp/grpcpp.h>
#include <thread>
#include <chrono>
#include <algorithm>
#include <cuda_runtime.h>
#include <filesystem>
#include <fstream>
#include <queue>
#include <condition_variable>
#include <future>
#include <any>
#include <numeric>

namespace vdb {
namespace server {

// QueryServiceImpl implementation
QueryServiceImpl::QueryServiceImpl(const Config& config) 
    : config_(config) {
    
    // Initialize transfer manager
    TransferManager::Config tm_config;
    tm_config.pinned_pool_size = 256 << 20;  // 256MB
    tm_config.device_pool_size = config_.gpu_memory_limit / 2;
    transfer_manager_ = std::make_unique<TransferManager>(tm_config);
    
    // Initialize prefetchers
    IOUringPrefetcher::Config io_config;
    io_config.ring_size = 1024;
    io_config.num_workers = 4;
    io_prefetcher_ = std::make_unique<IOUringPrefetcher>(io_config);
    
    AdaptivePrefetcher::Config adaptive_config;
    adaptive_config.prefetch_ratio = 0.3f;
    adaptive_config.max_prefetch_size = 64 << 20;  // 64MB
    adaptive_prefetcher_ = std::make_unique<AdaptivePrefetcher>(
        adaptive_config, io_prefetcher_.get());
    
    ListPrefetcher::Config list_config;
    list_config.max_cached_lists = 512;
    list_config.cache_size_bytes = config_.gpu_memory_limit / 4;
    list_prefetcher_ = std::make_unique<ListPrefetcher>(
        list_config, transfer_manager_.get());
    
    // Initialize storage managers
    storage::ShardManager::Config shard_config;
    shard_config.data_path = config_.data_path;
    shard_config.max_shard_size = 16ULL << 30;  // 16GB per shard
    shard_manager_ = std::make_unique<storage::ShardManager>(shard_config);
    
    storage::EpochManager::Config epoch_config;
    epoch_config.data_path = config_.data_path + "/epochs";
    epoch_manager_ = std::make_unique<storage::EpochManager>(epoch_config);
    
    // Start batch processor
    batch_processor_ = std::thread(&QueryServiceImpl::batch_processor_loop, this);
}

QueryServiceImpl::~QueryServiceImpl() {
    stop_ = true;
    queue_cv_.notify_all();
    if (batch_processor_.joinable()) {
        batch_processor_.join();
    }
}

grpc::Status QueryServiceImpl::Search(grpc::ServerContext* context,
                                     const vdb::SearchRequest* request,
                                     vdb::SearchResponse* response) {
    // Validate request
    if (request->queries().empty()) {
        return grpc::Status(grpc::StatusCode::INVALID_ARGUMENT, 
                          "No queries provided");
    }
    
    if (request->topk() <= 0 || request->topk() > 1000) {
        return grpc::Status(grpc::StatusCode::INVALID_ARGUMENT, 
                          "Invalid topk value");
    }
    
    if (request->index().empty()) {
        return grpc::Status(grpc::StatusCode::INVALID_ARGUMENT, 
                          "Index name required");
    }
    
    // Get index
    IndexState* index_state = get_index(request->index());
    if (!index_state) {
        return grpc::Status(grpc::StatusCode::NOT_FOUND, 
                          "Index not found: " + request->index());
    }
    
    // Prepare search parameters
    IVFFlatIndex::SearchParams params;
    params.k = request->topk();
    params.nprobe = request->nprobe() > 0 ? request->nprobe() : 8;
    
    // Parse metric
    if (request->metric() == "L2") {
        params.metric = kernels::Metric::L2;
    } else if (request->metric() == "InnerProduct") {
        params.metric = kernels::Metric::InnerProduct;
    } else if (request->metric() == "Cosine") {
        params.metric = kernels::Metric::Cosine;
    } else {
        params.metric = kernels::Metric::L2;  // Default
    }
    
    // Convert queries to flat array
    size_t num_queries = request->queries().size();
    uint32_t dimension = index_state->ivf_flat->get_dimension();
    
    std::vector<float> query_vectors(num_queries * dimension);
    for (size_t q = 0; q < num_queries; ++q) {
        const auto& query = request->queries(q);
        if (query.values().size() != dimension) {
            return grpc::Status(grpc::StatusCode::INVALID_ARGUMENT,
                              "Query dimension mismatch");
        }
        
        std::copy(query.values().begin(), query.values().end(),
                 query_vectors.begin() + q * dimension);
    }
    
    // Prepare result arrays
    std::vector<float> distances(num_queries * params.k);
    std::vector<uint64_t> indices(num_queries * params.k);
    
    try {
        // Perform search
        auto start_time = std::chrono::high_resolution_clock::now();
        
        index_state->ivf_flat->search(query_vectors.data(), num_queries, 
                                     params, distances.data(), indices.data());
        
        auto end_time = std::chrono::high_resolution_clock::now();
        double latency_ms = std::chrono::duration<double, std::milli>
            (end_time - start_time).count();
        
        // Convert results to protobuf
        response->mutable_results()->Reserve(num_queries);
        
        for (size_t q = 0; q < num_queries; ++q) {
            auto* result = response->add_results();
            result->mutable_neighbors()->Reserve(params.k);
            
            for (uint32_t k = 0; k < params.k; ++k) {
                size_t idx = q * params.k + k;
                if (indices[idx] != UINT64_MAX) {
                    auto* neighbor = result->add_neighbors();
                    neighbor->set_id(indices[idx]);
                    neighbor->set_distance(distances[idx]);
                }
            }
        }
        
        // Record metrics (placeholder - would integrate with actual metrics)
        // metrics_collector_->record_search_latency(request->index(), latency_ms);
        // metrics_collector_->increment_search_count(request->index());
        
        return grpc::Status::OK;
        
    } catch (const std::exception& e) {
        return grpc::Status(grpc::StatusCode::INTERNAL, 
                          "Search failed: " + std::string(e.what()));
    }
}

grpc::Status QueryServiceImpl::Warmup(grpc::ServerContext* context,
                                     const vdb::WarmupRequest* request,
                                     google::protobuf::Empty* response) {
    // Get index
    IndexState* index_state = get_index(request->index());
    if (!index_state) {
        return grpc::Status(grpc::StatusCode::NOT_FOUND, 
                          "Index not found: " + request->index());
    }
    
    try {
        // Warmup specified lists
        if (!request->lists().empty()) {
            std::vector<uint32_t> list_ids;
            for (int32_t list_id : request->lists()) {
                if (list_id >= 0) {
                    list_ids.push_back(static_cast<uint32_t>(list_id));
                }
            }
            
            if (!list_ids.empty()) {
                index_state->ivf_flat->warmup_lists(list_ids);
            }
        } else {
            // Warmup all lists
            index_state->ivf_flat->warmup_all();
        }
        
        return grpc::Status::OK;
        
    } catch (const std::exception& e) {
        return grpc::Status(grpc::StatusCode::INTERNAL, 
                          "Warmup failed: " + std::string(e.what()));
    }
}

grpc::Status QueryServiceImpl::LoadIndex(grpc::ServerContext* context,
                                        const vdb::LoadIndexRequest* request,
                                        google::protobuf::Empty* response) {
    return load_index_internal(request->index(), request->epoch());
}

IndexState* QueryServiceImpl::get_index(const std::string& name) {
    std::shared_lock<std::shared_mutex> lock(indices_mutex_);
    auto it = indices_.find(name);
    return (it != indices_.end()) ? it->second.get() : nullptr;
}

grpc::Status QueryServiceImpl::load_index_internal(const std::string& name,
                                                  const std::string& epoch) {
    try {
        // Load epoch
        auto epoch_ptr = epoch_manager_->load_epoch(epoch);
        if (!epoch_ptr) {
            return grpc::Status(grpc::StatusCode::NOT_FOUND, 
                              "Epoch not found: " + epoch);
        }
        
        // Parse manifest
        auto manifest = epoch_ptr->get_manifest();
        
        // Create index configuration
        IVFFlatIndex::Config config;
        config.dimension = manifest.dimension;
        config.nlist = manifest.nlist;
        config.metric = (manifest.metric == "L2") ? kernels::Metric::L2 :
                       (manifest.metric == "InnerProduct") ? kernels::Metric::InnerProduct :
                       kernels::Metric::Cosine;
        config.use_gpu = true;
        config.max_gpu_memory = config_.gpu_memory_limit / 2;
        
        // Create index
        auto index = std::make_unique<IVFFlatIndex>(config, transfer_manager_.get());
        
        // Load index data
        index->load_from_epoch(epoch_ptr);
        
        // Create index state
        auto state = std::make_unique<IndexState>();
        state->ivf_flat = std::move(index);
        state->epoch = epoch_ptr;
        state->loaded_at = std::chrono::steady_clock::now();
        
        // Update indices map
        {
            std::unique_lock<std::shared_mutex> lock(indices_mutex_);
            indices_[name] = std::move(state);
        }
        
        return grpc::Status::OK;
        
    } catch (const std::exception& e) {
        return grpc::Status(grpc::StatusCode::INTERNAL,
                          "Load failed: " + std::string(e.what()));
    }
}

void QueryServiceImpl::batch_processor_loop() {
    while (!stop_) {
        std::unique_lock<std::mutex> lock(queue_mutex_);
        
        queue_cv_.wait(lock, [this] { 
            return stop_ || !search_queue_.empty(); 
        });
        
        if (stop_) break;
        
        if (!search_queue_.empty()) {
            auto batch = std::move(search_queue_.front());
            search_queue_.pop();
            lock.unlock();
            
            process_batch(batch);
        }
    }
}

void QueryServiceImpl::process_batch(BatchedSearch& batch) {
    if (batch.requests.empty()) return;
    
    // Group requests by index for better batching
    std::unordered_map<std::string, std::vector<size_t>> index_groups;
    for (size_t i = 0; i < batch.requests.size(); ++i) {
        index_groups[batch.requests[i].index()].push_back(i);
    }
    
    // Process each index group
    for (const auto& [index_name, request_indices] : index_groups) {
        IndexState* index_state = get_index(index_name);
        if (!index_state) {
            // Handle missing index error for all requests in group
            for (size_t idx : request_indices) {
                batch.promises[idx].set_value(
                    grpc::Status(grpc::StatusCode::NOT_FOUND, 
                               "Index not found: " + index_name));
            }
            continue;
        }
        
        // Batch search parameters
        std::vector<float> all_queries;
        std::vector<uint32_t> query_offsets;
        std::vector<IVFFlatIndex::SearchParams> all_params;
        uint32_t total_queries = 0;
        uint32_t dimension = index_state->ivf_flat->get_dimension();
        
        // Collect all queries for this index
        for (size_t idx : request_indices) {
            const auto& request = batch.requests[idx];
            query_offsets.push_back(total_queries);
            total_queries += request.queries().size();
            
            // Add queries to flat array
            for (const auto& query : request.queries()) {
                for (float val : query.values()) {
                    all_queries.push_back(val);
                }
            }
            
            // Collect search params
            IVFFlatIndex::SearchParams params;
            params.k = request.topk();
            params.nprobe = request.nprobe() > 0 ? request.nprobe() : 8;
            
            if (request.metric() == "L2") {
                params.metric = kernels::Metric::L2;
            } else if (request.metric() == "InnerProduct") {
                params.metric = kernels::Metric::InnerProduct;
            } else if (request.metric() == "Cosine") {
                params.metric = kernels::Metric::Cosine;
            } else {
                params.metric = kernels::Metric::L2;
            }
            
            all_params.push_back(params);
        }
        
        try {
            // Perform batched search
            auto start_time = std::chrono::high_resolution_clock::now();
            
            // Assuming uniform search params for now (could be optimized further)
            const auto& first_params = all_params[0];
            std::vector<float> distances(total_queries * first_params.k);
            std::vector<uint64_t> indices(total_queries * first_params.k);
            
            index_state->ivf_flat->search(all_queries.data(), total_queries, 
                                         first_params, distances.data(), indices.data());
            
            auto end_time = std::chrono::high_resolution_clock::now();
            double latency_ms = std::chrono::duration<double, std::milli>
                (end_time - start_time).count();
            
            // Distribute results back to individual responses
            for (size_t i = 0; i < request_indices.size(); ++i) {
                size_t idx = request_indices[i];
                auto* response = batch.responses[idx];
                const auto& request = batch.requests[idx];
                const auto& params = all_params[i];
                
                uint32_t query_start = query_offsets[i];
                uint32_t num_queries = request.queries().size();
                
                response->mutable_results()->Reserve(num_queries);
                
                for (uint32_t q = 0; q < num_queries; ++q) {
                    auto* result = response->add_results();
                    result->mutable_neighbors()->Reserve(params.k);
                    
                    for (uint32_t k = 0; k < params.k; ++k) {
                        size_t result_idx = (query_start + q) * params.k + k;
                        if (indices[result_idx] != UINT64_MAX) {
                            auto* neighbor = result->add_neighbors();
                            neighbor->set_id(indices[result_idx]);
                            neighbor->set_distance(distances[result_idx]);
                        }
                    }
                }
                
                batch.promises[idx].set_value(grpc::Status::OK);
            }
            
        } catch (const std::exception& e) {
            // Handle batch error
            grpc::Status error_status(grpc::StatusCode::INTERNAL, 
                                    "Batch search failed: " + std::string(e.what()));
            for (size_t idx : request_indices) {
                batch.promises[idx].set_value(error_status);
            }
        }
    }
}

// AdminServiceImpl implementation
AdminServiceImpl::AdminServiceImpl(QueryServiceImpl* query_service)
    : query_service_(query_service) {
}

AdminServiceImpl::~AdminServiceImpl() {
    // Wait for all build jobs to complete
    std::lock_guard<std::mutex> lock(jobs_mutex_);
    for (auto& [name, job] : build_jobs_) {
        job->running = false;
        if (job->worker.joinable()) {
            job->worker.join();
        }
    }
}

grpc::Status AdminServiceImpl::CreateIndex(grpc::ServerContext* context,
                                          const vdb::CreateIndexRequest* request,
                                          google::protobuf::Empty* response) {
    // Validate request
    if (request->name().empty()) {
        return grpc::Status(grpc::StatusCode::INVALID_ARGUMENT,
                          "Index name required");
    }
    
    if (request->dimension() <= 0 || request->dimension() > 65536) {
        return grpc::Status(grpc::StatusCode::INVALID_ARGUMENT,
                          "Invalid dimension");
    }
    
    try {
        // Create index directory structure
        std::string index_path = query_service_->config_.data_path + "/" + request->name();
        std::filesystem::create_directories(index_path);
        
        // Save index metadata
        storage::IndexManifest manifest;
        manifest.name = request->name();
        manifest.dimension = request->dimension();
        manifest.metric = request->metric();
        manifest.nlist = request->nlist() > 0 ? request->nlist() : 
                        std::min(4096U, static_cast<uint32_t>(std::sqrt(1000000)));
        manifest.m = request->m();
        manifest.nbits = request->nbits() > 0 ? request->nbits() : 8;
        manifest.created_at = std::chrono::system_clock::now();
        
        // Save manifest
        std::string manifest_path = index_path + "/manifest.json";
        std::ofstream file(manifest_path);
        if (file) {
            // Would serialize manifest to JSON here
            file << "{\n";
            file << "  \"name\": \"" << manifest.name << "\",\n";
            file << "  \"dimension\": " << manifest.dimension << ",\n";
            file << "  \"metric\": \"" << manifest.metric << "\",\n";
            file << "  \"nlist\": " << manifest.nlist << ",\n";
            file << "  \"m\": " << manifest.m << ",\n";
            file << "  \"nbits\": " << manifest.nbits << "\n";
            file << "}\n";
        }
        
        return grpc::Status::OK;
        
    } catch (const std::exception& e) {
        return grpc::Status(grpc::StatusCode::INTERNAL,
                          "Create index failed: " + std::string(e.what()));
    }
}

grpc::Status AdminServiceImpl::BuildEpoch(grpc::ServerContext* context,
                                         const vdb::BuildEpochRequest* request,
                                         google::protobuf::Empty* response) {
    // Validate request
    if (request->index().empty() || request->source_path().empty()) {
        return grpc::Status(grpc::StatusCode::INVALID_ARGUMENT,
                          "Index name and source path required");
    }
    
    try {
        // Check if build is already in progress
        {
            std::lock_guard<std::mutex> lock(jobs_mutex_);
            if (build_jobs_.find(request->index()) != build_jobs_.end()) {
                return grpc::Status(grpc::StatusCode::ALREADY_EXISTS,
                                  "Build already in progress");
            }
        }
        
        // Create build job
        auto job = std::make_unique<BuildJob>();
        job->index_name = request->index();
        job->source_path = request->source_path();
        job->epoch_id = std::to_string(std::chrono::system_clock::now()
                                      .time_since_epoch().count());
        
        // Start build worker
        job->worker = std::thread(&AdminServiceImpl::build_index_worker, this, job.get());
        
        // Store job
        {
            std::lock_guard<std::mutex> lock(jobs_mutex_);
            build_jobs_[request->index()] = std::move(job);
        }
        
        return grpc::Status::OK;
        
    } catch (const std::exception& e) {
        return grpc::Status(grpc::StatusCode::INTERNAL,
                          "Build epoch failed: " + std::string(e.what()));
    }
}

grpc::Status AdminServiceImpl::ActivateEpoch(grpc::ServerContext* context,
                                            const vdb::ActivateEpochRequest* request,
                                            google::protobuf::Empty* response) {
    return query_service_->load_index_internal(request->index(), request->epoch());
}

grpc::Status AdminServiceImpl::GetStats(grpc::ServerContext* context,
                                       const vdb::StatsRequest* request,
                                       vdb::StatsResponse* response) {
    // Get index state
    IndexState* state = query_service_->get_index(request->index());
    if (!state) {
        return grpc::Status(grpc::StatusCode::NOT_FOUND,
                          "Index not found: " + request->index());
    }
    
    // Populate response
    if (state->epoch) {
        auto manifest = state->epoch->get_manifest();
        response->set_total_vectors(manifest.num_vectors);
        response->set_indexed_vectors(manifest.num_vectors);
    }
    
    if (state->ivf_flat) {
        response->set_gpu_memory_used(
            static_cast<float>(state->ivf_flat->get_gpu_memory_usage()) / (1024*1024*1024));
    }
    
    // Would add more detailed stats here
    response->set_nvme_usage(0.0f);
    
    return grpc::Status::OK;
}

void AdminServiceImpl::build_index_worker(BuildJob* job) {
    try {
        // Load source data (simplified - would use proper Arrow loading)
        job->progress = 0.1f;
        
        // Create index configuration based on manifest
        IVFFlatIndex::Config config;
        config.dimension = 128;  // Would read from manifest
        config.nlist = 128;
        config.metric = kernels::Metric::L2;
        config.use_gpu = true;
        
        // Create index
        TransferManager::Config tm_config;
        auto tm = std::make_unique<TransferManager>(tm_config);
        IVFFlatIndex index(config, tm.get());
        
        job->progress = 0.2f;
        
        // Train index (placeholder)
        // index.train(training_vectors, num_training);
        job->progress = 0.6f;
        
        // Add vectors (placeholder)
        // index.add(all_vectors, ids, num_vectors);
        job->progress = 0.9f;
        
        // Save epoch
        // index.save_to_epoch(job->epoch_id);
        job->progress = 1.0f;
        
    } catch (const std::exception& e) {
        // Log error
        job->running = false;
    }
}

// RequestCoalescer implementation
RequestCoalescer::RequestCoalescer(const Config& config) 
    : config_(config) {
}

RequestCoalescer::~RequestCoalescer() {
    stop();
}

void RequestCoalescer::start() {
    if (!running_) {
        running_ = true;
        processor_ = std::thread(&RequestCoalescer::processor_loop, this);
    }
}

void RequestCoalescer::stop() {
    if (running_) {
        running_ = false;
        queue_cv_.notify_all();
        if (processor_.joinable()) {
            processor_.join();
        }
    }
}

void RequestCoalescer::processor_loop() {
    while (running_) {
        std::unique_lock<std::mutex> lock(queue_mutex_);
        
        queue_cv_.wait_for(lock, config_.window_duration, [this] {
            return !running_ || queue_.size() >= config_.max_batch_size;
        });
        
        if (!running_) break;
        
        // Process current batch
        std::vector<PendingRequest> batch;
        while (!queue_.empty() && batch.size() < config_.max_batch_size) {
            batch.push_back(std::move(queue_.front()));
            queue_.pop();
        }
        
        lock.unlock();
        
        // Execute batch
        for (auto& req : batch) {
            req.process_func();
        }
    }
}

// RateLimiter implementation
RateLimiter::RateLimiter(const Config& config) 
    : max_tokens_(config.burst_size), 
      tokens_(config.burst_size),
      last_refill_(std::chrono::steady_clock::now()),
      refill_period_(std::chrono::seconds(1) / config.requests_per_second) {
}

bool RateLimiter::try_acquire(size_t tokens) {
    std::lock_guard<std::mutex> lock(mutex_);
    refill();
    
    if (tokens_ >= tokens) {
        tokens_ -= tokens;
        return true;
    }
    return false;
}

void RateLimiter::acquire(size_t tokens) {
    while (!try_acquire(tokens)) {
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
}

void RateLimiter::update_rate(size_t requests_per_second) {
    std::lock_guard<std::mutex> lock(mutex_);
    refill_period_ = std::chrono::seconds(1) / requests_per_second;
}

void RateLimiter::refill() {
    auto now = std::chrono::steady_clock::now();
    auto elapsed = now - last_refill_;
    
    if (elapsed >= refill_period_) {
        auto tokens_to_add = elapsed / refill_period_;
        tokens_ = std::min(max_tokens_, tokens_ + static_cast<size_t>(tokens_to_add));
        last_refill_ = now;
    }
}

// MetricsCollector implementation
MetricsCollector::MetricsCollector() 
    : start_time_(std::chrono::steady_clock::now()) {
}

MetricsCollector::~MetricsCollector() = default;

void MetricsCollector::record_search_latency(const std::string& index, double latency_ms) {
    std::lock_guard<std::mutex> lock(mutex_);
    latencies_[index].add(latency_ms);
}

void MetricsCollector::record_search_recall(const std::string& index, float recall) {
    std::lock_guard<std::mutex> lock(mutex_);
    recalls_[index].push_back(recall);
}

void MetricsCollector::increment_search_count(const std::string& index) {
    std::lock_guard<std::mutex> lock(mutex_);
    search_counts_[index]++;
}

void MetricsCollector::record_gpu_memory(size_t bytes) {
    gpu_memory_bytes_ = bytes;
}

void MetricsCollector::record_nvme_bandwidth(double gbps) {
    nvme_bandwidth_gbps_ = gbps;
}

MetricsCollector::Metrics MetricsCollector::get_metrics() const {
    std::lock_guard<std::mutex> lock(mutex_);
    
    Metrics metrics;
    
    for (const auto& [index, histogram] : latencies_) {
        auto& index_metrics = metrics.indices[index];
        index_metrics.search_count = search_counts_.at(index);
        index_metrics.avg_latency_ms = histogram.percentile(0.5);
        index_metrics.p50_latency_ms = histogram.percentile(0.5);
        index_metrics.p95_latency_ms = histogram.percentile(0.95);
        index_metrics.p99_latency_ms = histogram.percentile(0.99);
        
        if (recalls_.count(index)) {
            const auto& recall_vec = recalls_.at(index);
            if (!recall_vec.empty()) {
                float sum = std::accumulate(recall_vec.begin(), recall_vec.end(), 0.0f);
                index_metrics.avg_recall = sum / recall_vec.size();
            }
        }
    }
    
    metrics.gpu_memory_bytes = gpu_memory_bytes_;
    metrics.nvme_bandwidth_gbps = nvme_bandwidth_gbps_;
    
    // Calculate QPS
    auto elapsed = std::chrono::steady_clock::now() - start_time_;
    double elapsed_seconds = std::chrono::duration<double>(elapsed).count();
    
    uint64_t total_searches = 0;
    for (const auto& [index, count] : search_counts_) {
        total_searches += count;
    }
    
    metrics.qps = total_searches / elapsed_seconds;
    
    return metrics;
}

std::string MetricsCollector::prometheus_format() const {
    auto metrics = get_metrics();
    
    std::ostringstream ss;
    ss << "# HELP vdb_search_duration_milliseconds Search latency in milliseconds\n";
    ss << "# TYPE vdb_search_duration_milliseconds histogram\n";
    
    for (const auto& [index, index_metrics] : metrics.indices) {
        ss << "vdb_search_duration_milliseconds{index=\"" << index << "\",quantile=\"0.5\"} "
           << index_metrics.p50_latency_ms << "\n";
        ss << "vdb_search_duration_milliseconds{index=\"" << index << "\",quantile=\"0.95\"} "
           << index_metrics.p95_latency_ms << "\n";
        ss << "vdb_search_duration_milliseconds{index=\"" << index << "\",quantile=\"0.99\"} "
           << index_metrics.p99_latency_ms << "\n";
    }
    
    ss << "# HELP vdb_searches_total Total number of searches\n";
    ss << "# TYPE vdb_searches_total counter\n";
    for (const auto& [index, index_metrics] : metrics.indices) {
        ss << "vdb_searches_total{index=\"" << index << "\"} " 
           << index_metrics.search_count << "\n";
    }
    
    ss << "# HELP vdb_gpu_memory_bytes GPU memory usage in bytes\n";
    ss << "# TYPE vdb_gpu_memory_bytes gauge\n";
    ss << "vdb_gpu_memory_bytes " << metrics.gpu_memory_bytes << "\n";
    
    ss << "# HELP vdb_queries_per_second Current queries per second\n";
    ss << "# TYPE vdb_queries_per_second gauge\n";
    ss << "vdb_queries_per_second " << metrics.qps << "\n";
    
    return ss.str();
}

void MetricsCollector::LatencyHistogram::add(double value) {
    values.push_back(value);
    if (values.size() > 10000) {
        // Keep only recent values
        values.erase(values.begin(), values.begin() + 5000);
    }
}

double MetricsCollector::LatencyHistogram::percentile(double p) const {
    if (values.empty()) return 0.0;
    
    auto sorted_values = values;
    std::sort(sorted_values.begin(), sorted_values.end());
    
    size_t index = static_cast<size_t>(p * (sorted_values.size() - 1));
    return sorted_values[index];
}

}
}