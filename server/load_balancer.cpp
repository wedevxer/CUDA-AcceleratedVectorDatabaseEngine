#include "query_service.h"
#include <chrono>
#include <algorithm>

namespace vdb {
namespace server {

class LoadBalancer {
public:
    struct Config {
        size_t max_concurrent_requests = 100;
        double circuit_breaker_threshold = 0.5;  // 50% error rate
        std::chrono::milliseconds circuit_breaker_timeout{30000};  // 30s
        std::chrono::milliseconds health_check_interval{5000};     // 5s
    };
    
    explicit LoadBalancer(const Config& config) : config_(config) {
        // Initialize worker pools for different request types
        search_pool_ = std::make_unique<ThreadPool>(config_.max_concurrent_requests / 2);
        admin_pool_ = std::make_unique<ThreadPool>(config_.max_concurrent_requests / 4);
        warmup_pool_ = std::make_unique<ThreadPool>(config_.max_concurrent_requests / 4);
        
        // Start health checker
        health_checker_ = std::thread(&LoadBalancer::health_check_loop, this);
    }
    
    ~LoadBalancer() {
        stop_ = true;
        if (health_checker_.joinable()) {
            health_checker_.join();
        }
    }
    
    // Circuit breaker pattern for request handling
    template<typename Request, typename Response>
    grpc::Status handle_request(const std::string& service_name,
                               std::function<grpc::Status(const Request*, Response*)> handler,
                               const Request* request, Response* response) {
        
        // Check circuit breaker
        if (is_circuit_open(service_name)) {
            return grpc::Status(grpc::StatusCode::UNAVAILABLE, 
                              "Service temporarily unavailable");
        }
        
        // Check current load
        if (current_requests_.load() >= config_.max_concurrent_requests) {
            record_error(service_name);
            return grpc::Status(grpc::StatusCode::RESOURCE_EXHAUSTED, 
                              "Server overloaded");
        }
        
        // Execute request with monitoring
        current_requests_++;
        auto start_time = std::chrono::steady_clock::now();
        
        grpc::Status status = handler(request, response);
        
        auto end_time = std::chrono::steady_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
        
        current_requests_--;
        
        // Record metrics
        if (status.ok()) {
            record_success(service_name, duration);
        } else {
            record_error(service_name);
        }
        
        return status;
    }
    
    // Adaptive batching based on current load
    size_t get_optimal_batch_size() const {
        double load_factor = static_cast<double>(current_requests_) / config_.max_concurrent_requests;
        
        if (load_factor < 0.3) {
            return 16;  // Small batches for low load
        } else if (load_factor < 0.7) {
            return 32;  // Medium batches for medium load
        } else {
            return 64;  // Large batches for high load
        }
    }
    
    // Adaptive timeout based on historical performance
    std::chrono::milliseconds get_request_timeout(const std::string& service_name) const {
        std::lock_guard<std::mutex> lock(metrics_mutex_);
        
        auto it = service_metrics_.find(service_name);
        if (it == service_metrics_.end()) {
            return std::chrono::milliseconds(5000);  // Default 5s
        }
        
        // Set timeout to 3x average latency, but at least 1s, at most 30s
        auto timeout = std::max(std::chrono::milliseconds(1000),
                               std::min(std::chrono::milliseconds(30000),
                                       it->second.avg_latency * 3));
        return timeout;
    }
    
private:
    Config config_;
    std::atomic<size_t> current_requests_{0};
    std::atomic<bool> stop_{false};
    
    // Thread pools for different operations
    class ThreadPool {
    public:
        ThreadPool(size_t num_threads) : stop_(false) {
            for (size_t i = 0; i < num_threads; ++i) {
                workers_.emplace_back([this] {
                    while (!stop_) {
                        std::function<void()> task;
                        {
                            std::unique_lock<std::mutex> lock(queue_mutex_);
                            cv_.wait(lock, [this] { return stop_ || !tasks_.empty(); });
                            
                            if (stop_ && tasks_.empty()) break;
                            
                            task = std::move(tasks_.front());
                            tasks_.pop();
                        }
                        task();
                    }
                });
            }
        }
        
        ~ThreadPool() {
            stop_ = true;
            cv_.notify_all();
            for (auto& worker : workers_) {
                if (worker.joinable()) {
                    worker.join();
                }
            }
        }
        
        template<typename F>
        auto enqueue(F&& f) -> std::future<typename std::result_of<F()>::type> {
            using return_type = typename std::result_of<F()>::type;
            
            auto task = std::make_shared<std::packaged_task<return_type()>>(
                std::forward<F>(f));
                
            std::future<return_type> result = task->get_future();
            
            {
                std::unique_lock<std::mutex> lock(queue_mutex_);
                if (stop_) {
                    throw std::runtime_error("ThreadPool is stopped");
                }
                tasks_.emplace([task] { (*task)(); });
            }
            
            cv_.notify_one();
            return result;
        }
        
    private:
        std::vector<std::thread> workers_;
        std::queue<std::function<void()>> tasks_;
        std::mutex queue_mutex_;
        std::condition_variable cv_;
        std::atomic<bool> stop_;
    };
    
    std::unique_ptr<ThreadPool> search_pool_;
    std::unique_ptr<ThreadPool> admin_pool_;
    std::unique_ptr<ThreadPool> warmup_pool_;
    
    // Circuit breaker state
    struct ServiceMetrics {
        std::atomic<size_t> success_count{0};
        std::atomic<size_t> error_count{0};
        std::atomic<bool> circuit_open{false};
        std::chrono::steady_clock::time_point last_failure;
        std::chrono::milliseconds avg_latency{0};
        
        double error_rate() const {
            size_t total = success_count + error_count;
            return total == 0 ? 0.0 : static_cast<double>(error_count) / total;
        }
    };
    
    mutable std::mutex metrics_mutex_;
    std::unordered_map<std::string, ServiceMetrics> service_metrics_;
    
    std::thread health_checker_;
    
    bool is_circuit_open(const std::string& service_name) const {
        std::lock_guard<std::mutex> lock(metrics_mutex_);
        auto it = service_metrics_.find(service_name);
        
        if (it == service_metrics_.end()) return false;
        
        // Check if circuit should be closed (recovery)
        if (it->second.circuit_open) {
            auto now = std::chrono::steady_clock::now();
            if (now - it->second.last_failure > config_.circuit_breaker_timeout) {
                it->second.circuit_open = false;  // Try to recover
                return false;
            }
            return true;
        }
        
        return false;
    }
    
    void record_success(const std::string& service_name, 
                       std::chrono::milliseconds latency) {
        std::lock_guard<std::mutex> lock(metrics_mutex_);
        auto& metrics = service_metrics_[service_name];
        
        metrics.success_count++;
        
        // Update average latency (exponential moving average)
        if (metrics.avg_latency.count() == 0) {
            metrics.avg_latency = latency;
        } else {
            // EMA with alpha = 0.1
            metrics.avg_latency = std::chrono::milliseconds(
                static_cast<long>(metrics.avg_latency.count() * 0.9 + latency.count() * 0.1));
        }
        
        // Reset circuit if error rate drops below threshold
        if (metrics.error_rate() < config_.circuit_breaker_threshold) {
            metrics.circuit_open = false;
        }
    }
    
    void record_error(const std::string& service_name) {
        std::lock_guard<std::mutex> lock(metrics_mutex_);
        auto& metrics = service_metrics_[service_name];
        
        metrics.error_count++;
        metrics.last_failure = std::chrono::steady_clock::now();
        
        // Open circuit if error rate exceeds threshold
        if (metrics.error_rate() > config_.circuit_breaker_threshold) {
            metrics.circuit_open = true;
        }
    }
    
    void health_check_loop() {
        while (!stop_) {
            std::this_thread::sleep_for(config_.health_check_interval);
            
            // Decay old metrics to allow recovery
            std::lock_guard<std::mutex> lock(metrics_mutex_);
            for (auto& [service_name, metrics] : service_metrics_) {
                // Decay counters every health check interval
                metrics.success_count = metrics.success_count * 0.95;
                metrics.error_count = metrics.error_count * 0.95;
                
                // Auto-recovery for circuits that have been open too long
                if (metrics.circuit_open) {
                    auto now = std::chrono::steady_clock::now();
                    if (now - metrics.last_failure > config_.circuit_breaker_timeout * 2) {
                        metrics.circuit_open = false;
                        metrics.error_count = 0;
                        metrics.success_count = 1;  // Give it a chance
                    }
                }
            }
        }
    }
};

// Request queue with priority scheduling
class PriorityRequestQueue {
public:
    enum class Priority {
        LOW = 0,
        NORMAL = 1,  
        HIGH = 2,
        URGENT = 3
    };
    
    struct QueuedRequest {
        Priority priority;
        std::chrono::steady_clock::time_point enqueue_time;
        std::function<void()> handler;
        
        bool operator<(const QueuedRequest& other) const {
            // Higher priority first, then FIFO within same priority
            if (priority != other.priority) {
                return priority < other.priority;
            }
            return enqueue_time > other.enqueue_time;
        }
    };
    
    void enqueue(Priority priority, std::function<void()> handler) {
        std::lock_guard<std::mutex> lock(mutex_);
        
        queue_.emplace(QueuedRequest{
            priority, 
            std::chrono::steady_clock::now(),
            std::move(handler)
        });
        
        cv_.notify_one();
    }
    
    bool try_dequeue(QueuedRequest& request, std::chrono::milliseconds timeout) {
        std::unique_lock<std::mutex> lock(mutex_);
        
        if (cv_.wait_for(lock, timeout, [this] { return !queue_.empty(); })) {
            request = queue_.top();
            queue_.pop();
            return true;
        }
        
        return false;
    }
    
    size_t size() const {
        std::lock_guard<std::mutex> lock(mutex_);
        return queue_.size();
    }
    
private:
    mutable std::mutex mutex_;
    std::condition_variable cv_;
    std::priority_queue<QueuedRequest> queue_;
};

}
}