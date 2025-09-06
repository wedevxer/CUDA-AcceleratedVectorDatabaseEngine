#pragma once

#include <liburing.h>
#include <memory>
#include <vector>
#include <queue>
#include <thread>
#include <atomic>
#include <functional>
#include <unordered_map>

namespace vdb {

class IOUringPrefetcher {
public:
    struct Config {
        size_t queue_depth = 128;
        size_t max_batch_size = 32;
        size_t alignment = 4096;
        size_t read_ahead_size = 16 << 20;
        int sq_thread_cpu = -1;
        bool use_sqpoll = true;
        bool use_fixed_buffers = true;
        size_t fixed_buffer_size = 1 << 30;
    };
    
    struct IORequest {
        int fd;
        off_t offset;
        size_t size;
        void* buffer;
        std::function<void(int, size_t)> callback;
        uint64_t submit_time_ns;
    };
    
    explicit IOUringPrefetcher(const Config& config);
    ~IOUringPrefetcher();
    
    void* allocate_buffer(size_t size);
    void free_buffer(void* buffer);
    
    void submit_read(const IORequest& request);
    void submit_batch(const std::vector<IORequest>& requests);
    
    void prefetch(int fd, off_t offset, size_t size);
    void prefetch_pattern(int fd, const std::vector<std::pair<off_t, size_t>>& pattern);
    
    void wait_completion(int min_complete = 1);
    void process_completions();
    
    size_t get_pending_ios() const { return pending_ios_.load(); }
    double get_avg_latency_us() const;
    
private:
    Config config_;
    io_uring ring_;
    
    std::atomic<size_t> pending_ios_{0};
    std::atomic<uint64_t> total_latency_ns_{0};
    std::atomic<uint64_t> completed_ios_{0};
    
    struct BufferPool {
        void* base;
        size_t total_size;
        std::vector<bool> allocated;
        size_t block_size;
        std::mutex mutex;
        
        BufferPool(size_t size, size_t block_sz);
        ~BufferPool();
        
        void* allocate(size_t size);
        void free(void* ptr);
    };
    
    std::unique_ptr<BufferPool> buffer_pool_;
    std::unordered_map<uint64_t, IORequest> inflight_requests_;
    std::mutex requests_mutex_;
    
    std::thread completion_thread_;
    std::atomic<bool> stop_{false};
    
    void completion_loop();
    int setup_ring();
    void register_buffers();
    
    uint64_t get_time_ns() const;
};

class AdaptivePrefetcher {
public:
    struct AccessPattern {
        enum Type {
            Sequential,
            Random,
            Strided
        };
        
        Type type;
        size_t stride;
        size_t avg_size;
        double locality_score;
    };
    
    AdaptivePrefetcher(IOUringPrefetcher* io_prefetcher);
    ~AdaptivePrefetcher();
    
    void record_access(int fd, off_t offset, size_t size);
    
    void prefetch_adaptive(int fd, off_t current_offset, size_t current_size);
    
    AccessPattern analyze_pattern(int fd) const;
    
    void set_prefetch_depth(size_t depth) { prefetch_depth_ = depth; }
    void set_prefetch_distance(size_t distance) { prefetch_distance_ = distance; }
    
private:
    IOUringPrefetcher* io_prefetcher_;
    
    struct FileAccessHistory {
        std::vector<off_t> offsets;
        std::vector<size_t> sizes;
        std::vector<uint64_t> timestamps;
        AccessPattern pattern;
        uint64_t last_analysis_time;
    };
    
    std::unordered_map<int, FileAccessHistory> access_history_;
    std::mutex history_mutex_;
    
    size_t prefetch_depth_ = 4;
    size_t prefetch_distance_ = 2;
    
    void update_pattern(FileAccessHistory& history);
    std::vector<std::pair<off_t, size_t>> predict_next_accesses(
        const FileAccessHistory& history, off_t current_offset);
};

class ListPrefetcher {
public:
    struct Config {
        size_t max_prefetch_lists = 16;
        size_t list_chunk_size = 4 << 20;
        double prefetch_threshold = 0.7;
    };
    
    ListPrefetcher(const Config& config, 
                  IOUringPrefetcher* io_prefetcher,
                  AdaptivePrefetcher* adaptive_prefetcher);
    ~ListPrefetcher();
    
    void register_list_file(uint32_t list_id, int fd, 
                           off_t offset, size_t size);
    
    void access_list(uint32_t list_id);
    
    void prefetch_lists(const std::vector<uint32_t>& list_ids);
    
    void prefetch_hot_lists();
    
    std::vector<uint32_t> get_hot_lists(size_t n) const;
    
private:
    Config config_;
    IOUringPrefetcher* io_prefetcher_;
    AdaptivePrefetcher* adaptive_prefetcher_;
    
    struct ListInfo {
        int fd;
        off_t offset;
        size_t size;
        std::atomic<uint64_t> access_count{0};
        std::chrono::steady_clock::time_point last_access;
        bool prefetched = false;
    };
    
    std::unordered_map<uint32_t, ListInfo> lists_;
    std::priority_queue<std::pair<double, uint32_t>> hot_lists_;
    mutable std::shared_mutex mutex_;
    
    double calculate_hotness_score(const ListInfo& info) const;
    void update_hot_lists();
};

class PrefetchScheduler {
public:
    struct Config {
        size_t max_concurrent_prefetches = 32;
        size_t prefetch_window_ms = 10;
        double bandwidth_limit_gbps = 10.0;
    };
    
    PrefetchScheduler(const Config& config);
    ~PrefetchScheduler();
    
    void schedule_prefetch(uint32_t list_id, int priority,
                          std::function<void()> prefetch_func);
    
    void execute();
    
    void pause();
    void resume();
    
    double get_bandwidth_usage_gbps() const;
    
private:
    Config config_;
    
    struct PrefetchTask {
        uint32_t list_id;
        int priority;
        std::function<void()> func;
        uint64_t scheduled_time_ns;
    };
    
    std::priority_queue<PrefetchTask> task_queue_;
    std::mutex queue_mutex_;
    std::condition_variable queue_cv_;
    
    std::atomic<bool> paused_{false};
    std::atomic<bool> stop_{false};
    
    std::atomic<uint64_t> bytes_transferred_{0};
    std::atomic<uint64_t> transfer_start_time_ns_{0};
    
    std::thread scheduler_thread_;
    
    void scheduler_loop();
    bool should_throttle() const;
};

}