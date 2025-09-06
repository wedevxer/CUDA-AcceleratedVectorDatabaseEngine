#include "prefetcher.h"
#include <sys/mman.h>
#include <fcntl.h>
#include <unistd.h>
#include <iostream>
#include <algorithm>
#include <cstring>

// Only compile io_uring code on Linux
#ifdef __linux__
#include <liburing.h>
#else
// Stub implementations for non-Linux platforms
struct io_uring {};
static int io_uring_queue_init(unsigned entries, struct io_uring *ring, unsigned flags) { return -1; }
static void io_uring_queue_exit(struct io_uring *ring) {}
static int io_uring_submit(struct io_uring *ring) { return -1; }
static int io_uring_wait_cqe(struct io_uring *ring, struct io_uring_cqe **cqe_ptr) { return -1; }
static void io_uring_cqe_seen(struct io_uring *ring, struct io_uring_cqe *cqe) {}
struct io_uring_sqe {};
struct io_uring_cqe { int res; void *user_data; };
static struct io_uring_sqe *io_uring_get_sqe(struct io_uring *ring) { return nullptr; }
static void io_uring_prep_readv(struct io_uring_sqe *sqe, int fd, const struct iovec *iovecs, unsigned nr_vecs, off_t offset) {}
static void io_uring_sqe_set_data(struct io_uring_sqe *sqe, void *data) {}
#define IORING_SETUP_SQPOLL 0
#define IORING_SETUP_IOPOLL 0
#endif

namespace vdb {

// BufferPool implementation for IOUringPrefetcher
IOUringPrefetcher::BufferPool::BufferPool(size_t size, size_t block_sz) 
    : total_size_(size), block_size_(block_sz) {
    
    // Calculate number of blocks
    size_t num_blocks = size / block_sz;
    allocated.resize(num_blocks, false);
    
    // Allocate aligned memory
#ifdef __linux__
    base_ = mmap(nullptr, size, PROT_READ | PROT_WRITE, 
                MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    if (base_ == MAP_FAILED) {
        throw std::runtime_error("Failed to allocate buffer pool memory");
    }
#else
    // Fallback to regular allocation on non-Linux
    base_ = aligned_alloc(4096, size);
    if (!base_) {
        throw std::runtime_error("Failed to allocate buffer pool memory");
    }
#endif
}

IOUringPrefetcher::BufferPool::~BufferPool() {
    if (base_) {
#ifdef __linux__
        munmap(base_, total_size_);
#else
        free(base_);
#endif
    }
}

void* IOUringPrefetcher::BufferPool::allocate(size_t size) {
    std::lock_guard<std::mutex> lock(mutex);
    
    // Round up to block size
    size_t blocks_needed = (size + block_size_ - 1) / block_size_;
    
    // Find contiguous free blocks
    for (size_t i = 0; i <= allocated.size() - blocks_needed; ++i) {
        bool available = true;
        
        // Check if blocks are available
        for (size_t j = 0; j < blocks_needed; ++j) {
            if (allocated[i + j]) {
                available = false;
                break;
            }
        }
        
        if (available) {
            // Mark blocks as allocated
            for (size_t j = 0; j < blocks_needed; ++j) {
                allocated[i + j] = true;
            }
            
            return static_cast<char*>(base_) + i * block_size_;
        }
    }
    
    return nullptr;  // No space available
}

void IOUringPrefetcher::BufferPool::free(void* ptr) {
    if (!ptr) return;
    
    std::lock_guard<std::mutex> lock(mutex);
    
    // Calculate block index
    size_t offset = static_cast<char*>(ptr) - static_cast<char*>(base_);
    size_t block_idx = offset / block_size_;
    
    if (block_idx < allocated.size()) {
        // Find extent of allocation (assumes contiguous allocation)
        size_t start_block = block_idx;
        while (block_idx < allocated.size() && allocated[block_idx]) {
            allocated[block_idx] = false;
            ++block_idx;
        }
    }
}

// IOUringPrefetcher implementation
IOUringPrefetcher::IOUringPrefetcher(const Config& config) : config_(config) {
#ifdef __linux__
    // Initialize buffer pool
    buffer_pool_ = std::make_unique<BufferPool>(
        config_.fixed_buffer_size, config_.alignment);
    
    // Setup io_uring
    if (setup_ring() != 0) {
        throw std::runtime_error("Failed to setup io_uring");
    }
    
    // Register fixed buffers if enabled
    if (config_.use_fixed_buffers) {
        register_buffers();
    }
    
    // Start completion thread
    completion_thread_ = std::thread(&IOUringPrefetcher::completion_loop, this);
    
    std::cout << "IOUringPrefetcher initialized with queue depth " 
              << config_.queue_depth << std::endl;
#else
    std::cout << "IOUringPrefetcher: io_uring not available on this platform, "
              << "falling back to standard I/O" << std::endl;
#endif
}

IOUringPrefetcher::~IOUringPrefetcher() {
    stop_ = true;
    
#ifdef __linux__
    if (completion_thread_.joinable()) {
        completion_thread_.join();
    }
    
    io_uring_queue_exit(&ring_);
#endif
}

void* IOUringPrefetcher::allocate_buffer(size_t size) {
    if (buffer_pool_) {
        return buffer_pool_->allocate(size);
    }
    return nullptr;
}

void IOUringPrefetcher::free_buffer(void* buffer) {
    if (buffer_pool_) {
        buffer_pool_->free(buffer);
    }
}

void IOUringPrefetcher::submit_read(const IORequest& request) {
#ifdef __linux__
    std::lock_guard<std::mutex> lock(requests_mutex_);
    
    struct io_uring_sqe* sqe = io_uring_get_sqe(&ring_);
    if (!sqe) {
        std::cerr << "Failed to get io_uring SQE" << std::endl;
        return;
    }
    
    // Create iovec for the read
    struct iovec iov;
    iov.iov_base = request.buffer;
    iov.iov_len = request.size;
    
    // Setup read operation
    io_uring_prep_readv(sqe, request.fd, &iov, 1, request.offset);
    
    // Store request info for completion
    uint64_t request_id = reinterpret_cast<uint64_t>(&request);
    inflight_requests_[request_id] = request;
    io_uring_sqe_set_data(sqe, reinterpret_cast<void*>(request_id));
    
    pending_ios_++;
    
    // Submit immediately for small batches
    if (pending_ios_.load() >= config_.max_batch_size) {
        io_uring_submit(&ring_);
    }
#else
    // Fallback: synchronous read
    ssize_t bytes_read = pread(request.fd, request.buffer, request.size, request.offset);
    if (request.callback) {
        request.callback(bytes_read >= 0 ? 0 : -1, bytes_read >= 0 ? bytes_read : 0);
    }
#endif
}

void IOUringPrefetcher::submit_batch(const std::vector<IORequest>& requests) {
    for (const auto& request : requests) {
        submit_read(request);
    }
    
#ifdef __linux__
    // Submit all pending requests
    io_uring_submit(&ring_);
#endif
}

void IOUringPrefetcher::prefetch(int fd, off_t offset, size_t size) {
    // Allocate buffer for prefetch
    void* buffer = allocate_buffer(size);
    if (!buffer) {
        std::cerr << "Failed to allocate buffer for prefetch" << std::endl;
        return;
    }
    
    IORequest request;
    request.fd = fd;
    request.offset = offset;
    request.size = size;
    request.buffer = buffer;
    request.submit_time_ns = get_time_ns();
    request.callback = [this, buffer](int result, size_t bytes_transferred) {
        // Free buffer when done (fire and forget prefetch)
        free_buffer(buffer);
    };
    
    submit_read(request);
}

void IOUringPrefetcher::prefetch_pattern(int fd, 
    const std::vector<std::pair<off_t, size_t>>& pattern) {
    
    std::vector<IORequest> requests;
    requests.reserve(pattern.size());
    
    for (const auto& [offset, size] : pattern) {
        void* buffer = allocate_buffer(size);
        if (!buffer) continue;
        
        IORequest request;
        request.fd = fd;
        request.offset = offset;
        request.size = size;
        request.buffer = buffer;
        request.submit_time_ns = get_time_ns();
        request.callback = [this, buffer](int result, size_t bytes_transferred) {
            free_buffer(buffer);
        };
        
        requests.push_back(request);
    }
    
    submit_batch(requests);
}

void IOUringPrefetcher::wait_completion(int min_complete) {
#ifdef __linux__
    for (int i = 0; i < min_complete; ++i) {
        struct io_uring_cqe* cqe;
        int ret = io_uring_wait_cqe(&ring_, &cqe);
        if (ret < 0) {
            std::cerr << "io_uring_wait_cqe failed: " << strerror(-ret) << std::endl;
            return;
        }
        
        process_completion(cqe);
        io_uring_cqe_seen(&ring_, cqe);
    }
#endif
}

void IOUringPrefetcher::process_completions() {
#ifdef __linux__
    struct io_uring_cqe* cqe;
    
    // Process all available completions
    while (io_uring_wait_cqe(&ring_, &cqe) == 0) {
        process_completion(cqe);
        io_uring_cqe_seen(&ring_, cqe);
    }
#endif
}

double IOUringPrefetcher::get_avg_latency_us() const {
    uint64_t completed = completed_ios_.load();
    if (completed == 0) return 0.0;
    
    return static_cast<double>(total_latency_ns_.load()) / completed / 1000.0;
}

void IOUringPrefetcher::completion_loop() {
#ifdef __linux__
    while (!stop_) {
        struct io_uring_cqe* cqe;
        int ret = io_uring_wait_cqe(&ring_, &cqe);
        
        if (ret < 0) {
            if (!stop_) {
                std::cerr << "io_uring completion error: " << strerror(-ret) << std::endl;
            }
            continue;
        }
        
        process_completion(cqe);
        io_uring_cqe_seen(&ring_, cqe);
    }
#endif
}

#ifdef __linux__
void IOUringPrefetcher::process_completion(struct io_uring_cqe* cqe) {
    uint64_t request_id = reinterpret_cast<uint64_t>(cqe->user_data);
    
    std::lock_guard<std::mutex> lock(requests_mutex_);
    auto it = inflight_requests_.find(request_id);
    if (it != inflight_requests_.end()) {
        const auto& request = it->second;
        
        // Calculate latency
        uint64_t completion_time = get_time_ns();
        uint64_t latency_ns = completion_time - request.submit_time_ns;
        total_latency_ns_ += latency_ns;
        completed_ios_++;
        
        // Call callback if provided
        if (request.callback) {
            request.callback(cqe->res >= 0 ? 0 : cqe->res, 
                           cqe->res >= 0 ? cqe->res : 0);
        }
        
        inflight_requests_.erase(it);
    }
    
    pending_ios_--;
}
#endif

int IOUringPrefetcher::setup_ring() {
#ifdef __linux__
    unsigned flags = 0;
    
    if (config_.use_sqpoll) {
        flags |= IORING_SETUP_SQPOLL;
    }
    
    int ret = io_uring_queue_init(config_.queue_depth, &ring_, flags);
    if (ret < 0) {
        std::cerr << "io_uring_queue_init failed: " << strerror(-ret) << std::endl;
        return ret;
    }
    
    return 0;
#else
    return -1;
#endif
}

void IOUringPrefetcher::register_buffers() {
    // Buffer registration would go here for io_uring fixed buffers
    // This is an advanced optimization that can be added later
}

uint64_t IOUringPrefetcher::get_time_ns() const {
    auto now = std::chrono::steady_clock::now();
    return std::chrono::duration_cast<std::chrono::nanoseconds>
        (now.time_since_epoch()).count();
}

// AdaptivePrefetcher implementation
AdaptivePrefetcher::AdaptivePrefetcher(IOUringPrefetcher* io_prefetcher)
    : io_prefetcher_(io_prefetcher) {
}

AdaptivePrefetcher::~AdaptivePrefetcher() = default;

void AdaptivePrefetcher::record_access(int fd, off_t offset, size_t size) {
    std::lock_guard<std::mutex> lock(history_mutex_);
    
    auto& history = access_history_[fd];
    uint64_t now = std::chrono::duration_cast<std::chrono::microseconds>
        (std::chrono::steady_clock::now().time_since_epoch()).count();
    
    // Add access to history
    history.offsets.push_back(offset);
    history.sizes.push_back(size);
    history.timestamps.push_back(now);
    
    // Keep history bounded (last 100 accesses)
    const size_t max_history = 100;
    if (history.offsets.size() > max_history) {
        size_t excess = history.offsets.size() - max_history;
        history.offsets.erase(history.offsets.begin(), 
                             history.offsets.begin() + excess);
        history.sizes.erase(history.sizes.begin(), 
                           history.sizes.begin() + excess);
        history.timestamps.erase(history.timestamps.begin(), 
                                history.timestamps.begin() + excess);
    }
    
    // Update pattern analysis periodically
    if (now - history.last_analysis_time > 1000000) {  // 1 second
        update_pattern(history);
        history.last_analysis_time = now;
    }
}

void AdaptivePrefetcher::prefetch_adaptive(int fd, off_t current_offset, size_t current_size) {
    std::lock_guard<std::mutex> lock(history_mutex_);
    
    auto it = access_history_.find(fd);
    if (it == access_history_.end()) {
        return;  // No history for this file
    }
    
    const auto& history = it->second;
    auto predictions = predict_next_accesses(history, current_offset);
    
    // Submit prefetch requests for predictions
    for (const auto& [offset, size] : predictions) {
        io_prefetcher_->prefetch(fd, offset, size);
    }
}

AdaptivePrefetcher::AccessPattern AdaptivePrefetcher::analyze_pattern(int fd) const {
    std::lock_guard<std::mutex> lock(history_mutex_);
    
    auto it = access_history_.find(fd);
    if (it == access_history_.end()) {
        return {AccessPattern::Random, 0, 0, 0.0};
    }
    
    return it->second.pattern;
}

void AdaptivePrefetcher::update_pattern(FileAccessHistory& history) {
    if (history.offsets.size() < 3) {
        history.pattern = {AccessPattern::Random, 0, 0, 0.0};
        return;
    }
    
    // Analyze stride pattern
    std::vector<off_t> strides;
    for (size_t i = 1; i < history.offsets.size(); ++i) {
        strides.push_back(history.offsets[i] - history.offsets[i-1]);
    }
    
    // Check for sequential pattern (mostly positive strides)
    int positive_strides = 0;
    int consistent_strides = 0;
    off_t common_stride = 0;
    
    if (!strides.empty()) {
        // Find most common stride
        std::sort(strides.begin(), strides.end());
        
        off_t current_stride = strides[0];
        int current_count = 1;
        int max_count = 1;
        
        for (size_t i = 1; i < strides.size(); ++i) {
            if (strides[i] > 0) positive_strides++;
            
            if (strides[i] == current_stride) {
                current_count++;
            } else {
                if (current_count > max_count) {
                    max_count = current_count;
                    common_stride = current_stride;
                }
                current_stride = strides[i];
                current_count = 1;
            }
        }
        
        if (current_count > max_count) {
            common_stride = current_stride;
            max_count = current_count;
        }
        
        consistent_strides = max_count;
    }
    
    // Calculate average size
    size_t total_size = 0;
    for (size_t size : history.sizes) {
        total_size += size;
    }
    size_t avg_size = total_size / history.sizes.size();
    
    // Determine pattern type
    double consistency_ratio = static_cast<double>(consistent_strides) / strides.size();
    double sequential_ratio = static_cast<double>(positive_strides) / strides.size();
    
    if (consistency_ratio > 0.8 && common_stride > 0) {
        if (sequential_ratio > 0.8) {
            history.pattern = {AccessPattern::Sequential, 
                             static_cast<size_t>(common_stride), avg_size, consistency_ratio};
        } else {
            history.pattern = {AccessPattern::Strided, 
                             static_cast<size_t>(common_stride), avg_size, consistency_ratio};
        }
    } else {
        history.pattern = {AccessPattern::Random, 0, avg_size, 0.0};
    }
}

std::vector<std::pair<off_t, size_t>> AdaptivePrefetcher::predict_next_accesses(
    const FileAccessHistory& history, off_t current_offset) {
    
    std::vector<std::pair<off_t, size_t>> predictions;
    
    if (history.pattern.type == AccessPattern::Sequential) {
        // Predict sequential reads ahead
        for (size_t i = 1; i <= prefetch_depth_; ++i) {
            off_t predicted_offset = current_offset + i * history.pattern.stride;
            predictions.emplace_back(predicted_offset, history.pattern.avg_size);
        }
    } else if (history.pattern.type == AccessPattern::Strided) {
        // Predict strided access pattern
        for (size_t i = 1; i <= prefetch_depth_; ++i) {
            off_t predicted_offset = current_offset + i * history.pattern.stride;
            predictions.emplace_back(predicted_offset, history.pattern.avg_size);
        }
    }
    // No predictions for random access pattern
    
    return predictions;
}

} // namespace vdb