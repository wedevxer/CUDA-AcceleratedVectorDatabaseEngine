#include "transfer_manager.h"
#include <cuda_runtime_api.h>
#include <stdexcept>
#include <algorithm>
#include <chrono>
#include <sstream>
#include <unordered_map>
#include <iostream>

namespace vdb {

TransferManager::PinnedMemoryPool::PinnedMemoryPool(size_t size) 
    : total_size_(size) {
    cudaError_t err = cudaMallocHost(&base_, size);
    if (err != cudaSuccess) {
        throw std::runtime_error("Failed to allocate pinned memory");
    }
    blocks_.push_back({0, size, true});
}

TransferManager::PinnedMemoryPool::~PinnedMemoryPool() {
    if (base_) {
        cudaFreeHost(base_);
    }
}

void* TransferManager::PinnedMemoryPool::allocate(size_t size) {
    std::lock_guard<std::mutex> lock(mutex_);
    
    size = (size + 255) & ~255;
    
    for (auto& block : blocks_) {
        if (block.free && block.size >= size) {
            block.free = false;
            
            if (block.size > size) {
                blocks_.push_back({
                    block.offset + size,
                    block.size - size,
                    true
                });
                block.size = size;
            }
            
            return static_cast<char*>(base_) + block.offset;
        }
    }
    
    return nullptr;
}

void TransferManager::PinnedMemoryPool::free(void* ptr) {
    if (!ptr) return;
    
    std::lock_guard<std::mutex> lock(mutex_);
    
    size_t offset = static_cast<char*>(ptr) - static_cast<char*>(base_);
    
    for (auto& block : blocks_) {
        if (block.offset == offset) {
            block.free = true;
            coalesce_free_blocks();
            return;
        }
    }
}

void TransferManager::PinnedMemoryPool::coalesce_free_blocks() {
    std::sort(blocks_.begin(), blocks_.end(),
        [](const Block& a, const Block& b) { return a.offset < b.offset; });
    
    std::vector<Block> new_blocks;
    
    for (const auto& block : blocks_) {
        if (!new_blocks.empty() && 
            new_blocks.back().free && 
            block.free &&
            new_blocks.back().offset + new_blocks.back().size == block.offset) {
            new_blocks.back().size += block.size;
        } else {
            new_blocks.push_back(block);
        }
    }
    
    blocks_ = std::move(new_blocks);
}

TransferManager::DeviceMemoryPool::DeviceMemoryPool(size_t size)
    : total_size_(size) {
    cudaError_t err = cudaMalloc(&base_, size);
    if (err != cudaSuccess) {
        throw std::runtime_error("Failed to allocate device memory");
    }
    blocks_.push_back({0, size, true});
}

TransferManager::DeviceMemoryPool::~DeviceMemoryPool() {
    if (base_) {
        cudaFree(base_);
    }
}

void* TransferManager::DeviceMemoryPool::allocate(size_t size) {
    std::lock_guard<std::mutex> lock(mutex_);
    
    size = (size + 255) & ~255;
    
    for (auto& block : blocks_) {
        if (block.free && block.size >= size) {
            block.free = false;
            
            if (block.size > size) {
                blocks_.push_back({
                    block.offset + size,
                    block.size - size,
                    true
                });
                block.size = size;
            }
            
            return static_cast<char*>(base_) + block.offset;
        }
    }
    
    return nullptr;
}

void TransferManager::DeviceMemoryPool::free(void* ptr) {
    if (!ptr) return;
    
    std::lock_guard<std::mutex> lock(mutex_);
    
    size_t offset = static_cast<char*>(ptr) - static_cast<char*>(base_);
    
    for (auto& block : blocks_) {
        if (block.offset == offset) {
            block.free = true;
            coalesce_free_blocks();
            return;
        }
    }
}

void TransferManager::DeviceMemoryPool::coalesce_free_blocks() {
    std::sort(blocks_.begin(), blocks_.end(),
        [](const Block& a, const Block& b) { return a.offset < b.offset; });
    
    std::vector<Block> new_blocks;
    
    for (const auto& block : blocks_) {
        if (!new_blocks.empty() && 
            new_blocks.back().free && 
            block.free &&
            new_blocks.back().offset + new_blocks.back().size == block.offset) {
            new_blocks.back().size += block.size;
        } else {
            new_blocks.push_back(block);
        }
    }
    
    blocks_ = std::move(new_blocks);
}

TransferManager::TransferManager(const Config& config)
    : config_(config), pending_transfers_(0) {
    
    pinned_pool_ = std::make_unique<PinnedMemoryPool>(config.pinned_pool_size);
    device_pool_ = std::make_unique<DeviceMemoryPool>(config.device_pool_size);
    
    for (int i = 0; i < config.num_streams; ++i) {
        cudaStream_t stream;
        cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking);
        streams_.push_back(stream);
        available_streams_.push(stream);
    }
}

TransferManager::~TransferManager() {
    synchronize();
    
    for (auto stream : streams_) {
        cudaStreamDestroy(stream);
    }
}

void* TransferManager::allocate_pinned(size_t size) {
    return pinned_pool_->allocate(size);
}

void TransferManager::free_pinned(void* ptr) {
    pinned_pool_->free(ptr);
}

void* TransferManager::allocate_device(size_t size) {
    return device_pool_->allocate(size);
}

void TransferManager::free_device(void* ptr) {
    device_pool_->free(ptr);
}

cudaStream_t TransferManager::get_stream() {
    std::unique_lock<std::mutex> lock(stream_mutex_);
    stream_cv_.wait(lock, [this] { return !available_streams_.empty(); });
    
    cudaStream_t stream = available_streams_.front();
    available_streams_.pop();
    
    return stream;
}

void TransferManager::return_stream(cudaStream_t stream) {
    std::lock_guard<std::mutex> lock(stream_mutex_);
    available_streams_.push(stream);
    stream_cv_.notify_one();
}

void TransferManager::enqueue_transfer(const Transfer& transfer) {
    pending_transfers_++;
    process_transfer(transfer);
}

void TransferManager::enqueue_batch(const std::vector<Transfer>& transfers) {
    pending_transfers_ += transfers.size();
    
    for (const auto& transfer : transfers) {
        process_transfer(transfer);
    }
}

void TransferManager::process_transfer(const Transfer& transfer) {
    cudaStream_t stream = transfer.stream ? transfer.stream : get_stream();
    
    if (config_.use_async) {
        cudaMemcpyAsync(transfer.dst, transfer.src, transfer.size, 
                        transfer.kind, stream);
    } else {
        cudaMemcpy(transfer.dst, transfer.src, transfer.size, transfer.kind);
    }
    
    if (transfer.callback) {
        cudaLaunchHostFunc(stream, 
            [](void* data) { 
                auto* cb = static_cast<std::function<void()>*>(data);
                (*cb)();
                delete cb;
            },
            new std::function<void()>(transfer.callback));
    }
    
    cudaLaunchHostFunc(stream, 
        [](void* data) {
            auto* counter = static_cast<std::atomic<size_t>*>(data);
            (*counter)--;
        },
        &pending_transfers_);
    
    if (!transfer.stream) {
        return_stream(stream);
    }
}

void TransferManager::synchronize() {
    for (auto stream : streams_) {
        cudaStreamSynchronize(stream);
    }
}

void TransferManager::synchronize_stream(cudaStream_t stream) {
    cudaStreamSynchronize(stream);
}

DoubleBuffer::DoubleBuffer(size_t size, TransferManager* tm)
    : size_(size), tm_(tm), read_idx_(0), write_idx_(1) {
    
    buffers_[0] = tm_->allocate_device(size);
    buffers_[1] = tm_->allocate_device(size);
    
    cudaEventCreate(&events_[0]);
    cudaEventCreate(&events_[1]);
}

DoubleBuffer::~DoubleBuffer() {
    tm_->free_device(buffers_[0]);
    tm_->free_device(buffers_[1]);
    
    cudaEventDestroy(events_[0]);
    cudaEventDestroy(events_[1]);
}

void* DoubleBuffer::get_read_buffer() {
    return buffers_[read_idx_];
}

void* DoubleBuffer::get_write_buffer() {
    return buffers_[write_idx_];
}

void DoubleBuffer::swap() {
    std::swap(read_idx_, write_idx_);
}

void DoubleBuffer::start_h2d_transfer(const void* src, cudaStream_t stream) {
    cudaMemcpyAsync(buffers_[write_idx_], src, size_, 
                    cudaMemcpyHostToDevice, stream);
    cudaEventRecord(events_[write_idx_], stream);
}

void DoubleBuffer::wait_transfer(cudaStream_t stream) {
    cudaStreamWaitEvent(stream, events_[write_idx_], 0);
}

StreamScheduler::StreamScheduler(int num_streams) : stop_(false) {
    for (int i = 0; i < num_streams; ++i) {
        cudaStream_t stream;
        cudaStreamCreateWithPriority(&stream, cudaStreamNonBlocking, 
                                     i < num_streams/2 ? -1 : 0);
        streams_.push_back({stream, false, 
                           std::chrono::steady_clock::now()});
    }
    
    worker_ = std::thread(&StreamScheduler::worker_loop, this);
}

StreamScheduler::~StreamScheduler() {
    stop_ = true;
    cv_.notify_all();
    if (worker_.joinable()) {
        worker_.join();
    }
    
    for (auto& state : streams_) {
        cudaStreamDestroy(state.stream);
    }
}

void StreamScheduler::schedule(const Work& work) {
    {
        std::lock_guard<std::mutex> lock(mutex_);
        work_queue_.push(work);
    }
    cv_.notify_one();
}

void StreamScheduler::schedule_batch(const std::vector<Work>& batch) {
    {
        std::lock_guard<std::mutex> lock(mutex_);
        for (const auto& work : batch) {
            work_queue_.push(work);
        }
    }
    cv_.notify_all();
}

void StreamScheduler::wait_all() {
    for (auto& state : streams_) {
        cudaStreamSynchronize(state.stream);
    }
}

void StreamScheduler::worker_loop() {
    while (!stop_) {
        std::unique_lock<std::mutex> lock(mutex_);
        cv_.wait(lock, [this] { return !work_queue_.empty() || stop_; });
        
        if (stop_) break;
        
        if (!work_queue_.empty()) {
            Work work = work_queue_.top();
            work_queue_.pop();
            
            cudaStream_t stream = get_next_available_stream();
            
            lock.unlock();
            
            work.func(stream);
            
            lock.lock();
            for (auto& state : streams_) {
                if (state.stream == stream) {
                    state.busy = false;
                    state.available_at = std::chrono::steady_clock::now() + 
                        std::chrono::microseconds(work.estimated_time_us);
                    break;
                }
            }
        }
    }
}

cudaStream_t StreamScheduler::get_next_available_stream() {
    auto now = std::chrono::steady_clock::now();
    cudaStream_t best_stream = nullptr;
    auto earliest_time = std::chrono::steady_clock::time_point::max();
    
    for (auto& state : streams_) {
        if (!state.busy && state.available_at <= now) {
            state.busy = true;
            return state.stream;
        }
        if (state.available_at < earliest_time) {
            earliest_time = state.available_at;
            best_stream = state.stream;
        }
    }
    
    for (auto& state : streams_) {
        if (state.stream == best_stream) {
            state.busy = true;
            break;
        }
    }
    
    return best_stream;
}

// CUDA Error Handling and Memory Validation Implementation

static std::unordered_map<void*, size_t> g_device_allocations;
static std::unordered_map<void*, size_t> g_pinned_allocations;
static std::mutex g_allocation_mutex;
static std::atomic<size_t> g_total_device_allocated{0};
static std::atomic<size_t> g_total_pinned_allocated{0};
static std::atomic<size_t> g_peak_device_usage{0};
static std::atomic<size_t> g_peak_pinned_usage{0};

bool TransferManager::validate_cuda_pointer(void* ptr) {
    if (!ptr) return false;
    
    // Check if pointer is accessible from device
    cudaPointerAttributes attrs;
    cudaError_t err = cudaPointerGetAttributes(&attrs, ptr);
    
    if (err != cudaSuccess) {
        // Reset error state
        cudaGetLastError();
        return false;
    }
    
    // Verify pointer is either device or managed memory
    return (attrs.type == cudaMemoryTypeDevice || 
            attrs.type == cudaMemoryTypeManaged ||
            attrs.type == cudaMemoryTypeHost);
}

std::string TransferManager::get_cuda_error_string(cudaError_t error) {
    std::ostringstream ss;
    ss << "CUDA Error " << error << ": " << cudaGetErrorString(error);
    
    // Add additional context for common errors
    switch (error) {
        case cudaErrorOutOfMemory:
            ss << " (GPU out of memory - try reducing batch size or index cache)";
            break;
        case cudaErrorInvalidValue:
            ss << " (Invalid parameter passed to CUDA function)";
            break;
        case cudaErrorInvalidDevice:
            ss << " (Invalid device ordinal)";
            break;
        case cudaErrorInvalidDevicePointer:
            ss << " (Invalid device pointer - check memory allocation)";
            break;
        case cudaErrorLaunchFailure:
            ss << " (Kernel launch failed - check grid/block dimensions)";
            break;
        case cudaErrorMemoryAllocation:
            ss << " (Memory allocation failed - GPU memory exhausted)";
            break;
        case cudaErrorInvalidMemcpyDirection:
            ss << " (Invalid memory copy direction)";
            break;
        default:
            break;
    }
    
    return ss.str();
}

void TransferManager::check_cuda_error(cudaError_t error, const std::string& operation) {
    if (error != cudaSuccess) {
        std::string error_msg = "CUDA operation failed: " + operation + " - " + 
                               get_cuda_error_string(error);
        
        // Log additional debug information
        std::cerr << error_msg << std::endl;
        
        // Print GPU memory info for memory-related errors
        if (error == cudaErrorOutOfMemory || error == cudaErrorMemoryAllocation) {
            size_t free_mem, total_mem;
            if (cudaMemGetInfo(&free_mem, &total_mem) == cudaSuccess) {
                std::cerr << "GPU Memory - Free: " << (free_mem >> 20) << " MB, "
                         << "Total: " << (total_mem >> 20) << " MB" << std::endl;
            }
        }
        
        throw std::runtime_error(error_msg);
    }
}

TransferManager::MemoryStats TransferManager::get_memory_stats() const {
    std::lock_guard<std::mutex> lock(g_allocation_mutex);
    
    MemoryStats stats;
    stats.total_device_allocated = g_total_device_allocated;
    stats.total_pinned_allocated = g_total_pinned_allocated;
    stats.active_allocations = g_device_allocations.size() + g_pinned_allocations.size();
    stats.peak_device_usage = g_peak_device_usage;
    stats.peak_pinned_usage = g_peak_pinned_usage;
    
    return stats;
}

// Enhanced memory allocation with tracking
void* TransferManager::allocate_device(size_t size) {
    void* ptr = nullptr;
    cudaError_t err = cudaMalloc(&ptr, size);
    
    if (err != cudaSuccess) {
        // Try to free some cached memory and retry
        device_pool_->try_compact();
        err = cudaMalloc(&ptr, size);
        
        if (err != cudaSuccess) {
            check_cuda_error(err, "device memory allocation of " + 
                           std::to_string(size) + " bytes");
        }
    }
    
    // Track allocation
    {
        std::lock_guard<std::mutex> lock(g_allocation_mutex);
        g_device_allocations[ptr] = size;
        g_total_device_allocated += size;
        
        if (g_total_device_allocated > g_peak_device_usage) {
            g_peak_device_usage = g_total_device_allocated;
        }
    }
    
    // Validate pointer
    if (!validate_cuda_pointer(ptr)) {
        std::cerr << "Warning: Allocated device pointer failed validation" << std::endl;
    }
    
    return ptr;
}

void TransferManager::free_device(void* ptr) {
    if (!ptr) return;
    
    // Validate pointer before freeing
    if (!validate_cuda_pointer(ptr)) {
        std::cerr << "Warning: Attempting to free invalid device pointer" << std::endl;
        return;
    }
    
    // Remove from tracking
    size_t size = 0;
    {
        std::lock_guard<std::mutex> lock(g_allocation_mutex);
        auto it = g_device_allocations.find(ptr);
        if (it != g_device_allocations.end()) {
            size = it->second;
            g_device_allocations.erase(it);
            g_total_device_allocated -= size;
        }
    }
    
    cudaError_t err = cudaFree(ptr);
    if (err != cudaSuccess) {
        std::cerr << "Warning: Failed to free device memory: " 
                  << get_cuda_error_string(err) << std::endl;
        // Don't throw on free failure - could cause destructor issues
    }
}

void* TransferManager::allocate_pinned(size_t size) {
    void* ptr = pinned_pool_->allocate(size);
    
    if (!ptr) {
        // Pool exhausted, try direct allocation
        cudaError_t err = cudaMallocHost(&ptr, size);
        if (err != cudaSuccess) {
            check_cuda_error(err, "pinned memory allocation of " + 
                           std::to_string(size) + " bytes");
        }
        
        // Track direct allocation
        std::lock_guard<std::mutex> lock(g_allocation_mutex);
        g_pinned_allocations[ptr] = size;
    }
    
    g_total_pinned_allocated += size;
    if (g_total_pinned_allocated > g_peak_pinned_usage) {
        g_peak_pinned_usage = g_total_pinned_allocated;
    }
    
    return ptr;
}

void TransferManager::free_pinned(void* ptr) {
    if (!ptr) return;
    
    // Try pool first
    if (pinned_pool_->contains_pointer(ptr)) {
        size_t size = pinned_pool_->get_allocation_size(ptr);
        pinned_pool_->free(ptr);
        g_total_pinned_allocated -= size;
        return;
    }
    
    // Handle direct allocation
    size_t size = 0;
    {
        std::lock_guard<std::mutex> lock(g_allocation_mutex);
        auto it = g_pinned_allocations.find(ptr);
        if (it != g_pinned_allocations.end()) {
            size = it->second;
            g_pinned_allocations.erase(it);
            g_total_pinned_allocated -= size;
        }
    }
    
    cudaError_t err = cudaFreeHost(ptr);
    if (err != cudaSuccess) {
        std::cerr << "Warning: Failed to free pinned memory: " 
                  << get_cuda_error_string(err) << std::endl;
    }
}

}