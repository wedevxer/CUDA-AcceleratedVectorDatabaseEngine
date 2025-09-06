// Header guard to prevent multiple inclusions
#pragma once

// CUDA runtime API for memory operations and streams
#include <cuda_runtime.h>
// Smart pointers for automatic memory management
#include <memory>
// STL containers
#include <vector>
#include <queue>
// Threading primitives for concurrent access
#include <mutex>
#include <condition_variable>
#include <atomic>

// Main namespace for vector database
namespace vdb {

// TransferManager manages GPU/CPU memory pools and async transfers
// Optimizes Host-to-Device (H2D) and Device-to-Host (D2H) transfers
class TransferManager {
public:
    // Configuration structure for memory pools and streams
    struct Config {
        size_t pinned_pool_size = 1ULL << 30;  // 1GB pinned memory pool
        size_t device_pool_size = 4ULL << 30;  // 4GB device memory pool  
        int num_streams = 4;                    // Number of CUDA streams for parallelism
        bool use_async = true;                  // Use async transfers vs synchronous
    };
    
    // Transfer request structure containing all necessary information
    struct Transfer {
        void* src;                              // Source memory pointer
        void* dst;                              // Destination memory pointer
        size_t size;                            // Transfer size in bytes
        cudaMemcpyKind kind;                    // Transfer direction (H2D, D2H, etc.)
        cudaStream_t stream;                    // CUDA stream to use (optional)
        std::function<void()> callback;         // Optional completion callback
    };
    
    // Constructor initializes memory pools and streams
    explicit TransferManager(const Config& config);
    // Destructor ensures all transfers complete and frees resources
    ~TransferManager();
    
    // Allocate pinned (page-locked) host memory for fast transfers
    void* allocate_pinned(size_t size);
    // Free pinned memory back to pool
    void free_pinned(void* ptr);
    
    // CUDA error checking and memory validation
    static bool validate_cuda_pointer(void* ptr);
    static std::string get_cuda_error_string(cudaError_t error);
    static void check_cuda_error(cudaError_t error, const std::string& operation);
    
    // Memory validation and leak detection
    struct MemoryStats {
        size_t total_device_allocated = 0;
        size_t total_pinned_allocated = 0;
        size_t active_allocations = 0;
        size_t peak_device_usage = 0;
        size_t peak_pinned_usage = 0;
    };
    
    MemoryStats get_memory_stats() const;
    
    // Allocate GPU device memory from pool
    void* allocate_device(size_t size);
    // Free device memory back to pool
    void free_device(void* ptr);
    
    // Get an available CUDA stream for transfers
    cudaStream_t get_stream();
    // Return stream to pool when transfer complete
    void return_stream(cudaStream_t stream);
    
    // Queue a single transfer for execution
    void enqueue_transfer(const Transfer& transfer);
    // Queue multiple transfers as a batch
    void enqueue_batch(const std::vector<Transfer>& transfers);
    
    // Block until all pending transfers complete
    void synchronize();
    // Block until specific stream completes
    void synchronize_stream(cudaStream_t stream);
    
    // Get number of transfers currently in flight
    size_t get_pending_transfers() const { return pending_transfers_.load(); }
    
private:
    // Memory pool for pinned host memory (enables DMA transfers)
    class PinnedMemoryPool {
    public:
        // Constructor allocates large pinned memory block
        PinnedMemoryPool(size_t size);
        // Destructor frees the base allocation
        ~PinnedMemoryPool();
        
        // Allocate pinned memory from pool
        void* allocate(size_t size);
        // Return pinned memory to pool for reuse
        void free(void* ptr);
        
    private:
        // Block descriptor for free/used memory tracking
        struct Block {
            size_t offset;      // Offset from base pointer
            size_t size;        // Size of this block
            bool free;          // Whether block is available
        };
        
        void* base_;                    // Base pointer to large allocation
        size_t total_size_;             // Total pool size
        std::vector<Block> blocks_;     // List of memory blocks
        std::mutex mutex_;              // Thread-safe access
        
        // Merge adjacent free blocks to reduce fragmentation
        void coalesce_free_blocks();
    };
    
    // Memory pool for GPU device memory
    class DeviceMemoryPool {
    public:
        // Constructor allocates large device memory block
        DeviceMemoryPool(size_t size);
        // Destructor frees the base allocation
        ~DeviceMemoryPool();
        
        // Allocate device memory from pool
        void* allocate(size_t size);
        // Return device memory to pool for reuse
        void free(void* ptr);
        
    private:
        // Block descriptor (same structure as pinned pool)
        struct Block {
            size_t offset;      // Offset from base pointer
            size_t size;        // Size of this block
            bool free;          // Whether block is available
        };
        
        void* base_;                    // Base pointer to GPU allocation
        size_t total_size_;             // Total pool size
        std::vector<Block> blocks_;     // List of memory blocks
        std::mutex mutex_;              // Thread-safe access
        
        // Merge adjacent free blocks to reduce fragmentation
        void coalesce_free_blocks();
    };
    
    Config config_;                                 // Configuration settings
    std::unique_ptr<PinnedMemoryPool> pinned_pool_; // Pinned memory pool
    std::unique_ptr<DeviceMemoryPool> device_pool_; // Device memory pool
    
    std::vector<cudaStream_t> streams_;             // All CUDA streams
    std::queue<cudaStream_t> available_streams_;    // Available stream pool
    std::mutex stream_mutex_;                       // Protect stream access
    std::condition_variable stream_cv_;             // Signal stream availability
    
    std::atomic<size_t> pending_transfers_;         // Count of in-flight transfers
    
    // Internal method to execute a transfer request
    void process_transfer(const Transfer& transfer);
};

// DoubleBuffer implements producer-consumer pattern for pipelined transfers
// Enables overlap of compute and memory transfer operations
class DoubleBuffer {
public:
    // Constructor allocates two GPU buffers of specified size
    DoubleBuffer(size_t size, TransferManager* tm);
    // Destructor frees both buffers
    ~DoubleBuffer();
    
    // Get buffer currently being read from (for kernels)
    void* get_read_buffer();
    // Get buffer currently being written to (for transfers)
    void* get_write_buffer();
    // Swap read/write buffers atomically
    void swap();
    
    // Start async H2D transfer to write buffer
    void start_h2d_transfer(const void* src, cudaStream_t stream);
    // Wait for transfer to complete before accessing data
    void wait_transfer(cudaStream_t stream);
    
private:
    size_t size_;                       // Buffer size in bytes
    TransferManager* tm_;               // Transfer manager for allocation
    void* buffers_[2];                  // Two GPU buffers for double buffering
    int read_idx_;                      // Index of current read buffer (0 or 1)
    int write_idx_;                     // Index of current write buffer (0 or 1)
    cudaEvent_t events_[2];             // CUDA events for synchronization
};

// StreamScheduler manages work distribution across multiple CUDA streams
// Provides priority-based scheduling and load balancing
class StreamScheduler {
public:
    // Constructor creates streams and starts worker thread
    StreamScheduler(int num_streams);
    // Destructor stops worker and destroys streams
    ~StreamScheduler();
    
    // Work item containing function to execute and metadata
    struct Work {
        std::function<void(cudaStream_t)> func;  // Function to execute on stream
        int priority;                            // Higher number = higher priority
        size_t estimated_time_us;                // Estimated execution time in microseconds
    };
    
    // Schedule single work item for execution
    void schedule(const Work& work);
    // Schedule batch of work items
    void schedule_batch(const std::vector<Work>& batch);
    
    // Wait for all scheduled work to complete
    void wait_all();
    
private:
    // State tracking for each CUDA stream
    struct StreamState {
        cudaStream_t stream;                                    // CUDA stream handle
        bool busy;                                              // Whether stream is executing work
        std::chrono::steady_clock::time_point available_at;     // When stream becomes available
    };
    
    std::vector<StreamState> streams_;              // All stream states
    std::priority_queue<Work> work_queue_;          // Priority queue of pending work
    std::mutex mutex_;                              // Protect queue access
    std::condition_variable cv_;                    // Signal work availability
    std::atomic<bool> stop_;                        // Stop flag for worker thread
    std::thread worker_;                            // Background worker thread
    
    // Main worker loop that dispatches work to streams
    void worker_loop();
    // Select best available stream based on availability and load
    cudaStream_t get_next_available_stream();
};

} // namespace vdb