// Header guard to prevent multiple inclusions
#pragma once

// CUDA runtime API for device management and kernel execution
#include <cuda_runtime.h>
// CUB library for GPU primitives (reductions, scans, etc.)
#include <cub/cub.cuh>
// CUDA half-precision floating point support
#include <cuda_fp16.h>

// Main namespace for vector database
namespace vdb {
// Sub-namespace for CUDA kernel implementations
namespace kernels {

// CUDA warp size - fundamental unit of execution (32 threads)
constexpr int WARP_SIZE = 32;
// Tile size for blocked algorithms (optimized for memory coalescing)
constexpr int TILE_SIZE = 128;
// Maximum supported vector dimension
constexpr int MAX_DIM = 2048;

// Enumeration of supported distance metrics
enum class Metric {
    L2,              // Euclidean distance
    InnerProduct,    // Dot product (negated for minimization)
    Cosine          // Cosine similarity (1 - cos(θ))
};

// Template struct for distance computations on GPU
template<typename T>
struct Distance {
    // Compute L2 (Euclidean) distance between vectors a and b
    // __device__: Callable from GPU code only
    // __forceinline__: Force compiler to inline for performance
    __device__ __forceinline__ static float compute_l2(
        const T* a, const T* b, int dim) {
        float sum = 0.0f;  // Accumulator for squared differences
        // Unroll loop 4 times for instruction-level parallelism
        #pragma unroll 4
        for (int i = 0; i < dim; ++i) {
            // Cast to float and compute squared difference
            float diff = static_cast<float>(a[i]) - static_cast<float>(b[i]);
            sum += diff * diff;  // Accumulate squared difference
        }
        return sum;  // Return L2 distance (not sqrt for efficiency)
    }
    
    // Compute inner product (dot product) between vectors
    __device__ __forceinline__ static float compute_ip(
        const T* a, const T* b, int dim) {
        float sum = 0.0f;  // Accumulator for dot product
        // Unroll for better performance
        #pragma unroll 4
        for (int i = 0; i < dim; ++i) {
            // Multiply corresponding elements and accumulate
            sum += static_cast<float>(a[i]) * static_cast<float>(b[i]);
        }
        return -sum;  // Negate for minimization (larger dot product = smaller distance)
    }
    
    // Compute cosine distance (1 - cosine similarity)
    __device__ __forceinline__ static float compute_cosine(
        const T* a, const T* b, int dim) {
        // Three accumulators: dot product and norms
        float dot = 0.0f, norm_a = 0.0f, norm_b = 0.0f;
        #pragma unroll 4
        for (int i = 0; i < dim; ++i) {
            // Cache values to reduce memory accesses
            float ai = static_cast<float>(a[i]);
            float bi = static_cast<float>(b[i]);
            dot += ai * bi;      // Dot product
            norm_a += ai * ai;   // L2 norm of a
            norm_b += bi * bi;   // L2 norm of b
        }
        // Cosine similarity = dot / (||a|| * ||b||)
        // Return 1 - cosine for distance (0 = identical, 2 = opposite)
        // Add 1e-8f to prevent division by zero
        return 1.0f - dot / (sqrtf(norm_a) * sqrtf(norm_b) + 1e-8f);
    }
};

// Main bruteforce search kernel - compares all database vectors against queries
template<typename T, Metric metric>
__global__ void bruteforce_search_kernel(
    const T* __restrict__ database,      // Database vectors (n_vectors × dim)
    const T* __restrict__ queries,       // Query vectors (n_queries × dim)
    const uint64_t* __restrict__ ids,    // Optional vector IDs
    uint64_t n_vectors,                  // Number of database vectors
    uint32_t n_queries,                  // Number of query vectors
    uint32_t dim,                        // Vector dimension
    uint32_t k,                          // Number of neighbors to find
    float* __restrict__ out_distances,   // Output distances (n_queries × k)
    uint64_t* __restrict__ out_indices) {// Output indices (n_queries × k)
    
    // Declare dynamic shared memory for query vector caching
    extern __shared__ char shared_mem[];
    // Cast shared memory to appropriate type
    T* query_shared = reinterpret_cast<T*>(shared_mem);
    
    // Grid-stride loop: blockIdx.y indexes queries
    const uint32_t query_id = blockIdx.y;
    // Thread ID within block
    const uint32_t tid = threadIdx.x;
    // Block size for coalesced access
    const uint32_t block_size = blockDim.x;
    
    // Early exit if query index out of bounds
    if (query_id >= n_queries) return;
    
    // Cooperatively load query vector into shared memory
    // Each thread loads dim/block_size elements
    for (uint32_t i = tid; i < dim; i += block_size) {
        query_shared[i] = queries[query_id * dim + i];
    }
    // Synchronize to ensure all threads see loaded query
    __syncthreads();
    
    // Structure for maintaining top-K results
    struct TopK {
        float dist;      // Distance value
        uint64_t idx;    // Vector index
    };
    
    // Local top-K buffer per thread (max 32 for register usage)
    TopK local_topk[32];
    // Initialize with maximum values
    for (int i = 0; i < k && i < 32; ++i) {
        local_topk[i].dist = FLT_MAX;       // Largest float value
        local_topk[i].idx = UINT64_MAX;     // Invalid index
    }
    
    // Grid-stride loop over database vectors
    // Each block processes block_size vectors at a time
    for (uint64_t vec_base = blockIdx.x * block_size; 
         vec_base < n_vectors; 
         vec_base += gridDim.x * block_size) {
        
        // Each thread processes one vector
        uint64_t vec_id = vec_base + tid;
        if (vec_id < n_vectors) {
            // Pointer to current database vector
            const T* vec = &database[vec_id * dim];
            
            // Compute distance based on metric type
            float dist;
            if constexpr (metric == Metric::L2) {
                dist = Distance<T>::compute_l2(query_shared, vec, dim);
            } else if constexpr (metric == Metric::InnerProduct) {
                dist = Distance<T>::compute_ip(query_shared, vec, dim);
            } else {
                dist = Distance<T>::compute_cosine(query_shared, vec, dim);
            }
            
            // Update local top-K with insertion sort
            for (int i = 0; i < k && i < 32; ++i) {
                if (dist < local_topk[i].dist) {
                    // Shift larger distances down
                    for (int j = k - 1; j > i && j < 32; --j) {
                        local_topk[j] = local_topk[j-1];
                    }
                    // Insert new result
                    local_topk[i].dist = dist;
                    // Use provided IDs or vector index
                    local_topk[i].idx = ids ? ids[vec_id] : vec_id;
                    break;  // Inserted, done
                }
            }
        }
    }
    
    // Type definition for block-wide reduction
    typedef cub::BlockReduce<TopK, 256> BlockReduce;
    // Allocate shared memory for CUB reduction
    __shared__ typename BlockReduce::TempStorage temp_storage;
    
    // Write final results (only thread 0 for now - could be improved)
    for (int i = 0; i < k && i < 32; ++i) {
        if (tid == 0) {
            // Write distance and index to global memory
            out_distances[query_id * k + i] = local_topk[i].dist;
            out_indices[query_id * k + i] = local_topk[i].idx;
        }
    }
}

// Kernel for computing PQ distance lookup tables
template<typename T>
__global__ void pq_distance_table_kernel(
    const T* __restrict__ query,         // Query vectors (n_queries × d)
    const T* __restrict__ codebooks,     // PQ codebooks (m × ks × dsub)
    float* __restrict__ dist_tables,     // Output distance tables
    uint32_t n_queries,                  // Number of queries
    uint32_t m,                          // Number of subquantizers
    uint32_t ks,                         // Number of centroids per subquantizer
    uint32_t dsub) {                    // Dimension per subquantizer (d/m)
    
    // Each block handles one query
    const uint32_t query_id = blockIdx.x;
    // Each block row handles one subspace
    const uint32_t subspace = blockIdx.y;
    // Each thread handles one codeword
    const uint32_t codeword = threadIdx.x;
    
    // Bounds checking
    if (query_id >= n_queries || subspace >= m || codeword >= ks) return;
    
    // Pointer to query subvector for this subspace
    const T* q_sub = query + query_id * m * dsub + subspace * dsub;
    // Pointer to codebook centroid
    const T* cb_sub = codebooks + subspace * ks * dsub + codeword * dsub;
    
    // Compute L2 distance between query subvector and centroid
    float dist = 0.0f;
    for (uint32_t i = 0; i < dsub; ++i) {
        float diff = static_cast<float>(q_sub[i]) - static_cast<float>(cb_sub[i]);
        dist += diff * diff;
    }
    
    // Store distance in lookup table
    // Layout: [query][subspace][codeword]
    dist_tables[query_id * m * ks + subspace * ks + codeword] = dist;
}

// Kernel for scanning PQ-compressed vectors using distance tables
template<typename CodeT>
__global__ void pq_scan_kernel(
    const CodeT* __restrict__ codes,     // PQ codes (n_codes × m bytes)
    const float* __restrict__ dist_tables,// Distance lookup tables
    const uint64_t* __restrict__ ids,    // Vector IDs
    uint64_t n_codes,                    // Number of PQ codes
    uint32_t n_queries,                  // Number of queries
    uint32_t m,                          // Number of subquantizers
    uint32_t ks,                         // Centroids per subquantizer
    uint32_t k,                          // Top-K to return
    float* __restrict__ out_distances,   // Output distances
    uint64_t* __restrict__ out_indices) {// Output indices
    
    // Allocate shared memory for distance table
    extern __shared__ float shared_table[];
    
    // Each y-block handles one query
    const uint32_t query_id = blockIdx.y;
    const uint32_t tid = threadIdx.x;
    const uint32_t block_size = blockDim.x;
    
    // Bounds check
    if (query_id >= n_queries) return;
    
    // Cooperatively load distance table into shared memory
    for (uint32_t i = tid; i < m * ks; i += block_size) {
        shared_table[i] = dist_tables[query_id * m * ks + i];
    }
    // Ensure all threads see loaded table
    __syncthreads();
    
    // Top-K tracking structure
    struct TopK {
        float dist;
        uint64_t idx;
    };
    
    // Initialize local top-K buffer
    TopK local_topk[32];
    for (int i = 0; i < k && i < 32; ++i) {
        local_topk[i].dist = FLT_MAX;
        local_topk[i].idx = UINT64_MAX;
    }
    
    // Grid-stride loop over PQ codes
    for (uint64_t code_base = blockIdx.x * block_size;
         code_base < n_codes;
         code_base += gridDim.x * block_size) {
        
        uint64_t code_id = code_base + tid;
        if (code_id < n_codes) {
            // Pointer to PQ code for this vector
            const CodeT* code = &codes[code_id * m];
            
            // Sum distances from lookup table
            float dist = 0.0f;
            // Unroll for better performance
            #pragma unroll 8
            for (uint32_t i = 0; i < m; ++i) {
                // code[i] is the centroid index for subspace i
                // Look up precomputed distance from shared memory
                dist += shared_table[i * ks + code[i]];
            }
            
            // Update local top-K using insertion sort
            for (int i = 0; i < k && i < 32; ++i) {
                if (dist < local_topk[i].dist) {
                    // Shift and insert
                    for (int j = k - 1; j > i && j < 32; --j) {
                        local_topk[j] = local_topk[j-1];
                    }
                    local_topk[i].dist = dist;
                    local_topk[i].idx = ids ? ids[code_id] : code_id;
                    break;
                }
            }
        }
    }
    
    // Write results (single thread for simplicity)
    if (tid == 0) {
        for (int i = 0; i < k && i < 32; ++i) {
            out_distances[query_id * k + i] = local_topk[i].dist;
            out_indices[query_id * k + i] = local_topk[i].idx;
        }
    }
}

// K-means assignment kernel - assigns vectors to nearest centroids
template<typename T>
__global__ void kmeans_assign_kernel(
    const T* __restrict__ vectors,       // Input vectors
    const T* __restrict__ centroids,     // Cluster centroids
    uint32_t* __restrict__ assignments,  // Output cluster assignments
    float* __restrict__ distances,       // Optional output distances
    uint64_t n_vectors,                  // Number of vectors
    uint32_t n_centroids,                // Number of centroids
    uint32_t dim) {                     // Vector dimension
    
    // Each thread processes one vector
    const uint64_t vec_id = blockIdx.x * blockDim.x + threadIdx.x;
    // Bounds check
    if (vec_id >= n_vectors) return;
    
    // Pointer to current vector
    const T* vec = &vectors[vec_id * dim];
    
    // Track minimum distance and best centroid
    float min_dist = FLT_MAX;
    uint32_t best_centroid = 0;
    
    // Linear search through all centroids
    for (uint32_t c = 0; c < n_centroids; ++c) {
        // Pointer to current centroid
        const T* centroid = &centroids[c * dim];
        // Compute L2 distance
        float dist = Distance<T>::compute_l2(vec, centroid, dim);
        // Update if closer
        if (dist < min_dist) {
            min_dist = dist;
            best_centroid = c;
        }
    }
    
    // Write assignment
    assignments[vec_id] = best_centroid;
    // Optionally write distance
    if (distances) distances[vec_id] = min_dist;
}

// Vector normalization kernel for cosine similarity
template<typename T>
__global__ void normalize_vectors_kernel(
    T* __restrict__ vectors,             // Vectors to normalize (in-place)
    uint64_t n_vectors,                  // Number of vectors
    uint32_t dim) {                     // Vector dimension
    
    // Each thread handles one vector
    const uint64_t vec_id = blockIdx.x * blockDim.x + threadIdx.x;
    // Bounds check
    if (vec_id >= n_vectors) return;
    
    // Pointer to vector
    T* vec = &vectors[vec_id * dim];
    
    // Compute L2 norm
    float norm = 0.0f;
    for (uint32_t i = 0; i < dim; ++i) {
        float val = static_cast<float>(vec[i]);
        norm += val * val;  // Sum of squares
    }
    // Reciprocal square root for efficiency
    // Add 1e-8f to prevent division by zero
    norm = rsqrtf(norm + 1e-8f);
    
    // Normalize vector in-place
    for (uint32_t i = 0; i < dim; ++i) {
        vec[i] = static_cast<T>(static_cast<float>(vec[i]) * norm);
    }
}

// Kernel launcher functions (implemented in kernels.cu)
// These functions provide C++ interface to CUDA kernels

// Launch bruteforce search kernel with appropriate template instantiation
template<typename T>
void launch_bruteforce_search(
    const T* database, const T* queries, const uint64_t* ids,
    uint64_t n_vectors, uint32_t n_queries, uint32_t dim, uint32_t k,
    float* distances, uint64_t* indices, Metric metric, cudaStream_t stream = 0);

// Launch PQ distance table computation kernel
template<typename T>
void launch_pq_distance_table(
    const T* queries, const T* codebooks, float* dist_tables,
    uint32_t n_queries, uint32_t m, uint32_t ks, uint32_t dsub,
    cudaStream_t stream = 0);

// Launch PQ code scanning kernel
template<typename CodeT>
void launch_pq_scan(
    const CodeT* codes, const float* dist_tables, const uint64_t* ids,
    uint64_t n_codes, uint32_t n_queries, uint32_t m, uint32_t ks, uint32_t k,
    float* distances, uint64_t* indices, cudaStream_t stream = 0);

// Launch k-means assignment kernel
template<typename T>
void launch_kmeans_assign(
    const T* vectors, const T* centroids, uint32_t* assignments,
    float* distances, uint64_t n_vectors, uint32_t n_centroids, uint32_t dim,
    cudaStream_t stream = 0);

// Launch vector normalization kernel
template<typename T>
void launch_normalize_vectors(
    T* vectors, uint64_t n_vectors, uint32_t dim, cudaStream_t stream = 0);

// Explicit declarations for common types
extern template void launch_bruteforce_search<float>(
    const float*, const float*, const uint64_t*, uint64_t, uint32_t, uint32_t, uint32_t,
    float*, uint64_t*, Metric, cudaStream_t);

extern template void launch_pq_distance_table<float>(
    const float*, const float*, float*, uint32_t, uint32_t, uint32_t, uint32_t, cudaStream_t);

extern template void launch_pq_scan<uint8_t>(
    const uint8_t*, const float*, const uint64_t*, uint64_t, uint32_t, uint32_t, uint32_t, uint32_t,
    float*, uint64_t*, cudaStream_t);

extern template void launch_kmeans_assign<float>(
    const float*, const float*, uint32_t*, float*, uint64_t, uint32_t, uint32_t, cudaStream_t);

extern template void launch_normalize_vectors<float>(
    float*, uint64_t, uint32_t, cudaStream_t);

} // namespace kernels
} // namespace vdb