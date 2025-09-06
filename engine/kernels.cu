// CUDA kernel implementations
// This file contains the actual __global__ kernel function implementations
// that are declared in kernels.cuh

#include "kernels.cuh"
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

namespace vdb {
namespace kernels {

// Kernel launcher for bruteforce search
template<typename T>
void launch_bruteforce_search(
    const T* database, const T* queries, const uint64_t* ids,
    uint64_t n_vectors, uint32_t n_queries, uint32_t dim, uint32_t k,
    float* distances, uint64_t* indices, Metric metric, cudaStream_t stream) {
    
    // Grid configuration
    // Each block handles one query (blockIdx.y)
    // Grid-stride loop over database vectors (blockIdx.x)
    dim3 block(256);  // 256 threads per block (good for most GPUs)
    dim3 grid(std::min(65535u, (uint32_t)((n_vectors + block.x - 1) / block.x)), n_queries);
    
    // Shared memory for query vector caching
    size_t shared_mem_size = dim * sizeof(T);
    
    // Launch appropriate kernel based on metric
    switch (metric) {
        case Metric::L2:
            bruteforce_search_kernel<T, Metric::L2><<<grid, block, shared_mem_size, stream>>>(
                database, queries, ids, n_vectors, n_queries, dim, k, distances, indices);
            break;
        case Metric::InnerProduct:
            bruteforce_search_kernel<T, Metric::InnerProduct><<<grid, block, shared_mem_size, stream>>>(
                database, queries, ids, n_vectors, n_queries, dim, k, distances, indices);
            break;
        case Metric::Cosine:
            bruteforce_search_kernel<T, Metric::Cosine><<<grid, block, shared_mem_size, stream>>>(
                database, queries, ids, n_vectors, n_queries, dim, k, distances, indices);
            break;
    }
}

// Kernel launcher for PQ distance table computation
template<typename T>
void launch_pq_distance_table(
    const T* queries, const T* codebooks, float* dist_tables,
    uint32_t n_queries, uint32_t m, uint32_t ks, uint32_t dsub,
    cudaStream_t stream) {
    
    // Grid: (n_queries, m, 1) - each block handles one query-subspace pair
    // Block: (ks, 1, 1) - each thread handles one codeword
    dim3 block(ks);
    dim3 grid(n_queries, m);
    
    pq_distance_table_kernel<T><<<grid, block, 0, stream>>>(
        queries, codebooks, dist_tables, n_queries, m, ks, dsub);
}

// Kernel launcher for PQ code scanning
template<typename CodeT>
void launch_pq_scan(
    const CodeT* codes, const float* dist_tables, const uint64_t* ids,
    uint64_t n_codes, uint32_t n_queries, uint32_t m, uint32_t ks, uint32_t k,
    float* distances, uint64_t* indices, cudaStream_t stream) {
    
    // Grid configuration similar to bruteforce
    dim3 block(256);
    dim3 grid(std::min(65535u, (uint32_t)((n_codes + block.x - 1) / block.x)), n_queries);
    
    // Shared memory for distance table
    size_t shared_mem_size = m * ks * sizeof(float);
    
    pq_scan_kernel<CodeT><<<grid, block, shared_mem_size, stream>>>(
        codes, dist_tables, ids, n_codes, n_queries, m, ks, k, distances, indices);
}

// Kernel launcher for k-means assignment
template<typename T>
void launch_kmeans_assign(
    const T* vectors, const T* centroids, uint32_t* assignments,
    float* distances, uint64_t n_vectors, uint32_t n_centroids, uint32_t dim,
    cudaStream_t stream) {
    
    // Simple 1D grid - each thread handles one vector
    dim3 block(256);
    dim3 grid((n_vectors + block.x - 1) / block.x);
    
    kmeans_assign_kernel<T><<<grid, block, 0, stream>>>(
        vectors, centroids, assignments, distances, n_vectors, n_centroids, dim);
}

// Kernel launcher for vector normalization
template<typename T>
void launch_normalize_vectors(
    T* vectors, uint64_t n_vectors, uint32_t dim, cudaStream_t stream) {
    
    // Simple 1D grid - each thread handles one vector
    dim3 block(256);
    dim3 grid((n_vectors + block.x - 1) / block.x);
    
    normalize_vectors_kernel<T><<<grid, block, 0, stream>>>(
        vectors, n_vectors, dim);
}

// Explicit template instantiations for common types
template void launch_bruteforce_search<float>(
    const float*, const float*, const uint64_t*, uint64_t, uint32_t, uint32_t, uint32_t,
    float*, uint64_t*, Metric, cudaStream_t);

template void launch_bruteforce_search<half>(
    const half*, const half*, const uint64_t*, uint64_t, uint32_t, uint32_t, uint32_t,
    float*, uint64_t*, Metric, cudaStream_t);

template void launch_pq_distance_table<float>(
    const float*, const float*, float*, uint32_t, uint32_t, uint32_t, uint32_t, cudaStream_t);

template void launch_pq_scan<uint8_t>(
    const uint8_t*, const float*, const uint64_t*, uint64_t, uint32_t, uint32_t, uint32_t, uint32_t,
    float*, uint64_t*, cudaStream_t);

template void launch_kmeans_assign<float>(
    const float*, const float*, uint32_t*, float*, uint64_t, uint32_t, uint32_t, cudaStream_t);

template void launch_normalize_vectors<float>(
    float*, uint64_t, uint32_t, cudaStream_t);

} // namespace kernels
} // namespace vdb