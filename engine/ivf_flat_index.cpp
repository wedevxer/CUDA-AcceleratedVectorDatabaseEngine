#include "ivf_flat_index.h"
#include "kernels.cuh"
#include <cuda_runtime.h>
#include <algorithm>
#include <iostream>
#include <random>
#include <cstring>
#include <limits>

namespace vdb {

// Constructor initializes IVF-Flat index with given configuration
IVFFlatIndex::IVFFlatIndex(const Config& config, TransferManager* tm)
    : config_(config), tm_(tm) {
    
    // Validate configuration parameters
    if (config_.dimension == 0 || config_.nlist == 0) {
        throw std::invalid_argument("Invalid configuration: dimension and nlist must be > 0");
    }
    
    // Allocate centroid storage (nlist × dimension)
    centroids_.resize(config_.nlist * config_.dimension);
    
    // Initialize inverted lists
    lists_.resize(config_.nlist);
    for (uint32_t i = 0; i < config_.nlist; ++i) {
        lists_[i] = std::make_unique<InvertedList>();
        lists_[i]->last_access = std::chrono::steady_clock::now();
    }
    
    std::cout << "IVF-Flat index created: " << config_.nlist 
              << " lists, " << config_.dimension << "D" << std::endl;
}

// Destructor cleans up GPU memory
IVFFlatIndex::~IVFFlatIndex() {
    // Free GPU memory for all lists
    for (auto& list : lists_) {
        if (list->gpu_vectors) {
            tm_->free_device(list->gpu_vectors);
        }
        if (list->gpu_ids) {
            tm_->free_device(list->gpu_ids);
        }
    }
}

// Train IVF centroids using k-means clustering
void IVFFlatIndex::train(const float* vectors, uint64_t n_vectors) {
    std::cout << "Training IVF centroids with " << n_vectors << " vectors..." << std::endl;
    
    // Use simple k-means++ initialization
    std::mt19937 gen(42);  // Deterministic seed for reproducibility
    std::uniform_int_distribution<uint64_t> dist(0, n_vectors - 1);
    
    // Initialize first centroid randomly
    uint64_t first_idx = dist(gen);
    std::memcpy(centroids_.data(), 
                vectors + first_idx * config_.dimension,
                config_.dimension * sizeof(float));
    
    // K-means++ initialization for remaining centroids
    for (uint32_t c = 1; c < config_.nlist; ++c) {
        std::vector<float> distances(n_vectors);
        float total_dist = 0.0f;
        
        // Compute distances to nearest existing centroid
        for (uint64_t v = 0; v < n_vectors; ++v) {
            const float* vec = vectors + v * config_.dimension;
            float min_dist = std::numeric_limits<float>::max();
            
            // Find distance to nearest centroid
            for (uint32_t existing_c = 0; existing_c < c; ++existing_c) {
                const float* centroid = centroids_.data() + existing_c * config_.dimension;
                float dist = 0.0f;
                
                // Compute L2 distance
                for (uint32_t d = 0; d < config_.dimension; ++d) {
                    float diff = vec[d] - centroid[d];
                    dist += diff * diff;
                }
                
                min_dist = std::min(min_dist, dist);
            }
            
            distances[v] = min_dist;
            total_dist += min_dist;
        }
        
        // Sample next centroid proportional to squared distance
        std::uniform_real_distribution<float> prob_dist(0.0f, total_dist);
        float target = prob_dist(gen);
        float cumsum = 0.0f;
        
        for (uint64_t v = 0; v < n_vectors; ++v) {
            cumsum += distances[v];
            if (cumsum >= target) {
                std::memcpy(centroids_.data() + c * config_.dimension,
                           vectors + v * config_.dimension,
                           config_.dimension * sizeof(float));
                break;
            }
        }
    }
    
    // Refine centroids with Lloyd's algorithm (simplified version)
    std::vector<uint32_t> assignments(n_vectors);
    
    for (int iter = 0; iter < 10; ++iter) {  // 10 iterations
        // Assignment step - assign vectors to nearest centroids
        if (config_.use_gpu) {
            assign_to_lists_gpu(vectors, n_vectors, assignments);
        } else {
            assign_to_lists(vectors, n_vectors, assignments);
        }
        
        // Update step - recompute centroids
        std::vector<std::vector<float>> new_centroids(config_.nlist, 
            std::vector<float>(config_.dimension, 0.0f));
        std::vector<uint32_t> counts(config_.nlist, 0);
        
        // Accumulate vectors for each centroid
        for (uint64_t v = 0; v < n_vectors; ++v) {
            uint32_t cluster = assignments[v];
            const float* vec = vectors + v * config_.dimension;
            
            for (uint32_t d = 0; d < config_.dimension; ++d) {
                new_centroids[cluster][d] += vec[d];
            }
            counts[cluster]++;
        }
        
        // Compute new centroids by averaging
        for (uint32_t c = 0; c < config_.nlist; ++c) {
            if (counts[c] > 0) {
                for (uint32_t d = 0; d < config_.dimension; ++d) {
                    centroids_[c * config_.dimension + d] = 
                        new_centroids[c][d] / counts[c];
                }
            }
        }
    }
    
    std::cout << "Training completed" << std::endl;
}

// Add vectors to the index by assigning to inverted lists
void IVFFlatIndex::add(const float* vectors, const uint64_t* ids, uint64_t n_vectors) {
    std::cout << "Adding " << n_vectors << " vectors to index..." << std::endl;
    
    // Assign vectors to lists
    std::vector<uint32_t> assignments(n_vectors);
    if (config_.use_gpu) {
        assign_to_lists_gpu(vectors, n_vectors, assignments);
    } else {
        assign_to_lists(vectors, n_vectors, assignments);
    }
    
    // Group vectors by list assignment
    std::vector<std::vector<uint64_t>> list_vectors(config_.nlist);
    std::vector<std::vector<uint64_t>> list_ids(config_.nlist);
    
    // Collect vectors for each list
    for (uint64_t v = 0; v < n_vectors; ++v) {
        uint32_t list_id = assignments[v];
        list_vectors[list_id].push_back(v);
        list_ids[list_id].push_back(ids[v]);
    }
    
    // Add vectors to each inverted list
    for (uint32_t list_id = 0; list_id < config_.nlist; ++list_id) {
        if (list_vectors[list_id].empty()) continue;
        
        auto& list = lists_[list_id];
        size_t old_count = list->count;
        size_t new_count = old_count + list_vectors[list_id].size();
        
        // Resize storage
        list->vectors.resize(new_count * config_.dimension);
        list->ids.resize(new_count);
        
        // Copy new vectors
        for (size_t i = 0; i < list_vectors[list_id].size(); ++i) {
            uint64_t vec_idx = list_vectors[list_id][i];
            const float* src = vectors + vec_idx * config_.dimension;
            float* dst = list->vectors.data() + (old_count + i) * config_.dimension;
            std::memcpy(dst, src, config_.dimension * sizeof(float));
            
            list->ids[old_count + i] = list_ids[list_id][i];
        }
        
        list->count = new_count;
        
        // Evict from GPU if list was cached (will reload with new data)
        if (list->on_gpu) {
            evict_list_from_gpu(list_id);
        }
    }
    
    total_vectors_ += n_vectors;
    std::cout << "Added " << n_vectors << " vectors. Total: " << total_vectors_ << std::endl;
}

// Perform similarity search on the index
void IVFFlatIndex::search(const float* queries, uint32_t n_queries,
                         const SearchParams& params,
                         float* distances, uint64_t* indices) {
    
    // Allocate temporary storage for results from each list
    std::vector<std::vector<float>> list_distances(params.nprobe);
    std::vector<std::vector<uint64_t>> list_indices(params.nprobe);
    
    // Process each query
    for (uint32_t q = 0; q < n_queries; ++q) {
        const float* query = queries + q * config_.dimension;
        
        // Select nprobe lists to search
        std::vector<uint32_t> probe_lists = select_nprobe_lists(query, params.nprobe);
        
        // Search each selected list
        for (uint32_t p = 0; p < params.nprobe; ++p) {
            uint32_t list_id = probe_lists[p];
            auto& list = lists_[list_id];
            
            if (list->count == 0) continue;  // Skip empty lists
            
            // Update access statistics
            list->access_count++;
            list->last_access = std::chrono::steady_clock::now();
            
            // Resize result vectors
            list_distances[p].resize(std::min(params.k, (uint32_t)list->count));
            list_indices[p].resize(std::min(params.k, (uint32_t)list->count));
            
            // Search this list using GPU or CPU
            if (config_.use_gpu) {
                // Ensure list is loaded to GPU
                load_list_to_gpu(list_id);
                
                // Use GPU-accelerated search
                cudaStream_t stream = tm_->get_stream();
                search_list_gpu(list_id, query, std::min(params.k, (uint32_t)list->count),
                               list_distances[p].data(), list_indices[p].data(), stream);
                tm_->return_stream(stream);
            } else {
                // Use CPU fallback
                search_list_cpu(list_id, query, std::min(params.k, (uint32_t)list->count),
                               list_distances[p].data(), list_indices[p].data());
            }
        }
        
        // Merge results from all lists
        merge_results(list_distances, list_indices, params.k,
                     distances + q * params.k, indices + q * params.k);
    }
}

// Assign vectors to nearest centroids (lists)
void IVFFlatIndex::assign_to_lists(const float* vectors, uint64_t n_vectors,
                                  std::vector<uint32_t>& assignments) {
    assignments.resize(n_vectors);
    
    // For each vector, find nearest centroid
    for (uint64_t v = 0; v < n_vectors; ++v) {
        const float* vec = vectors + v * config_.dimension;
        float min_dist = std::numeric_limits<float>::max();
        uint32_t best_list = 0;
        
        // Check all centroids
        for (uint32_t c = 0; c < config_.nlist; ++c) {
            const float* centroid = centroids_.data() + c * config_.dimension;
            float dist = 0.0f;
            
            // Compute distance based on metric
            if (config_.metric == kernels::Metric::L2) {
                for (uint32_t d = 0; d < config_.dimension; ++d) {
                    float diff = vec[d] - centroid[d];
                    dist += diff * diff;
                }
            } else if (config_.metric == kernels::Metric::InnerProduct) {
                for (uint32_t d = 0; d < config_.dimension; ++d) {
                    dist += vec[d] * centroid[d];
                }
                dist = -dist;  // Negate for minimization
            }
            
            if (dist < min_dist) {
                min_dist = dist;
                best_list = c;
            }
        }
        
        assignments[v] = best_list;
    }
}

// Select nprobe lists closest to query
std::vector<uint32_t> IVFFlatIndex::select_nprobe_lists(const float* query, uint32_t nprobe) {
    std::vector<std::pair<float, uint32_t>> centroid_distances;
    centroid_distances.reserve(config_.nlist);
    
    // Compute distances to all centroids
    for (uint32_t c = 0; c < config_.nlist; ++c) {
        const float* centroid = centroids_.data() + c * config_.dimension;
        float dist = 0.0f;
        
        // Compute distance based on metric
        if (config_.metric == kernels::Metric::L2) {
            for (uint32_t d = 0; d < config_.dimension; ++d) {
                float diff = query[d] - centroid[d];
                dist += diff * diff;
            }
        } else if (config_.metric == kernels::Metric::InnerProduct) {
            for (uint32_t d = 0; d < config_.dimension; ++d) {
                dist += query[d] * centroid[d];
            }
            dist = -dist;  // Negate for minimization
        }
        
        centroid_distances.emplace_back(dist, c);
    }
    
    // Sort by distance and select top nprobe
    std::partial_sort(centroid_distances.begin(),
                     centroid_distances.begin() + std::min(nprobe, config_.nlist),
                     centroid_distances.end());
    
    std::vector<uint32_t> probe_lists;
    probe_lists.reserve(nprobe);
    
    for (uint32_t p = 0; p < std::min(nprobe, config_.nlist); ++p) {
        probe_lists.push_back(centroid_distances[p].second);
    }
    
    return probe_lists;
}

// CPU-based search within a single list (fallback implementation)
void IVFFlatIndex::search_list_cpu(uint32_t list_id, const float* query, uint32_t k,
                                  float* distances, uint64_t* indices) {
    auto& list = lists_[list_id];
    
    std::vector<std::pair<float, uint64_t>> candidates;
    candidates.reserve(list->count);
    
    // Compute distances to all vectors in the list
    for (size_t i = 0; i < list->count; ++i) {
        const float* vec = list->vectors.data() + i * config_.dimension;
        float dist = 0.0f;
        
        // Compute distance based on metric
        if (config_.metric == kernels::Metric::L2) {
            for (uint32_t d = 0; d < config_.dimension; ++d) {
                float diff = query[d] - vec[d];
                dist += diff * diff;
            }
        } else if (config_.metric == kernels::Metric::InnerProduct) {
            for (uint32_t d = 0; d < config_.dimension; ++d) {
                dist += query[d] * vec[d];
            }
            dist = -dist;  // Negate for minimization
        }
        
        candidates.emplace_back(dist, list->ids[i]);
    }
    
    // Sort and select top-k
    std::partial_sort(candidates.begin(),
                     candidates.begin() + std::min(k, (uint32_t)candidates.size()),
                     candidates.end());
    
    // Copy results
    uint32_t result_count = std::min(k, (uint32_t)candidates.size());
    for (uint32_t i = 0; i < result_count; ++i) {
        distances[i] = candidates[i].first;
        indices[i] = candidates[i].second;
    }
    
    // Fill remaining slots if needed
    for (uint32_t i = result_count; i < k; ++i) {
        distances[i] = std::numeric_limits<float>::max();
        indices[i] = UINT64_MAX;
    }
}

// Load inverted list to GPU memory
void IVFFlatIndex::load_list_to_gpu(uint32_t list_id) {
    auto& list = lists_[list_id];
    
    if (list->on_gpu || list->count == 0) return;  // Already loaded or empty
    
    // Calculate required GPU memory
    size_t vectors_size = list->count * config_.dimension * sizeof(float);
    size_t ids_size = list->count * sizeof(uint64_t);
    size_t total_size = vectors_size + ids_size;
    
    // Check if we have enough GPU memory
    if (gpu_memory_used_ + total_size > config_.max_gpu_memory) {
        // Evict some lists to make space
        // TODO: Implement LFU eviction policy
        return;  // Skip for now
    }
    
    // Allocate GPU memory
    list->gpu_vectors = tm_->allocate_device(vectors_size);
    list->gpu_ids = tm_->allocate_device(ids_size);
    
    if (!list->gpu_vectors || !list->gpu_ids) {
        // Allocation failed, clean up
        if (list->gpu_vectors) tm_->free_device(list->gpu_vectors);
        if (list->gpu_ids) tm_->free_device(list->gpu_ids);
        list->gpu_vectors = nullptr;
        list->gpu_ids = nullptr;
        return;
    }
    
    // Transfer data to GPU
    TransferManager::Transfer vec_transfer{
        .src = list->vectors.data(),
        .dst = list->gpu_vectors,
        .size = vectors_size,
        .kind = cudaMemcpyHostToDevice,
        .stream = nullptr,
        .callback = nullptr
    };
    
    TransferManager::Transfer id_transfer{
        .src = list->ids.data(),
        .dst = list->gpu_ids,
        .size = ids_size,
        .kind = cudaMemcpyHostToDevice,
        .stream = nullptr,
        .callback = nullptr
    };
    
    tm_->enqueue_transfer(vec_transfer);
    tm_->enqueue_transfer(id_transfer);
    tm_->synchronize();
    
    // Update state
    list->on_gpu = true;
    list->gpu_capacity = list->count;
    gpu_memory_used_ += total_size;
}

// Evict inverted list from GPU memory
void IVFFlatIndex::evict_list_from_gpu(uint32_t list_id) {
    auto& list = lists_[list_id];
    
    if (!list->on_gpu) return;  // Not on GPU
    
    // Calculate freed memory
    size_t vectors_size = list->gpu_capacity * config_.dimension * sizeof(float);
    size_t ids_size = list->gpu_capacity * sizeof(uint64_t);
    size_t total_size = vectors_size + ids_size;
    
    // Free GPU memory
    if (list->gpu_vectors) {
        tm_->free_device(list->gpu_vectors);
        list->gpu_vectors = nullptr;
    }
    if (list->gpu_ids) {
        tm_->free_device(list->gpu_ids);
        list->gpu_ids = nullptr;
    }
    
    // Update state
    list->on_gpu = false;
    list->gpu_capacity = 0;
    gpu_memory_used_ -= total_size;
}

// Merge results from multiple inverted lists
void IVFFlatIndex::merge_results(const std::vector<std::vector<float>>& list_distances,
                                const std::vector<std::vector<uint64_t>>& list_indices,
                                uint32_t k, float* distances, uint64_t* indices) {
    
    // Collect all candidates
    std::vector<std::pair<float, uint64_t>> all_candidates;
    
    for (size_t list_idx = 0; list_idx < list_distances.size(); ++list_idx) {
        const auto& dists = list_distances[list_idx];
        const auto& ids = list_indices[list_idx];
        
        for (size_t i = 0; i < dists.size(); ++i) {
            if (ids[i] != UINT64_MAX) {  // Valid result
                all_candidates.emplace_back(dists[i], ids[i]);
            }
        }
    }
    
    // Sort all candidates
    std::sort(all_candidates.begin(), all_candidates.end());
    
    // Remove duplicates (same vector found in multiple lists)
    std::vector<std::pair<float, uint64_t>> unique_candidates;
    std::unordered_set<uint64_t> seen_ids;
    
    for (const auto& candidate : all_candidates) {
        if (seen_ids.find(candidate.second) == seen_ids.end()) {
            unique_candidates.push_back(candidate);
            seen_ids.insert(candidate.second);
        }
    }
    
    // Copy top-k results
    uint32_t result_count = std::min(k, (uint32_t)unique_candidates.size());
    for (uint32_t i = 0; i < result_count; ++i) {
        distances[i] = unique_candidates[i].first;
        indices[i] = unique_candidates[i].second;
    }
    
    // Fill remaining slots
    for (uint32_t i = result_count; i < k; ++i) {
        distances[i] = std::numeric_limits<float>::max();
        indices[i] = UINT64_MAX;
    }
}

// GPU-accelerated search within a single list
void IVFFlatIndex::search_list_gpu(uint32_t list_id, const float* query, uint32_t k,
                                   float* distances, uint64_t* indices, cudaStream_t stream) {
    auto& list = lists_[list_id];
    
    // Ensure list is on GPU
    if (!list->on_gpu || list->count == 0) {
        // Fall back to CPU if GPU load failed
        search_list_cpu(list_id, query, k, distances, indices);
        return;
    }
    
    // Allocate GPU memory for query and results
    void* d_query = tm_->allocate_device(config_.dimension * sizeof(float));
    void* d_distances = tm_->allocate_device(k * sizeof(float));
    void* d_indices = tm_->allocate_device(k * sizeof(uint64_t));
    
    if (!d_query || !d_distances || !d_indices) {
        // GPU memory allocation failed, fall back to CPU
        if (d_query) tm_->free_device(d_query);
        if (d_distances) tm_->free_device(d_distances);
        if (d_indices) tm_->free_device(d_indices);
        
        search_list_cpu(list_id, query, k, distances, indices);
        return;
    }
    
    // Transfer query to GPU
    TransferManager::Transfer query_transfer{
        .src = const_cast<float*>(query),
        .dst = d_query,
        .size = config_.dimension * sizeof(float),
        .kind = cudaMemcpyHostToDevice,
        .stream = stream,
        .callback = nullptr
    };
    tm_->enqueue_transfer(query_transfer);
    
    // Launch GPU kernel for bruteforce search on this list
    try {
        kernels::launch_bruteforce_search<float>(
            static_cast<const float*>(list->gpu_vectors),  // Database vectors on GPU
            static_cast<const float*>(d_query),            // Query vector on GPU
            static_cast<const uint64_t*>(list->gpu_ids),   // Vector IDs on GPU
            list->count,                                    // Number of vectors in list
            1,                                              // Single query
            config_.dimension,                              // Vector dimension
            k,                                              // Top-K to return
            static_cast<float*>(d_distances),               // Output distances
            static_cast<uint64_t*>(d_indices),              // Output indices
            config_.metric,                                 // Distance metric
            stream                                          // CUDA stream
        );
        
        // Check for kernel launch errors
        cudaError_t launch_error = cudaGetLastError();
        if (launch_error != cudaSuccess) {
            std::cerr << "CUDA kernel launch failed: " << cudaGetErrorString(launch_error) << std::endl;
            // Fall back to CPU
            search_list_cpu(list_id, query, k, distances, indices);
        } else {
            // Transfer results back to host
            TransferManager::Transfer dist_transfer{
                .src = d_distances,
                .dst = distances,
                .size = k * sizeof(float),
                .kind = cudaMemcpyDeviceToHost,
                .stream = stream,
                .callback = nullptr
            };
            
            TransferManager::Transfer idx_transfer{
                .src = d_indices,
                .dst = indices,
                .size = k * sizeof(uint64_t),
                .kind = cudaMemcpyDeviceToHost,
                .stream = stream,
                .callback = nullptr
            };
            
            tm_->enqueue_transfer(dist_transfer);
            tm_->enqueue_transfer(idx_transfer);
            
            // Wait for completion
            tm_->synchronize_stream(stream);
        }
        
    } catch (const std::exception& e) {
        std::cerr << "GPU search failed: " << e.what() << std::endl;
        // Fall back to CPU
        search_list_cpu(list_id, query, k, distances, indices);
    }
    
    // Clean up GPU memory
    tm_->free_device(d_query);
    tm_->free_device(d_distances);
    tm_->free_device(d_indices);
}

// GPU-accelerated k-means assignment for training
void IVFFlatIndex::assign_to_lists_gpu(const float* vectors, uint64_t n_vectors,
                                       std::vector<uint32_t>& assignments) {
    assignments.resize(n_vectors);
    
    // Allocate GPU memory
    size_t vectors_size = n_vectors * config_.dimension * sizeof(float);
    size_t centroids_size = config_.nlist * config_.dimension * sizeof(float);
    size_t assignments_size = n_vectors * sizeof(uint32_t);
    
    void* d_vectors = tm_->allocate_device(vectors_size);
    void* d_centroids = tm_->allocate_device(centroids_size);
    void* d_assignments = tm_->allocate_device(assignments_size);
    
    if (!d_vectors || !d_centroids || !d_assignments) {
        // GPU allocation failed, use CPU fallback
        if (d_vectors) tm_->free_device(d_vectors);
        if (d_centroids) tm_->free_device(d_centroids);
        if (d_assignments) tm_->free_device(d_assignments);
        
        assign_to_lists(vectors, n_vectors, assignments);
        return;
    }
    
    try {
        // Transfer data to GPU
        cudaStream_t stream = tm_->get_stream();
        
        TransferManager::Transfer vec_transfer{
            .src = const_cast<float*>(vectors),
            .dst = d_vectors,
            .size = vectors_size,
            .kind = cudaMemcpyHostToDevice,
            .stream = stream,
            .callback = nullptr
        };
        
        TransferManager::Transfer centroid_transfer{
            .src = centroids_.data(),
            .dst = d_centroids,
            .size = centroids_size,
            .kind = cudaMemcpyHostToDevice,
            .stream = stream,
            .callback = nullptr
        };
        
        tm_->enqueue_transfer(vec_transfer);
        tm_->enqueue_transfer(centroid_transfer);
        
        // Launch k-means assignment kernel
        kernels::launch_kmeans_assign<float>(
            static_cast<const float*>(d_vectors),
            static_cast<const float*>(d_centroids),
            static_cast<uint32_t*>(d_assignments),
            nullptr,  // Don't need distances
            n_vectors,
            config_.nlist,
            config_.dimension,
            stream
        );
        
        // Transfer results back
        TransferManager::Transfer assign_transfer{
            .src = d_assignments,
            .dst = assignments.data(),
            .size = assignments_size,
            .kind = cudaMemcpyDeviceToHost,
            .stream = stream,
            .callback = nullptr
        };
        
        tm_->enqueue_transfer(assign_transfer);
        tm_->synchronize_stream(stream);
        tm_->return_stream(stream);
        
    } catch (const std::exception& e) {
        std::cerr << "GPU k-means assignment failed: " << e.what() << std::endl;
        // Fall back to CPU
        assign_to_lists(vectors, n_vectors, assignments);
    }
    
    // Clean up
    tm_->free_device(d_vectors);
    tm_->free_device(d_centroids);
    tm_->free_device(d_assignments);
}

// Get current GPU memory usage
size_t IVFFlatIndex::get_gpu_memory_usage() const {
    return gpu_memory_used_;
}

} // namespace vdb