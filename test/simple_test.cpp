// Simple test to validate CUDA kernels and basic functionality
#include <iostream>
#include <vector>
#include <random>
#include <chrono>
#include <cuda_runtime.h>
#include "../engine/ivf_flat_index.h"
#include "../engine/transfer_manager.h"

using namespace vdb;

// Test CUDA device availability
bool test_cuda_device() {
    std::cout << "Testing CUDA device..." << std::endl;
    
    int device_count;
    cudaError_t err = cudaGetDeviceCount(&device_count);
    
    if (err != cudaSuccess) {
        std::cerr << "CUDA error: " << cudaGetErrorString(err) << std::endl;
        return false;
    }
    
    if (device_count == 0) {
        std::cerr << "No CUDA devices found" << std::endl;
        return false;
    }
    
    std::cout << "Found " << device_count << " CUDA device(s)" << std::endl;
    
    // Get device properties
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    std::cout << "Device 0: " << prop.name 
              << " (Compute " << prop.major << "." << prop.minor << ")"
              << " Memory: " << prop.totalGlobalMem / (1024*1024*1024) << " GB"
              << std::endl;
    
    return true;
}

// Test basic memory allocation
bool test_memory_allocation() {
    std::cout << "Testing memory allocation..." << std::endl;
    
    // Test CUDA memory allocation
    float* d_test;
    size_t test_size = 1024 * sizeof(float);
    
    cudaError_t err = cudaMalloc(&d_test, test_size);
    if (err != cudaSuccess) {
        std::cerr << "CUDA malloc failed: " << cudaGetErrorString(err) << std::endl;
        return false;
    }
    
    cudaFree(d_test);
    std::cout << "CUDA memory allocation: OK" << std::endl;
    
    return true;
}

// Test TransferManager
bool test_transfer_manager() {
    std::cout << "Testing TransferManager..." << std::endl;
    
    try {
        TransferManager::Config tm_config;
        tm_config.pinned_pool_size = 64 << 20;  // 64MB
        tm_config.device_pool_size = 128 << 20; // 128MB
        
        TransferManager tm(tm_config);
        
        // Test pinned memory allocation
        void* pinned_ptr = tm.allocate_pinned(1024);
        if (!pinned_ptr) {
            std::cerr << "Failed to allocate pinned memory" << std::endl;
            return false;
        }
        tm.free_pinned(pinned_ptr);
        
        // Test device memory allocation
        void* device_ptr = tm.allocate_device(1024);
        if (!device_ptr) {
            std::cerr << "Failed to allocate device memory" << std::endl;
            return false;
        }
        tm.free_device(device_ptr);
        
        std::cout << "TransferManager: OK" << std::endl;
        return true;
        
    } catch (const std::exception& e) {
        std::cerr << "TransferManager test failed: " << e.what() << std::endl;
        return false;
    }
}

// Test IVF-Flat index with small dataset
bool test_ivf_flat_index() {
    std::cout << "Testing IVF-Flat index..." << std::endl;
    
    try {
        // Create transfer manager
        TransferManager::Config tm_config;
        tm_config.pinned_pool_size = 64 << 20;
        tm_config.device_pool_size = 128 << 20;
        TransferManager tm(tm_config);
        
        // Create index configuration
        IVFFlatIndex::Config config;
        config.dimension = 64;
        config.nlist = 16;
        config.metric = kernels::Metric::L2;
        config.use_gpu = false;  // Use CPU for now
        
        IVFFlatIndex index(config, &tm);
        
        // Generate small test dataset
        const size_t n_vectors = 1000;
        const size_t n_queries = 10;
        
        std::vector<float> vectors(n_vectors * config.dimension);
        std::vector<uint64_t> ids(n_vectors);
        std::vector<float> queries(n_queries * config.dimension);
        
        // Fill with random data
        std::mt19937 gen(42);
        std::normal_distribution<float> dist(0.0f, 1.0f);
        
        for (size_t i = 0; i < vectors.size(); ++i) {
            vectors[i] = dist(gen);
        }
        for (size_t i = 0; i < queries.size(); ++i) {
            queries[i] = dist(gen);
        }
        for (size_t i = 0; i < n_vectors; ++i) {
            ids[i] = i;
        }
        
        // Train index
        auto train_start = std::chrono::high_resolution_clock::now();
        index.train(vectors.data(), std::min(n_vectors, size_t(100)));
        auto train_end = std::chrono::high_resolution_clock::now();
        
        auto train_time = std::chrono::duration_cast<std::chrono::milliseconds>
            (train_end - train_start).count();
        std::cout << "Training time: " << train_time << " ms" << std::endl;
        
        // Add vectors
        auto add_start = std::chrono::high_resolution_clock::now();
        index.add(vectors.data(), ids.data(), n_vectors);
        auto add_end = std::chrono::high_resolution_clock::now();
        
        auto add_time = std::chrono::duration_cast<std::chrono::milliseconds>
            (add_end - add_start).count();
        std::cout << "Adding time: " << add_time << " ms" << std::endl;
        
        // Search
        const uint32_t k = 5;
        std::vector<float> distances(n_queries * k);
        std::vector<uint64_t> result_ids(n_queries * k);
        
        IVFFlatIndex::SearchParams params;
        params.nprobe = 4;
        params.k = k;
        
        auto search_start = std::chrono::high_resolution_clock::now();
        index.search(queries.data(), n_queries, params,
                    distances.data(), result_ids.data());
        auto search_end = std::chrono::high_resolution_clock::now();
        
        auto search_time = std::chrono::duration_cast<std::chrono::microseconds>
            (search_end - search_start).count();
        std::cout << "Search time: " << search_time << " μs" << std::endl;
        
        // Validate results
        bool valid_results = true;
        for (size_t q = 0; q < n_queries; ++q) {
            std::cout << "Query " << q << " results: ";
            for (uint32_t i = 0; i < k; ++i) {
                uint64_t result_id = result_ids[q * k + i];
                float distance = distances[q * k + i];
                
                std::cout << "(" << result_id << ", " << distance << ") ";
                
                if (result_id >= n_vectors && result_id != UINT64_MAX) {
                    valid_results = false;
                }
            }
            std::cout << std::endl;
        }
        
        if (!valid_results) {
            std::cerr << "Invalid search results detected" << std::endl;
            return false;
        }
        
        std::cout << "IVF-Flat index: OK" << std::endl;
        std::cout << "Total vectors: " << index.get_total_vectors() << std::endl;
        std::cout << "GPU memory used: " << index.get_gpu_memory_usage() << " bytes" << std::endl;
        
        return true;
        
    } catch (const std::exception& e) {
        std::cerr << "IVF-Flat index test failed: " << e.what() << std::endl;
        return false;
    }
}

int main() {
    std::cout << "=== CUDA Vector Database Simple Test ===" << std::endl;
    
    bool all_passed = true;
    
    // Run tests
    all_passed &= test_cuda_device();
    all_passed &= test_memory_allocation();
    all_passed &= test_transfer_manager();
    all_passed &= test_ivf_flat_index();
    
    std::cout << "\n=== Test Results ===" << std::endl;
    if (all_passed) {
        std::cout << "All tests PASSED ✓" << std::endl;
        return 0;
    } else {
        std::cout << "Some tests FAILED ✗" << std::endl;
        return 1;
    }
}