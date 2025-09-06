// Performance comparison test between GPU and CPU implementations
#include <iostream>
#include <vector>
#include <random>
#include <chrono>
#include <iomanip>
#include <cuda_runtime.h>
#include "../engine/ivf_flat_index.h"
#include "../engine/transfer_manager.h"

using namespace vdb;

struct BenchmarkResult {
    double train_time_ms;
    double add_time_ms;
    double search_time_ms;
    double qps;
    double avg_latency_ms;
    size_t gpu_memory_bytes;
};

class GPUvsCPUBenchmark {
public:
    struct Config {
        size_t num_vectors = 50000;
        size_t num_queries = 1000;
        uint32_t dimension = 128;
        uint32_t nlist = 128;
        uint32_t nprobe = 8;
        uint32_t k = 10;
        kernels::Metric metric = kernels::Metric::L2;
    };
    
    explicit GPUvsCPUBenchmark(const Config& config) : config_(config) {
        // Setup transfer manager
        tm_config_.pinned_pool_size = 256 << 20;  // 256MB
        tm_config_.device_pool_size = 512 << 20;  // 512MB
        tm_ = std::make_unique<TransferManager>(tm_config_);
        
        // Generate test data
        generate_test_data();
    }
    
    void run_comparison() {
        std::cout << "=== GPU vs CPU Performance Comparison ===" << std::endl;
        std::cout << "Dataset: " << config_.num_vectors << " vectors, " 
                  << config_.dimension << "D" << std::endl;
        std::cout << "Queries: " << config_.num_queries << std::endl;
        std::cout << "Index params: nlist=" << config_.nlist 
                  << ", nprobe=" << config_.nprobe << ", k=" << config_.k << std::endl;
        std::cout << std::endl;
        
        // Test CPU implementation
        std::cout << "Running CPU benchmark..." << std::endl;
        auto cpu_result = run_benchmark(false);
        
        // Test GPU implementation (if available)
        std::cout << "Running GPU benchmark..." << std::endl;
        auto gpu_result = run_benchmark(true);
        
        // Compare results
        print_comparison(cpu_result, gpu_result);
    }
    
private:
    Config config_;
    TransferManager::Config tm_config_;
    std::unique_ptr<TransferManager> tm_;
    
    std::vector<float> vectors_;
    std::vector<uint64_t> ids_;
    std::vector<float> queries_;
    
    void generate_test_data() {
        std::cout << "Generating test data..." << std::endl;
        
        // Resize vectors
        vectors_.resize(config_.num_vectors * config_.dimension);
        ids_.resize(config_.num_vectors);
        queries_.resize(config_.num_queries * config_.dimension);
        
        // Random number generator with fixed seed for reproducibility
        std::mt19937 gen(12345);
        std::normal_distribution<float> dist(0.0f, 1.0f);
        
        // Generate database vectors
        for (size_t i = 0; i < vectors_.size(); ++i) {
            vectors_[i] = dist(gen);
        }
        
        // Generate query vectors
        for (size_t i = 0; i < queries_.size(); ++i) {
            queries_[i] = dist(gen);
        }
        
        // Generate IDs
        for (size_t i = 0; i < config_.num_vectors; ++i) {
            ids_[i] = i;
        }
        
        // Normalize for cosine similarity if needed
        if (config_.metric == kernels::Metric::Cosine) {
            normalize_vectors(vectors_);
            normalize_vectors(queries_);
        }
        
        std::cout << "Test data generated successfully" << std::endl;
    }
    
    void normalize_vectors(std::vector<float>& vecs) {
        size_t n_vecs = vecs.size() / config_.dimension;
        
        for (size_t i = 0; i < n_vecs; ++i) {
            float* vec = vecs.data() + i * config_.dimension;
            
            // Compute L2 norm
            float norm = 0.0f;
            for (uint32_t d = 0; d < config_.dimension; ++d) {
                norm += vec[d] * vec[d];
            }
            norm = std::sqrt(norm + 1e-8f);
            
            // Normalize
            for (uint32_t d = 0; d < config_.dimension; ++d) {
                vec[d] /= norm;
            }
        }
    }
    
    BenchmarkResult run_benchmark(bool use_gpu) {
        BenchmarkResult result = {};
        
        try {
            // Create index configuration
            IVFFlatIndex::Config index_config;
            index_config.dimension = config_.dimension;
            index_config.nlist = config_.nlist;
            index_config.metric = config_.metric;
            index_config.use_gpu = use_gpu;
            index_config.max_gpu_memory = 256 << 20;  // 256MB
            
            IVFFlatIndex index(index_config, tm_.get());
            
            // Training phase
            auto train_start = std::chrono::high_resolution_clock::now();
            // Use subset for training
            size_t train_size = std::min(config_.num_vectors, size_t(10000));
            index.train(vectors_.data(), train_size);
            auto train_end = std::chrono::high_resolution_clock::now();
            
            result.train_time_ms = std::chrono::duration<double, std::milli>
                (train_end - train_start).count();
            
            // Adding phase
            auto add_start = std::chrono::high_resolution_clock::now();
            index.add(vectors_.data(), ids_.data(), config_.num_vectors);
            auto add_end = std::chrono::high_resolution_clock::now();
            
            result.add_time_ms = std::chrono::duration<double, std::milli>
                (add_end - add_start).count();
            
            // Search phase
            std::vector<float> distances(config_.num_queries * config_.k);
            std::vector<uint64_t> result_ids(config_.num_queries * config_.k);
            
            IVFFlatIndex::SearchParams params;
            params.nprobe = config_.nprobe;
            params.k = config_.k;
            
            // Warmup run
            index.search(queries_.data(), std::min(config_.num_queries, size_t(10)), 
                        params, distances.data(), result_ids.data());
            
            // Timed search run
            auto search_start = std::chrono::high_resolution_clock::now();
            index.search(queries_.data(), config_.num_queries, params,
                        distances.data(), result_ids.data());
            auto search_end = std::chrono::high_resolution_clock::now();
            
            result.search_time_ms = std::chrono::duration<double, std::milli>
                (search_end - search_start).count();
            
            // Calculate metrics
            result.qps = config_.num_queries * 1000.0 / result.search_time_ms;
            result.avg_latency_ms = result.search_time_ms / config_.num_queries;
            result.gpu_memory_bytes = index.get_gpu_memory_usage();
            
            // Validate results
            validate_results(distances, result_ids);
            
        } catch (const std::exception& e) {
            std::cerr << (use_gpu ? "GPU" : "CPU") << " benchmark failed: " 
                      << e.what() << std::endl;
            result = {};  // Zero out results on failure
        }
        
        return result;
    }
    
    void validate_results(const std::vector<float>& distances, 
                         const std::vector<uint64_t>& indices) {
        bool valid = true;
        size_t invalid_count = 0;
        
        for (size_t q = 0; q < config_.num_queries; ++q) {
            for (uint32_t k = 0; k < config_.k; ++k) {
                size_t idx = q * config_.k + k;
                
                // Check for valid indices
                if (indices[idx] >= config_.num_vectors && indices[idx] != UINT64_MAX) {
                    valid = false;
                    invalid_count++;
                }
                
                // Check for reasonable distances (not infinite/NaN)
                if (!std::isfinite(distances[idx]) || distances[idx] < 0) {
                    valid = false;
                    invalid_count++;
                }
            }
        }
        
        if (!valid) {
            std::cout << "WARNING: " << invalid_count << " invalid results detected" << std::endl;
        }
    }
    
    void print_comparison(const BenchmarkResult& cpu_result, 
                         const BenchmarkResult& gpu_result) {
        
        std::cout << std::fixed << std::setprecision(2);
        std::cout << "\n=== Benchmark Results ===" << std::endl;
        
        // Print table header
        std::cout << std::left << std::setw(15) << "Metric" 
                  << std::setw(15) << "CPU" << std::setw(15) << "GPU" 
                  << std::setw(15) << "Speedup" << std::endl;
        std::cout << std::string(60, '-') << std::endl;
        
        // Training time
        double train_speedup = cpu_result.train_time_ms > 0 ? 
            cpu_result.train_time_ms / gpu_result.train_time_ms : 0.0;
        std::cout << std::setw(15) << "Train (ms)" 
                  << std::setw(15) << cpu_result.train_time_ms
                  << std::setw(15) << gpu_result.train_time_ms
                  << std::setw(15) << (train_speedup > 0 ? std::to_string(train_speedup) + "x" : "N/A")
                  << std::endl;
        
        // Add time
        double add_speedup = cpu_result.add_time_ms > 0 ? 
            cpu_result.add_time_ms / gpu_result.add_time_ms : 0.0;
        std::cout << std::setw(15) << "Add (ms)" 
                  << std::setw(15) << cpu_result.add_time_ms
                  << std::setw(15) << gpu_result.add_time_ms
                  << std::setw(15) << (add_speedup > 0 ? std::to_string(add_speedup) + "x" : "N/A")
                  << std::endl;
        
        // Search time
        double search_speedup = cpu_result.search_time_ms > 0 ? 
            cpu_result.search_time_ms / gpu_result.search_time_ms : 0.0;
        std::cout << std::setw(15) << "Search (ms)" 
                  << std::setw(15) << cpu_result.search_time_ms
                  << std::setw(15) << gpu_result.search_time_ms
                  << std::setw(15) << (search_speedup > 0 ? std::to_string(search_speedup) + "x" : "N/A")
                  << std::endl;
        
        // QPS
        double qps_speedup = cpu_result.qps > 0 ? 
            gpu_result.qps / cpu_result.qps : 0.0;
        std::cout << std::setw(15) << "QPS" 
                  << std::setw(15) << (int)cpu_result.qps
                  << std::setw(15) << (int)gpu_result.qps
                  << std::setw(15) << (qps_speedup > 0 ? std::to_string(qps_speedup) + "x" : "N/A")
                  << std::endl;
        
        // Latency
        std::cout << std::setw(15) << "Latency (ms)" 
                  << std::setw(15) << cpu_result.avg_latency_ms
                  << std::setw(15) << gpu_result.avg_latency_ms
                  << std::setw(15) << "-"
                  << std::endl;
        
        // GPU Memory
        std::cout << std::setw(15) << "GPU Mem (MB)" 
                  << std::setw(15) << "-"
                  << std::setw(15) << (gpu_result.gpu_memory_bytes / (1024 * 1024))
                  << std::setw(15) << "-"
                  << std::endl;
        
        std::cout << std::endl;
        
        // Summary
        if (search_speedup > 1.1) {
            std::cout << "🚀 GPU provides " << std::setprecision(1) 
                      << search_speedup << "x speedup for search!" << std::endl;
        } else if (gpu_result.search_time_ms == 0) {
            std::cout << "❌ GPU benchmark failed - check CUDA installation" << std::endl;
        } else {
            std::cout << "⚠️  GPU performance similar to CPU - check GPU utilization" << std::endl;
        }
    }
};

int main(int argc, char** argv) {
    std::cout << "=== CUDA Vector Database GPU vs CPU Benchmark ===" << std::endl;
    
    // Check CUDA availability
    int device_count;
    cudaError_t err = cudaGetDeviceCount(&device_count);
    if (err != cudaSuccess || device_count == 0) {
        std::cout << "No CUDA devices available - running CPU-only test" << std::endl;
        return 1;
    }
    
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    std::cout << "GPU: " << prop.name << std::endl;
    std::cout << "CUDA Compute: " << prop.major << "." << prop.minor << std::endl;
    std::cout << "Global Memory: " << prop.totalGlobalMem / (1024*1024*1024) << " GB" << std::endl;
    std::cout << std::endl;
    
    // Parse command line arguments
    GPUvsCPUBenchmark::Config config;
    if (argc > 1) config.num_vectors = std::stoul(argv[1]);
    if (argc > 2) config.num_queries = std::stoul(argv[2]);
    if (argc > 3) config.dimension = std::stoul(argv[3]);
    if (argc > 4) config.nlist = std::stoul(argv[4]);
    
    try {
        GPUvsCPUBenchmark benchmark(config);
        benchmark.run_comparison();
        
    } catch (const std::exception& e) {
        std::cerr << "Benchmark failed: " << e.what() << std::endl;
        return 1;
    }
    
    return 0;
}