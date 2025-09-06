// Standard I/O for console output
#include <iostream>
// STL containers
#include <vector>
// Random number generation
#include <random>
// High-resolution timing
#include <chrono>
// File I/O for results
#include <fstream>
// Threading support (unused but available)
#include <thread>
// IVF-Flat index implementation
#include "../engine/ivf_flat_index.h"
// Transfer manager for GPU memory
#include "../engine/transfer_manager.h"

// Main namespace for vector database
namespace vdb {
// Benchmark namespace for performance testing
namespace bench {

// Benchmark class for comprehensive performance testing
class Benchmark {
public:
    // Configuration structure for benchmark parameters
    struct Config {
        size_t num_vectors = 1000000;                       // Number of database vectors to index
        size_t num_queries = 10000;                         // Number of query vectors to search
        uint32_t dimension = 128;                           // Vector dimension (affects memory/compute)
        uint32_t nlist = 1024;                              // Number of IVF clusters
        uint32_t nprobe = 10;                               // Number of clusters to probe during search
        uint32_t k = 10;                                    // Number of nearest neighbors to return
        kernels::Metric metric = kernels::Metric::L2;       // Distance metric for similarity
        bool use_gpu = true;                                // Whether to use GPU acceleration
        std::string output_file = "benchmark_results.csv";  // File to save results
    };
    
    // Constructor initializes transfer manager with memory pools
    explicit Benchmark(const Config& config) : config_(config) {
        tm_config_.pinned_pool_size = 1ULL << 30;          // 1GB pinned memory for fast H2D transfers
        tm_config_.device_pool_size = 4ULL << 30;          // 4GB GPU memory pool
        tm_ = std::make_unique<TransferManager>(tm_config_);// Create transfer manager instance
    }
    
    // Main benchmark execution function
    void run() {
        // Generate synthetic data for testing
        std::cout << "Generating random vectors..." << std::endl;
        auto vectors = generate_random_vectors(config_.num_vectors);  // Database vectors
        auto queries = generate_random_vectors(config_.num_queries);   // Query vectors  
        auto ids = generate_ids(config_.num_vectors);                  // Vector IDs
        
        // Create and configure IVF-Flat index
        std::cout << "Creating index..." << std::endl;
        IVFFlatIndex::Config index_config;                  // Index configuration
        index_config.dimension = config_.dimension;        // Vector dimension
        index_config.nlist = config_.nlist;                // Number of inverted lists
        index_config.metric = config_.metric;              // Distance metric
        index_config.use_gpu = config_.use_gpu;            // GPU acceleration flag
        
        // Instantiate index with configuration and transfer manager
        IVFFlatIndex index(index_config, tm_.get());
        
        // Phase 1: Train IVF centroids using k-means clustering
        std::cout << "Training index..." << std::endl;
        auto train_start = std::chrono::high_resolution_clock::now();      // Start timing
        // Use subset for training (100k vectors max for faster training)
        index.train(vectors.data(), std::min(size_t(100000), config_.num_vectors));
        auto train_end = std::chrono::high_resolution_clock::now();        // End timing
        // Calculate training time in seconds
        double train_time = std::chrono::duration<double>(train_end - train_start).count();
        
        // Phase 2: Add all vectors to inverted lists
        std::cout << "Adding vectors..." << std::endl;
        auto add_start = std::chrono::high_resolution_clock::now();        // Start timing
        index.add(vectors.data(), ids.data(), config_.num_vectors);        // Add all vectors
        auto add_end = std::chrono::high_resolution_clock::now();          // End timing
        // Calculate indexing time in seconds
        double add_time = std::chrono::duration<double>(add_end - add_start).count();
        
        // Phase 3: Execute search queries
        std::cout << "Running searches..." << std::endl;
        // Allocate output arrays for results
        std::vector<float> distances(config_.num_queries * config_.k);     // Distance results
        std::vector<uint64_t> indices(config_.num_queries * config_.k);    // Index results
        
        // Configure search parameters
        IVFFlatIndex::SearchParams params;
        params.nprobe = config_.nprobe;                     // Number of lists to probe
        params.k = config_.k;                               // Top-K neighbors to return
        
        // Execute batch search and measure time
        auto search_start = std::chrono::high_resolution_clock::now();     // Start timing
        index.search(queries.data(), config_.num_queries, params,          // Execute search
                    distances.data(), indices.data());
        auto search_end = std::chrono::high_resolution_clock::now();       // End timing
        // Calculate search time in seconds
        double search_time = std::chrono::duration<double>(search_end - search_start).count();
        
        // Calculate performance metrics
        double qps = config_.num_queries / search_time;                    // Queries per second
        double latency_ms = (search_time / config_.num_queries) * 1000;    // Average latency in ms
        
        // Display benchmark results
        std::cout << "\n=== Benchmark Results ===" << std::endl;
        std::cout << "Vectors: " << config_.num_vectors << std::endl;
        std::cout << "Dimension: " << config_.dimension << std::endl;
        std::cout << "nlist: " << config_.nlist << std::endl;
        std::cout << "nprobe: " << config_.nprobe << std::endl;
        std::cout << "Train time: " << train_time << " s" << std::endl;
        std::cout << "Add time: " << add_time << " s" << std::endl;
        std::cout << "Search time: " << search_time << " s" << std::endl;
        std::cout << "QPS: " << qps << std::endl;
        std::cout << "Latency: " << latency_ms << " ms" << std::endl;
        // Convert GPU memory usage from bytes to MB
        std::cout << "GPU memory: " << index.get_gpu_memory_usage() / (1024.0 * 1024.0) 
                  << " MB" << std::endl;
        
        // Save results to CSV file for analysis
        save_results(train_time, add_time, search_time, qps, latency_ms);
    }
    
private:
    Config config_;                                         // Benchmark configuration
    TransferManager::Config tm_config_;                     // Transfer manager config
    std::unique_ptr<TransferManager> tm_;                   // Transfer manager instance
    
    // Generate random vectors for testing
    std::vector<float> generate_random_vectors(size_t n) {
        std::vector<float> vectors(n * config_.dimension);  // Allocate vector storage
        std::mt19937 gen(42);                               // Deterministic random generator
        std::normal_distribution<float> dist(0.0f, 1.0f);  // Normal distribution (mean=0, std=1)
        
        // Fill vector with random values
        for (size_t i = 0; i < vectors.size(); ++i) {
            vectors[i] = dist(gen);                         // Sample from normal distribution
        }
        
        // Normalize vectors for cosine similarity if needed
        if (config_.metric == kernels::Metric::Cosine) {
            for (size_t i = 0; i < n; ++i) {
                float norm = 0.0f;                          // Accumulator for L2 norm
                // Compute squared L2 norm
                for (size_t j = 0; j < config_.dimension; ++j) {
                    float val = vectors[i * config_.dimension + j];
                    norm += val * val;
                }
                norm = std::sqrt(norm);                     // Take square root to get L2 norm
                // Normalize each component by the norm
                for (size_t j = 0; j < config_.dimension; ++j) {
                    vectors[i * config_.dimension + j] /= norm;
                }
            }
        }
        
        return vectors;                                     // Return generated vectors
    }
    
    // Generate sequential vector IDs
    std::vector<uint64_t> generate_ids(size_t n) {
        std::vector<uint64_t> ids(n);                       // Allocate ID storage
        for (size_t i = 0; i < n; ++i) {
            ids[i] = i;                                     // Sequential IDs starting from 0
        }
        return ids;                                         // Return ID array
    }
    
    // Save benchmark results to CSV file for analysis
    void save_results(double train_time, double add_time, double search_time,
                     double qps, double latency_ms) {
        std::ofstream file(config_.output_file, std::ios::app);  // Open file in append mode
        if (!file.is_open()) {                              // Check if file opened successfully
            std::cerr << "Failed to open output file" << std::endl;
            return;
        }
        
        // Write CSV header only once
        static bool header_written = false;
        if (!header_written) {
            file << "vectors,dimension,nlist,nprobe,k,train_time,add_time,"
                 << "search_time,qps,latency_ms\n";         // CSV column headers
            header_written = true;                          // Mark header as written
        }
        
        // Write benchmark results as CSV row
        file << config_.num_vectors << ","                  // Number of vectors
             << config_.dimension << ","                    // Vector dimension
             << config_.nlist << ","                        // Number of lists
             << config_.nprobe << ","                       // Number of probes
             << config_.k << ","                            // Top-K value
             << train_time << ","                           // Training time
             << add_time << ","                             // Indexing time
             << search_time << ","                          // Search time
             << qps << ","                                  // Queries per second
             << latency_ms << "\n";                         // Average latency
    }
};

} // namespace bench
} // namespace vdb

// Main function - entry point for benchmark executable
int main(int argc, char** argv) {
    vdb::bench::Benchmark::Config config;                  // Default configuration
    
    // Parse command-line arguments to override defaults
    if (argc > 1) config.num_vectors = std::stoul(argv[1]); // Number of vectors
    if (argc > 2) config.dimension = std::stoul(argv[2]);   // Vector dimension
    if (argc > 3) config.nlist = std::stoul(argv[3]);       // Number of lists
    if (argc > 4) config.nprobe = std::stoul(argv[4]);      // Number of probes
    
    // Create and run benchmark
    vdb::bench::Benchmark benchmark(config);               // Create benchmark instance
    benchmark.run();                                        // Execute benchmark
    
    return 0;                                               // Success exit code
}