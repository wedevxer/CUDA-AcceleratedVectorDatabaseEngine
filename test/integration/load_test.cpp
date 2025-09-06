#include <iostream>
#include <thread>
#include <vector>
#include <chrono>
#include <atomic>
#include <random>
#include <grpcpp/grpcpp.h>
#include "../../proto/vdb.grpc.pb.h"

class VDBLoadTest {
public:
    struct Config {
        std::string server_address = "localhost:50051";
        int num_threads = 8;
        int requests_per_thread = 100;
        int query_dimension = 128;
        int queries_per_request = 1;
        int topk = 10;
        std::chrono::seconds duration{60};
        bool use_duration = false;
    };
    
    explicit VDBLoadTest(const Config& config) : config_(config) {
        // Create gRPC channel
        channel_ = grpc::CreateChannel(config_.server_address, 
                                      grpc::InsecureChannelCredentials());
        
        // Wait for server to be ready
        if (!WaitForServer()) {
            throw std::runtime_error("Failed to connect to server");
        }
    }
    
    void RunLoadTest() {
        std::cout << "=== VDB Load Test ===" << std::endl;
        std::cout << "Server: " << config_.server_address << std::endl;
        std::cout << "Threads: " << config_.num_threads << std::endl;
        std::cout << "Requests per thread: " << config_.requests_per_thread << std::endl;
        std::cout << "Query dimension: " << config_.query_dimension << std::endl;
        std::cout << std::endl;
        
        // Statistics
        std::atomic<int> total_requests{0};
        std::atomic<int> successful_requests{0};
        std::atomic<int> failed_requests{0};
        std::atomic<long long> total_latency_ms{0};
        
        auto start_time = std::chrono::high_resolution_clock::now();
        
        // Start worker threads
        std::vector<std::thread> threads;
        for (int i = 0; i < config_.num_threads; ++i) {
            threads.emplace_back([this, i, &total_requests, &successful_requests, 
                                &failed_requests, &total_latency_ms]() {
                RunWorker(i, total_requests, successful_requests, 
                         failed_requests, total_latency_ms);
            });
        }
        
        // Monitor progress
        std::thread monitor([&]() {
            while (true) {
                std::this_thread::sleep_for(std::chrono::seconds(5));
                
                auto current_time = std::chrono::high_resolution_clock::now();
                auto elapsed = std::chrono::duration_cast<std::chrono::seconds>
                              (current_time - start_time).count();
                
                int completed = total_requests.load();
                int success = successful_requests.load();
                int failed = failed_requests.load();
                
                if (completed > 0) {
                    double qps = static_cast<double>(completed) / elapsed;
                    double success_rate = static_cast<double>(success) / completed * 100;
                    double avg_latency = static_cast<double>(total_latency_ms.load()) / success;
                    
                    std::cout << "Progress: " << completed << " requests, "
                             << qps << " QPS, "
                             << success_rate << "% success rate, "
                             << avg_latency << "ms avg latency" << std::endl;
                }
                
                // Check if all threads are done
                bool all_done = true;
                for (const auto& thread : threads) {
                    if (thread.joinable()) {
                        all_done = false;
                        break;
                    }
                }
                if (all_done) break;
            }
        });
        
        // Wait for workers to complete
        for (auto& thread : threads) {
            thread.join();
        }
        
        // Stop monitor
        if (monitor.joinable()) {
            monitor.detach();
        }
        
        auto end_time = std::chrono::high_resolution_clock::now();
        auto total_duration = std::chrono::duration_cast<std::chrono::milliseconds>
                             (end_time - start_time).count();
        
        // Print final results
        PrintResults(total_requests.load(), successful_requests.load(), 
                    failed_requests.load(), total_latency_ms.load(), total_duration);
    }
    
private:
    Config config_;
    std::shared_ptr<grpc::Channel> channel_;
    
    bool WaitForServer() {
        auto deadline = std::chrono::system_clock::now() + std::chrono::seconds(30);
        
        while (std::chrono::system_clock::now() < deadline) {
            auto state = channel_->GetState(true);
            if (state == GRPC_CHANNEL_READY) {
                return true;
            }
            std::this_thread::sleep_for(std::chrono::milliseconds(100));
        }
        
        return false;
    }
    
    void RunWorker(int worker_id, 
                  std::atomic<int>& total_requests,
                  std::atomic<int>& successful_requests,
                  std::atomic<int>& failed_requests,
                  std::atomic<long long>& total_latency_ms) {
        
        auto stub = vdb::QueryService::NewStub(channel_);
        std::mt19937 gen(12345 + worker_id);
        std::normal_distribution<float> dist(0.0f, 1.0f);
        
        for (int i = 0; i < config_.requests_per_thread; ++i) {
            auto start = std::chrono::high_resolution_clock::now();
            
            // Create search request
            vdb::SearchRequest request;
            request.set_index("test_index");
            request.set_topk(config_.topk);
            request.set_nprobe(8);
            request.set_metric("L2");
            
            // Add queries
            for (int q = 0; q < config_.queries_per_request; ++q) {
                auto* query = request.add_queries();
                query->set_id(worker_id * 10000 + i * 100 + q);
                
                for (int d = 0; d < config_.query_dimension; ++d) {
                    query->add_values(dist(gen));
                }
            }
            
            // Execute request
            grpc::ClientContext context;
            context.set_deadline(std::chrono::system_clock::now() + 
                               std::chrono::seconds(10));
            
            vdb::SearchResponse response;
            grpc::Status status = stub->Search(&context, request, &response);
            
            auto end = std::chrono::high_resolution_clock::now();
            auto latency = std::chrono::duration_cast<std::chrono::milliseconds>
                          (end - start).count();
            
            total_requests++;
            
            if (status.ok()) {
                successful_requests++;
                total_latency_ms += latency;
            } else {
                failed_requests++;
                
                // Log occasional failures for debugging
                if (failed_requests.load() % 100 == 1) {
                    std::cerr << "Request failed: " << status.error_code() 
                             << " - " << status.error_message() << std::endl;
                }
            }
        }
    }
    
    void PrintResults(int total, int success, int failed, 
                     long long total_latency, long long duration_ms) {
        std::cout << std::endl;
        std::cout << "=== Load Test Results ===" << std::endl;
        std::cout << "Total requests: " << total << std::endl;
        std::cout << "Successful: " << success << " (" 
                 << (100.0 * success / total) << "%)" << std::endl;
        std::cout << "Failed: " << failed << " (" 
                 << (100.0 * failed / total) << "%)" << std::endl;
        std::cout << "Duration: " << (duration_ms / 1000.0) << "s" << std::endl;
        
        if (total > 0) {
            double qps = static_cast<double>(total) / (duration_ms / 1000.0);
            std::cout << "Throughput: " << qps << " QPS" << std::endl;
        }
        
        if (success > 0) {
            double avg_latency = static_cast<double>(total_latency) / success;
            std::cout << "Average latency: " << avg_latency << "ms" << std::endl;
        }
        
        std::cout << std::endl;
        
        // Performance assessment
        if (success == 0) {
            std::cout << "❌ All requests failed - server may not be working" << std::endl;
        } else if (static_cast<double>(success) / total < 0.95) {
            std::cout << "⚠️  High failure rate - check server health" << std::endl;
        } else {
            double avg_latency = static_cast<double>(total_latency) / success;
            if (avg_latency < 10) {
                std::cout << "🚀 Excellent performance!" << std::endl;
            } else if (avg_latency < 50) {
                std::cout << "✅ Good performance" << std::endl;
            } else if (avg_latency < 100) {
                std::cout << "⚠️  Acceptable performance" << std::endl;
            } else {
                std::cout << "❌ Poor performance - optimize server" << std::endl;
            }
        }
    }
};

int main(int argc, char** argv) {
    VDBLoadTest::Config config;
    
    // Parse command line arguments
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        
        if (arg == "--help" || arg == "-h") {
            std::cout << "VDB Load Test Tool" << std::endl;
            std::cout << std::endl;
            std::cout << "Usage: " << argv[0] << " [OPTIONS]" << std::endl;
            std::cout << std::endl;
            std::cout << "Options:" << std::endl;
            std::cout << "  --server HOST:PORT     Server address (default: localhost:50051)" << std::endl;
            std::cout << "  --threads N            Number of threads (default: 8)" << std::endl;
            std::cout << "  --requests N           Requests per thread (default: 100)" << std::endl;
            std::cout << "  --dimension N          Query dimension (default: 128)" << std::endl;
            std::cout << "  --queries N            Queries per request (default: 1)" << std::endl;
            std::cout << "  --topk N               Top-k results (default: 10)" << std::endl;
            std::cout << "  --help, -h             Show this help" << std::endl;
            return 0;
        }
        else if (arg == "--server" && i + 1 < argc) {
            config.server_address = argv[++i];
        }
        else if (arg == "--threads" && i + 1 < argc) {
            config.num_threads = std::stoi(argv[++i]);
        }
        else if (arg == "--requests" && i + 1 < argc) {
            config.requests_per_thread = std::stoi(argv[++i]);
        }
        else if (arg == "--dimension" && i + 1 < argc) {
            config.query_dimension = std::stoi(argv[++i]);
        }
        else if (arg == "--queries" && i + 1 < argc) {
            config.queries_per_request = std::stoi(argv[++i]);
        }
        else if (arg == "--topk" && i + 1 < argc) {
            config.topk = std::stoi(argv[++i]);
        }
    }
    
    try {
        VDBLoadTest load_test(config);
        load_test.RunLoadTest();
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "Load test failed: " << e.what() << std::endl;
        return 1;
    }
}