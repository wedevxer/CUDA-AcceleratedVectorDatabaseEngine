#include <iostream>
#include <memory>
#include <string>
#include <signal.h>
#include <grpcpp/grpcpp.h>
#include <grpcpp/health_check_service_interface.h>
#include <grpcpp/ext/proto_server_reflection_plugin.h>
#include "query_service.h"

using grpc::Server;
using grpc::ServerBuilder;

namespace {
    std::unique_ptr<Server> g_server;
    
    void signal_handler(int signal) {
        std::cout << "\nReceived signal " << signal << ", shutting down server..." << std::endl;
        if (g_server) {
            g_server->Shutdown();
        }
    }
}

void print_banner() {
    std::cout << R"(
    ╔════════════════════════════════════════════════╗
    ║              CUDA Vector Database              ║
    ║           High-Performance gRPC Server         ║
    ╚════════════════════════════════════════════════╝
    )" << std::endl;
}

void print_config(const vdb::server::QueryServiceImpl::Config& config) {
    std::cout << "Configuration:" << std::endl;
    std::cout << "  Data Path: " << config.data_path << std::endl;
    std::cout << "  Max Batch Size: " << config.max_batch_size << std::endl;
    std::cout << "  Coalesce Window: " << config.coalesce_window_ms << "ms" << std::endl;
    std::cout << "  Max Concurrent Searches: " << config.max_concurrent_searches << std::endl;
    std::cout << "  GPU Memory Limit: " << (config.gpu_memory_limit >> 30) << "GB" << std::endl;
    std::cout << std::endl;
}

void check_cuda_availability() {
    int device_count = 0;
    cudaError_t error = cudaGetDeviceCount(&device_count);
    
    if (error != cudaSuccess || device_count == 0) {
        std::cout << "⚠️  WARNING: No CUDA devices found - running in CPU-only mode" << std::endl;
        std::cout << "   CUDA Error: " << cudaGetErrorString(error) << std::endl;
        return;
    }
    
    std::cout << "🚀 CUDA Acceleration Enabled" << std::endl;
    std::cout << "   Devices Found: " << device_count << std::endl;
    
    for (int i = 0; i < device_count; ++i) {
        cudaDeviceProp prop;
        cudaGetDeviceProperties(&prop, i);
        
        std::cout << "   GPU " << i << ": " << prop.name << std::endl;
        std::cout << "     Compute: " << prop.major << "." << prop.minor << std::endl;
        std::cout << "     Memory: " << (prop.totalGlobalMem >> 30) << " GB" << std::endl;
        std::cout << "     SM Count: " << prop.multiProcessorCount << std::endl;
    }
    std::cout << std::endl;
}

void run_server(const std::string& server_address, 
               const vdb::server::QueryServiceImpl::Config& config) {
    
    // Create services
    auto query_service = std::make_unique<vdb::server::QueryServiceImpl>(config);
    auto admin_service = std::make_unique<vdb::server::AdminServiceImpl>(query_service.get());
    
    // Enable health checking and server reflection
    grpc::EnableDefaultHealthCheckService(true);
    grpc::reflection::InitProtoReflectionServerBuilderPlugin();
    
    // Build server
    ServerBuilder builder;
    
    // Configure server options
    builder.AddListeningPort(server_address, grpc::InsecureServerCredentials());
    builder.RegisterService(query_service.get());
    builder.RegisterService(admin_service.get());
    
    // Set max message sizes (100MB for large batches)
    builder.SetMaxReceiveMessageSize(100 * 1024 * 1024);
    builder.SetMaxSendMessageSize(100 * 1024 * 1024);
    
    // Set threading options
    builder.SetSyncServerOption(ServerBuilder::SyncServerOption::NUM_CQS, 4);
    builder.SetSyncServerOption(ServerBuilder::SyncServerOption::MIN_POLLERS, 2);
    builder.SetSyncServerOption(ServerBuilder::SyncServerOption::MAX_POLLERS, 8);
    
    // Build and start server
    g_server = builder.BuildAndStart();
    
    if (!g_server) {
        std::cerr << "❌ Failed to start server on " << server_address << std::endl;
        exit(1);
    }
    
    std::cout << "✅ Server listening on " << server_address << std::endl;
    std::cout << std::endl;
    
    std::cout << "Available Services:" << std::endl;
    std::cout << "  vdb.QueryService - Search operations and index warmup" << std::endl;
    std::cout << "  vdb.AdminService - Index management and statistics" << std::endl;
    std::cout << "  grpc.health.v1.Health - Health checking" << std::endl;
    std::cout << "  grpc.reflection.v1alpha.ServerReflection - Service reflection" << std::endl;
    std::cout << std::endl;
    
    std::cout << "Example gRPC calls:" << std::endl;
    std::cout << "  grpcurl -plaintext " << server_address << " list" << std::endl;
    std::cout << "  grpcurl -plaintext " << server_address << " describe vdb.QueryService" << std::endl;
    std::cout << std::endl;
    
    std::cout << "Press Ctrl+C to stop the server..." << std::endl;
    
    // Wait for server to shutdown
    g_server->Wait();
    
    std::cout << "✅ Server shut down gracefully" << std::endl;
}

int main(int argc, char** argv) {
    print_banner();
    
    // Parse command line arguments
    std::string server_address = "0.0.0.0:50051";
    vdb::server::QueryServiceImpl::Config config;
    
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        
        if (arg == "--help" || arg == "-h") {
            std::cout << "CUDA Vector Database Server" << std::endl;
            std::cout << std::endl;
            std::cout << "Usage: " << argv[0] << " [OPTIONS]" << std::endl;
            std::cout << std::endl;
            std::cout << "Options:" << std::endl;
            std::cout << "  --address HOST:PORT    Server address (default: 0.0.0.0:50051)" << std::endl;
            std::cout << "  --data-path PATH       Data directory path (default: /mnt/nvme/vdb)" << std::endl;
            std::cout << "  --gpu-memory SIZE      GPU memory limit in GB (default: 8)" << std::endl;
            std::cout << "  --batch-size N         Max batch size (default: 64)" << std::endl;
            std::cout << "  --coalesce-window MS   Batch coalesce window in ms (default: 2)" << std::endl;
            std::cout << "  --help, -h             Show this help message" << std::endl;
            std::cout << std::endl;
            std::cout << "Examples:" << std::endl;
            std::cout << "  " << argv[0] << " --address 0.0.0.0:8080 --gpu-memory 16" << std::endl;
            std::cout << "  " << argv[0] << " --data-path /data/vdb --batch-size 128" << std::endl;
            return 0;
        }
        else if (arg == "--address" && i + 1 < argc) {
            server_address = argv[++i];
        }
        else if (arg == "--data-path" && i + 1 < argc) {
            config.data_path = argv[++i];
        }
        else if (arg == "--gpu-memory" && i + 1 < argc) {
            uint64_t gpu_gb = std::stoull(argv[++i]);
            config.gpu_memory_limit = gpu_gb << 30;  // Convert GB to bytes
        }
        else if (arg == "--batch-size" && i + 1 < argc) {
            config.max_batch_size = std::stoull(argv[++i]);
        }
        else if (arg == "--coalesce-window" && i + 1 < argc) {
            config.coalesce_window_ms = std::stoull(argv[++i]);
        }
        else {
            std::cerr << "Unknown option: " << arg << std::endl;
            std::cerr << "Use --help for usage information" << std::endl;
            return 1;
        }
    }
    
    // Validate configuration
    if (config.max_batch_size == 0 || config.max_batch_size > 1000) {
        std::cerr << "❌ Invalid batch size: " << config.max_batch_size << std::endl;
        return 1;
    }
    
    if (config.gpu_memory_limit < (1ULL << 30)) {  // Minimum 1GB
        std::cerr << "❌ GPU memory limit too low: " << (config.gpu_memory_limit >> 30) << "GB" << std::endl;
        return 1;
    }
    
    // Create data directory if it doesn't exist
    try {
        std::filesystem::create_directories(config.data_path);
        std::filesystem::create_directories(config.data_path + "/epochs");
        std::filesystem::create_directories(config.data_path + "/indices");
    } catch (const std::exception& e) {
        std::cerr << "❌ Failed to create data directory: " << e.what() << std::endl;
        return 1;
    }
    
    print_config(config);
    check_cuda_availability();
    
    // Setup signal handling
    signal(SIGINT, signal_handler);
    signal(SIGTERM, signal_handler);
    
    try {
        run_server(server_address, config);
    } catch (const std::exception& e) {
        std::cerr << "❌ Server failed: " << e.what() << std::endl;
        return 1;
    }
    
    return 0;
}