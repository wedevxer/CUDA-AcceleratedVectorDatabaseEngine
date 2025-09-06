#include "query_service.h"
#include <grpcpp/health_check_service_interface.h>
#include <cuda_runtime.h>

namespace vdb {
namespace server {

class HealthCheckServiceImpl : public grpc::HealthCheckServiceInterface {
public:
    explicit HealthCheckServiceImpl(QueryServiceImpl* query_service)
        : query_service_(query_service) {}
    
    grpc::Status Check(grpc::ServerContext* context,
                      const grpc::health::v1::HealthCheckRequest* request,
                      grpc::health::v1::HealthCheckResponse* response) override {
        
        std::string service = request->service();
        
        // Check overall system health
        if (service.empty() || service == "") {
            response->set_status(check_system_health());
        }
        // Check specific service health
        else if (service == "vdb.QueryService") {
            response->set_status(check_query_service_health());
        }
        else if (service == "vdb.AdminService") {
            response->set_status(check_admin_service_health());
        }
        else {
            response->set_status(grpc::health::v1::HealthCheckResponse::SERVICE_UNKNOWN);
        }
        
        return grpc::Status::OK;
    }
    
    grpc::Status Watch(grpc::ServerContext* context,
                      const grpc::health::v1::HealthCheckRequest* request,
                      grpc::ServerWriter<grpc::health::v1::HealthCheckResponse>* writer) override {
        
        std::string service = request->service();
        
        // Send initial status
        grpc::health::v1::HealthCheckResponse response;
        if (service.empty()) {
            response.set_status(check_system_health());
        } else if (service == "vdb.QueryService") {
            response.set_status(check_query_service_health());
        } else if (service == "vdb.AdminService") {
            response.set_status(check_admin_service_health());
        } else {
            response.set_status(grpc::health::v1::HealthCheckResponse::SERVICE_UNKNOWN);
        }
        
        if (!writer->Write(response)) {
            return grpc::Status(grpc::StatusCode::ABORTED, "Stream closed");
        }
        
        // Monitor health changes (simplified implementation)
        auto last_status = response.status();
        while (!context->IsCancelled()) {
            std::this_thread::sleep_for(std::chrono::seconds(5));
            
            auto current_status = service.empty() ? check_system_health() :
                                 service == "vdb.QueryService" ? check_query_service_health() :
                                 service == "vdb.AdminService" ? check_admin_service_health() :
                                 grpc::health::v1::HealthCheckResponse::SERVICE_UNKNOWN;
            
            if (current_status != last_status) {
                response.set_status(current_status);
                if (!writer->Write(response)) {
                    break;
                }
                last_status = current_status;
            }
        }
        
        return grpc::Status::OK;
    }
    
private:
    QueryServiceImpl* query_service_;
    
    grpc::health::v1::HealthCheckResponse::ServingStatus check_system_health() {
        // Check CUDA availability
        if (!check_cuda_health()) {
            return grpc::health::v1::HealthCheckResponse::NOT_SERVING;
        }
        
        // Check memory health
        if (!check_memory_health()) {
            return grpc::health::v1::HealthCheckResponse::NOT_SERVING;
        }
        
        // Check service health
        if (check_query_service_health() == grpc::health::v1::HealthCheckResponse::NOT_SERVING) {
            return grpc::health::v1::HealthCheckResponse::NOT_SERVING;
        }
        
        return grpc::health::v1::HealthCheckResponse::SERVING;
    }
    
    grpc::health::v1::HealthCheckResponse::ServingStatus check_query_service_health() {
        if (!query_service_) {
            return grpc::health::v1::HealthCheckResponse::NOT_SERVING;
        }
        
        // Check if any indices are loaded
        // This is a simplified check - in practice you'd check circuit breakers, etc.
        
        return grpc::health::v1::HealthCheckResponse::SERVING;
    }
    
    grpc::health::v1::HealthCheckResponse::ServingStatus check_admin_service_health() {
        // Admin service is stateless, so if the process is running it's healthy
        return grpc::health::v1::HealthCheckResponse::SERVING;
    }
    
    bool check_cuda_health() {
        int device_count;
        cudaError_t error = cudaGetDeviceCount(&device_count);
        
        if (error != cudaSuccess) {
            return false;
        }
        
        if (device_count == 0) {
            return false;
        }
        
        // Check if we can access the first GPU
        cudaSetDevice(0);
        error = cudaGetLastError();
        
        return error == cudaSuccess;
    }
    
    bool check_memory_health() {
        // Check GPU memory
        size_t free_mem, total_mem;
        cudaError_t error = cudaMemGetInfo(&free_mem, &total_mem);
        
        if (error != cudaSuccess) {
            return false;
        }
        
        // Consider unhealthy if less than 10% GPU memory is free
        double free_ratio = static_cast<double>(free_mem) / total_mem;
        if (free_ratio < 0.1) {
            return false;
        }
        
        // Check system memory (simplified)
        // In production, you'd use more sophisticated memory monitoring
        
        return true;
    }
};

// Metrics HTTP server implementation
class MetricsServer {
public:
    explicit MetricsServer(int port, MetricsCollector* collector)
        : port_(port), collector_(collector), stop_(false) {}
    
    ~MetricsServer() {
        stop();
    }
    
    void start() {
        if (running_) return;
        
        running_ = true;
        server_thread_ = std::thread(&MetricsServer::run, this);
    }
    
    void stop() {
        if (!running_) return;
        
        stop_ = true;
        if (server_thread_.joinable()) {
            server_thread_.join();
        }
        running_ = false;
    }
    
private:
    int port_;
    MetricsCollector* collector_;
    std::atomic<bool> stop_{false};
    std::atomic<bool> running_{false};
    std::thread server_thread_;
    
    void run() {
        // Simplified HTTP server for metrics endpoint
        // In production, you'd use a proper HTTP library like cpp-httplib
        
        // This is a placeholder implementation
        // The actual metrics server would:
        // 1. Listen on the specified port
        // 2. Serve GET /metrics with Prometheus format
        // 3. Handle health check endpoints
        
        while (!stop_) {
            std::this_thread::sleep_for(std::chrono::seconds(1));
            
            // In real implementation, this would handle HTTP requests
            // For now, we just keep the thread alive
        }
    }
    
    std::string handle_metrics_request() {
        if (collector_) {
            return collector_->prometheus_format();
        }
        return "# No metrics available\n";
    }
};

}
}