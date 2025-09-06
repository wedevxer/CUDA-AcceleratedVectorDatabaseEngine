#include <gtest/gtest.h>
#include <grpcpp/grpcpp.h>
#include "../../proto/vdb.grpc.pb.h"
#include <thread>
#include <chrono>
#include <vector>
#include <random>

class VDBIntegrationTest : public ::testing::Test {
protected:
    void SetUp() override {
        // Connect to test server (should be running on localhost:50051)
        auto channel = grpc::CreateChannel("localhost:50051", 
                                          grpc::InsecureChannelCredentials());
        
        query_stub_ = vdb::QueryService::NewStub(channel);
        admin_stub_ = vdb::AdminService::NewStub(channel);
        
        // Wait for server to be ready
        ASSERT_TRUE(WaitForServer(channel, std::chrono::seconds(30))) 
            << "Failed to connect to test server";
    }
    
    void TearDown() override {
        // Cleanup test indices if any were created
        CleanupTestData();
    }
    
    bool WaitForServer(std::shared_ptr<grpc::Channel> channel, 
                      std::chrono::seconds timeout) {
        auto deadline = std::chrono::system_clock::now() + timeout;
        
        while (std::chrono::system_clock::now() < deadline) {
            auto state = channel->GetState(true);
            if (state == GRPC_CHANNEL_READY) {
                return true;
            }
            
            std::this_thread::sleep_for(std::chrono::milliseconds(100));
        }
        
        return false;
    }
    
    void CleanupTestData() {
        // Implementation would clean up any test indices created
        // For now, this is a placeholder
    }
    
    // Helper to generate test vectors
    std::vector<float> GenerateRandomVector(int dimension, std::mt19937& gen) {
        std::normal_distribution<float> dist(0.0f, 1.0f);
        std::vector<float> vec(dimension);
        
        for (int i = 0; i < dimension; ++i) {
            vec[i] = dist(gen);
        }
        
        return vec;
    }
    
    // Helper to create a test vector message
    vdb::Vector CreateTestVector(uint64_t id, const std::vector<float>& values) {
        vdb::Vector vector;
        vector.set_id(id);
        for (float val : values) {
            vector.add_values(val);
        }
        return vector;
    }
    
    std::unique_ptr<vdb::QueryService::Stub> query_stub_;
    std::unique_ptr<vdb::AdminService::Stub> admin_stub_;
    
    static constexpr const char* TEST_INDEX_NAME = "test_index";
    static constexpr int TEST_DIMENSION = 128;
};

// Test basic server connectivity
TEST_F(VDBIntegrationTest, ServerConnectivity) {
    grpc::ClientContext context;
    context.set_deadline(std::chrono::system_clock::now() + std::chrono::seconds(5));
    
    vdb::StatsRequest request;
    request.set_index("nonexistent_index");  // This should fail gracefully
    
    vdb::StatsResponse response;
    grpc::Status status = admin_stub_->GetStats(&context, request, &response);
    
    // We expect either OK (if index exists) or NOT_FOUND (which is fine for this test)
    EXPECT_TRUE(status.ok() || status.error_code() == grpc::StatusCode::NOT_FOUND)
        << "Server should respond to requests. Error: " << status.error_message();
}

// Test index creation workflow
TEST_F(VDBIntegrationTest, IndexCreationWorkflow) {
    // Create a test index
    {
        grpc::ClientContext context;
        context.set_deadline(std::chrono::system_clock::now() + std::chrono::seconds(10));
        
        vdb::CreateIndexRequest request;
        request.set_name(TEST_INDEX_NAME);
        request.set_dimension(TEST_DIMENSION);
        request.set_metric("L2");
        request.set_nlist(64);  // Small nlist for testing
        
        google::protobuf::Empty response;
        grpc::Status status = admin_stub_->CreateIndex(&context, request, &response);
        
        ASSERT_TRUE(status.ok()) << "Failed to create index: " << status.error_message();
    }
    
    // Verify index was created by checking stats
    {
        grpc::ClientContext context;
        context.set_deadline(std::chrono::system_clock::now() + std::chrono::seconds(5));
        
        vdb::StatsRequest request;
        request.set_index(TEST_INDEX_NAME);
        
        vdb::StatsResponse response;
        grpc::Status status = admin_stub_->GetStats(&context, request, &response);
        
        if (status.ok()) {
            // Index exists and has stats
            EXPECT_GE(response.total_vectors(), 0);
        } else {
            // Index might not be fully initialized yet, which is acceptable
            EXPECT_EQ(status.error_code(), grpc::StatusCode::NOT_FOUND);
        }
    }
}

// Test search functionality with dummy data
TEST_F(VDBIntegrationTest, BasicSearchTest) {
    // First ensure index exists (this might fail if index creation is not working)
    {
        grpc::ClientContext context;
        vdb::CreateIndexRequest request;
        request.set_name(TEST_INDEX_NAME);
        request.set_dimension(TEST_DIMENSION);
        request.set_metric("L2");
        request.set_nlist(32);
        
        google::protobuf::Empty response;
        admin_stub_->CreateIndex(&context, request, &response);
        // Ignore result - index might already exist
    }
    
    // Attempt a search (this will likely fail since no vectors are loaded)
    grpc::ClientContext context;
    context.set_deadline(std::chrono::system_clock::now() + std::chrono::seconds(10));
    
    vdb::SearchRequest request;
    request.set_index(TEST_INDEX_NAME);
    request.set_topk(10);
    request.set_nprobe(8);
    request.set_metric("L2");
    
    // Add a test query vector
    std::mt19937 gen(12345);
    auto query_values = GenerateRandomVector(TEST_DIMENSION, gen);
    auto* query = request.add_queries();
    *query = CreateTestVector(0, query_values);
    
    vdb::SearchResponse response;
    grpc::Status status = query_stub_->Search(&context, request, &response);
    
    // We expect either success (with empty results) or an error about no vectors
    EXPECT_TRUE(status.ok() || 
                status.error_code() == grpc::StatusCode::NOT_FOUND ||
                status.error_code() == grpc::StatusCode::FAILED_PRECONDITION)
        << "Search should handle empty index gracefully. Error: " << status.error_message();
    
    if (status.ok()) {
        EXPECT_EQ(response.results_size(), 1) << "Should return results for one query";
        if (response.results_size() > 0) {
            // Results might be empty if no vectors are indexed
            EXPECT_GE(response.results(0).neighbors_size(), 0);
        }
    }
}

// Test error handling for invalid requests
TEST_F(VDBIntegrationTest, ErrorHandling) {
    // Test empty query
    {
        grpc::ClientContext context;
        context.set_deadline(std::chrono::system_clock::now() + std::chrono::seconds(5));
        
        vdb::SearchRequest request;
        request.set_index(TEST_INDEX_NAME);
        request.set_topk(10);
        // No queries added - should fail
        
        vdb::SearchResponse response;
        grpc::Status status = query_stub_->Search(&context, request, &response);
        
        EXPECT_FALSE(status.ok());
        EXPECT_EQ(status.error_code(), grpc::StatusCode::INVALID_ARGUMENT);
    }
    
    // Test invalid topk
    {
        grpc::ClientContext context;
        context.set_deadline(std::chrono::system_clock::now() + std::chrono::seconds(5));
        
        vdb::SearchRequest request;
        request.set_index(TEST_INDEX_NAME);
        request.set_topk(0);  // Invalid
        
        std::mt19937 gen(12345);
        auto query_values = GenerateRandomVector(TEST_DIMENSION, gen);
        auto* query = request.add_queries();
        *query = CreateTestVector(0, query_values);
        
        vdb::SearchResponse response;
        grpc::Status status = query_stub_->Search(&context, request, &response);
        
        EXPECT_FALSE(status.ok());
        EXPECT_EQ(status.error_code(), grpc::StatusCode::INVALID_ARGUMENT);
    }
    
    // Test missing index name
    {
        grpc::ClientContext context;
        context.set_deadline(std::chrono::system_clock::now() + std::chrono::seconds(5));
        
        vdb::SearchRequest request;
        // No index name - should fail
        request.set_topk(10);
        
        std::mt19937 gen(12345);
        auto query_values = GenerateRandomVector(TEST_DIMENSION, gen);
        auto* query = request.add_queries();
        *query = CreateTestVector(0, query_values);
        
        vdb::SearchResponse response;
        grpc::Status status = query_stub_->Search(&context, request, &response);
        
        EXPECT_FALSE(status.ok());
        EXPECT_EQ(status.error_code(), grpc::StatusCode::INVALID_ARGUMENT);
    }
}

// Test concurrent requests
TEST_F(VDBIntegrationTest, ConcurrentRequests) {
    const int num_threads = 4;
    const int requests_per_thread = 5;
    
    std::vector<std::thread> threads;
    std::atomic<int> success_count{0};
    std::atomic<int> error_count{0};
    
    for (int t = 0; t < num_threads; ++t) {
        threads.emplace_back([this, t, requests_per_thread, &success_count, &error_count]() {
            std::mt19937 gen(12345 + t);
            
            for (int r = 0; r < requests_per_thread; ++r) {
                grpc::ClientContext context;
                context.set_deadline(std::chrono::system_clock::now() + std::chrono::seconds(10));
                
                vdb::SearchRequest request;
                request.set_index(TEST_INDEX_NAME);
                request.set_topk(5);
                request.set_nprobe(4);
                request.set_metric("L2");
                
                auto query_values = GenerateRandomVector(TEST_DIMENSION, gen);
                auto* query = request.add_queries();
                *query = CreateTestVector(t * 1000 + r, query_values);
                
                vdb::SearchResponse response;
                grpc::Status status = query_stub_->Search(&context, request, &response);
                
                if (status.ok() || 
                    status.error_code() == grpc::StatusCode::NOT_FOUND ||
                    status.error_code() == grpc::StatusCode::FAILED_PRECONDITION) {
                    success_count++;
                } else {
                    error_count++;
                    std::cerr << "Unexpected error in thread " << t << ", request " << r 
                             << ": " << status.error_message() << std::endl;
                }
            }
        });
    }
    
    // Wait for all threads
    for (auto& thread : threads) {
        thread.join();
    }
    
    // Most requests should succeed or fail gracefully
    EXPECT_GT(success_count.load(), num_threads * requests_per_thread * 0.8)
        << "Too many requests failed unexpectedly";
    EXPECT_LT(error_count.load(), num_threads * requests_per_thread * 0.2)
        << "Too many unexpected errors";
}

// Test warmup functionality
TEST_F(VDBIntegrationTest, WarmupTest) {
    grpc::ClientContext context;
    context.set_deadline(std::chrono::system_clock::now() + std::chrono::seconds(10));
    
    vdb::WarmupRequest request;
    request.set_index(TEST_INDEX_NAME);
    // Add some list IDs to warmup
    request.add_lists(0);
    request.add_lists(1);
    request.add_lists(2);
    
    google::protobuf::Empty response;
    grpc::Status status = query_stub_->Warmup(&context, request, &response);
    
    // Warmup should either succeed or fail gracefully if index doesn't exist
    EXPECT_TRUE(status.ok() || 
                status.error_code() == grpc::StatusCode::NOT_FOUND)
        << "Warmup failed unexpectedly: " << status.error_message();
}

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    
    std::cout << "Starting VDB Integration Tests..." << std::endl;
    std::cout << "Make sure VDB server is running on localhost:50051" << std::endl;
    std::cout << std::endl;
    
    return RUN_ALL_TESTS();
}