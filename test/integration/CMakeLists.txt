# Integration tests for gRPC APIs
# These tests require a running VDB server

# Find Google Test
find_package(GTest QUIET)
if(NOT GTest_FOUND)
    # Use FetchContent to get GTest
    include(FetchContent)
    FetchContent_Declare(
        googletest
        GIT_REPOSITORY https://github.com/google/googletest.git
        GIT_TAG v1.14.0
    )
    FetchContent_MakeAvailable(googletest)
endif()

# Define integration test executable
add_executable(vdb_integration_test
    grpc_integration_test.cpp
    ${PROTO_SRCS}  # Use generated protobuf sources from parent
    ${GRPC_SRCS}   # Use generated gRPC sources from parent
)

# Set C++ standard
set_target_properties(vdb_integration_test PROPERTIES
    CXX_STANDARD 20
    CXX_STANDARD_REQUIRED ON
)

# Link required libraries
target_link_libraries(vdb_integration_test
    GTest::gtest
    GTest::gtest_main
    gRPC::grpc++
    protobuf::libprotobuf
    Threads::Threads
)

# Include directories
target_include_directories(vdb_integration_test PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/../../  # For proto includes
    ${GENERATED_PROTOBUF_PATH}          # For generated headers
)

# Add test
add_test(NAME integration_test 
         COMMAND vdb_integration_test)

# Set test properties
set_tests_properties(integration_test PROPERTIES 
    TIMEOUT 300
    LABELS "integration"
)

# Custom test target that starts server and runs tests
add_custom_target(run_integration_tests
    COMMENT "Running integration tests with server"
    COMMAND ${CMAKE_COMMAND} -E echo "Starting VDB server for integration tests..."
    COMMAND ${CMAKE_CURRENT_SOURCE_DIR}/run_integration_tests.sh
    DEPENDS vdb_integration_test vdb_server_bin
)

# Load test executable for performance testing
add_executable(vdb_load_test
    load_test.cpp
    ${PROTO_SRCS}
    ${GRPC_SRCS}
)

set_target_properties(vdb_load_test PROPERTIES
    CXX_STANDARD 20
    CXX_STANDARD_REQUIRED ON
)

target_link_libraries(vdb_load_test
    gRPC::grpc++
    protobuf::libprotobuf
    Threads::Threads
)

target_include_directories(vdb_load_test PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/../../
    ${GENERATED_PROTOBUF_PATH}
)