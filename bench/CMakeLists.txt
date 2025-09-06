# Benchmark library and executable - Performance testing
# Provides comprehensive benchmarking suite for the vector database

# Define benchmark executable
add_executable(vdb_benchmark
    benchmark.cpp                  # Main benchmark implementation
)

# Set C++ standard
set_target_properties(vdb_benchmark PROPERTIES
    CXX_STANDARD 20               # C++20 features
    CXX_STANDARD_REQUIRED ON     # Enforce standard
    OUTPUT_NAME benchmark         # Executable name
)

# Link required libraries
target_link_libraries(vdb_benchmark
    vdb_engine                    # Engine for index operations
    vdb_format                    # Format for storage
    cxxopts                       # Command-line parsing
)

# Include directories
target_include_directories(vdb_benchmark PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}   # Benchmark headers
)

# Define recall evaluation executable
add_executable(vdb_recall_test
    recall_test.cpp               # Recall evaluation
)

# Set properties for recall test
set_target_properties(vdb_recall_test PROPERTIES
    CXX_STANDARD 20               # C++20 features
    CXX_STANDARD_REQUIRED ON     # Enforce standard
    OUTPUT_NAME recall_test       # Executable name
)

# Link recall test
target_link_libraries(vdb_recall_test
    vdb_engine                    # Engine library
    cxxopts                       # Command-line parsing
)

# Add benchmark as a test
add_test(NAME benchmark_small
    COMMAND vdb_benchmark 1000 64 32 5
)

# Set test properties
set_tests_properties(benchmark_small PROPERTIES
    TIMEOUT 60                    # 60 second timeout
)