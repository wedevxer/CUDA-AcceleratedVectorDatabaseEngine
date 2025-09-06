# Test suite for vector database components
# Simple validation tests for core functionality

# Find required packages
find_package(CUDAToolkit REQUIRED)

# Define simple test executable
add_executable(vdb_simple_test
    simple_test.cpp
)

# Set C++ standard
set_target_properties(vdb_simple_test PROPERTIES
    CXX_STANDARD 20
    CXX_STANDARD_REQUIRED ON
)

# Link required libraries
target_link_libraries(vdb_simple_test
    vdb_engine              # Engine library with IVF index
    CUDA::cudart           # CUDA runtime
)

# Include directories
target_include_directories(vdb_simple_test PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/../
    ${CUDAToolkit_INCLUDE_DIRS}
)

# Define GPU vs CPU comparison test
add_executable(vdb_gpu_vs_cpu_test
    gpu_vs_cpu_test.cpp
)

# Set properties for GPU vs CPU test
set_target_properties(vdb_gpu_vs_cpu_test PROPERTIES
    CXX_STANDARD 20
    CXX_STANDARD_REQUIRED ON
    OUTPUT_NAME gpu_vs_cpu_test
)

# Link GPU vs CPU test
target_link_libraries(vdb_gpu_vs_cpu_test
    vdb_engine              # Engine library
    CUDA::cudart           # CUDA runtime
)

# Include directories for GPU vs CPU test
target_include_directories(vdb_gpu_vs_cpu_test PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/../
    ${CUDAToolkit_INCLUDE_DIRS}
)

# Add tests
add_test(NAME simple_test COMMAND vdb_simple_test)
add_test(NAME gpu_vs_cpu_test COMMAND vdb_gpu_vs_cpu_test 10000 100 64 32)

# Set test timeouts
set_tests_properties(simple_test PROPERTIES TIMEOUT 120)
set_tests_properties(gpu_vs_cpu_test PROPERTIES TIMEOUT 300)