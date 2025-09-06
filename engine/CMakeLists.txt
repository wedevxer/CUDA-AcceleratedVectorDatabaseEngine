# Engine library - CUDA kernels and core algorithms
# Contains IVF index implementations, transfer manager, and prefetcher

# Find required packages for engine components
find_package(CUDAToolkit REQUIRED)  # CUDA runtime and libraries
find_package(Arrow REQUIRED)        # Arrow for memory-mapped storage
find_package(Threads REQUIRED)      # Threading for async operations

# Define engine library with CUDA kernels and C++ implementations
add_library(vdb_engine SHARED
    # CUDA kernel implementations
    kernels.cu                       # Distance computations and search kernels
    
    # C++ implementations  
    ivf_flat_index.cpp              # IVF-Flat index with GPU acceleration
    transfer_manager.cpp            # Memory pools and async transfers
    prefetcher.cpp                  # io_uring based NVMe prefetcher
)

# Set target properties for CUDA compilation
set_target_properties(vdb_engine PROPERTIES
    CUDA_STANDARD 17                # CUDA C++17 standard
    CUDA_STANDARD_REQUIRED ON      # Enforce CUDA standard
    CXX_STANDARD 20                 # C++20 for host code
    CXX_STANDARD_REQUIRED ON       # Enforce C++ standard
    CUDA_ARCHITECTURES "70;75;80;86;90"  # Support multiple GPU generations
)

# Link required libraries
target_link_libraries(vdb_engine
    CUDA::cudart                    # CUDA runtime
    CUDA::cub                       # CUB primitives library
    arrow                           # Apache Arrow
    Threads::Threads               # Threading support
    uring                          # liburing for async I/O
)

# Include directories for headers
target_include_directories(vdb_engine PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}    # Engine headers
    ${CUDAToolkit_INCLUDE_DIRS}    # CUDA headers
)

# Compile definitions for conditional compilation
target_compile_definitions(vdb_engine PRIVATE
    VDB_USE_CUDA=1                 # Enable CUDA code paths
    VDB_USE_URING=1                # Enable io_uring support
)