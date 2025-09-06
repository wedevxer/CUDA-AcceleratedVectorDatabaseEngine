# Specify minimum CMake version required (3.24 for better CUDA support)
cmake_minimum_required(VERSION 3.24)
# Define project name and specify both C++ and CUDA as project languages
project(CudaVectorDB LANGUAGES CXX CUDA)

# Set C++ standard to C++20 for modern features (concepts, coroutines, etc.)
set(CMAKE_CXX_STANDARD 20)
# Enforce C++20 standard (fail if compiler doesn't support it)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
# Set CUDA C++ standard to C++17 (highest supported by CUDA 12.x)
set(CMAKE_CUDA_STANDARD 17)
# Enforce CUDA C++17 standard
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# Find required packages with fallback to pkg-config
find_package(CUDAToolkit REQUIRED)
find_package(Threads REQUIRED)

# Find Protocol Buffers - try CMake first, then pkg-config
find_package(Protobuf QUIET)
if(NOT Protobuf_FOUND)
    find_package(PkgConfig REQUIRED)
    pkg_check_modules(Protobuf REQUIRED IMPORTED_TARGET protobuf)
    add_library(protobuf::libprotobuf ALIAS PkgConfig::Protobuf)
    find_program(Protobuf_PROTOC_EXECUTABLE protoc REQUIRED)
endif()

# Find gRPC with multiple fallback strategies
find_package(gRPC QUIET CONFIG)
if(NOT gRPC_FOUND)
    find_package(PkgConfig QUIET)
    if(PkgConfig_FOUND)
        pkg_check_modules(gRPC QUIET IMPORTED_TARGET grpc++)
    endif()
    
    if(NOT gRPC_FOUND)
        # Use FetchContent as last resort
        message(STATUS "gRPC not found, fetching from source...")
        FetchContent_Declare(
            gRPC
            GIT_REPOSITORY https://github.com/grpc/grpc.git
            GIT_TAG v1.60.0
            GIT_SHALLOW TRUE
        )
        set(gRPC_BUILD_TESTS OFF CACHE BOOL "" FORCE)
        FetchContent_MakeAvailable(gRPC)
    endif()
endif()

# Find Apache Arrow - try CMake first
find_package(Arrow QUIET)
if(NOT Arrow_FOUND)
    find_package(PkgConfig QUIET)
    if(PkgConfig_FOUND)
        pkg_check_modules(Arrow QUIET IMPORTED_TARGET arrow)
    endif()
    
    if(NOT Arrow_FOUND)
        # Create dummy target for now - can be replaced with FetchContent
        add_library(Arrow::arrow INTERFACE IMPORTED)
        message(WARNING "Arrow not found - some storage features may be disabled")
    endif()
endif()

# Find RocksDB - optional for now
find_package(RocksDB QUIET)
if(NOT RocksDB_FOUND)
    find_package(PkgConfig QUIET)
    if(PkgConfig_FOUND)
        pkg_check_modules(RocksDB QUIET IMPORTED_TARGET rocksdb)
    endif()
    
    if(NOT RocksDB_FOUND)
        add_library(RocksDB::rocksdb INTERFACE IMPORTED)
        message(WARNING "RocksDB not found - some storage features may be disabled")
    endif()
endif()

# Use FetchContent to download dependencies at configure time
include(FetchContent)
# Declare cxxopts library for command-line parsing
FetchContent_Declare(
  cxxopts
  GIT_REPOSITORY https://github.com/jarro2783/cxxopts.git
  GIT_TAG v3.0.0  # Specific version for reproducibility
)
# Download and make cxxopts available
FetchContent_MakeAvailable(cxxopts)

# Set CUDA compilation flags:
# -O3: Maximum optimization level
# --use_fast_math: Enable fast math operations (trades precision for speed)
# -lineinfo: Add line number info for profiling
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -O3 --use_fast_math -lineinfo")
# Target multiple GPU architectures:
# 70: V100, 75: Turing, 80: A100, 86: RTX 30xx, 90: H100
set(CMAKE_CUDA_ARCHITECTURES 70 75 80 86 90)

# Add subdirectories containing component CMakeLists.txt files
add_subdirectory(engine)      # CUDA kernels and core algorithms
add_subdirectory(server)      # gRPC server implementation
add_subdirectory(indexmgr)    # Index building and management
add_subdirectory(format)      # Storage format and I/O

# Enable CTest testing framework
enable_testing()
# Add test directory for unit tests
add_subdirectory(test)
# Add benchmark directory for performance testing
add_subdirectory(bench)