#!/bin/bash

# Build script for CUDA Vector Database
# This script sets up the build environment and compiles the project

set -e  # Exit on any error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Print colored output
print_status() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check if running from project root
if [ ! -f "CMakeLists.txt" ]; then
    print_error "Please run this script from the project root directory"
    exit 1
fi

print_status "Starting CUDA Vector Database build..."

# Configuration
BUILD_TYPE=${1:-Release}
BUILD_DIR="build"
CUDA_ARCH=${CUDA_ARCH:-"70;75;80;86;90"}
INSTALL_DIR=${INSTALL_DIR:-"/opt/vdb"}

print_status "Build configuration:"
echo "  Build type: $BUILD_TYPE"
echo "  Build directory: $BUILD_DIR"
echo "  CUDA architectures: $CUDA_ARCH"
echo "  Install directory: $INSTALL_DIR"

# Check dependencies
print_status "Checking system dependencies..."

# Check CUDA installation
if ! command -v nvcc &> /dev/null; then
    print_error "CUDA compiler (nvcc) not found. Please install CUDA Toolkit 12.x"
    exit 1
fi

CUDA_VERSION=$(nvcc --version | grep "release" | sed 's/.*release \([0-9]\+\.[0-9]\+\).*/\1/')
print_success "Found CUDA $CUDA_VERSION"

# Check CMake
if ! command -v cmake &> /dev/null; then
    print_error "CMake not found. Please install CMake 3.24 or later"
    exit 1
fi

CMAKE_VERSION=$(cmake --version | head -n1 | sed 's/cmake version //')
print_success "Found CMake $CMAKE_VERSION"

# Check GPU
print_status "Checking GPU availability..."
if command -v nvidia-smi &> /dev/null; then
    GPU_COUNT=$(nvidia-smi -L | wc -l)
    if [ $GPU_COUNT -gt 0 ]; then
        print_success "Found $GPU_COUNT GPU(s)"
        nvidia-smi -L
    else
        print_warning "No GPUs detected"
    fi
else
    print_warning "nvidia-smi not found, cannot check GPU status"
fi

# Create build directory
print_status "Setting up build directory..."
if [ -d "$BUILD_DIR" ]; then
    print_warning "Build directory exists, cleaning..."
    rm -rf "$BUILD_DIR"
fi
mkdir -p "$BUILD_DIR"

# Configure with CMake
print_status "Configuring with CMake..."
cd "$BUILD_DIR"

cmake .. \
    -DCMAKE_BUILD_TYPE="$BUILD_TYPE" \
    -DCMAKE_CUDA_ARCHITECTURES="$CUDA_ARCH" \
    -DCMAKE_INSTALL_PREFIX="$INSTALL_DIR" \
    -DCMAKE_EXPORT_COMPILE_COMMANDS=ON \
    -DCMAKE_VERBOSE_MAKEFILE=OFF

if [ $? -eq 0 ]; then
    print_success "CMake configuration completed"
else
    print_error "CMake configuration failed"
    exit 1
fi

# Build
print_status "Building project..."
CPU_COUNT=$(nproc 2>/dev/null || sysctl -n hw.ncpu 2>/dev/null || echo 4)
print_status "Using $CPU_COUNT parallel jobs"

make -j$CPU_COUNT

if [ $? -eq 0 ]; then
    print_success "Build completed successfully"
else
    print_error "Build failed"
    exit 1
fi

# Run tests
print_status "Running tests..."
ctest --output-on-failure

if [ $? -eq 0 ]; then
    print_success "All tests passed"
else
    print_warning "Some tests failed - check output above"
fi

cd ..

# Print build summary
print_status "Build Summary:"
echo "  Build directory: $BUILD_DIR"
echo "  Executables:"
echo "    - $BUILD_DIR/test/vdb_simple_test (unit tests)"
echo "    - $BUILD_DIR/bench/benchmark (performance testing)"

print_status "Next steps:"
echo "  1. Run unit tests: ./$BUILD_DIR/test/vdb_simple_test"
echo "  2. Run benchmark: ./$BUILD_DIR/bench/benchmark [num_vectors] [dimension] [nlist] [nprobe]"
echo "  3. Install system-wide: sudo make -C $BUILD_DIR install"

print_success "Build script completed!"