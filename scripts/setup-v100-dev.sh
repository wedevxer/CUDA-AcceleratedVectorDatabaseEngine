#!/bin/bash

# Development environment setup script for GCP V100 instances
# This script sets up the complete development environment for CUDA Vector DB

set -e

RED='\033[0;31m'
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m'

print_status() { echo -e "${BLUE}[SETUP]${NC} $1"; }
print_success() { echo -e "${GREEN}[SUCCESS]${NC} $1"; }
print_error() { echo -e "${RED}[ERROR]${NC} $1"; }
print_warning() { echo -e "${YELLOW}[WARNING]${NC} $1"; }

print_status "CUDA Vector Database - V100 Development Environment Setup"
echo

# Check if we're on GCP with V100
print_status "Checking environment..."
if command -v nvidia-smi &> /dev/null; then
    GPU_INFO=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)
    print_success "GPU detected: $GPU_INFO"
    
    if [[ "$GPU_INFO" == *"V100"* ]]; then
        print_success "NVIDIA V100 detected - perfect for development!"
    else
        print_warning "Non-V100 GPU detected. Code will work but may have different performance characteristics."
    fi
else
    print_error "No NVIDIA GPU detected. This setup is optimized for GPU development."
    exit 1
fi

# Update system
print_status "Updating system packages..."
sudo apt update -y

# Install essential build tools
print_status "Installing build essentials..."
sudo apt install -y \
    build-essential \
    cmake \
    ninja-build \
    git \
    wget \
    curl \
    pkg-config \
    software-properties-common \
    ca-certificates \
    gnupg \
    lsb-release

# Install CUDA if not present
print_status "Checking CUDA installation..."
if ! command -v nvcc &> /dev/null; then
    print_status "Installing CUDA Toolkit 12.3..."
    wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.1-1_all.deb
    sudo dpkg -i cuda-keyring_1.1-1_all.deb
    sudo apt update
    sudo apt install -y cuda-toolkit-12-3
    
    # Add CUDA to PATH
    echo 'export PATH=/usr/local/cuda-12.3/bin:$PATH' >> ~/.bashrc
    echo 'export LD_LIBRARY_PATH=/usr/local/cuda-12.3/lib64:$LD_LIBRARY_PATH' >> ~/.bashrc
    source ~/.bashrc
    
    print_success "CUDA Toolkit installed"
else
    CUDA_VERSION=$(nvcc --version | grep "release" | sed 's/.*release \([0-9]\+\.[0-9]\+\).*/\1/')
    print_success "CUDA $CUDA_VERSION already installed"
fi

# Install gRPC and Protocol Buffers
print_status "Installing gRPC and Protocol Buffers..."
sudo apt install -y \
    libgrpc++-dev \
    libprotobuf-dev \
    protobuf-compiler \
    protobuf-compiler-grpc

# Install Arrow (may need to build from source on some systems)
print_status "Installing Apache Arrow..."
sudo apt install -y libarrow-dev libparquet-dev || {
    print_warning "Arrow not available via apt, you may need to build from source"
}

# Install additional dependencies
print_status "Installing additional dependencies..."
sudo apt install -y \
    libgoogle-glog-dev \
    libgflags-dev \
    libssl-dev \
    zlib1g-dev \
    libc-ares-dev \
    libgtest-dev \
    libbenchmark-dev

# Install Docker for containerization
print_status "Installing Docker..."
if ! command -v docker &> /dev/null; then
    # Add Docker's official GPG key
    curl -fsSL https://download.docker.com/linux/ubuntu/gpg | sudo gpg --dearmor -o /usr/share/keyrings/docker-archive-keyring.gpg
    
    # Add Docker repository
    echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/docker-archive-keyring.gpg] https://download.docker.com/linux/ubuntu $(lsb_release -cs) stable" | sudo tee /etc/apt/sources.list.d/docker.list > /dev/null
    
    sudo apt update
    sudo apt install -y docker-ce docker-ce-cli containerd.io docker-compose-plugin
    
    # Add current user to docker group
    sudo usermod -aG docker $USER
    print_success "Docker installed - please logout and login again to use docker without sudo"
else
    print_success "Docker already installed"
fi

# Install NVIDIA Container Toolkit
print_status "Installing NVIDIA Container Toolkit..."
if ! dpkg -l | grep -q nvidia-container-toolkit; then
    curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
    curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list | \
        sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
        sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list
    
    sudo apt update
    sudo apt install -y nvidia-container-toolkit
    sudo nvidia-ctk runtime configure --runtime=docker
    sudo systemctl restart docker
    
    print_success "NVIDIA Container Toolkit installed"
else
    print_success "NVIDIA Container Toolkit already installed"
fi

# Install development tools
print_status "Installing development tools..."
sudo apt install -y \
    vim \
    htop \
    tree \
    jq \
    bc

# Install grpcurl for API testing
print_status "Installing grpcurl..."
if ! command -v grpcurl &> /dev/null; then
    go install github.com/fullstorydev/grpcurl/cmd/grpcurl@latest
    echo 'export PATH=$HOME/go/bin:$PATH' >> ~/.bashrc
fi

# Clone the repository if not already present
if [ ! -d "CUDA-AcceleratedVectorDatabaseEngine" ]; then
    print_status "Cloning CUDA Vector Database repository..."
    git clone https://github.com/wedevxer/CUDA-AcceleratedVectorDatabaseEngine.git
    cd CUDA-AcceleratedVectorDatabaseEngine
else
    print_success "Repository already cloned"
    cd CUDA-AcceleratedVectorDatabaseEngine
fi

# Build the project
print_status "Building CUDA Vector Database..."
mkdir -p build
cd build

cmake -GNinja \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_CUDA_ARCHITECTURES="70" \
    -DCMAKE_CXX_FLAGS="-O3 -DNDEBUG" \
    -DCMAKE_CUDA_FLAGS="-O3 --use_fast_math" \
    ..

if ninja -j$(nproc); then
    print_success "Build completed successfully!"
else
    print_error "Build failed. Check the error messages above."
    print_status "Common fixes:"
    print_status "  - Make sure all dependencies are installed"
    print_status "  - Check CUDA installation: nvcc --version"
    print_status "  - Verify GPU is accessible: nvidia-smi"
    exit 1
fi

cd ..

# Run quick validation
print_status "Running validation tests..."
if ./scripts/quick-test.sh; then
    print_success "Validation passed!"
else
    print_warning "Some validation tests failed - this is normal if CUDA/GPU drivers need a reboot"
fi

echo
print_success "🎉 Development environment setup complete!"
echo
print_status "Next steps:"
print_status "1. Logout and login again to use Docker without sudo"
print_status "2. Source your bashrc: source ~/.bashrc"
print_status "3. Test the server: ./build/server/vdb_server --help"
print_status "4. Run integration tests: ./test/integration/run_integration_tests.sh"
print_status "5. Build Docker image: docker build -t cuda-vector-db ."
echo
print_status "Development commands:"
print_status "  ninja -C build                    # Rebuild"
print_status "  ctest --test-dir build             # Run unit tests"
print_status "  ./scripts/healthcheck.sh           # Health check"
print_status "  ./build/server/vdb_server --help   # Server options"
echo
print_success "Happy coding! 🚀"