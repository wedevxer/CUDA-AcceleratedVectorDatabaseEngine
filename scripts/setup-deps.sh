#!/bin/bash

# Dependency setup script for CUDA Vector Database
# Installs required system packages and libraries

set -e

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

print_status() { echo -e "${BLUE}[INFO]${NC} $1"; }
print_success() { echo -e "${GREEN}[SUCCESS]${NC} $1"; }
print_warning() { echo -e "${YELLOW}[WARNING]${NC} $1"; }
print_error() { echo -e "${RED}[ERROR]${NC} $1"; }

# Detect OS
if [ -f /etc/os-release ]; then
    . /etc/os-release
    OS=$NAME
    VER=$VERSION_ID
elif type lsb_release >/dev/null 2>&1; then
    OS=$(lsb_release -si)
    VER=$(lsb_release -sr)
else
    OS=$(uname -s)
    VER=$(uname -r)
fi

print_status "Detected OS: $OS $VER"

# Ubuntu/Debian setup
setup_ubuntu() {
    print_status "Setting up dependencies for Ubuntu/Debian..."
    
    # Update package list
    sudo apt update
    
    # Install build essentials
    sudo apt install -y \
        build-essential \
        cmake \
        git \
        pkg-config
    
    # Install CUDA if not present
    if ! command -v nvcc &> /dev/null; then
        print_status "Installing CUDA Toolkit..."
        
        # Add NVIDIA package repository
        wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.0-1_all.deb
        sudo dpkg -i cuda-keyring_1.0-1_all.deb
        sudo apt update
        
        # Install CUDA
        sudo apt install -y cuda-toolkit-12-2
        
        # Add CUDA to PATH
        echo 'export PATH=/usr/local/cuda/bin:$PATH' >> ~/.bashrc
        echo 'export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH' >> ~/.bashrc
        
        print_success "CUDA Toolkit installed - please restart shell or source ~/.bashrc"
    else
        print_success "CUDA already installed"
    fi
    
    # Install gRPC and protobuf
    sudo apt install -y \
        libgrpc++-dev \
        libgrpc-dev \
        libprotobuf-dev \
        protobuf-compiler \
        protobuf-compiler-grpc
    
    # Install Arrow
    sudo apt install -y \
        libarrow-dev \
        libarrow-dataset-dev \
        libarrow-flight-dev
    
    # Install RocksDB
    sudo apt install -y librocksdb-dev
    
    # Install liburing for io_uring support
    sudo apt install -y liburing-dev
    
    # Install jsoncpp
    sudo apt install -y libjsoncpp-dev
    
    # Install monitoring tools
    sudo apt install -y \
        htop \
        nvtop \
        iotop
    
    print_success "Ubuntu/Debian dependencies installed"
}

# CentOS/RHEL/Rocky setup
setup_centos() {
    print_status "Setting up dependencies for CentOS/RHEL/Rocky..."
    
    # Enable EPEL
    sudo dnf install -y epel-release
    
    # Install build tools
    sudo dnf groupinstall -y "Development Tools"
    sudo dnf install -y cmake git pkg-config
    
    # Install CUDA
    if ! command -v nvcc &> /dev/null; then
        print_status "Please install CUDA Toolkit manually from NVIDIA website"
        print_status "https://developer.nvidia.com/cuda-downloads"
    fi
    
    # Install gRPC/protobuf
    sudo dnf install -y \
        grpc-devel \
        grpc-plugins \
        protobuf-devel \
        protobuf-compiler
    
    # Note: Arrow and other dependencies may need manual installation
    print_warning "Some dependencies may need manual installation on CentOS/RHEL"
    print_warning "Consider using conda or building from source"
}

# macOS setup
setup_macos() {
    print_status "Setting up dependencies for macOS..."
    
    # Install Homebrew if not present
    if ! command -v brew &> /dev/null; then
        print_status "Installing Homebrew..."
        /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
    fi
    
    # Install dependencies
    brew install \
        cmake \
        grpc \
        protobuf \
        apache-arrow \
        rocksdb \
        jsoncpp
    
    print_warning "CUDA support on macOS is limited - consider using CPU-only build"
    print_success "macOS dependencies installed"
}

# Docker setup
setup_docker() {
    print_status "Creating Docker development environment..."
    
    cat > Dockerfile << 'EOF'
FROM nvidia/cuda:12.2-devel-ubuntu22.04

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    cmake \
    git \
    pkg-config \
    libgrpc++-dev \
    libgrpc-dev \
    libprotobuf-dev \
    protobuf-compiler \
    protobuf-compiler-grpc \
    libarrow-dev \
    librocksdb-dev \
    liburing-dev \
    libjsoncpp-dev \
    htop \
    nvtop \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /workspace

# Copy project files
COPY . .

# Build project
RUN ./scripts/build.sh

# Default command
CMD ["./build/test/vdb_simple_test"]
EOF

    print_success "Dockerfile created - build with: docker build -t vdb ."
}

# Main setup logic
case "$OS" in
    *"Ubuntu"*|*"Debian"*)
        setup_ubuntu
        ;;
    *"CentOS"*|*"Red Hat"*|*"Rocky"*)
        setup_centos
        ;;
    *"Darwin"*)
        setup_macos
        ;;
    *)
        print_warning "Unsupported OS: $OS"
        print_status "Available options:"
        echo "  --docker    : Create Docker development environment"
        echo "  --manual    : Show manual installation instructions"
        
        if [[ "$1" == "--docker" ]]; then
            setup_docker
        elif [[ "$1" == "--manual" ]]; then
            echo ""
            echo "Manual Installation Requirements:"
            echo "  - CUDA Toolkit 12.x"
            echo "  - CMake 3.24+"
            echo "  - gRPC and Protocol Buffers"
            echo "  - Apache Arrow"
            echo "  - RocksDB"
            echo "  - liburing (Linux only)"
            echo "  - jsoncpp"
        fi
        ;;
esac

print_status "Setup completed! Next steps:"
echo "  1. Restart your shell or source ~/.bashrc"
echo "  2. Run ./scripts/build.sh to build the project"
echo "  3. Run ./build/test/vdb_simple_test to validate installation"