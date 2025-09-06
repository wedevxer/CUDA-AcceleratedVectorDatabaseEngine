# CUDA-Accelerated Vector Database Engine

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![CUDA](https://img.shields.io/badge/CUDA-12.3-green.svg)](https://developer.nvidia.com/cuda-toolkit)
[![C++](https://img.shields.io/badge/C++-20-blue.svg)](https://en.cppreference.com/w/cpp/20)
[![gRPC](https://img.shields.io/badge/gRPC-1.60-orange.svg)](https://grpc.io/)

A high-performance, production-ready vector database engine optimized for NVIDIA GPUs, designed for billion-scale approximate nearest neighbor (ANN) search with enterprise-grade features.

## 🚀 Key Features

### **GPU-Accelerated Performance**
- **CUDA 12.3+ Support** - Optimized for V100, A100, RTX series, and H100 GPUs
- **IVF-Flat Index** - GPU-accelerated inverted file indexing with k-means clustering
- **Batch Processing** - Intelligent request batching for optimal GPU utilization
- **Memory Management** - Advanced GPU memory pools with automatic fallback to CPU

### **Production-Ready Architecture**
- **gRPC API** - High-performance RPC with streaming support
- **Kubernetes Native** - Complete K8s manifests with auto-scaling
- **Docker Containerized** - Multi-stage builds with security hardening
- **Health Monitoring** - Comprehensive health checks and Prometheus metrics

### **Enterprise Features**
- **Multi-GPU Support** - Automatic GPU detection and utilization
- **Circuit Breakers** - Fault tolerance with automatic recovery
- **Load Balancing** - Advanced request scheduling with priority queues
- **Arrow Storage** - Efficient columnar data format with zero-copy access
- **Epoch Management** - Versioned indexes with zero-downtime updates

### **Performance Optimizations**
- **io_uring Prefetching** - High-performance asynchronous I/O on Linux
- **Memory Validation** - Complete CUDA error handling with leak detection  
- **Adaptive Caching** - LFU eviction with access pattern detection
- **NVMe Optimization** - Direct storage integration for billion-scale datasets

## 📊 Performance Benchmarks

| Configuration | QPS | Latency (p99) | GPU Memory | Throughput |
|---------------|-----|---------------|------------|------------|
| V100 16GB     | 15K | 12ms         | 14GB       | 1.2M vec/min |
| A100 40GB     | 45K | 6ms          | 35GB       | 3.8M vec/min |
| RTX 4090 24GB | 25K | 8ms          | 20GB       | 2.1M vec/min |

*Benchmarks on 10M vectors, 768D, batch_size=64, nprobe=32*

## 🛠️ Quick Start

### Prerequisites
- **NVIDIA GPU** with Compute Capability 7.0+ (V100, RTX, A100, H100)
- **CUDA Toolkit 12.3+**
- **CMake 3.24+**
- **C++ Compiler** with C++20 support
- **Docker** (for containerized deployment)

### Local Development Build

```bash
# Clone the repository
git clone https://github.com/wedevxer/CUDA-AcceleratedVectorDatabaseEngine.git
cd CUDA-AcceleratedVectorDatabaseEngine

# Quick validation
./scripts/quick-test.sh

# Install dependencies (Ubuntu/Debian)
sudo apt update && sudo apt install -y \
    build-essential cmake ninja-build \
    libgrpc++-dev libprotobuf-dev protobuf-compiler \
    libarrow-dev libgoogle-glog-dev

# Build the project
mkdir build && cd build
cmake -GNinja -DCMAKE_BUILD_TYPE=Release ..
ninja -j$(nproc)

# Run tests
ctest -j$(nproc) --output-on-failure

# Start the server
./server/vdb_server --data-path=/tmp/vdb --gpu-memory=8
```

### Docker Deployment

```bash
# Build Docker image
docker build -t cuda-vector-db .

# Run with GPU support
docker run --gpus all -p 50051:50051 -p 8080:8080 \
    -v /data/vdb:/data/vdb \
    cuda-vector-db
```

### GCP Production Deployment

```bash
# Set your GCP project
export PROJECT_ID="your-gcp-project-id"

# Deploy to GKE with GPU support
./scripts/deploy-gcp.sh
```

See [README-DEPLOYMENT.md](README-DEPLOYMENT.md) for complete deployment guide.

## 🔧 API Usage

### Creating an Index
```bash
grpcurl -plaintext -d '{
  "name": "embeddings_index",
  "dimension": 768,
  "metric": "L2", 
  "nlist": 256
}' localhost:50051 vdb.AdminService/CreateIndex
```

### Searching Vectors
```bash
grpcurl -plaintext -d '{
  "index": "embeddings_index",
  "topk": 10,
  "nprobe": 32,
  "queries": [{
    "id": 1,
    "values": [0.1, 0.2, 0.3, ...]
  }]
}' localhost:50051 vdb.QueryService/Search
```

### Monitoring
```bash
# Health check
curl http://localhost:8080/health

# Prometheus metrics
curl http://localhost:8080/metrics
```

## 📈 Performance Tuning

### GPU Memory Configuration
```bash
# Configure GPU memory pools
./vdb_server \
    --gpu-memory=12 \              # 12GB GPU memory limit
    --batch-size=128 \             # Larger batches for higher throughput
    --coalesce-window=5            # 5ms batching window
```

### Index Optimization
```yaml
# Optimal parameters for different scales
Small Dataset (< 1M vectors):
  nlist: 128
  nprobe: 16

Medium Dataset (1M-100M vectors):
  nlist: 4096  
  nprobe: 32

Large Dataset (100M+ vectors):
  nlist: 16384
  nprobe: 64
```

## 🧪 Testing

```bash
# Unit tests
cd build && ctest

# Integration tests (requires running server)
./test/integration/run_integration_tests.sh

# Load testing
./build/test/integration/vdb_load_test \
    --server=localhost:50051 \
    --threads=8 \
    --requests=10000
```

## 🔍 Monitoring & Observability

### Key Metrics
- `vdb_search_duration_milliseconds` - Search latency percentiles
- `vdb_searches_total` - Total search request count
- `vdb_gpu_memory_bytes` - GPU memory utilization
- `vdb_queries_per_second` - Current QPS rate

### Health Checks
```bash
# Comprehensive health check
./scripts/healthcheck.sh

# gRPC health check
grpcurl -plaintext localhost:50051 grpc.health.v1.Health/Check
```

## 🏛️ Project Structure

```
├── engine/              # Core CUDA engine
│   ├── kernels.cu      # CUDA kernels for distance computation
│   ├── ivf_flat_index.* # IVF-Flat index implementation
│   ├── transfer_manager.* # GPU memory management
│   └── prefetcher.*    # io_uring async I/O
├── server/             # gRPC server implementation
│   ├── query_service.* # Query and admin services
│   ├── health_service.* # Health check implementation
│   └── main.cpp        # Server entry point
├── format/             # Storage format
│   └── storage.*       # Arrow-based persistence
├── proto/              # Protocol Buffer definitions
├── test/               # Test suites
│   ├── integration/    # gRPC API tests
│   └── *.cpp          # Unit tests
├── k8s/               # Kubernetes manifests
├── scripts/           # Deployment and utility scripts
└── configs/           # Configuration files
```

## 🤝 Contributing

We welcome contributions! Please see our [Contributing Guide](CONTRIBUTING.md) for details.

### Development Setup
```bash
# Install pre-commit hooks
pip install pre-commit
pre-commit install

# Run linting
./scripts/lint.sh

# Run all tests
./scripts/test-all.sh
```

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🔗 Links

- **Documentation**: [Full API Documentation](docs/api.md)
- **Deployment Guide**: [Production Deployment](README-DEPLOYMENT.md)
- **Performance Guide**: [Optimization Tips](docs/performance.md)
- **Issue Tracker**: [GitHub Issues](https://github.com/wedevxer/CUDA-AcceleratedVectorDatabaseEngine/issues)

## 🙏 Acknowledgments

- NVIDIA CUDA team for GPU computing frameworks
- gRPC and Protocol Buffers for efficient RPC
- Apache Arrow for columnar data formats
- Google Test for testing frameworks
- The open-source community for inspiration and tools

---

**Built with ❤️ for high-performance vector search**