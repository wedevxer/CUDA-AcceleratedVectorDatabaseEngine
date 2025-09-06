# Multi-stage Docker build for CUDA Vector Database
# Stage 1: Development environment with all build dependencies
FROM nvidia/cuda:12.3-devel-ubuntu22.04 AS builder

# Avoid interactive prompts during package installation
ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=UTC

# Install system dependencies
RUN apt-get update && apt-get install -y \
    # Build tools
    build-essential \
    cmake \
    ninja-build \
    pkg-config \
    git \
    wget \
    curl \
    # gRPC/Protobuf dependencies
    libgrpc++-dev \
    libprotobuf-dev \
    protobuf-compiler \
    protobuf-compiler-grpc \
    # Arrow dependencies  
    libarrow-dev \
    libparquet-dev \
    # Other dependencies
    libssl-dev \
    zlib1g-dev \
    libc-ares-dev \
    libgflags-dev \
    libgoogle-glog-dev \
    && rm -rf /var/lib/apt/lists/*

# Set up build environment
WORKDIR /workspace

# Copy source code
COPY . .

# Create build directory
RUN mkdir -p build

# Configure CMake build
RUN cd build && \
    cmake -GNinja \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_CUDA_ARCHITECTURES="70;75;80;86;90" \
    -DCMAKE_INSTALL_PREFIX=/opt/vdb \
    -DCMAKE_CXX_FLAGS="-O3 -DNDEBUG" \
    -DCMAKE_CUDA_FLAGS="-O3 --use_fast_math" \
    ..

# Build the project
RUN cd build && ninja -j$(nproc)

# Install to staging area
RUN cd build && ninja install

# Stage 2: Runtime environment
FROM nvidia/cuda:12.3-runtime-ubuntu22.04 AS runtime

# Avoid interactive prompts
ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=UTC

# Install runtime dependencies only
RUN apt-get update && apt-get install -y \
    # gRPC runtime
    libgrpc++1.46 \
    libprotobuf32 \
    # Arrow runtime
    libarrow1300 \
    libparquet1300 \
    # System libraries
    libssl3 \
    libgoogle-glog0v5 \
    libc-ares2 \
    zlib1g \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Create non-root user for security
RUN groupadd -g 1000 vdb && \
    useradd -r -u 1000 -g vdb -m -s /bin/bash vdb

# Create necessary directories
RUN mkdir -p /data/vdb /data/epochs /data/indices /var/log/vdb && \
    chown -R vdb:vdb /data /var/log/vdb

# Copy built binaries from builder stage
COPY --from=builder /opt/vdb /opt/vdb
COPY --from=builder /workspace/scripts /opt/vdb/scripts

# Make scripts executable
RUN chmod +x /opt/vdb/scripts/*.sh

# Set up PATH
ENV PATH="/opt/vdb/bin:$PATH"
ENV LD_LIBRARY_PATH="/opt/vdb/lib:$LD_LIBRARY_PATH"

# Create config directory and default config
RUN mkdir -p /etc/vdb
COPY --from=builder /workspace/configs/production.yaml /etc/vdb/config.yaml

# Health check script
COPY <<EOF /opt/vdb/bin/healthcheck.sh
#!/bin/bash
# gRPC health check using grpc_health_probe
exec /opt/vdb/bin/grpc_health_probe -addr=localhost:50051 -service=vdb.QueryService
EOF

RUN chmod +x /opt/vdb/bin/healthcheck.sh

# Switch to non-root user
USER vdb
WORKDIR /data/vdb

# Expose gRPC port
EXPOSE 50051
# Expose metrics port
EXPOSE 8080

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD /opt/vdb/bin/healthcheck.sh || exit 1

# Default command
CMD ["/opt/vdb/bin/vdb_server", \
     "--address=0.0.0.0:50051", \
     "--data-path=/data/vdb", \
     "--gpu-memory=6", \
     "--batch-size=64"]