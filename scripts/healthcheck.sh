#!/bin/bash

# Health check script for VDB server
# This script performs comprehensive health checks for the CUDA Vector Database

set -e

# Configuration
GRPC_HOST=${GRPC_HOST:-"localhost"}
GRPC_PORT=${GRPC_PORT:-"50051"}
METRICS_HOST=${METRICS_HOST:-"localhost"}
METRICS_PORT=${METRICS_PORT:-"8080"}
TIMEOUT=${TIMEOUT:-"10"}

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

print_check() { echo -e "${YELLOW}[CHECK]${NC} $1"; }
print_pass() { echo -e "${GREEN}[PASS]${NC} $1"; }
print_fail() { echo -e "${RED}[FAIL]${NC} $1"; }

exit_code=0

# Function to check gRPC health
check_grpc_health() {
    print_check "Checking gRPC server health..."
    
    # Try to connect to gRPC server
    if command -v grpc_health_probe &> /dev/null; then
        if grpc_health_probe -addr="$GRPC_HOST:$GRPC_PORT" -rpc-timeout="${TIMEOUT}s" &>/dev/null; then
            print_pass "gRPC server is healthy"
            return 0
        else
            print_fail "gRPC health check failed"
            return 1
        fi
    else
        # Fallback to grpcurl if available
        if command -v grpcurl &> /dev/null; then
            if timeout "$TIMEOUT" grpcurl -plaintext "$GRPC_HOST:$GRPC_PORT" grpc.health.v1.Health/Check &>/dev/null; then
                print_pass "gRPC server is responding"
                return 0
            else
                print_fail "gRPC server not responding"
                return 1
            fi
        else
            # Basic TCP connectivity check
            if timeout "$TIMEOUT" bash -c "</dev/tcp/$GRPC_HOST/$GRPC_PORT" &>/dev/null; then
                print_pass "gRPC port is open"
                return 0
            else
                print_fail "Cannot connect to gRPC port"
                return 1
            fi
        fi
    fi
}

# Function to check metrics endpoint
check_metrics_health() {
    print_check "Checking metrics endpoint..."
    
    if command -v curl &> /dev/null; then
        if curl -s -f --connect-timeout "$TIMEOUT" "http://$METRICS_HOST:$METRICS_PORT/metrics" > /dev/null; then
            print_pass "Metrics endpoint is healthy"
            return 0
        else
            print_fail "Metrics endpoint not responding"
            return 1
        fi
    else
        # Basic TCP connectivity check
        if timeout "$TIMEOUT" bash -c "</dev/tcp/$METRICS_HOST/$METRICS_PORT" &>/dev/null; then
            print_pass "Metrics port is open"
            return 0
        else
            print_fail "Cannot connect to metrics port"
            return 1
        fi
    fi
}

# Function to check CUDA availability
check_cuda_health() {
    print_check "Checking CUDA availability..."
    
    if command -v nvidia-smi &> /dev/null; then
        if nvidia-smi &>/dev/null; then
            gpu_count=$(nvidia-smi -L 2>/dev/null | wc -l)
            if [ "$gpu_count" -gt 0 ]; then
                print_pass "CUDA is available ($gpu_count GPU(s) detected)"
                return 0
            else
                print_fail "No GPUs detected"
                return 1
            fi
        else
            print_fail "nvidia-smi failed"
            return 1
        fi
    else
        print_check "nvidia-smi not available, skipping GPU check"
        return 0
    fi
}

# Function to check memory usage
check_memory_health() {
    print_check "Checking memory usage..."
    
    # Check system memory
    if command -v free &> /dev/null; then
        mem_usage=$(free | grep Mem | awk '{print ($3/$2) * 100.0}')
        if (( $(echo "$mem_usage < 90" | bc -l) )); then
            print_pass "System memory usage: ${mem_usage}%"
        else
            print_fail "High system memory usage: ${mem_usage}%"
            return 1
        fi
    fi
    
    # Check GPU memory if available
    if command -v nvidia-smi &> /dev/null; then
        if nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader,nounits 2>/dev/null | head -1 | while IFS=', ' read -r used total; do
            if [ -n "$used" ] && [ -n "$total" ] && [ "$total" -gt 0 ]; then
                gpu_usage=$(echo "scale=1; $used * 100 / $total" | bc)
                if (( $(echo "$gpu_usage < 95" | bc -l) )); then
                    print_pass "GPU memory usage: ${gpu_usage}%"
                else
                    print_fail "High GPU memory usage: ${gpu_usage}%"
                    exit 1
                fi
            fi
        done
        
        # Check exit status from the subshell
        if [ $? -ne 0 ]; then
            return 1
        fi
    fi
    
    return 0
}

# Function to check disk space
check_disk_health() {
    print_check "Checking disk space..."
    
    data_path=${VDB_DATA_PATH:-"/data/vdb"}
    
    if [ -d "$data_path" ]; then
        disk_usage=$(df "$data_path" | tail -1 | awk '{print $5}' | sed 's/%//')
        if [ "$disk_usage" -lt 90 ]; then
            print_pass "Disk usage: ${disk_usage}%"
            return 0
        else
            print_fail "High disk usage: ${disk_usage}%"
            return 1
        fi
    else
        print_check "Data path $data_path not found, skipping disk check"
        return 0
    fi
}

# Function to check process health
check_process_health() {
    print_check "Checking process health..."
    
    if pgrep -f "vdb_server" > /dev/null; then
        print_pass "VDB server process is running"
        
        # Check if process is responsive (not zombie/hung)
        cpu_usage=$(ps -o %cpu -p $(pgrep -f "vdb_server") --no-headers | tr -d ' ')
        if [ -n "$cpu_usage" ]; then
            print_pass "Process CPU usage: ${cpu_usage}%"
        fi
        
        return 0
    else
        print_fail "VDB server process not found"
        return 1
    fi
}

# Function to perform a basic functional test
check_functional_health() {
    print_check "Performing functional test..."
    
    if command -v grpcurl &> /dev/null; then
        # Test service discovery
        if grpcurl -plaintext "$GRPC_HOST:$GRPC_PORT" list 2>/dev/null | grep -q "vdb.QueryService"; then
            print_pass "QueryService is available"
        else
            print_fail "QueryService not available"
            return 1
        fi
        
        if grpcurl -plaintext "$GRPC_HOST:$GRPC_PORT" list 2>/dev/null | grep -q "vdb.AdminService"; then
            print_pass "AdminService is available"
        else
            print_fail "AdminService not available"
            return 1
        fi
        
        return 0
    else
        print_check "grpcurl not available, skipping functional test"
        return 0
    fi
}

# Main health check execution
echo "=== CUDA Vector Database Health Check ==="
echo

# Run all health checks
if ! check_process_health; then exit_code=1; fi
if ! check_grpc_health; then exit_code=1; fi
if ! check_metrics_health; then exit_code=1; fi
if ! check_cuda_health; then exit_code=1; fi
if ! check_memory_health; then exit_code=1; fi
if ! check_disk_health; then exit_code=1; fi
if ! check_functional_health; then exit_code=1; fi

echo
if [ $exit_code -eq 0 ]; then
    print_pass "All health checks passed ✅"
else
    print_fail "Some health checks failed ❌"
fi

exit $exit_code