#!/bin/bash

# Integration test runner that starts VDB server and runs tests

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
BUILD_DIR="$PROJECT_ROOT/build"

RED='\033[0;31m'
GREEN='\033[0;32m'
BLUE='\033[0;34m'
NC='\033[0m'

print_status() { echo -e "${BLUE}[TEST]${NC} $1"; }
print_success() { echo -e "${GREEN}[PASS]${NC} $1"; }
print_error() { echo -e "${RED}[FAIL]${NC} $1"; }

# Configuration
SERVER_PORT=${SERVER_PORT:-50051}
METRICS_PORT=${METRICS_PORT:-8080}
TEST_TIMEOUT=${TEST_TIMEOUT:-300}
SERVER_STARTUP_TIMEOUT=${SERVER_STARTUP_TIMEOUT:-30}

cleanup() {
    if [ -n "$SERVER_PID" ]; then
        print_status "Stopping VDB server (PID: $SERVER_PID)"
        kill $SERVER_PID 2>/dev/null || true
        wait $SERVER_PID 2>/dev/null || true
    fi
    
    if [ -n "$TEST_DATA_DIR" ] && [ -d "$TEST_DATA_DIR" ]; then
        print_status "Cleaning up test data"
        rm -rf "$TEST_DATA_DIR"
    fi
}

trap cleanup EXIT

print_status "VDB Integration Test Runner"
echo

# Check if server binary exists
SERVER_BINARY="$BUILD_DIR/server/vdb_server"
if [ ! -f "$SERVER_BINARY" ]; then
    print_error "Server binary not found: $SERVER_BINARY"
    print_status "Please build the project first:"
    print_status "  mkdir -p build && cd build"
    print_status "  cmake .. && make -j\$(nproc)"
    exit 1
fi

# Check if test binary exists
TEST_BINARY="$BUILD_DIR/test/integration/vdb_integration_test"
if [ ! -f "$TEST_BINARY" ]; then
    print_error "Test binary not found: $TEST_BINARY"
    print_status "Please build the integration tests first"
    exit 1
fi

# Create temporary test data directory
TEST_DATA_DIR=$(mktemp -d)
print_status "Using test data directory: $TEST_DATA_DIR"

# Start VDB server
print_status "Starting VDB server on port $SERVER_PORT"
$SERVER_BINARY \
    --address="0.0.0.0:$SERVER_PORT" \
    --data-path="$TEST_DATA_DIR" \
    --gpu-memory=2 \
    --batch-size=32 \
    > "$TEST_DATA_DIR/server.log" 2>&1 &

SERVER_PID=$!
print_status "VDB server started (PID: $SERVER_PID)"

# Wait for server to be ready
print_status "Waiting for server to be ready..."
for i in $(seq 1 $SERVER_STARTUP_TIMEOUT); do
    if timeout 2 bash -c "</dev/tcp/localhost/$SERVER_PORT" 2>/dev/null; then
        print_success "Server is ready"
        break
    fi
    
    if ! kill -0 $SERVER_PID 2>/dev/null; then
        print_error "Server process died"
        echo "Server log:"
        cat "$TEST_DATA_DIR/server.log"
        exit 1
    fi
    
    sleep 1
done

# Final check if server is responding
if ! timeout 5 bash -c "</dev/tcp/localhost/$SERVER_PORT" 2>/dev/null; then
    print_error "Server is not responding after $SERVER_STARTUP_TIMEOUT seconds"
    echo "Server log:"
    cat "$TEST_DATA_DIR/server.log"
    exit 1
fi

# Run health check
print_status "Running health check"
if [ -f "$PROJECT_ROOT/scripts/healthcheck.sh" ]; then
    GRPC_HOST=localhost GRPC_PORT=$SERVER_PORT METRICS_HOST=localhost METRICS_PORT=$METRICS_PORT \
        "$PROJECT_ROOT/scripts/healthcheck.sh" || {
        print_error "Health check failed"
        echo "Server log:"
        cat "$TEST_DATA_DIR/server.log"
        exit 1
    }
else
    print_status "Health check script not found, skipping"
fi

# Run integration tests
print_status "Running integration tests"
if timeout $TEST_TIMEOUT "$TEST_BINARY" --gtest_output=xml:"$TEST_DATA_DIR/test_results.xml"; then
    print_success "Integration tests passed"
    test_result=0
else
    print_error "Integration tests failed"
    test_result=1
fi

# Show server logs if tests failed
if [ $test_result -ne 0 ]; then
    echo
    print_status "Server logs:"
    cat "$TEST_DATA_DIR/server.log"
fi

# Optionally run load test
if [ "${RUN_LOAD_TEST:-false}" = "true" ]; then
    print_status "Running load test"
    
    LOAD_TEST_BINARY="$BUILD_DIR/test/integration/vdb_load_test"
    if [ -f "$LOAD_TEST_BINARY" ]; then
        $LOAD_TEST_BINARY \
            --server="localhost:$SERVER_PORT" \
            --threads=4 \
            --requests=50 \
            --dimension=128
    else
        print_status "Load test binary not found, skipping"
    fi
fi

# Show test results location
if [ -f "$TEST_DATA_DIR/test_results.xml" ]; then
    print_status "Test results saved to: $TEST_DATA_DIR/test_results.xml"
fi

exit $test_result