#!/bin/bash
set -e

# CUDA Vector Database GCP Deployment Script
# This script builds and deploys the VDB to Google Cloud Platform

# Configuration
PROJECT_ID=${PROJECT_ID:-"your-gcp-project-id"}
REGION=${REGION:-"us-central1"}
CLUSTER_NAME=${CLUSTER_NAME:-"vdb-cluster"}
IMAGE_NAME="gcr.io/$PROJECT_ID/cuda-vector-db"
IMAGE_TAG=${IMAGE_TAG:-"latest"}

RED='\033[0;31m'
GREEN='\033[0;32m'
BLUE='\033[0;34m'
NC='\033[0m'

print_status() { echo -e "${BLUE}[DEPLOY]${NC} $1"; }
print_success() { echo -e "${GREEN}[SUCCESS]${NC} $1"; }
print_error() { echo -e "${RED}[ERROR]${NC} $1"; }

print_status "CUDA Vector Database - GCP Deployment"
echo

# Check prerequisites
print_status "Checking prerequisites..."

# Check if gcloud is installed
if ! command -v gcloud &> /dev/null; then
    print_error "gcloud CLI not found. Please install Google Cloud SDK."
    exit 1
fi

# Check if kubectl is installed
if ! command -v kubectl &> /dev/null; then
    print_error "kubectl not found. Please install kubectl."
    exit 1
fi

# Check if docker is installed
if ! command -v docker &> /dev/null; then
    print_error "Docker not found. Please install Docker."
    exit 1
fi

# Verify project ID is set
if [ "$PROJECT_ID" = "your-gcp-project-id" ]; then
    print_error "Please set PROJECT_ID environment variable to your GCP project ID"
    exit 1
fi

print_success "All prerequisites found"

# Authenticate with Google Cloud
print_status "Authenticating with Google Cloud..."
gcloud auth configure-docker --quiet

# Build Docker image
print_status "Building Docker image..."
docker build -t "$IMAGE_NAME:$IMAGE_TAG" .

# Push image to Google Container Registry
print_status "Pushing image to GCR..."
docker push "$IMAGE_NAME:$IMAGE_TAG"
print_success "Image pushed: $IMAGE_NAME:$IMAGE_TAG"

# Create or update GKE cluster
print_status "Setting up GKE cluster..."

# Check if cluster exists
if gcloud container clusters describe "$CLUSTER_NAME" --region="$REGION" --project="$PROJECT_ID" &>/dev/null; then
    print_status "Cluster $CLUSTER_NAME already exists"
else
    print_status "Creating GKE cluster with GPU support..."
    gcloud container clusters create "$CLUSTER_NAME" \
        --project="$PROJECT_ID" \
        --region="$REGION" \
        --machine-type=n1-standard-4 \
        --num-nodes=2 \
        --enable-autoscaling \
        --min-nodes=1 \
        --max-nodes=10 \
        --enable-autorepair \
        --enable-autoupgrade \
        --maintenance-window-start="2023-01-01T09:00:00Z" \
        --maintenance-window-end="2023-01-01T17:00:00Z" \
        --maintenance-window-recurrence="FREQ=WEEKLY;BYDAY=SA" \
        --enable-network-policy \
        --enable-ip-alias \
        --disk-size=100GB \
        --disk-type=pd-ssd \
        --image-type=COS_CONTAINERD \
        --logging=SYSTEM,WORKLOAD \
        --monitoring=SYSTEM \
        --enable-shielded-nodes
    
    print_success "GKE cluster created"
fi

# Create GPU node pool
print_status "Creating GPU node pool..."
GPU_POOL_NAME="gpu-pool"

if gcloud container node-pools describe "$GPU_POOL_NAME" --cluster="$CLUSTER_NAME" --region="$REGION" --project="$PROJECT_ID" &>/dev/null; then
    print_status "GPU node pool already exists"
else
    gcloud container node-pools create "$GPU_POOL_NAME" \
        --project="$PROJECT_ID" \
        --region="$REGION" \
        --cluster="$CLUSTER_NAME" \
        --machine-type=n1-standard-4 \
        --accelerator=type=nvidia-tesla-v100,count=1 \
        --num-nodes=1 \
        --enable-autoscaling \
        --min-nodes=0 \
        --max-nodes=5 \
        --enable-autorepair \
        --enable-autoupgrade \
        --disk-size=100GB \
        --disk-type=pd-ssd \
        --image-type=COS_CONTAINERD \
        --node-taints=nvidia.com/gpu=true:NoSchedule
    
    print_success "GPU node pool created"
fi

# Get cluster credentials
print_status "Getting cluster credentials..."
gcloud container clusters get-credentials "$CLUSTER_NAME" --region="$REGION" --project="$PROJECT_ID"

# Install NVIDIA GPU operator
print_status "Installing NVIDIA GPU operator..."
kubectl apply -f https://raw.githubusercontent.com/GoogleCloudPlatform/container-engine-accelerators/master/nvidia-driver-installer/cos/daemonset-preloaded.yaml

# Wait for GPU operator to be ready
print_status "Waiting for GPU operator to be ready..."
kubectl wait --for=condition=ready pod -l name=nvidia-driver-installer --timeout=300s -n kube-system

# Update Kubernetes manifests with correct image
print_status "Updating Kubernetes manifests..."
sed -i.bak "s|gcr.io/YOUR_PROJECT_ID/cuda-vector-db:latest|$IMAGE_NAME:$IMAGE_TAG|g" k8s/deployment.yaml
print_success "Manifests updated"

# Deploy to Kubernetes
print_status "Deploying to Kubernetes..."

# Apply manifests in order
kubectl apply -f k8s/namespace.yaml
kubectl apply -f k8s/rbac.yaml
kubectl apply -f k8s/configmap.yaml
kubectl apply -f k8s/persistent-volume.yaml

# Wait for PVCs to be bound
print_status "Waiting for storage to be provisioned..."
kubectl wait --for=condition=Bound pvc/vdb-data-pvc -n vdb-system --timeout=300s
kubectl wait --for=condition=Bound pvc/vdb-logs-pvc -n vdb-system --timeout=300s

# Deploy application
kubectl apply -f k8s/deployment.yaml
kubectl apply -f k8s/service.yaml
kubectl apply -f k8s/hpa.yaml

# Wait for deployment to be ready
print_status "Waiting for deployment to be ready..."
kubectl wait --for=condition=available deployment/vdb-server -n vdb-system --timeout=600s

# Get service information
print_status "Getting service information..."
kubectl get services -n vdb-system

# Get external IP
EXTERNAL_IP=""
print_status "Waiting for LoadBalancer IP..."
for i in {1..30}; do
    EXTERNAL_IP=$(kubectl get service vdb-server -n vdb-system -o jsonpath='{.status.loadBalancer.ingress[0].ip}' 2>/dev/null)
    if [ -n "$EXTERNAL_IP" ]; then
        break
    fi
    echo "Waiting for LoadBalancer IP... ($i/30)"
    sleep 10
done

if [ -n "$EXTERNAL_IP" ]; then
    print_success "VDB Server deployed successfully!"
    echo
    echo "🚀 Access Information:"
    echo "  gRPC Endpoint: $EXTERNAL_IP:50051"
    echo "  Metrics: http://$EXTERNAL_IP:8080/metrics"
    echo
    echo "📋 Management Commands:"
    echo "  kubectl get pods -n vdb-system"
    echo "  kubectl logs -f deployment/vdb-server -n vdb-system"
    echo "  kubectl describe service vdb-server -n vdb-system"
    echo
    echo "🧪 Test Connection:"
    echo "  grpcurl -plaintext $EXTERNAL_IP:50051 list"
    echo
else
    print_error "Failed to get LoadBalancer IP. Check service status:"
    kubectl describe service vdb-server -n vdb-system
    exit 1
fi

# Optional: Install monitoring
if [ "${INSTALL_MONITORING:-false}" = "true" ]; then
    print_status "Installing Prometheus monitoring..."
    kubectl apply -f k8s/servicemonitor.yaml
    print_success "Monitoring installed"
fi

# Restore original manifests
mv k8s/deployment.yaml.bak k8s/deployment.yaml

print_success "Deployment completed successfully! 🎉"