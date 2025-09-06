# CUDA Vector Database - Production Deployment Guide

## 🚀 GCP Production Deployment

This guide covers deploying the CUDA Vector Database to Google Cloud Platform in a production-ready configuration.

## Prerequisites

### Required Tools
```bash
# Google Cloud SDK
curl https://sdk.cloud.google.com | bash
gcloud auth login
gcloud config set project YOUR_PROJECT_ID

# Kubectl
gcloud components install kubectl

# Docker
# Install Docker from https://docs.docker.com/get-docker/

# Optional: grpcurl for testing
go install github.com/fullstorydev/grpcurl/cmd/grpcurl@latest
```

### GCP Setup
```bash
# Enable required APIs
gcloud services enable container.googleapis.com
gcloud services enable compute.googleapis.com
gcloud services enable storage-api.googleapis.com
gcloud services enable logging.googleapis.com
gcloud services enable monitoring.googleapis.com

# Set up authentication for Docker
gcloud auth configure-docker
```

## Quick Deployment

### 1. Automated Deployment
```bash
# Set your project ID
export PROJECT_ID="your-gcp-project-id"

# Run automated deployment
./scripts/deploy-gcp.sh
```

### 2. Manual Deployment

#### Build and Push Docker Image
```bash
export PROJECT_ID="your-gcp-project-id"
export IMAGE_TAG="v1.0.0"

# Build image
docker build -t gcr.io/$PROJECT_ID/cuda-vector-db:$IMAGE_TAG .

# Push to GCR
docker push gcr.io/$PROJECT_ID/cuda-vector-db:$IMAGE_TAG
```

#### Create GKE Cluster
```bash
# Create cluster with GPU support
gcloud container clusters create vdb-cluster \
    --project=$PROJECT_ID \
    --region=us-central1 \
    --machine-type=n1-standard-4 \
    --num-nodes=2 \
    --enable-autoscaling \
    --min-nodes=1 \
    --max-nodes=10 \
    --enable-autorepair \
    --enable-autoupgrade \
    --enable-ip-alias \
    --disk-size=100GB \
    --disk-type=pd-ssd

# Add GPU node pool
gcloud container node-pools create gpu-pool \
    --project=$PROJECT_ID \
    --region=us-central1 \
    --cluster=vdb-cluster \
    --machine-type=n1-standard-4 \
    --accelerator=type=nvidia-tesla-v100,count=1 \
    --num-nodes=1 \
    --enable-autoscaling \
    --min-nodes=0 \
    --max-nodes=5 \
    --node-taints=nvidia.com/gpu=true:NoSchedule
```

#### Deploy Application
```bash
# Get credentials
gcloud container clusters get-credentials vdb-cluster --region=us-central1

# Install NVIDIA GPU drivers
kubectl apply -f https://raw.githubusercontent.com/GoogleCloudPlatform/container-engine-accelerators/master/nvidia-driver-installer/cos/daemonset-preloaded.yaml

# Update image in deployment manifest
sed -i 's|gcr.io/YOUR_PROJECT_ID|gcr.io/'$PROJECT_ID'|g' k8s/deployment.yaml

# Deploy application
kubectl apply -f k8s/
```

## Configuration

### Environment Variables
```yaml
# In k8s/configmap.yaml or via environment variables
VDB_CONFIG_PATH: "/etc/vdb/production.yaml"
CUDA_VISIBLE_DEVICES: "0"
NVIDIA_VISIBLE_DEVICES: "all"
```

### Resource Requirements
```yaml
# Minimum resources per pod
resources:
  requests:
    memory: "8Gi"
    cpu: "4"
    nvidia.com/gpu: 1
  limits:
    memory: "16Gi"
    cpu: "8"
    nvidia.com/gpu: 1
```

### Storage Configuration
- **Data Volume**: 1TB SSD for vector storage
- **Log Volume**: 100GB SSD for application logs
- **Storage Class**: `ssd-retain` for production data persistence

## Monitoring & Observability

### Health Checks
```bash
# Manual health check
./scripts/healthcheck.sh

# Check via Kubernetes
kubectl get pods -n vdb-system
kubectl describe pod -n vdb-system -l app.kubernetes.io/name=cuda-vector-db
```

### Metrics & Monitoring
- **Prometheus Metrics**: Available at `:8080/metrics`
- **Grafana Dashboard**: Import `monitoring/grafana-dashboard.json`
- **Alerts**: Configure via `monitoring/prometheus-rules.yaml`

### Logging
```bash
# View application logs
kubectl logs -f deployment/vdb-server -n vdb-system

# View all pods logs
kubectl logs -f -l app.kubernetes.io/name=cuda-vector-db -n vdb-system
```

## Testing Deployment

### Integration Tests
```bash
# Run integration tests against deployed server
export VDB_SERVER="EXTERNAL_IP:50051"
./test/integration/run_integration_tests.sh
```

### Load Testing
```bash
# Run load test
./build/test/integration/vdb_load_test \
    --server=EXTERNAL_IP:50051 \
    --threads=8 \
    --requests=1000
```

### Basic API Testing
```bash
# Test service discovery
grpcurl -plaintext EXTERNAL_IP:50051 list

# Test health check
grpcurl -plaintext EXTERNAL_IP:50051 grpc.health.v1.Health/Check

# Create test index
grpcurl -plaintext -d '{
  "name": "test_index", 
  "dimension": 128, 
  "metric": "L2", 
  "nlist": 256
}' EXTERNAL_IP:50051 vdb.AdminService/CreateIndex
```

## Scaling & Performance

### Horizontal Pod Autoscaler
```bash
# HPA is configured to scale based on:
# - CPU utilization (70%)
# - Memory utilization (80%) 
# - Custom QPS metric (100 per pod)

kubectl get hpa -n vdb-system
```

### GPU Node Scaling
```bash
# Scale GPU node pool
gcloud container clusters resize vdb-cluster \
    --node-pool=gpu-pool \
    --num-nodes=3 \
    --region=us-central1
```

### Performance Optimization
1. **Batch Size Tuning**: Adjust `--batch-size` based on GPU memory
2. **Memory Pools**: Configure GPU memory pools via `--gpu-memory`
3. **Concurrent Searches**: Tune `max_concurrent_searches` in config

## Security

### Network Security
- Internal LoadBalancer by default
- Network policies restrict pod-to-pod communication
- TLS termination at load balancer level

### RBAC
- Minimal service account permissions
- Pod security contexts with non-root user
- Read-only filesystem where possible

### Secrets Management
```bash
# Store sensitive configuration in secrets
kubectl create secret generic vdb-secrets \
    --from-literal=database-password=<password> \
    -n vdb-system
```

## Troubleshooting

### Common Issues

#### Pod Stuck in Pending
```bash
# Check node resources and GPU availability
kubectl describe nodes
kubectl get pods -n vdb-system -o wide

# Check GPU driver installation
kubectl get daemonset nvidia-driver-installer -n kube-system
```

#### High Memory Usage
```bash
# Check memory metrics
kubectl top pods -n vdb-system
kubectl exec -it deployment/vdb-server -n vdb-system -- nvidia-smi
```

#### gRPC Connection Issues
```bash
# Check service and endpoints
kubectl get svc -n vdb-system
kubectl get endpoints -n vdb-system

# Test internal connectivity
kubectl run -it --rm debug --image=busybox --restart=Never -- sh
# Inside pod: telnet vdb-server.vdb-system.svc.cluster.local 50051
```

### Performance Issues
```bash
# Check GPU utilization
kubectl exec deployment/vdb-server -n vdb-system -- nvidia-smi

# Check application metrics
curl http://EXTERNAL_IP:8080/metrics | grep vdb_

# Check resource limits
kubectl describe pod -n vdb-system -l app.kubernetes.io/name=cuda-vector-db
```

## Backup & Recovery

### Data Backup
```bash
# Backup persistent volumes
gcloud compute snapshots create vdb-data-snapshot \
    --source-disk=<pv-disk-name> \
    --zone=<zone>
```

### Configuration Backup
```bash
# Export current configuration
kubectl get all,configmap,secret -n vdb-system -o yaml > vdb-backup.yaml
```

## Cleanup

### Remove Deployment
```bash
# Delete application
kubectl delete -f k8s/

# Delete namespace
kubectl delete namespace vdb-system

# Delete cluster (if no longer needed)
gcloud container clusters delete vdb-cluster --region=us-central1
```

### Cost Optimization
- Use preemptible nodes for development/testing
- Configure aggressive cluster autoscaling
- Monitor GPU usage and scale down unused nodes
- Use committed use discounts for production workloads

## Next Steps

1. **Set up monitoring dashboards** in Grafana/GCP Monitoring
2. **Configure alerting** for critical metrics
3. **Implement backup automation** for data persistence  
4. **Set up CI/CD pipeline** for automated deployments
5. **Performance tuning** based on workload patterns
6. **Security hardening** with TLS, authentication, and network policies

## Support

For issues and questions:
- Check application logs: `kubectl logs -f deployment/vdb-server -n vdb-system`
- Run health checks: `./scripts/healthcheck.sh`
- Review metrics: `http://EXTERNAL_IP:8080/metrics`
- Test with integration suite: `./test/integration/run_integration_tests.sh`