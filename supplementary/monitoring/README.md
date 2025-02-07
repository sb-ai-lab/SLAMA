# Monitoring Stack

This directory contains configurations for the monitoring stack used to monitor Spark applications:
- Grafana for visualization
- InfluxDB for time-series data storage
- JVM Profiler for collecting Spark metrics

## Prerequisites
- Kubernetes cluster
- kubectl configured with cluster access

## Installation

### Kubernetes Setup

First, set up your deployment files:
```bash
# Copy template files and update with your values
cp k8s/grafana-deployment.yaml.template k8s/grafana-deployment.yaml
cp k8s/influxdb-deployment.yaml.template k8s/influxdb-deployment.yaml
```

The template files contain placeholders for sensitive data. Update the copied files with your actual values before deploying.

Then deploy:
```bash
# Create namespace if not exists
kubectl create namespace slama

# Deploy InfluxDB
kubectl apply -f k8s/influxdb-deployment.yaml
kubectl apply -f k8s/influxdb-service.yaml

# Deploy Grafana
kubectl apply -f k8s/grafana-deployment.yaml
kubectl apply -f k8s/grafana-service.yaml
```

## Access
- Grafana is available at NodePort 30833
- InfluxDB is available at NodePort 30855
- Default credentials:
  - Username: admin
  - Password: admin

## JVM Profiler
The monitoring stack includes a custom JVM profiler based on Uber's jvm-profiler, modified to support our specific metrics:
- CPU and Memory usage
- Process information
- Stack traces

### Building the Profiler
```bash
cd monitoring/jvm-profiler/src
mvn -P influxdb clean package

# Copy the built JAR to the jars directory
mkdir -p /root/jars
cp target/jvm-profiler-1.0.0.jar /root/jars/
```

The profiler JAR will be used automatically by the Spark jobs through the configuration in `slamactl.sh`. The JAR must be in `/root/jars/` directory as this path is configured in the Spark job settings.

## Debugging
For debugging purposes, you can deploy a debug pod:
```bash
# Copy template and update with your values
cp debug-pod.yaml.template debug-pod.yaml
# Update the storage path in the copied file

# Deploy the debug pod
kubectl apply -f debug-pod.yaml
```

This pod can be used to:
- Test connectivity to InfluxDB
- Query metrics directly
- Verify data persistence

## Useful Commands
```bash
# Check pod status
kubectl get pods -n slama

# Get Grafana URL
kubectl get svc -n slama grafana

# Get InfluxDB URL
kubectl get svc -n slama influxdb

# Access debug pod
kubectl exec -it -n slama debug-pod -- sh

# Query InfluxDB (from debug pod)
wget -qO- --post-data="q=SHOW MEASUREMENTS" "http://influxdb:8086/query?db=metrics&u=admin&p=admin"
```

## Cleanup
```bash
# Delete all resources
kubectl delete -f k8s/
kubectl delete pod debug-pod -n slama  # if deployed
```

## Note on Sensitive Data
The repository includes template files (*.template) for Kubernetes deployments. These templates contain placeholders for sensitive data. When deploying:

1. Copy the template files and remove the .template extension
2. Update the copied files with your actual values
3. The actual deployment files (without .template extension) are gitignored to prevent committing sensitive data
