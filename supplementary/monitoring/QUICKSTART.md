# Quick Start Guide for SLAMA Monitoring

## 1. Infrastructure Setup

### Deploy Grafana + InfluxDB
```bash
# Create namespace
kubectl create namespace slama

# Deploy InfluxDB
cp k8s/influxdb-deployment.yaml.template k8s/influxdb-deployment.yaml
kubectl apply -f k8s/influxdb-deployment.yaml
kubectl apply -f k8s/influxdb-service.yaml

# Deploy Grafana
cp k8s/grafana-deployment.yaml.template k8s/grafana-deployment.yaml
kubectl apply -f k8s/grafana-deployment.yaml
kubectl apply -f k8s/grafana-service.yaml
```

Default credentials:
- Username: admin
- Password: admin
- Grafana port: 30833
- InfluxDB port: 30855

## 2. Environment Configuration

Create/edit `.env` file in project root:
```bash
# InfluxDB Configuration
INFLUXDB_HOST=node11.bdcl         # Your InfluxDB host
INFLUXDB_PORT=30855              # Your InfluxDB port
INFLUXDB_DATABASE=spark_metrics  # Database name
INFLUXDB_USERNAME=admin         # InfluxDB username
INFLUXDB_PASSWORD=admin         # InfluxDB password

# Spark Configuration
SPARK_DRIVER_CORES=2
SPARK_DRIVER_MEMORY=16g
SPARK_EXECUTOR_INSTANCES=6
SPARK_EXECUTOR_CORES=4
SPARK_EXECUTOR_MEMORY=16g
SPARK_MEMORY_OVERHEAD_FACTOR=0.1073741824
SPARK_MEMORY_STORAGE_FRACTION=0.05

# Dataset Configuration
DATASET_NAME=used_cars_dataset_100x      # Full dataset name
DATASET_NAME_SHORT=u100x                 # Short dataset identifier (keep it short!)
PIPELINE_NAME_SHORT=lsim                 # Short pipeline identifier (keep it short!)

# Note: Keep DATASET_NAME_SHORT and PIPELINE_NAME_SHORT concise 
# to prevent Spark from auto-renaming with generic "spark-xxx" names
```

## 3. Run Spark Job

```bash
./bin/slamactl.sh submit-job-k8s ./examples/spark/spark-test-lgb-on-prep-dataset.py
```

## 4. Monitor Pods

Watch pods in slama namespace:
```bash
kubectl get pods -n slama --sort-by=.metadata.creationTimestamp
```

## 5. Access Grafana Dashboard

### Option 1: Quick Access Script
Use the provided script to open Grafana directly from pod name:
```bash
# Open dashboard using full pod name
./scripts/open_grafana.py lsim-u100x-241225-1558-e6i005msf16g4-5f-9cec2d93fe8ab116-exec-1

# Or just use the prefix part
./scripts/open_grafana.py lsim-u100x-241225-1558-e6i005msf16g4-5f
```

The script will automatically:
- Extract the correct pattern from pod name
- Convert hyphens to underscores for Grafana
- Open the dashboard in your default browser

### Option 2: Manual Access

1. Open Grafana dashboard:
```
http://node11.bdcl:30833/d/de7qag4j0qsqoe/tagged-spark-jvm-profile
```

2. Use the pod name prefix as metaId filter (replace hyphens with underscores for Grafana):
```
# Example pod name (in Kubernetes):
lsim-u100x-241225-1558-e6i005msf16g4-5f-[hash]-driver

# Use this part as metaId (with underscores for Grafana):
lsim_u100x_241225_1558_e6i005msf16g4_5f
```

3. Direct link template (just replace the last part with your metaId):
```
http://node11.bdcl:30833/d/de7qag4j0qsqoe/tagged-spark-jvm-profile?orgId=1&from=now-24h&to=now&timezone=browser&var-appId=$__all&var-executorId=$__all&refresh=5s&var-metaId=lsim_u100x_241225_1558_e6i005msf16g4_5f
```

> **Note**: 
> - If metaId doesn't appear in dropdown, refresh the page
> - Remember to convert hyphens (-) to underscores (_) when using the pod name in Grafana
