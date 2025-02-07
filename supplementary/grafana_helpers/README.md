# Scripts Documentation

## process_driver_pods.py

A utility script for managing Kubernetes pods in the SLAMA project, specifically designed to handle Spark driver and executor pods.

### Purpose

This script helps with collecting logs from Spark driver and executor pods before optionally deleting them. It's particularly useful for:
- Gathering logs from completed or failed Spark jobs
- Managing pod cleanup
- Preserving execution history

### Features

- Collects logs from both driver and executor pods
- Intelligently matches executor pods to their corresponding driver
- Creates timestamped log files
- Optional pod deletion with `--delete-pods` flag
- Skips pods that are already in Terminating state
- Works with pods in a specified Kubernetes namespace (default: 'slama')

### Usage

Basic usage (collect logs only):
```bash
python process_driver_pods.py
```

Collect logs and delete pods:
```bash
python process_driver_pods.py --delete-pods
```

### Configuration

#### Kubernetes Namespace
The script is configured to work with the 'slama' namespace by default. To change the namespace:

1. Open `process_driver_pods.py`
2. Find all occurrences of `-n slama` in the commands (there are three places):
   ```python
   # In save_pod_logs function
   logs = subprocess.run(['kubectl', 'logs', '-n', 'slama', pod_name], ...)
   
   # In get_executor_pods function
   cmd = ['kubectl', 'get', 'pods', '-n', 'slama', '--no-headers']
   
   # In process_driver_pods function
   cmd = ['kubectl', 'get', 'pods', '-n', 'slama', '--no-headers']
   ```
3. Replace 'slama' with your desired namespace

### Log File Format

Logs are saved in the `logs` directory with the following naming convention:
```
[pod-name]_[YYYYMMDD_HHMMSS].log
```

Example:
```
lgbm-u100x-241224-0611-e6i005msf100g04f-da7fc593f74a6d69-driver_20241224_123456.log
lgbm-u100x-241224-0611-e6i005msf100g04f-f2677b93f74b3077-exec-1_20241224_123456.log
```

### Pod Matching Logic

The script matches executor pods to their driver pod by:
1. Extracting the base name up to the timestamp from the driver pod name
2. Finding all executor pods that share this base name
3. Excluding any pods in Terminating state

Example matching:
```
Driver:   lgbm-u100x-241224-0611-e6i005msf100g04f-[hash1]-driver
Executor: lgbm-u100x-241224-0611-e6i005msf100g04f-[hash2]-exec-1
```

### Requirements

- Python 3.x
- Access to Kubernetes cluster with `kubectl` configured
- Appropriate permissions to read pod logs and optionally delete pods in the target namespace
