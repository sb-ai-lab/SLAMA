#!/usr/bin/env python3

import subprocess
import os
import argparse
from datetime import datetime

def parse_args():
    parser = argparse.ArgumentParser(description='Process Kubernetes driver and executor pods')
    parser.add_argument('--delete-pods', action='store_true', default=False,
                      help='Delete pods after collecting logs (default: False)')
    return parser.parse_args()

def run_kubectl_command(command, wait=False):
    try:
        if isinstance(command, str):
            cmd_list = command.split()
        else:
            cmd_list = command
            
        if wait:
            result = subprocess.run(cmd_list, check=True, capture_output=True, text=True)
            return result.stdout
        else:
            # Run command without waiting
            subprocess.Popen(cmd_list, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            return None
    except subprocess.CalledProcessError as e:
        print(f"Error executing command: {' '.join(cmd_list)}")
        print(f"Error message: {e.stderr}")
        return None

def save_pod_logs(pod_name, logs_dir, pod_type="driver"):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(logs_dir, f"{pod_name}_{timestamp}.log")
    
    print(f"Getting logs for {pod_type} pod: {pod_name}")
    try:
        logs = subprocess.run(['kubectl', 'logs', '-n', 'slama', pod_name], 
                           capture_output=True, 
                           text=True, 
                           check=True).stdout
        with open(log_file, 'w') as f:
            f.write(logs)
        print(f"Logs saved to {log_file}")
    except subprocess.CalledProcessError as e:
        print(f"Error getting logs for {pod_name}: {e.stderr}")

def get_executor_pods(driver_pod_name):
    # Extract the base name up to the timestamp
    # e.g., from 'plgb-u100x-241224-0611-e6i005msf100g04f-da7fc593f74a6d69-driver'
    # get 'plgb-u100x-241224-0611-e6i005msf100g04f'
    base_name = '-'.join(driver_pod_name.split('-')[:-2])  # Remove the hash and 'driver' parts
    
    # Get all pods with kubectl
    cmd = ['kubectl', 'get', 'pods', '-n', 'slama', '--no-headers']
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        pods = result.stdout.splitlines()
        
        # Find all executor pods for this driver
        executor_pods = []
        for pod in pods:
            if not pod.strip():
                continue
            
            fields = pod.split()
            pod_name = fields[0]
            pod_status = fields[2]  # STATUS field
            
            # Check if pod name starts with the base name and is an executor
            if pod_name.startswith(base_name) and '-exec-' in pod_name and pod_status != 'Terminating':
                executor_pods.append(pod_name)
                
        return executor_pods
    except subprocess.CalledProcessError as e:
        print(f"Error getting pod list: {e.stderr}")
        return []

def process_driver_pods(delete_pods=False):
    # Create logs directory if it doesn't exist
    logs_dir = "logs"
    os.makedirs(logs_dir, exist_ok=True)

    # Get list of all pods
    cmd = ['kubectl', 'get', 'pods', '-n', 'slama', '--no-headers']
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        pods = result.stdout.splitlines()
    except subprocess.CalledProcessError as e:
        print(f"Failed to get pod list: {e.stderr}")
        return

    # Process each line
    for pod_line in pods:
        if not pod_line.strip():
            continue
            
        # Split the line into columns
        fields = pod_line.split()
        if not fields:
            continue

        pod_name = fields[0]
        pod_status = fields[2]  # STATUS field
        
        # Check if it's a driver pod and not in Terminating state
        if '-driver' in pod_name and pod_status != 'Terminating':
            print(f"\nProcessing driver pod: {pod_name}")
            
            # Save driver logs
            save_pod_logs(pod_name, logs_dir, "driver")
            
            # Get and save executor logs
            executor_pods = get_executor_pods(pod_name)
            if executor_pods:
                print(f"Found {len(executor_pods)} executor pods")
                for exec_pod in executor_pods:
                    save_pod_logs(exec_pod, logs_dir, "executor")
            else:
                print("No active executor pods found")
            
            # Delete pod if requested
            if delete_pods:
                print(f"Deleting pod {pod_name}")
                delete_command = ['kubectl', 'delete', 'pod', '-n', 'slama', pod_name]
                run_kubectl_command(delete_command, wait=False)
            else:
                print(f"Skipping deletion of pod {pod_name} (--delete-pods not specified)")

if __name__ == "__main__":
    args = parse_args()
    process_driver_pods(delete_pods=args.delete_pods)
