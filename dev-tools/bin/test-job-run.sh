#!/usr/bin/env bash

set -ex

job_name=$1
launch_script_name=$2
out_file_path=$3
cfg_file="/tmp/${job_name}-config.yaml"
kube_ns="spark-lama-exps"

kubectl -n ${kube_ns} delete configmap "${job_name}-scripts" --ignore-not-found
kubectl -n ${kube_ns} create configmap "${job_name}-scripts" \
  --from-file=dev-tools/performance_tests/ \
  --from-file=config.yaml="${cfg_file}"

kubectl -n ${kube_ns} delete job "${job_name}" --ignore-not-found
kubectl -n ${kube_ns} delete svc "${job_name}" --ignore-not-found
sed -e "s/{{jobname}}/${job_name}/g" -e "s/{{launchscript}}/${launch_script_name}/g" dev-tools/config/spark-job.yaml.j2 | kubectl apply -f -

echo "Waiting for spark-job to complete..."
until \
  (kubectl -n ${kube_ns} get job/${job_name} -o jsonpath='{.status.conditions[?(@.type=="Complete")].status}' | grep True) \
  || (kubectl -n ${kube_ns} get job/${job_name} -o jsonpath='{.status.conditions[?(@.type=="Failed")].status}' | grep True); \
  do sleep 1 ; done

echo "Getting logs..."
kubectl -n ${kube_ns} logs job/${job_name} &> "${out_file_path}"

res=$(kubectl -n ${kube_ns} get job/${job_name} -o jsonpath='{.status.conditions[?(@.type=="Failed")].status}' | grep True)

if [[ "$res" == "True" ]]; then
    echo "Job failed"
    exit 1
else
    echo "Job succeseded"
    exit 0
fi
