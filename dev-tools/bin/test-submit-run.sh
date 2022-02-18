#!/usr/bin/env bash

set -e

APISERVER=$(kubectl config view --minify -o jsonpath='{.clusters[0].cluster.server}')

KUBE_NAMESPACE=spark-lama-exps
# TODO: volumes
#  --conf 'spark.kubernetes.file.upload.path=' \
# --conf "spark.jars.repositories=https://mmlspark.azureedge.net/maven"\
#  --conf 'spark.kubernetes.driver.podTemplateFile=dev-tools/config/spark-pod-driver-template.yaml' \
#  --conf 'spark.kubernetes.executor.podTemplateFile=dev-tools/config/spark-pod-exec-template.yaml' \

#  --conf 'spark.kubernetes.driver.volumes.configMap.scripts-volume.options.name=exp-test-scripts' \
#  --conf 'spark.kubernetes.driver.volumes.configMap.scripts-volume.mount.path=/scripts' \
#  --conf 'spark.kubernetes.driver.volumes.configMap.scripts-volume.mount.readOnly=true' \

remote_script_path=spark_used_cars.py
scp dev-tools/performance_tests/spark_used_cars.py \
  node2.bdcl:/mnt/ess_storage/DN_1/tmp/scripts-shared-vol/${remote_script_path}

spark-submit \
  --master k8s://${APISERVER} \
  --deploy-mode cluster \
  --conf 'spark.kubernetes.container.image=node2.bdcl:5000/spark-lama-k8s:3.9-3.2.0' \
  --conf 'spark.kubernetes.container.image.pullPolicy=Always' \
  --conf 'spark.kubernetes.driver.volumes.persistentVolumeClaim.scripts.options.claimName=scripts-shared-vol' \
  --conf 'spark.kubernetes.driver.volumes.persistentVolumeClaim.scripts.options.storageClass=local-hdd' \
  --conf 'spark.kubernetes.driver.volumes.persistentVolumeClaim.scripts.mount.path=/scripts' \
  --conf 'spark.kubernetes.driver.volumes.persistentVolumeClaim.scripts.mount.readOnly=true' \
  --conf 'spark.kubernetes.driver.volumes.persistentVolumeClaim.spark-lama-data.options.claimName=spark-lama-data' \
  --conf 'spark.kubernetes.driver.volumes.persistentVolumeClaim.spark-lama-data.options.storageClass=local-hdd' \
  --conf 'spark.kubernetes.driver.volumes.persistentVolumeClaim.spark-lama-data.mount.path=/spark_data' \
  --conf 'spark.kubernetes.driver.volumes.persistentVolumeClaim.spark-lama-data.mount.readOnly=true' \
  --conf 'spark.kubernetes.executor.volumes.persistentVolumeClaim.spark-lama-data.options.claimName=spark-lama-data' \
  --conf 'spark.kubernetes.executor.volumes.persistentVolumeClaim.spark-lama-data.options.storageClass=local-hdd' \
  --conf 'spark.kubernetes.executor.volumes.persistentVolumeClaim.spark-lama-data.mount.path=/spark_data' \
  --conf 'spark.kubernetes.executor.volumes.persistentVolumeClaim.spark-lama-data.mount.readOnly=true' \
  --conf "spark.kubernetes.namespace=${KUBE_NAMESPACE}" \
  --conf 'spark.kubernetes.authenticate.driver.serviceAccountName=spark' \
  --conf 'spark.kubernetes.memoryOverheadFactor=0.1' \
  --conf 'spark.kubernetes.driver.label.appname=driver-test-submit-run' \
  --conf 'spark.kubernetes.executor.label.appname=executor-test-submit-run' \
  --conf 'spark.kubernetes.executor.deleteOnTermination=false' \
  --conf "spark.jars.packages=com.microsoft.azure:synapseml_2.12:0.9.4"\
  --conf 'spark.driver.cores=4' \
  --conf 'spark.driver.memory=16g' \
  --conf 'spark.executor.instances=4' \
  --conf 'spark.executor.cores=4' \
  --conf 'spark.executor.memory=16g' \
  --conf 'spark.cores.max=16' \
  --conf 'spark.memory.fraction= 0.6' \
  --conf 'spark.memory.storageFraction=0.5' \
  --conf 'spark.sql.autoBroadcastJoinThreshold=100MB' \
  --conf 'spark.sql.execution.arrow.pyspark.enabled=true' \
  local:///scripts/${remote_script_path}


# spark-submit \
#   --master k8s://${APISERVER} \
#   --deploy-mode cluster \
#   --conf 'spark.kubernetes.container.image=node2.bdcl:5000/spark-lama-k8s:3.9-3.2.0' \
#   #--conf 'spark.kubernetes.file.upload.path=/mnt/ess_storage/DN_1/storage/sber_LAMA/scripts/' \
#   --conf 'spark.kubernetes.container.image.pullPolicy=Always' \
#   --conf "spark.kubernetes.namespace=${KUBE_NAMESPACE}" \
#   --conf 'spark.kubernetes.authenticate.driver.serviceAccountName=spark' \
#   --conf 'spark.kubernetes.memoryOverheadFactor=0.1' \
#   --conf 'spark.kubernetes.driver.label.appname=driver-test-submit-run' \
#   --conf 'spark.kubernetes.executor.label.appname=executor-test-submit-run' \
#   --conf 'spark.kubernetes.executor.deleteOnTermination=false' \
#   --conf 'spark.kubernetes.driver.volumes.persistentVolumeClaim.spark-lama-data.options.claimName=spark-lama-data' \
#   --conf 'spark.kubernetes.driver.volumes.persistentVolumeClaim.spark-lama-data.options.storageClass=local-hdd' \
#   --conf 'spark.kubernetes.driver.volumes.persistentVolumeClaim.spark-lama-data.mount.path=/spark_data' \
#   --conf 'spark.kubernetes.driver.volumes.persistentVolumeClaim.spark-lama-data.mount.readOnly=true' \
#   --conf 'spark.kubernetes.executor.volumes.persistentVolumeClaim.spark-lama-data.options.claimName=spark-lama-data' \
#   --conf 'spark.kubernetes.executor.volumes.persistentVolumeClaim.spark-lama-data.options.storageClass=local-hdd' \
#   --conf 'spark.kubernetes.executor.volumes.persistentVolumeClaim.spark-lama-data.mount.path=/spark_data' \
#   --conf 'spark.kubernetes.executor.volumes.persistentVolumeClaim.spark-lama-data.mount.readOnly=true' \
#   --conf 'spark.driver.cores=4' \
#   --conf 'spark.driver.memory=16g' \
#   --conf 'spark.executor.instances=4' \
#   --conf 'spark.executor.cores=4' \
#   --conf 'spark.executor.memory=16g' \
#   --conf 'spark.cores.max=16' \
#   --conf 'spark.memory.fraction= 0.6' \
#   --conf 'spark.memory.storageFraction=0.5' \
#   --conf 'spark.sql.autoBroadcastJoinThreshold=100MB' \
#   --conf 'spark.sql.execution.arrow.pyspark.enabled=true' \
#   file:///mnt/ess_storage/DN_1/storage/sber_LAMA/scripts/spark_used_cars.py