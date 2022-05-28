#!/bin/bash

set -ex

script=$1

EXEC_CORES="${EXEC_CORES:-6}"
EXEC_INST="${EXEC_INST:-4}"
EXEC_MEM="${EXEC_MEM:-128g}"
CORES_MAX=$((EXEC_CORES * EXEC_INST))

export SCRIPT_ENV=cluster
spark-submit \
  --master spark://node3.bdcl:7077 \
  --deploy-mode client \
  --conf 'spark.driver.host=node3.bdcl' \
  --conf 'spark.jars.packages=com.microsoft.azure:synapseml_2.12:0.9.5' \
  --conf 'spark.jars.repositories=https://mmlspark.azureedge.net/maven' \
  --conf 'spark.kryoserializer.buffer.max=512m' \
  --conf 'spark.driver.cores=10' \
  --conf 'spark.driver.memory=20g' \
  --conf "spark.executor.instances=${EXEC_INST}" \
  --conf "spark.executor.cores=${EXEC_CORES}" \
  --conf "spark.executor.memory=${EXEC_MEM}" \
  --conf "spark.cores.max=${CORES_MAX}" \
  --conf 'spark.memory.fraction=0.6' \
  --conf 'spark.memory.storageFraction=0.5' \
  --conf 'spark.sql.autoBroadcastJoinThreshold=100MB' \
  --conf 'spark.sql.execution.arrow.pyspark.enabled=true' \
  --conf 'spark.cleaner.referenceTracking.cleanCheckpoints=true' \
  --conf 'spark.cleaner.referenceTracking=true' \
  --conf 'spark.cleaner.periodicGC.interval=1min' \
  --conf 'spark.scheduler.minRegisteredResourcesRatio=1.0' \
  --conf 'spark.scheduler.maxRegisteredResourcesWaitingTime=180s' \
  --jars /opt/spark-lightautoml_2.12-0.1.jar \
  --py-files /opt/LightAutoML-0.3.0.tar.gz \
  ${script}
