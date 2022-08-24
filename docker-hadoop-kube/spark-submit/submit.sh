#!/usr/bin/env bash

set -ex

export SPARK_HOME=/usr/local/lib/python3.8/site-packages/pyspark
export PYSPARK_PYTHON=/usr/bin/python3

spark-submit \
  --master yarn \
  --deploy-mode cluster \
  --conf 'spark.kryoserializer.buffer.max=512m' \
  --conf 'spark.driver.cores=1' \
  --conf 'spark.driver.memory=3g' \
  --conf 'spark.executor.instances=3' \
  --conf 'spark.executor.cores=4' \
  --conf 'spark.executor.memory=2g' \
  --conf 'spark.cores.max=12' \
  --conf 'spark.sql.autoBroadcastJoinThreshold=100MB' \
  --conf 'spark.sql.execution.arrow.pyspark.enabled=true' \
  --conf 'spark.scheduler.minRegisteredResourcesRatio=1.0' \
  --conf 'spark.scheduler.maxRegisteredResourcesWaitingTime=180s' \
  submit_example.py
