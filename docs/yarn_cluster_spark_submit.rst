Running spark lama app on Spark YARN with spark_submit
====================================

Prerequisites
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

1. Create a folder and put there the following files from the repository:
* <project_root>/examples/spark/* -> examples-spark/*
* <project_root>/sparklightautoml/automl/presets/tabular_config.yml -> tabular_config.yml

2. Install sparklightautoml in your python env on cluster ::

.. code-block::

    <python env on your cluster>/bin/pip install sparklightautoml

Launching
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
To launch example 'tabular-preset-automl.py' (the most comprehensive example) run the following command

.. code-block:: bash

    PYSPARK_PYTHON_PATH=<python env on your cluster>
    WAREHOUSE_DIR=<hdfs folder>
    DRIVER_CORES=1
    DRIVER_MEMORY="4g"
    DRIVER_MAX_RESULT_SIZE="1g"
    EXECUTOR_INSTANCES=4
    EXECUTOR_CORES=4
    EXECUTOR_MEMORY="10g"
    CORES_MAX=$(($EXECUTOR_CORES * $EXECUTOR_INSTANCES))
    # PARTITION_NUM and BUCKET_NUMS should be equal
    PARTITION_NUM=$CORES_MAX
    BUCKET_NUMS=$PARTITION_NUM
    SCRIPT="examples-spark/tabular-preset-automl.py"

    # Notes:

    # "spark.kryoserializer.buffer.max=512m"
    # is required when there are a lot of categorical variables with very high cardinality

    # "spark.sql.autoBroadcastJoinThreshold=100MB" depends on your dataset

    # if you run on jdk11
    #--conf "spark.driver.extraJavaOptions=-Dio.netty.tryReflectionSetAccessible=true" \
    #--conf "spark.executor.extraJavaOptions=-Dio.netty.tryReflectionSetAccessible=true"

    spark-submit \
    --master yarn \
    --deploy-mode cluster \
    --conf "spark.yarn.appMasterEnv.SCRIPT_ENV=cluster" \
    --conf "spark.yarn.appMasterEnv.PYSPARK_PYTHON=${PYSPARK_PYTHON_PATH}" \
    --conf "spark.yarn.appMasterEnv.PERSISTENCE_MANAGER=CompositeBucketedPersistenceManager" \
    --conf "spark.yarn.appMasterEnv.BUCKET_NUMS=${BUCKET_NUMS}" \
    --conf "spark.kryoserializer.buffer.max=512m" \
    --conf "spark.driver.cores=${DRIVER_CORES}" \
    --conf "spark.driver.memory=${DRIVER_MEMORY}" \
    --conf "spark.driver.maxResultSize=${DRIVER_MAX_RESULT_SIZE}" \
    --conf "spark.executor.instances=${EXECUTOR_INSTANCES}" \
    --conf "spark.executor.cores=${EXECUTOR_CORES}" \
    --conf "spark.executor.memory=${EXECUTOR_MEMORY}" \
    --conf "spark.cores.max=${CORES_MAX}" \
    --conf "spark.memory.fraction=0.8" \
    --conf "spark.sql.shuffle.partitions=${PARTITION_NUM}" \
    --conf "spark.default.parallelism=${PARTITION_NUM}" \
    --conf "spark.rpc.message.maxSize=1024" \
    --conf "spark.sql.autoBroadcastJoinThreshold=100MB" \
    --conf "spark.sql.execution.arrow.pyspark.enabled=true" \
    --conf "spark.driver.extraJavaOptions=-Dio.netty.tryReflectionSetAccessible=true" \
    --conf "spark.executor.extraJavaOptions=-Dio.netty.tryReflectionSetAccessible=true" \
    --conf "spark.jars.repositories=https://mmlspark.azureedge.net/maven"
    --conf "spark.jars.packages=com.microsoft.azure:synapseml_2.12:0.9.5,io.github.fonhorst:spark-lightautoml_2.12:0.1.1"
    --conf "spark.sql.warehouse.dir=${WAREHOUSE_DIR}" \
    --py-files "examples-spark/*,tabular_config.yml" \
    --num-executors "${EXECUTOR_INSTANCES}" \
    --jars "spark-lightautoml_2.12-0.1.jar" \
    "${SCRIPT}"
