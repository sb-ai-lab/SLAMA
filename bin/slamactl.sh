#!/usr/bin/env bash

set -e

export SPARK_VERSION=${SPARK_VERSION:-3.5.4}
export HADOOP_VERSION=3
REPORT_TO_INFLUX=${REPORT_TO_INFLUX:-false}
SYNAPSEML_VERSION=${SYNAPSEML_VERSION:-1.0.8}
SLAMA_VERSION=${SLAMA_VERSION:-0.5.0}
LIGHTGBM_VERSION=3.3.5
BASE_IMAGE_TAG="slama-${SYNAPSEML_VERSION}-spark${SPARK_VERSION}"

if [[ -z "${KUBE_NAMESPACE}" ]]
then
  KUBE_NAMESPACE=default
fi

if [[ -z "${IMAGE_TAG}" ]]
then
  IMAGE_TAG=${BASE_IMAGE_TAG}
fi


if [[ -z "${REPO}" ]]
then
  echo "REPO var is not defined!"
  REPO=""
  IMAGE=spark-py-lama:${IMAGE_TAG}
  BASE_SPARK_IMAGE=spark-py:${BASE_IMAGE_TAG}
else
  IMAGE=${REPO}/spark-py-lama:${IMAGE_TAG}
  BASE_SPARK_IMAGE=${REPO}/spark-py:${BASE_IMAGE_TAG}
fi


function build_jars() {
  cur_dir=$(pwd)

  echo "Building docker image for lama-jar-builder"
  docker build -t lama-jar-builder -f docker/jar-builder/scala.dockerfile docker/jar-builder

  echo "Building jars"
  docker run -it \
    -v "${cur_dir}/scala-lightautoml-transformers:/scala-lightautoml-transformers" \
    -v "${cur_dir}/jars:/jars" \
    lama-jar-builder
}

function build_pyspark_images() {
  filename="spark-${SPARK_VERSION}-bin-hadoop${HADOOP_VERSION}"

  mkdir -p /tmp/spark-build-dir
  cd /tmp/spark-build-dir

  if [ ! -f "${filename}.tgz" ]; then
    rm -rf spark \
      && rm -rf ${filename} \
      && wget "https://archive.apache.org/dist/spark/spark-${SPARK_VERSION}/${filename}.tgz" \
      && tar -xvzf${filename}.tgz \
      && mv ${filename} spark
  fi

  # create images with names:
  # - ${REPO}/spark:${BASE_IMAGE_TAG}
  # - ${REPO}/spark-py:${BASE_IMAGE_TAG}
  # the last is equal to BASE_SPARK_IMAGE

  if [[ ! -z "${REPO}" ]]
  then
    DOCKER_DEFAULT_PLATFORM=linux/amd64 ./spark/bin/docker-image-tool.sh -r ${REPO} -t ${BASE_IMAGE_TAG} \
      -p spark/kubernetes/dockerfiles/spark/bindings/python/Dockerfile \
      build

#    ./spark/bin/docker-image-tool.sh -r ${REPO} -t ${BASE_IMAGE_TAG} push
  else
      DOCKER_DEFAULT_PLATFORM=linux/amd64 ./spark/bin/docker-image-tool.sh -t ${BASE_IMAGE_TAG} \
      -p spark/kubernetes/dockerfiles/spark/bindings/python/Dockerfile \
      build
  fi
}

function build_lama_dist() {
  # shellcheck disable=SC2094
  poetry export -f requirements.txt > requirements.txt
  poetry build
}

function build_lama_image() {
  # shellcheck disable=SC2094
  poetry export -f requirements.txt > requirements.txt
  poetry build

  DOCKER_DEFAULT_PLATFORM=linux/amd64  docker build \
    --build-arg base_image=${BASE_SPARK_IMAGE} \
    --build-arg SPARK_VER=${SPARK_VERSION} \
    --build-arg SYNAPSEML_VER=${SYNAPSEML_VERSION} \
    --build-arg SLAMA_VER=${SLAMA_VERSION} \
    --build-arg LIGHTGBM_VER=3.2.1 \
    -t ${IMAGE} \
    -f docker/spark-lama/spark-py-lama.dockerfile \
    .

  if [[ ! -z "${REPO}" ]]
  then
    docker push ${IMAGE}
  fi

  rm -rf dist
}

function push_images() {
    docker push ${BASE_SPARK_IMAGE}
    docker push ${IMAGE}
}

function build_dist() {
    build_jars
    build_pyspark_images
    build_lama_image
}

function submit_job_k8s() {
  load_env

  APISERVER=$(kubectl config view --minify -o jsonpath='{.clusters[0].cluster.server}')

  script_path=$1

  filename=$(echo ${script_path} | python -c 'import os; path = input(); print(os.path.splitext(os.path.basename(path))[0]);')

  shift 1

  local TIMESTAMP=$(date +%y%m%d_%H%M)

  # Format the memory storage fraction (remove dot and pad with zeros)
  local MSF=$(echo ${SPARK_MEMORY_STORAGE_FRACTION} | sed 's/0\.//' | awk '{printf "%03d", $1}')

  # Format memory overhead factor to show as X_Y (e.g., 3.81 -> 3_8)
  local MOF=$(echo ${SPARK_MEMORY_OVERHEAD_FACTOR} | awk -F '.' '{printf "%d_%d", $1, substr($2, 1, 1)}')

  # Remove 'g' from executor memory
  local MEM=$(echo ${SPARK_EXECUTOR_MEMORY} | sed 's/g$//')

  # Create full and short versions of CONFIG_SUFFIX
  local FULL_SUFFIX="${PIPELINE_NAME_SHORT}_${DATASET_NAME_SHORT}_${TIMESTAMP}_e${SPARK_EXECUTOR_INSTANCES}i${MSF}msf${MEM}g${MOF}f"
  local SHORT_SUFFIX="${PIPELINE_NAME_SHORT}_${DATASET_NAME_SHORT}_${TIMESTAMP}"

  # Use short version if full version is too long, this is a system limitation
  if [ ${#FULL_SUFFIX} -ge 63 ]; then
    local CONFIG_SUFFIX="${SHORT_SUFFIX}"
  else
    local CONFIG_SUFFIX="${FULL_SUFFIX}"
  fi

  if [[ "${REPORT_TO_INFLUX}" = true ]]; then
    extra_java_options="
      -javaagent:\"/root/jars/jvm-profiler-1.0.0.jar=
        reporter=com.uber.profiling.reporters.InfluxDBOutputReporter,
        metricInterval=2500,
        sampleInterval=2500,
        ioProfiling=false,
        influxdb.host=${INFLUXDB_HOST},
        influxdb.port=${INFLUXDB_PORT},
        influxdb.database=${INFLUXDB_DATABASE},
        influxdb.username=${INFLUXDB_USERNAME},
        influxdb.password=${INFLUXDB_PASSWORD},
        metaId.override=${CONFIG_SUFFIX}\"
    "
    extra_java_options=$(echo "${extra_java_options}" | tr -d '\n' | tr -d ' ')
    extra_java_options="-Dlog4j.logger.com.uber.profiling=DEBUG ${extra_java_options}"
    jars="jars/spark-lightautoml_2.12-0.1.1.jar,jars/jvm-profiler-1.0.0.jar"
  else
    jars="jars/spark-lightautoml_2.12-0.1.1.jar"
  fi

  extra_java_options="-Dlog4j.configuration=log4j2.properties ${extra_java_options}"

  # TODO: spark.sql.warehouse.dir should be customizable
  spark-submit \
    --master k8s://${APISERVER} \
    --deploy-mode cluster \
    --conf "spark.kryoserializer.buffer.max=512m" \
    --conf "spark.log.level=INFO" \
    --conf "spark.app.name=${CONFIG_SUFFIX}" \
    --conf "spark.scheduler.minRegisteredResourcesRatio=1.0" \
    --conf "spark.scheduler.maxRegisteredResourcesWaitingTime=180s" \
    --conf "spark.task.maxFailures=1" \
    --conf "spark.driver.cores=${SPARK_DRIVER_CORES:-2}" \
    --conf "spark.driver.memory=${SPARK_DRIVER_MEMORY:-8g}" \
    --conf "spark.driver.extraJavaOptions=${extra_java_options}" \
    --conf "spark.driver.extraClassPath=/root/.ivy2/jars/*" \
    --conf "spark.executor.extraJavaOptions=${extra_java_options}" \
    --conf "spark.executor.extraClassPath=/root/.ivy2/jars/*" \
    --conf "spark.executor.instances=${SPARK_EXECUTOR_INSTANCES:-1}" \
    --conf "spark.executor.cores=${SPARK_EXECUTOR_CORES:-4}" \
    --conf "spark.executor.memory=${SPARK_EXECUTOR_MEMORY:-16g}" \
    --conf "spark.cores.max=${SPARK_CORES_MAX:-4}" \
    --conf "spark.memory.fraction=${SPARK_MEMORY_FRACTION:-0.6}" \
    --conf "spark.memory.storageFraction=${SPARK_MEMORY_STORAGE_FRACTION:-0.6}" \
    --conf "spark.sql.autoBroadcastJoinThreshold=100MB" \
    --conf "spark.sql.execution.arrow.pyspark.enabled=true" \
    --conf "spark.sql.warehouse.dir=${SPARK_WAREHOUSE_DIR:-/tmp/spark-warehouse}" \
    --conf "spark.kubernetes.container.image=${IMAGE}" \
    --conf "spark.kubernetes.namespace=${KUBE_NAMESPACE}" \
    --conf "spark.kubernetes.authenticate.driver.serviceAccountName=${SPARK_K8S_SERVICE_ACCOUNT:-spark}" \
    --conf "spark.kubernetes.memoryOverheadFactor=${SPARK_MEMORY_OVERHEAD_FACTOR:-0.4}" \
    --conf "spark.kubernetes.driver.label.appname=${CONFIG_SUFFIX}" \
    --conf "spark.kubernetes.executor.label.appname=${CONFIG_SUFFIX}" \
    --conf "spark.kubernetes.executor.deleteOnTermination=false" \
    --conf "spark.kubernetes.container.image.pullPolicy=Always" \
    --conf "spark.kubernetes.driverEnv.SCRIPT_ENV=cluster" \
    --conf "spark.kubernetes.driverEnv.BASE_HDFS_PREFIX=${BASE_HDFS_PREFIX:-''}" \
    --conf "spark.kubernetes.file.upload.path=${SPARK_K8S_FILE_UPLOAD_PATH:-/tmp/spark-upload-dir}" \
    --jars "${jars}" \
    --files "examples/spark/log4j2.properties" \
    --py-files "examples/spark/examples_utils.py" \
    ${script_path} "${@}"
}

function submit_job_yarn() {
  py_files=$1

  script_path=$2

  filename=$(echo ${script_path} | python -c 'import os; path = input(); print(os.path.splitext(os.path.basename(path))[0]);')

  spark-submit \
    --master yarn \
    --deploy-mode cluster \
    --conf 'spark.jars.packages=com.microsoft.azure:synapseml_2.12:0.9.5' \
    --conf 'spark.jars.repositories=https://mmlspark.azureedge.net/maven' \
    --conf 'spark.yarn.appMasterEnv.SCRIPT_ENV=cluster' \
    --conf 'spark.kryoserializer.buffer.max=512m' \
    --conf 'spark.driver.cores=4' \
    --conf 'spark.driver.memory=5g' \
    --conf 'spark.executor.instances=8' \
    --conf 'spark.executor.cores=8' \
    --conf 'spark.executor.memory=5g' \
    --conf 'spark.cores.max=8' \
    --conf 'spark.memory.fraction=0.6' \
    --conf 'spark.memory.storageFraction=0.5' \
    --conf 'spark.sql.autoBroadcastJoinThreshold=100MB' \
    --conf 'spark.sql.execution.arrow.pyspark.enabled=true' \
    --conf 'spark.scheduler.minRegisteredResourcesRatio=1.0' \
    --conf 'spark.scheduler.maxRegisteredResourcesWaitingTime=180s' \
    --jars jars/spark-lightautoml_2.12-0.1.jar \
    --py-files ${py_files} ${script_path}
}

function submit_job_spark() {
  if [[ -z "${SPARK_MASTER_URL}" ]]
  then
    SPARK_MASTER_URL="spark://node21.bdcl:7077"
  fi

  if [[ -z "${HADOOP_DEFAULT_FS}" ]]
  then
    HADOOP_DEFAULT_FS="hdfs://node21.bdcl:9000"
  fi

  script_path=$1

  filename=$(echo ${script_path} | python -c 'import os; path = input(); print(os.path.splitext(os.path.basename(path))[0]);')

spark-submit \
  --master ${SPARK_MASTER_URL} \
  --conf 'spark.hadoop.fs.defaultFS='${HADOOP_DEFAULT_FS} \
  --conf 'spark.jars.packages=com.microsoft.azure:synapseml_2.12:0.9.5' \
  --conf 'spark.jars.repositories=https://mmlspark.azureedge.net/maven' \
  --conf 'spark.yarn.appMasterEnv.SCRIPT_ENV=cluster' \
  --conf 'spark.kryoserializer.buffer.max=512m' \
  --conf 'spark.driver.cores=4' \
  --conf 'spark.driver.memory=5g' \
  --conf 'spark.executor.instances=8' \
  --conf 'spark.executor.cores=8' \
  --conf 'spark.executor.memory=5g' \
  --conf 'spark.cores.max=8' \
  --conf 'spark.memory.fraction=0.6' \
  --conf 'spark.memory.storageFraction=0.5' \
  --conf 'spark.sql.autoBroadcastJoinThreshold=100MB' \
  --conf 'spark.sql.execution.arrow.pyspark.enabled=true' \
  --conf 'spark.scheduler.minRegisteredResourcesRatio=1.0' \
  --conf 'spark.scheduler.maxRegisteredResourcesWaitingTime=180s' \
  --jars jars/spark-lightautoml_2.12-0.1.jar \
  --py-files dist/LightAutoML-0.3.0.tar.gz ${script_path}
}

function help() {
  echo "
  Required env variables:
    KUBE_NAMESPACE - a kubernetes namespace to make actions in
    REPO - a private docker repository to push images to. It should be accessible by the cluster.

  List of commands.
    build-jars - Builds scala-based components of Slama and creates appropriate jar files in jar folder of the project
    build-pyspark-images - Builds and pushes base pyspark images required to start pyspark on cluster.
      Pushing requires remote docker repo address accessible from the cluster.
    build-lama-dist - Builds SLAMA .wheel
    build-lama-image - Builds and pushes a docker image to be used for running lama remotely on the cluster.
    build-dist - build_jars, build_pyspark_images, build_lama_image in a sequence
    submit-job - Submit a pyspark application with script that represent SLAMA automl app.
    submit-job-yarn - Submit a pyspark application to YARN cluster to execution.
    port-forward - Forwards port 4040 of the driver to 9040 port
    help - prints this message

  Examples:
  1. Start job
     KUBE_NAMESPACE=spark-lama-exps REPO=node2.bdcl:5000 ./bin/slamactl.sh submit-job ./examples/spark/tabular-preset-automl.py
  2. Forward Spark WebUI on local port
     KUBE_NAMESPACE=spark-lama-exps REPO=node2.bdcl:5000 ./bin/slamactl.sh port-forward ./examples/spark/tabular-preset-automl.py
  "
}


function main () {
    cmd="$1"

    if [ -z "${cmd}" ]
    then
      echo "No command is provided."
      help
      exit 1
    fi

    shift 1

    echo "Executing command: ${cmd}"

    case "${cmd}" in
    "build-jars")
        build_jars
        ;;

    "build-pyspark-images")
        build_pyspark_images
        ;;

    "build-lama-dist")
        build_lama_dist
        ;;

    "build-lama-image")
        build_lama_image
        ;;

    "build-dist")
        build_dist
        ;;

    "push-images")
        push_images
        ;;

    "submit-job-yarn")
        submit_job_yarn "${@}"
        ;;

    "submit-job-spark")
        submit_job_spark "${@}"
        ;;

    "submit-job-k8s")
        submit_job_k8s "${@}"
        ;;

    "help")
        help
        ;;

    *)
        echo "Unknown command: ${cmd}"
        ;;

    esac
}

load_env() {
  if [ -f ".env" ]; then
    set -a
    source ".env"
    set +a
  elif [ -f "../.env" ]; then
    set -a
    source "../.env"
    set +a
  else
    echo "Warning: .env file not found in current directory or project root directory"
    exit 1
  fi
}

main "${@}"
