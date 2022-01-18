#!/usr/bin/env bash
# This script is used for test runs on Spark Cluster

set -ex

docker pull node2.bdcl:5000/spark-lama:3.9-3.2.0

docker run -it --net=host \
        -v  /tmp/ivy2_cache:/root/.ivy2/cache \
        -v /mnt/ess_storage/DN_1/storage/sber_LAMA/kaggle_used_cars_dataset:/spark_data \
        -v /nfshome/nbutakov/used_cars.py:/code/examples/spark/simple_tabular_classification_used_cars.py \
        node2.bdcl:5000/spark-lama:base-3.9 \
        python examples/spark/simple_tabular_classification_used_cars.py
