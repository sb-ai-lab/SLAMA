#!/usr/bin/env bash

set -e

cur_dir=$(pwd)

echo "Cur dir: ${cur_dir}"

docker run -it \
        -v  ${cur_dir}:/src \
        -v /opt:/spark_data \
        spark-lama:3.9-3.2.0 \
        python /src/dev-tools/performance_tests/launch.py
