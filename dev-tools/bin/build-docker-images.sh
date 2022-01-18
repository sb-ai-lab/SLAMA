#!/usr/bin/env bash

set -e

docker build -t spark-pyspark-python:3.9-3.2.0 -f dev-tools/docker/spark-pyspark-python.dockerfile .
# shellcheck disable=SC2094
poetry export -f requirements.txt > requirements.txt
docker build -t spark-lama:3.9-3.2.0 -f dev-tools/docker/spark-lama.dockerfile .
