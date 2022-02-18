#!/usr/bin/env bash

set -e

cp -r "${HOME}/.ivy2/cache" jars_cache
docker build -t spark-pyspark-python:3.9-3.2.0 -f dev-tools/docker/spark-pyspark-python.dockerfile .
rm -rf jars_cache
