#!/usr/bin/env bash

set -e

docker_repo=node2.bdcl:5000

docker tag spark-pyspark-python:3.9-3.2.0 ${docker_repo}/spark-pyspark-python:3.9-3.2.0
docker tag spark-lama:3.9-3.2.0 ${docker_repo}/spark-lama:3.9-3.2.0

docker push ${docker_repo}/spark-pyspark-python:3.9-3.2.0
docker push ${docker_repo}/spark-lama:3.9-3.2.0