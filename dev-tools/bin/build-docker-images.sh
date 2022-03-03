#!/usr/bin/env bash

set -e

# shellcheck disable=SC2094
poetry export -f requirements.txt > requirements.txt
poetry build

docker build \
  -t spark-lama:3.9-3.2.0-1 \
  -f dev-tools/docker/spark-lama.dockerfile \
  .

docker build -t spark-lama-k8s:3.9-3.2.0-1 -f dev-tools/docker/spark-lama-k8s.dockerfile .

rm -rf dist
