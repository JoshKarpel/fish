#!/usr/bin/env bash

set -e

TAG=$1

docker build --pull -t "${TAG}" -f docker/Dockerfile .
docker push "${TAG}"
