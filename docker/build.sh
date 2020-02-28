#!/usr/bin/env bash

set -e

USERNAME=maventree
IMAGE=fish

docker build --pull --build-arg CACHEBUST=$(date +%s) -t ${USERNAME}/${IMAGE}:$1 .
docker push ${USERNAME}/${IMAGE}:$1
