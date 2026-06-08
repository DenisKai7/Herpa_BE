#!/usr/bin/env sh
set -eu

export DOCKER_BUILDKIT=1
export COMPOSE_DOCKER_CLI_BUILD=1

docker build \
  -f Dockerfile.backend-base \
  -t medical-backend-base:py311-v1 .

docker build \
  -f Dockerfile.ocr-base \
  -t medical-ocr-base:torch-cpu-v1 .
