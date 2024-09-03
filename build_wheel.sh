#!/bin/bash

CUDA_VERSION=${CUDA_VERSION:-12.4.1}
MAX_JOBS=${MAX_JOBS:-}
NVCC_THREADS=${NVCC_THREADS:-}
TORCH_CUDA_ARCH_LIST="6.0 6.1 7.0 7.5 8.0 8.6 8.9 9.0+PTX"

DOCKER_BUILDKIT=1 docker build . --target build --tag alpindale/aphrodite-build \
    --build-arg CUDA_VERSION=$CUDA_VERSION \
    ${MAX_JOBS:+--build-arg max_jobs=$MAX_JOBS} \
    ${NVCC_THREADS:+--build-arg nvcc_threads=$NVCC_THREADS}

docker run -d --name aphrodite-build-container alpindale/aphrodite-build tail -f /dev/null
# copies to dist/ within working directory
docker cp aphrodite-build-container:/workspace/dist .
docker stop aphrodite-build-container && docker rm aphrodite-build-container