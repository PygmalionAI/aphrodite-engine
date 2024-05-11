#!/bin/bash

python_executable=python$1
cuda_home=/usr/local/cuda-$2

# Update paths
PATH=${cuda_home}/bin:$PATH
LD_LIBRARY_PATH=${cuda_home}/lib64:$LD_LIBRARY_PATH

# Limit Ninja's number of processes to 1 to prevent OOM
# TODO: Can we do 2 jobs?
export MAX_JOBS=1

export TORCH_CUDA_ARCH_LIST="6.0 6.1 7.0 7.5 8.0 8.6 8.9 9.0+PTX"
# Install requirements
$python_executable -m pip install wheel packaging
$python_executable -m pip install -r requirements-cuda.txt

# Build
$python_executable setup.py bdist_wheel --dist-dir=dist