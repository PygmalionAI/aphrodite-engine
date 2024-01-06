#!/bin/bash

wget -qO- https://micromamba.snakepit.net/api/micromamba/linux-64/latest | tar -xvj bin/micromamba
if [ ! -f "conda/envs/aphrodite-runtime/bin/python" ]; then
 bin/micromamba create --no-shortcuts -r conda -n aphrodite-runtime -f environment.yaml -y
fi
bin/micromamba create --no-shortcuts -r conda -n aphrodite-runtime -f environment.yaml -y
bin/micromamba install -r conda -n aphrodite-runtime gxx=10 -c conda-forge -y
bin/micromamba run -r conda -n aphrodite-runtime pip install packaging
bin/micromamba run -r conda -n aphrodite-runtime pip install -r requirements.txt
# Make it so the correct NVCC is found. Looks only within the current working
# directory, since find will return the *first* result, leading to conflicts
# if you have multiple environments and one of them does not contain CUDA runtime.
export CUDA_HOME=$(find / -type d -path "*/conda/envs/aphrodite-runtime" 2>/dev/null | head -n 1)
bin/micromamba run -r conda -n aphrodite-runtime pip install -vvv -e .

