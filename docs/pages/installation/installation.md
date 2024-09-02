
# Installation with NVIDIA

The installation process for Aphrodite is different depending on your hardware. The primary hardware for Aphrodite is NVIDIA GPUs, with x86_64 CPU architecture; we provide pre-built binaries for that combination. If you're using other types of hardware, you may need to build from source. We have provided Docker files to ease the process. You can also find detailed instructions for building manually if you cannot use Docker.


## Requirements

- Linux (or WSL on Windows)
- Python 3.8 - 3.11
- NVIDIA GPU (compute capability 6.0 or higher)

To find out what compute capability your GPU has, you can run this in a python interpreter:

```py
>>> import torch
>>> torch.cuda.get_device_capability()
(8, 6)
```

The (8, 6) indicates the GPU has a compute capability of 8.6, for example.


## Installation with pip

You can install Aphrodite using pip:

```sh
# Create a new environment (optional)
python -m venv ./aphrodite-venv --prompt "aphrodite"
# You will need to run this every time you close the terminal
source ./aphrodite-venv/bin/activate
# Install Aphrodite with CUDA 12.1
pip install aphrodite-engine
```

:::warning
Since our binaries are compiled with CUDA 12.1, you may need to build from source for different CUDA versions. Please see below for instructions.
:::

## Building from source

You can build Aphrodite from source if needed.

```sh
git clone https://github.com/PygmalionAI/aphrodite-engine.git
cd aphrodite-engine

python -m venv ./venv --prompt "aphrodite"
source ./venv/bin/activate  # do this every time

pip install -e .  # this may take a while
```

If you don't have enough RAM, you may need to run `export MAX_JOBS=n`, where `n` is the number of jobs.

You can also use the embedded micromamba runtime for one-command installation of Aphrodite:

```sh
./update-runtime.sh
```

Afterwards, prefix every Aphrodite-related command with `./runtime.sh`. e.g.:
```sh
./runtime.sh aphrodite run -h
```

## Installation with Docker
We provide both a pre-built docker image, and a Dockerfile.

### Using the pre-built Docker image

```sh
docker run --runtime nvidia --gpus all \
    -v ~/.cache/huggingface:/root/.cache/huggingface \
    #--env "CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7" \
    -p 2242:2242 \
    --ipc=host \
    alpindale/aphrodite-openai:latest \
    --model NousResearch/Meta-Llama-3.1-8B-Instruct \
    --tensor-parallel-size 8 \
    --api-keys "sk-empty"
```

### Building via Docker image

```sh
DOCKER_BUILDKIT=1 docker build . \
    --target aphrodite-openai \
    --tag alpindale/aphrodite-openai
    # optionally: \
    --build-arg max_jobs=8 \
    --build-arg nvcc_threads 2
```

This Dockerfile will build for all CUDA arches, which may take hours. You can limit the arch to your GPU by adding this flag:

```sh
--build-arg torch_cuda_arch_list='8.9'  # Ada lovelace, e.g. RTX 4090
```

You can run your built image using the command in the previous section.



