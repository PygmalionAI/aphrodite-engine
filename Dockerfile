# The Aphrodite Dockerfile is used to construct Aphrodite image that can be directly used
# to run the OpenAI compatible server.

#################### BASE BUILD IMAGE ####################
# prepare basic build environment
FROM nvidia/cuda:12.4.1-devel-ubuntu22.04 AS dev

RUN apt-get update -y \
    && apt-get install -y python3-pip git

# Workaround for https://github.com/openai/triton/issues/2507 and
# https://github.com/pytorch/pytorch/issues/107960 -- hopefully
# this won't be needed for future versions of this docker image
# or future versions of triton.
RUN ldconfig /usr/local/cuda-12.4/compat/

WORKDIR /workspace

# install build and runtime dependencies
COPY requirements-common.txt requirements-common.txt
COPY requirements-cuda.txt requirements-cuda.txt
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install -r requirements-cuda.txt

# install development dependencies
COPY requirements-dev.txt requirements-dev.txt
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install -r requirements-dev.txt

# cuda arch list used by torch
# can be useful for both `dev` and `test`
# explicitly set the list to avoid issues with torch 2.2
# see https://github.com/pytorch/pytorch/pull/123243
ARG torch_cuda_arch_list='6.0 6.1 7.0 7.5 8.0 8.6 8.9 9.0+PTX'
ENV TORCH_CUDA_ARCH_LIST=${torch_cuda_arch_list}
#################### BASE BUILD IMAGE ####################


#################### WHEEL BUILD IMAGE ####################
FROM dev AS build

# install build dependencies
COPY requirements-build.txt requirements-build.txt
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install -r requirements-build.txt

# install compiler cache to speed up compilation leveraging local or remote caching
RUN apt-get update -y && apt-get install -y ccache

# files and directories related to build wheels
COPY kernels kernels
COPY setup.py setup.py
COPY cmake cmake
COPY CMakeLists.txt CMakeLists.txt
COPY requirements-common.txt requirements-common.txt
COPY requirements-cuda.txt requirements-cuda.txt
COPY pyproject.toml pyproject.toml
COPY aphrodite aphrodite

# max jobs used by Ninja to build extensions
ARG max_jobs=2
ENV MAX_JOBS=${max_jobs}
# number of threads used by nvcc
ARG nvcc_threads=8
ENV NVCC_THREADS=${nvcc_threads}
# make sure punica kernels are built (for LoRA)
ENV APHRODITE_INSTALL_PUNICA_KERNELS=1
# make sure quant kernels are built
ENV APHRODITE_INSTALL_QUANT_KERNELS=1
# make sure hadamard kernels are build
ENV APHRODITE_INSTALL_HADAMARD_KERNELS=1

ENV CCACHE_DIR=/root/.cache/ccache
RUN --mount=type=cache,target=/root/.cache/ccache \
    --mount=type=cache,target=/root/.cache/pip \
    python3 setup.py bdist_wheel --dist-dir=dist

#################### EXTENSION Build IMAGE ####################

#################### Aphrodite installation IMAGE ####################
# image with Aphrodite installed
FROM nvidia/cuda:12.4.1-base-ubuntu22.04 AS aphrodite-base
WORKDIR /aphrodite-workspace

RUN apt-get update -y \
    && apt-get install -y python3-pip git vim

# Workaround for https://github.com/openai/triton/issues/2507 and
# https://github.com/pytorch/pytorch/issues/107960 -- hopefully
# this won't be needed for future versions of this docker image
# or future versions of triton.
RUN ldconfig /usr/local/cuda-12.4/compat/

# install vllm wheel first, so that torch etc will be installed
RUN --mount=type=bind,from=build,src=/workspace/dist,target=/aphrodite-workspace/dist \
    --mount=type=cache,target=/root/.cache/pip \
    pip install dist/*.whl --verbose
#################### Aphrodite installation IMAGE ####################


#################### OPENAI API SERVER ####################
# openai api server alternative
FROM aphrodite-base AS aphrodite-openai

# install additional dependencies for openai api server
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install accelerate hf_transfer modelscope

ENTRYPOINT ["python3", "-m", "aphrodite.endpoints.openai.api_server"]
#################### OPENAI API SERVER ####################