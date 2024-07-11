FROM nvidia/cuda:12.1.1-devel-ubuntu22.04

# Adjust UID/GID when mounting volumes need to match the host UID/GID
ARG UID=1000
ARG GID=1000

ARG APHRODITE_BRANCH=main

# Setting MAX_JOBS allows build server to limit ninja build jobs. For reference
# see https://github.com/PygmalionAI/aphrodite-engine/wiki/1.-Installation#build-from-source
ARG MAX_JOBS

# Setting TORCH_CUDA_ARCH_LIST specifies the CUDA architectures to compile for. 
# a list of possible values can be found at:
# https://docs.nvidia.com/cuda/cuda-compiler-driver-nvcc/index.html#gpu-feature-list
ARG TORCH_CUDA_ARCH_LIST="6.0 6.1 7.0 7.5 8.0 8.6 8.7 8.9 9.0+PTX"

ARG FLASH_ATTN_VERSION="2.5.8"

# Upgrade OS Packages + Prepare Python Environment
RUN set -eux; \
    export DEBIAN_FRONTEND=noninteractive; \
    apt-get update && \
    apt-get upgrade -y && \
    apt-get install -y \
    bzip2 \
    g++ \
    git \
    make \
    python3-pip \
    tzdata && \
    python3 -m pip install --no-cache-dir --upgrade pip && \
    ln -s /usr/bin/python3 /usr/bin/python && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*

# Add image service account
ENV USER=aphrodite-engine
ENV APP_HOME=/app
RUN groupadd -g ${GID} ${USER} && \
    useradd -u ${UID} -g ${GID} -d /home/${USER} -m -s /bin/bash ${USER}

# Install aphrodite-engine, creates APP_HOME
RUN git clone -b ${APHRODITE_BRANCH} https://github.com/PygmalionAI/aphrodite-engine.git ${APP_HOME} && \
    chmod +x ${APP_HOME}/docker/entrypoint.sh

# Install aphrodite-engine dependencies
ENV MAX_JOBS=${MAX_JOBS}
ENV CUDA_HOME=/usr/local/cuda
ENV HF_HOME=/tmp
ENV NUMBA_CACHE_DIR=$HF_HOME/numba_cache
ENV TORCH_CUDA_ARCH_LIST=${TORCH_CUDA_ARCH_LIST}
RUN python3 -m pip install --no-cache-dir -e ${APP_HOME}


# Workaround to properly install flash-attn. This needs to be executed after installing aphrodite-engine 
# build dependencies. For reference see: https://github.com/Dao-AILab/flash-attention/issues/453
RUN python3 -m pip install "flash-attn>=${FLASH_ATTN_VERSION}" --no-build-isolation --no-cache-dir

EXPOSE 7860

VOLUME ["/tmp"]

USER ${USER}
WORKDIR ${APP_HOME}
ENTRYPOINT ["/bin/sh", "-c", "${APP_HOME}/docker/entrypoint.sh"]
