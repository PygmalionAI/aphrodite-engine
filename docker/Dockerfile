FROM nvidia/cuda:12.1.1-devel-ubuntu22.04

WORKDIR /workspace/aphrodite-engine

# Upgrade OS Packages
RUN set -eux; \
    apt-get update \
    && apt-get upgrade -y \
    && rm -rf /var/lib/apt/lists/*

# Preparing Conda Environment
RUN apt-get update \
    && apt-get install -y git build-essential \
    && apt-get install python3 python3-pip -y \
    && rm -fr /var/lib/apt/lists/*

ENV PATH /opt/conda/envs/aphrodite-engine/bin:$PATH

# alias python3 to python
RUN ln -s /usr/bin/python3 /usr/bin/python

RUN python3 -m pip install --upgrade pip

RUN git clone https://github.com/PygmalionAI/aphrodite-engine.git /tmp/aphrodite-engine
RUN mv /tmp/aphrodite-engine/* /workspace/aphrodite-engine/
RUN rm -rf /tmp/aphrodite-engine

# Export the CUDA_HOME variable correctly
ENV CUDA_HOME=/usr/local/cuda

RUN python3 -m pip install --no-cache-dir -e .
