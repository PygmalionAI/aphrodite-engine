FROM nvidia/cuda:12.1.1-devel-ubuntu22.04

ENV HOME=/app/aphrodite-engine

WORKDIR $HOME

# Upgrade OS Packages + Prepare Python Environment
RUN set -eux; \
    export DEBIAN_FRONTEND=noninteractive \
    && apt-get update \
    && apt-get upgrade -y \
    && apt-get install -y bzip2 g++ git make python3-pip tzdata \
    && rm -fr /var/lib/apt/lists/*

# Alias python3 to python
RUN ln -s /usr/bin/python3 /usr/bin/python

RUN python3 -m pip install --no-cache-dir --upgrade pip

RUN git clone https://github.com/PygmalionAI/aphrodite-engine.git /tmp/aphrodite-engine \
    && mv /tmp/aphrodite-engine/* . \
    && rm -fr /tmp/aphrodite-engine \
    && chmod +x docker/entrypoint.sh

# Export the CUDA_HOME variable correctly
ENV CUDA_HOME=/usr/local/cuda

ENV HF_HOME=/tmp
ENV TORCH_CUDA_ARCH_LIST="6.1 7.0 7.5 8.0 8.6 8.9 9.0+PTX"
RUN python3 -m pip install --no-cache-dir -e .

# Entrypoint exec form doesn't do variable substitution automatically ($HOME)
ENTRYPOINT ["/app/aphrodite-engine/docker/entrypoint.sh"]

EXPOSE 5000

USER 1000:0

VOLUME ["/tmp"]
