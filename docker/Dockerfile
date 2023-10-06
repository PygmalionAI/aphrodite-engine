FROM continuumio/miniconda3:23.5.2-0

WORKDIR /app/aphrodite-engine

# Upgrade OS Packages
RUN set -eux; \
    apt-get update \
    && apt-get upgrade -y \
    && rm -rf /var/lib/apt/lists/*

COPY environment.yml /app/aphrodite-engine

# Preparing Conda Environment
RUN apt-get update \
    && apt-get install -y build-essential \
    && conda env create \
    && apt-get purge -y --auto-remove build-essential \
    && pip install --upgrade ray pyarrow pandas \
    && rm -fr /var/lib/apt/lists/*

COPY entrypoint.sh /app/aphrodite-engine

ENV PATH /opt/conda/envs/aphrodite-engine/bin:$PATH

ENTRYPOINT [ "/app/aphrodite-engine/entrypoint.sh" ]