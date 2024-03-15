#!/bin/bash -e

export NUMBA_CACHE_DIR="/tmp/numba_cache"
echo 'Starting Aphrodite Engine API server...'

CMD="python3 -m aphrodite.endpoints.openai.api_server
             --host ${HOST:-0.0.0.0}
             --port ${PORT:-7860}
             --download-dir ${HF_HOME:?}/hub
             ${MODEL_NAME:+--model $MODEL_NAME}
             ${REVISION:+--revision $REVISION}
             ${DATATYPE:+--dtype $DATATYPE}
             ${KVCACHE:+--kv-cache-dtype $KVCACHE}
             ${CONTEXT_LENGTH:+--max-model-len $CONTEXT_LENGTH}
             ${NUM_GPUS:+--tensor-parallel-size $NUM_GPUS}
             ${GPU_MEMORY_UTILIZATION:+--gpu-memory-utilization $GPU_MEMORY_UTILIZATION}
             ${QUANTIZATION:+--quantization $QUANTIZATION}
             ${ENFORCE_EAGER:+--enforce-eager}
             ${KOBOLD:+--launch-kobold-api}
             ${CMD_ADDITIONAL_ARGUMENTS}"

# set umask to ensure group read / write at runtime
umask 002

set -x

exec $CMD
