#!/bin/bash -e

echo 'Starting Aphrodite Engine API server...'

CMD="python3 -m aphrodite.endpoints.${ENDPOINT:-openai}.api_server
             --host 0.0.0.0
             --port 5000
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
             ${CMD_ADDITIONAL_ARGUMENTS}"

# Only the 'openai' endpoint currently supports api-keys and ssl
if [ "$ENDPOINT" = "openai" ]; then
  CMD+=" ${API_KEY:+--api-keys "$API_KEY"} ${SSL_KEYFILE:+--ssl-keyfile /etc/ssl/private/server.key} ${SSL_CERTFILE:+--ssl-certfile /etc/ssl/certs/server.crt}"
fi

# set umask to ensure group read / write at runtime
umask 002

set -x

exec $CMD
