#!/bin/bash

set -xe

cd /app/aphrodite-engine
echo 'Starting Aphrodite Engine API server...'
CMD="python3 -m aphrodite.endpoints.openai.api_server \
             --host 0.0.0.0 \
             --port 7860 \
             --model $MODEL_NAME \
             --tensor-parallel-size $NUM_GPUS \
             --dtype $DATATYPE \
             --max-model-len $CONTEXT_LENGTH \
             --gmu $GPU_MEMORY_UTILIZATION"

if [ -n "$QUANTIZATION" ]; then
    CMD="$CMD --quantization $QUANTIZATION --dtype half"
fi
if [ -n "$API_KEY" ]; then
    CMD="$CMD --api-keys $API_KEY"
fi
if [ -n "$ENFORCE_EAGER" ]; then
    CMD="$CMD --enforce-eager"
fi
if [ -n "$KVCACHE" ]; then
    CMD="$CMD --kv-cache-dtype $KVCACHE"
fi

exec $CMD