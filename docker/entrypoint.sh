#!/bin/bash

set -xe

cd /app/aphrodite-engine
source activate aphrodite-engine
echo 'Starting Aphrodite Engine API server...'
CMD="python -u -m aphrodite.endpoints.kobold.api_server \
             --host 0.0.0.0 \
             --port 2242 \
             --model $MODEL_NAME \
             --tensor-parallel-size $NUM_GPUS \
             --dtype $DATATYPE"

if [ -n "$QUANTIZATION" ]; then
    CMD="$CMD --quantization $QUANTIZATION"
fi

exec $CMD
