#!/bin/bash

set -xe

cd /app/aphrodite-engine
source activate aphrodite-engine
echo 'Starting Aphrodite Engine API server...'
python -u -m aphrodite.endpoints.api_server_kobold \
             --host 0.0.0.0 \
             --port 2242 \
             --model $MODEL_NAME \
             --tensor-parallel-size $NUM_GPUS \
             --quantization $QUANTIZATION \
             --dtype $DATATYPE