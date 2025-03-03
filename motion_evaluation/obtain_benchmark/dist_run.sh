#!/bin/bash

set -x

CONFIG=$1
GPUS=$2
PT_SCRIPT=$3
DATA_TYPE=$4
RANDOM_PORT=$((49152 + RANDOM % 16384))

torchrun \
    --nproc_per_node=$GPUS \
    --master_port=$RANDOM_PORT \
    ${PT_SCRIPT} \
        --data_type=${DATA_TYPE} \
        --config=${CONFIG} \
        --launcher=pytorch \
        --port=${RANDOM_PORT}