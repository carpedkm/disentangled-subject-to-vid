#!/bin/bash

export CUDA_LAUNCH_BLOCKING=1
export NCCL_P2P_DISABLE=1
export NCCL_IB_DISABLE=1

bash dist_run.sh configs/curate_pexels.yaml 4 curate_data.py pexels