#!/bin/bash

PEXEL_PATH="/video_data/"
JSON_PATH="./output/flow_curation_metadata_10K_fps8/curate_pexels_10K/flow_statistics.json"
SAVE_PATH="sample_video/Pexels_subset_10K_fps8_flow-25-50_sample100"
NUM=100
sample_fps=8

python divide_benchmark_dataset.py \
    --data_root_path ${PEXEL_PATH} \
    --json_path ${JSON_PATH} \
    --save_path ${SAVE_PATH} \
    --num ${NUM} \
    --sample_fps ${sample_fps} \