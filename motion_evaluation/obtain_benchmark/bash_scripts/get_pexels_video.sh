#!/bin/bash
JSON_PATH=
NUM_SAMPLE=500
STRIDE?orFPS?
PEXEL_PATH=
SAVE_PATH=

python get_pexels_video.py \
    --json_path ${JSON_PATH} \
    --num ${NUM_SAMPLE} \
    --current_sample_stride ${FPS} \
    --data_root_path ${} \
    --save_path ${SAVE_PATH} \
