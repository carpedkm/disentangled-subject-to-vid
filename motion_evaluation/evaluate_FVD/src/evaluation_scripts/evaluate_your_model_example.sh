#!/bin/bash

# TARGET='small'       # small
# TARGET='medium'      # medium
# TARGET='large'     # large

#---------------------------------------------------------------------------------------------------------
TARGET='small' # For small, medium, and large motion video benchmark sets

video_path_GT="../obtain_benchmark/sample_video/Pexels_subset_10K_fps8_flow-25-50_sample100//${TARGET}/video_frames" # Path to benchmark dataset
video_path_pred="../obtain_benchmark/sample_video/Pexels_subset_10K_fps8_flow-25-50_sample100//${TARGET}/video_frames" # Path to synthesized samples
save_path="./Results/example/${TARGET}/" # Path to save dir.

python src/scripts/calc_metrics_for_dataset.py \
    --metrics fvd500_49f \
    --real_data_path "${video_path_GT}" \
    --fake_data_path "${video_path_pred}" \
    --resolution 512 \
    --run_dir "${save_path}" \
    --verbose true  \



TARGET='medium' # For small, medium, and large motion video benchmark sets

video_path_GT="../obtain_benchmark/sample_video/Pexels_subset_10K_fps8_flow-25-50_sample100//${TARGET}/video_frames" # Path to benchmark dataset
video_path_pred="../obtain_benchmark/sample_video/Pexels_subset_10K_fps8_flow-25-50_sample100//${TARGET}/video_frames" # Path to synthesized samples
save_path="./Results/example/${TARGET}/" # Path to save dir.

python src/scripts/calc_metrics_for_dataset.py \
    --metrics fvd500_49f \
    --real_data_path "${video_path_GT}" \
    --fake_data_path "${video_path_pred}" \
    --resolution 512 \
    --run_dir "${save_path}" \
    --verbose true  \



TARGET='large' # For small, medium, and large motion video benchmark sets

video_path_GT="../obtain_benchmark/sample_video/Pexels_subset_10K_fps8_flow-25-50_sample100//${TARGET}/video_frames" # Path to benchmark dataset
video_path_pred="../obtain_benchmark/sample_video/Pexels_subset_10K_fps8_flow-25-50_sample100//${TARGET}/video_frames" # Path to synthesized samples
save_path="./Results/example/${TARGET}/" # Path to save dir.

python src/scripts/calc_metrics_for_dataset.py \
    --metrics fvd500_49f \
    --real_data_path "${video_path_GT}" \
    --fake_data_path "${video_path_pred}" \
    --resolution 512 \
    --run_dir "${save_path}" \
    --verbose true  \
