#!/bin/bash

# Define dataset paths
Path_to_Benchmark="/root/daneul/projects/refactored/CogVideo/Pexels_subset_100K_fps8_flow-25-50_sample500"
Path_to_Synthesized_Video="/root/daneul/projects/refactored/CogVideo/Pexels_evaluation_small_100K_200_0225_original"
Save_Path="original_0225_100"

# Loop through dataset sizes
for TARGET in small medium large; do
    video_path_GT="${Path_to_Benchmark}/${TARGET}/video_frames"  # Path to real video dataset
    video_path_pred="${Path_to_Synthesized_Video}/${TARGET}/video_frames"  # Path to generated samples
    save_path="./Results/${Save_Path}/${TARGET}/"  # Path to save results

    # Verify paths exist
    if [ ! -d "$video_path_GT" ]; then
        echo "Error: Ground truth path does not exist: $video_path_GT"
        exit 1
    fi
    if [ ! -d "$video_path_pred" ]; then
        echo "Error: Predicted video path does not exist: $video_path_pred"
        exit 1
    fi

    # Run metric computation
    python src/scripts/calc_metrics_for_dataset.py \
        --metrics fvd500_49f \
        --real_data_path "${video_path_GT}" \
        --fake_data_path "${video_path_pred}" \
        --resolution 512 \
        --run_dir "${save_path}" \
        --verbose true
done