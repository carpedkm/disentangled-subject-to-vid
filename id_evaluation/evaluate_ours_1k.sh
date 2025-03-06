
export CUDA_VISIBLE_DEVICES=0
#adaptively compute world_size from CUDA_VISIBLE_DEVICES
export WORLD_SIZE=$(echo $CUDA_VISIBLE_DEVICES | tr ',' '\n' | wc -l)


torchrun --nproc_per_node $WORLD_SIZE --standalone evaluate.py \
    --output_path /root/daneul/projects/refactored/CogVideo/id_evaluation/ours_1k_output\
    --video_dir /root/daneul/projects/refactored/CogVideo/id_evaluation/ours_1k \
    --image_dir /root/daneul/projects/refactored/CogVideo/id_evaluation/processed_white_720x480 \
    --json_path /root/daneul/projects/refactored/CogVideo/id_evaluation/ours_1k_vid_to_img.json \
    --prompt_json /root/daneul/projects/refactored/CogVideo/id_evaluation/ours_first_frame_in_i2v_only_vid_to_prompt.json \
    --mode 'default' \
    --n_frames 16
    # --regional_suffix_image regional_sam \

    # --dimension