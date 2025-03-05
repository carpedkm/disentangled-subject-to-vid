
export CUDA_VISIBLE_DEVICES=2
#adaptively compute world_size from CUDA_VISIBLE_DEVICES
export WORLD_SIZE=$(echo $CUDA_VISIBLE_DEVICES | tr ',' '\n' | wc -l)


torchrun --nproc_per_node $WORLD_SIZE --standalone evaluate.py \
    --output_path /root/daneul/projects/refactored/CogVideo/id_evaluation/ip_adapter_output \
    --video_dir /root/daneul/projects/refactored/CogVideo/id_evaluation/CogVideoX_I2V_results_ip_adapter \
    --image_dir /root/daneul/projects/refactored/CogVideo/id_evaluation/processed_white_720x480 \
    --json_path /root/daneul/projects/refactored/CogVideo/id_evaluation/ip_vid_to_img.json \
    --prompt_json /root/daneul/projects/refactored/CogVideo/id_evaluation/ip_vid_to_prompt.json \
    --mode 'default' \
    --n_frames 16
    # --regional_suffix_image regional_sam \

    # --dimension