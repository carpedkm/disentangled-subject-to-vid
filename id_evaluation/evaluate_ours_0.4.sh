
export CUDA_VISIBLE_DEVICES=2
#adaptively compute world_size from CUDA_VISIBLE_DEVICES
export WORLD_SIZE=$(echo $CUDA_VISIBLE_DEVICES | tr ',' '\n' | wc -l)


torchrun --nproc_per_node $WORLD_SIZE --standalone evaluate.py \
    --output_path /root/daneul/projects/refactored/CogVideo/id_evaluation/ours_0.4_output\
    --video_dir /mnt/carpedkm_data/result250227/joint_finetune_random_frame_select_8fps_prob04_dropfull_prob05_palisades_40G32/seed_42 \
    --image_dir /root/daneul/projects/refactored/CogVideo/id_evaluation/processed_white_720x480 \
    --json_path /root/daneul/projects/refactored/CogVideo/id_evaluation/ours_0.4_vid_to_img.json \
    --prompt_json /root/daneul/projects/refactored/CogVideo/id_evaluation/ours_0.4_vid_to_prompt.json \
    --mode 'default' \
    --n_frames 16
    # --regional_suffix_image regional_sam \

    # --dimension