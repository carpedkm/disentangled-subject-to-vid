
export CUDA_VISIBLE_DEVICES=0
#adaptively compute world_size from CUDA_VISIBLE_DEVICES
export WORLD_SIZE=$(echo $CUDA_VISIBLE_DEVICES | tr ',' '\n' | wc -l)


torchrun --nproc_per_node $WORLD_SIZE --standalone evaluate.py \
    --output_path ./test_output \
    --video_dir ./test_video \
    --image_dir ./test_image \
    --json_path ./test.json \
    --prompt_json ./test_prompt.json \
    --mode 'default' \
    --n_frames 16 \
    # --regional_suffix_image regional_sam \

    # --dimension