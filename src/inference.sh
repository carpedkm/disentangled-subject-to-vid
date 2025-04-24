export OUTPUT_PATH="./test_output" 
export REF_IMG_PATH="./samples/cat.jpg" 
export TEST_PROMPT_PATH="../zs_prompts_new.json" 
export CHECKPOINT_PATH="/root/daneul/projects/videocustom/ckpts_best_ours" 
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True 
export CUDA_VISIBLE_DEVICES=0
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1 
export WANDB_API_KEY=b524799f98b5a09033fe24848862dcb2a68af571

python inference.py \
  --enable_tiling \
  --enable_slicing \
  --ref_img_path $REF_IMG_PATH \
  --output_dir $OUTPUT_PATH \
  --checkpoint_path $CHECKPOINT_PATH \
  --height 480 \
  --width 720 \
  --resume_from_checkpoint checkpoint-4000 \
  --phase_name test \
  --prompt 