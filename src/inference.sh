export OUTPUT_PATH="./test_output" 
export REF_IMG_PATH="/root/daneul/projects/videocustom/samples/chosen_bluecar.png" 
export CHECKPOINT_PATH="/root/daneul/projects/videocustom/ckpts_best_ours/checkpoint-4000" 
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True 
export CUDA_VISIBLE_DEVICES=0
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1 
export WANDB_API_KEY=b524799f98b5a09033fe24848862dcb2a68af571
export PROMPT="a blue car driving through the desert at sunset"

python inference.py \
  --enable_tiling \
  --enable_slicing \
  --ref_img_path $REF_IMG_PATH \
  --output_dir $OUTPUT_PATH \
  --checkpoint_path $CHECKPOINT_PATH \
  --prompt $PROMPT