export MODEL_PATH="THUDM/CogVideoX-5b" 
export CACHE_PATH="~/.cache" 
export OUTPUT_PATH="./test_output" 
export VALIDATION_REF_PATH="../zs_samples/" 
export TEST_PROMPT_PATH="../zs_prompts_new.json" 
export CHECKPOINT_PATH="/root/daneul/projects/videocustom/ckpts_best_ours" 
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True 
export CUDA_VISIBLE_DEVICES=0
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1 
export WANDB_API_KEY=b524799f98b5a09033fe24848862dcb2a68af571

python inference.py \
  --pretrained_model_name_or_path $MODEL_PATH \
  --cache_dir $CACHE_PATH \
  --enable_tiling \
  --enable_slicing \
  --validation_reference_image $VALIDATION_REF_PATH \
  --seed 2025 \
  --rank 128 \
  --lora_alpha 64 \
  --output_dir $OUTPUT_PATH \
  --checkpoint_path $CHECKPOINT_PATH \
  --height 480 \
  --width 720 \
  --fps 8 \
  --max_num_frames 49 \
  --skip_frames_start 0 \
  --skip_frames_end 0 \
  --enable_slicing \
  --enable_tiling \
  --t5_first \
  --vae_add \
  --pos_embed \
  --pos_embed_inf_match \
  --non_shared_pos_embed \
  --add_special \
  --layernorm_fix \
  --inference \
  --resume_from_checkpoint checkpoint-4000 \
  --phase_name test \
  --test_prompt_path $TEST_PROMPT_PATH \
  --sampling_for_quali \
  --num_of_prompts 8 \