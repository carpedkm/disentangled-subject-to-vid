#!/bin/bash

export MODEL_PATH="THUDM/CogVideoX-5b"
export CACHE_PATH="~/.cache"
export DATASET_PATH="/mnt/carpedkm_data/image_gen_ds/omini200k"
export ANNO_PATH="/mnt/carpedkm_data/image_gen_ds/omini200k/metadata_omini200k_update_refined.json"
export OUTPUT_PATH="/mnt/carpedkm_data/result250110/result_check_200k_512x512_embedding_refined_prompt_without_subject_moviegenstyle_vaeadd"
export VALIDATION_REF_PATH="./val_samples_im/"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export WANDB_API_KEY=b524799f98b5a09033fe24848862dcb2a68af571
# if you are not using wth 8 gus, change `accelerate_config_machine_single.yaml` num_processes as your gpu number
accelerate launch --config_file accelerate_config_machine_single_4gpu.yaml --multi_gpu \
  train_cogvideox_for_videocustom_wo_vae_250106_image_vae.py \
  --gradient_checkpointing \
  --pretrained_model_name_or_path $MODEL_PATH \
  --cache_dir $CACHE_PATH \
  --enable_tiling \
  --enable_slicing \
  --instance_data_root $DATASET_PATH \
  --anno_root $ANNO_PATH \
  --validation_epochs 5 \
  --num_validation_videos 1 \
  --validation_reference_image $VALIDATION_REF_PATH \
  --seed 42 \
  --rank 128 \
  --lora_alpha 64 \
  --mixed_precision bf16 \
  --output_dir $OUTPUT_PATH \
  --height 512 \
  --width 512 \
  --fps 8 \
  --max_num_frames 1 \
  --skip_frames_start 0 \
  --skip_frames_end 0 \
  --train_batch_size 48 \
  --num_train_epochs 30 \
  --checkpointing_steps 50 \
  --gradient_accumulation_steps 1 \
  --learning_rate 5e-5 \
  --lr_scheduler cosine_with_restarts \
  --lr_warmup_steps 200 \
  --lr_num_cycles 1 \
  --enable_slicing \
  --enable_tiling \
  --gradient_checkpointing \
  --optimizer AdamW \
  --adam_beta1 0.9 \
  --adam_beta2 0.95 \
  --max_grad_norm 1.0 \
  --allow_tf32 \
  --t5_first \
  --use_latent \
  --vae_add \
  --load_to_ram \
  --latent_data_root /mnt/carpedkm_data/pexels_4k_updatd_vae_latents\
  --report_to wandb 