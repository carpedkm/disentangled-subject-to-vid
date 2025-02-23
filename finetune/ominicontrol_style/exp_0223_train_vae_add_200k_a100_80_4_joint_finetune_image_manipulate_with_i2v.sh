
export MODEL_PATH="THUDM/CogVideoX-5b"
export CACHE_PATH="~/.cache"
export DATASET_PATH="/mnt/carpedkm_data/image_gen_ds/omini200k_720p_full"
export ANNO_PATH="/mnt/carpedkm_data/image_gen_ds/omini200k/metadata_omini200k_update_refined.json"
export OUTPUT_PATH="/mnt/carpedkm_data/result250223/image_man_80_4_prob05_with_i2v_debug"
export VALIDATION_REF_PATH="../val_samples_im/"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export WANDB_API_KEY=b524799f98b5a09033fe24848862dcb2a68af571

accelerate launch --config_file ../accelerate_config_machine_single_4gpu.yaml --multi_gpu \
  ../train_0223_imageman.py \
  --gradient_checkpointing \
  --pretrained_model_name_or_path $MODEL_PATH \
  --cache_dir $CACHE_PATH \
  --enable_tiling \
  --enable_slicing \
  --instance_data_root $DATASET_PATH \
  --anno_root $ANNO_PATH \
  --validation_epochs 1 \
  --num_validation_videos 1 \
  --validation_reference_image $VALIDATION_REF_PATH \
  --seed 42 \
  --rank 128 \
  --lora_alpha 64 \
  --mixed_precision bf16 \
  --output_dir $OUTPUT_PATH \
  --height 480 \
  --width 720 \
  --fps 8 \
  --max_num_frames 49 \
  --skip_frames_start 0 \
  --skip_frames_end 0 \
  --train_batch_size 4 \
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
  --pos_embed \
  --pos_embed_inf_match \
  --non_shared_pos_embed \
  --add_special \
  --layernorm_fix \
  --image_man \
  --image_man_with_i2v \
  --joint_train \
  --frame_weighted_loss \
  --prob_sample_video 0.2 \
  --random_drop_full \
  --random_drop_prob 0.0 \
  --video_anno /mnt/carpedkm_data/image_gen_ds/second_stage_video_train/second_stage_video_filtered_data_dict_sampled_4k.json \
  --video_instance_root /mnt/carpedkm_data/image_gen_ds/second_stage_video_train_pexels_8fps \
  --video_ref_root /mnt/carpedkm_data/image_gen_ds/second_stage_video_train_pexels_8fps_rand \
  --latent_data_root /mnt/carpedkm_data/pexels_4k_updatd_vae_latents\
  --report_to wandb \
  # --inference 채
  # --resume_from_checkpoint /mnt/carpedkm_data/result250215/special_tk_layernorm_fix_pos_embed_fix_40_16_non_shared_random_fix/checkpoint-3000 