
export MODEL_PATH="THUDM/CogVideoX-5b"
export CACHE_PATH="~/.cache"
export DATASET_PATH="/mnt/carpedkm_data/image_gen_ds/omini200k_720p_full"
export ANNO_PATH="/mnt/carpedkm_data/image_gen_ds/omini200k/metadata_omini200k_update_refined.json"
export OUTPUT_PATH="/mnt/carpedkm_data/result250225/joint_finetune_random_frame_select_8fps_prob01_dropfull_prob05_palisades_40G32"
export VALIDATION_REF_PATH="../dreambooth_test_white/"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export CUDA_VISIBLE_DEVICES=0
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export WANDB_API_KEY=b524799f98b5a09033fe24848862dcb2a68af571


accelerate launch --config_file ../accelerate_config_machine_single_inf.yaml \
  ../train_0219_randomdrop.py \
  --gradient_checkpointing \
  --pretrained_model_name_or_path $MODEL_PATH \
  --cache_dir $CACHE_PATH \
  --enable_tiling \
  --enable_slicing \
  --instance_data_root $DATASET_PATH \
  --anno_root $ANNO_PATH \
  --validation_epochs 100 \
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
  --train_batch_size 8 \
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
  --joint_train \
  --prob_sample_video 0.2 \
  --random_drop_full \
  --video_anno /mnt/carpedkm_data/image_gen_ds/second_stage_video_train/second_stage_video_filtered_data_dict_sampled_4k.json \
  --video_instance_root /mnt/carpedkm_data/image_gen_ds/second_stage_video_train_pexels_8fps \
  --video_ref_root /mnt/carpedkm_data/image_gen_ds/second_stage_video_train_pexels_8fps_rand_multi \
  --load_to_ram \
  --latent_data_root /mnt/carpedkm_data/pexels_4k_updatd_vae_latents\
  --report_to wandb \
  --inference \
  --resume_from_checkpoint checkpoint-6000 \
  --phase_name test
  # --inference 
  # --resume_from_checkpoint /mnt/carpedkm_data/result250215/special_tk_layernorm_fix_pos_embed_fix_40_16_non_shared_random_fix/checkpoint-3000 