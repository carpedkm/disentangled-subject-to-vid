
export MODEL_PATH="THUDM/CogVideoX-5b"
export CACHE_PATH="~/.cache"
export DATASET_PATH="/mnt/carpedkm_data/image_gen_ds/omini200k_720p_full"
export ANNO_PATH="/mnt/carpedkm_data/image_gen_ds/omini200k/metadata_omini200k_update_refined.json"
export OUTPUT_PATH="/mnt/carpedkm_data/result250223/image_man_40_32_prob05_with_i2v"
export VALIDATION_REF_PATH="../val_samples_im/"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export WANDB_API_KEY=b524799f98b5a09033fe24848862dcb2a68af571

export NCCL_IB_DISABLE=0
export NCCL_IB_PCI_RELAXED_ORDERING=1
export NCCL_SOCKET_IFNAME=eth0
export NCCL_NET_GDR_LEVEL=5
export NCCL_TOPO_FILE=/opt/microsoft/ndv4-topo.xml
export NCCL_TIMEOUT=600  # Increase the timeout to 600 seconds

RANDOM_PORT=$((49152 + RANDOM % 16384))

accelerate launch --config_file ../accelerate_config_machine_single_inf.yaml \
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
  --train_batch_size 2 \
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
  --video_anno /mnt/carpedkm_data/image_gen_ds/second_stage_video_train/second_stage_video_filtered_data_dict_sampled_4k.json \
  --video_instance_root /mnt/carpedkm_data/image_gen_ds/second_stage_video_train_pexels_8fps \
  --video_ref_root /mnt/carpedkm_data/image_gen_ds/second_stage_video_train_pexels_8fps_rand \
  --latent_data_root /mnt/carpedkm_data/pexels_4k_updatd_vae_latents\
  --report_to wandb \
  --inference \
  --resume_from_checkpoint checkpoint-2200 \
  --inference_num_frames 49
  # --inference 채
  # --resume_from_checkpoint /mnt/carpedkm_data/result250215/special_tk_layernorm_fix_pos_embed_fix_40_16_non_shared_random_fix/checkpoint-3000 