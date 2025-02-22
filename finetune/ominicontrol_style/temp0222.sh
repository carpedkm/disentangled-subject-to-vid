#!/bin/bash

while true; do
    # Get the list of processes running on GPU 0 (excluding header lines)
    GPU_PROCESS=$(nvidia-smi --query-compute-apps=gpu_uuid --format=csv,noheader | wc -l)

    # If no processes are running on GPU 0, start the training script
    if [ "$GPU_PROCESS" -eq 0 ]; then
        echo "$(date): No process found on GPU 0. Running the script..."
        bash exp_0221_train_vae_add_200k_a100_40_16_joint_finetune_random_full_drop_0.9_rand_init_frame_8fps_inf_ckpt3000.sh
    else
        echo "$(date): Process running on GPU 0, waiting..."
    fi

    # Wait for 30 seconds before checking again
    sleep 30
done
