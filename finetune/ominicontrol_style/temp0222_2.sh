#!/bin/bash

while true; do
    # Check if there are any processes running on GPU 1
    GPU_PROCESS=$(nvidia-smi --query-compute-apps=gpu_bus_id --format=csv,noheader | awk 'NR>0 {print}' | wc -l)

    # If no processes are running on GPU 1, start the training script
    if [ "$GPU_PROCESS" -eq 0 ]; then
        echo "$(date): No process found on GPU 1. Running the script..."
        bash exp_0221_train_vae_add_200k_a100_40_16_joint_finetune_random_full_drop_0.9_rand_init_frame_8fps_inf_ckpt6000.sh
    else
        echo "$(date): Process running on GPU 1, waiting..."
    fi

    # Wait for 30 seconds before checking again
    sleep 30
done
