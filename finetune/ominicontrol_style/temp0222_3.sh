#!/bin/bash

while true; do
    # Check if there are any processes running on GPU 2
    GPU_PROCESS=$(nvidia-smi --query-compute-apps|grep "2" | wc -l)

    # If no processes are running on GPU 2, start the training script
    if [ "$GPU_PROCESS" -eq 0 ]; then
        echo "$(date): No process found on GPU 2. Running the script..."
        bash exp_0220_train_vae_add_200k_a100_40_16_joint_finetune_palisades_random_8fps_inf_ckpt9000.sh
    else
        echo "$(date): Process running on GPU 2, waiting..."
    fi

    # Wait for 30 seconds before checking again
    sleep 30
done
