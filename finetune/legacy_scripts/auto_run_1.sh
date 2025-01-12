#!/bin/bash

# Infinite loop to check GPU processes every 30 seconds
while true; do
    # Check if there are any active processes using the GPU
    PROCESS_COUNT=$(nvidia-smi | grep -c "No running processes found")
    
    if [ "$PROCESS_COUNT" -gt 0 ]; then
        echo "No GPU processes found. Running abc.sh..."
        # Run abc.sh script
        bash refactored_a100_40_8_moviegen_1600_5b_w_latent_multi24.sh 1
        echo "Script executed. Exiting..."
        exit 0  # Exit the loop and script after running
    else
        echo "GPU is in use. Checking again in 30 seconds..."
    fi

    # Wait for 30 seconds before checking again
    sleep 30
done