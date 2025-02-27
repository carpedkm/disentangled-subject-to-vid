#!/bin/bash

while true; do
	    # Check for GPU processes
	        if ! nvidia-smi | grep -q "python"; then
			        echo "No GPU process found. Running the script..."
				        bash exp_0224_train_vae_add_200k_a100_40_16_joint_finetune_image_manipulate_with_i2v_with_first_frame_schedule_and_13f_jw_inf.sh
					        exit 0  # Exit after running the script
						    else
							            echo "GPU is in use. Checking again in 30 seconds..."
								        fi
									    sleep 30
								    done
