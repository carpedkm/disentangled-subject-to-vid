export OUTPUT_PATH="./test_output" 
export REF_IMG_PATH="./samples/pig.png" 
export CHECKPOINT_PATH="../ckpts_best_ours/checkpoint-4000" 
export CUDA_VISIBLE_DEVICES=0

python src/inference.py \
  --ref_img_path $REF_IMG_PATH \
  --output_dir $OUTPUT_PATH \
  --checkpoint_path $CHECKPOINT_PATH \
  --prompt "<cls> A pig eagerly chases a butterfly in a meadow."