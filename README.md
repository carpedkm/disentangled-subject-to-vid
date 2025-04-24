# [Official Implementation] Subject-driven Video Generation via Disentangled Identity and Motion
---
<p align="center">
  <img src="./assets/s2v_teaser.gif" width="500"/>
</p>

---
## <span style="color:red"><strong> Currently Code is under maintenance -- It would be made available this week. Stay tuned!</strong></span>
---

This repository provides the code for the paper "Subject-driven Video Generation via Disentangled Identity and Motion." The method enables the generation of high-quality videos based on a subject image and a text prompt, without requiring large annotated video datasets. By leveraging an image customization dataset and a small set of unannotated videos, this approach achieves robust subject consistency and temporal coherence in a zero-shot setting.

**Note:** This repository currently only includes the inference code. Fine-tuning code is not provided at this time but is planned for a future release.

## Installation

To set up the environment and install dependencies, follow these steps:

1. **Install PyTorch:**
   ```bash
   pip install torch==2.4.1+cu121 -f https://download.pytorch.org/whl/torch_stable.html
   ```

2. **Install other dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Install diffusers in editable mode:**
   ```bash
   git clone https://github.com/huggingface/diffusers.git
   cd diffusers
   pip install -e .
   cd ..
   ```

The `requirements.txt` file includes:
- transformers
- numpy
- Pillow
- tqdm
- peft

## Usage

To run inference and generate videos using the provided pre-trained model, follow these steps:

1. **Set environment variables (optional):**
   ```bash
   export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
   export CUDA_VISIBLE_DEVICES=0
   export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
   ```
   Adjust `CUDA_VISIBLE_DEVICES` based on your GPU setup.

2. **Download the pre-trained model checkpoint:**
   Download the best checkpoint from [Google Drive Link] and extract it to a directory, e.g., `./ckpts_best_ours`.

3. **Prepare validation reference images and test prompts:**
   - Place your validation reference images in a directory, e.g., `../zs_samples/`.
   - Prepare a JSON file with test prompts, e.g., `../zs_prompts_new.json`.

4. **Run the inference script:**
   ```bash
   python inference.py \
     --pretrained_model_name_or_path "THUDM/CogVideoX-5b" \
     --cache_dir "~/.cache" \
     --enable_tiling \
     --enable_slicing \
     --validation_reference_image "../zs_samples/" \
     --seed 2025 \
     --rank 128 \
     --lora_alpha 64 \
     --output_dir "./test_output" \
     --checkpoint_path "./ckpts_best_ours" \
     --height 480 \
     --width 720 \
     --fps 8 \
     --max_num_frames 49 \
     --skip_frames_start 0 \
     --skip_frames_end 0 \
     --enable_slicing \
     --enable_tiling \
     --t5_first \
     --vae_add \
     --pos_embed \
     --pos_embed_inf_match \
     --non_shared_pos_embed \
     --add_special \
     --layernorm_fix \
     --inference \
     --resume_from_checkpoint checkpoint-4000 \
     --phase_name test \
     --test_prompt_path "../zs_prompts_new.json"
   ```

   Ensure that the paths (`validation_reference_image`, `checkpoint_path`, `test_prompt_path`, etc.) are adjusted according to your local setup.

Alternatively, you can use the provided shell script:
```bash
bash s2v_inference.sh
```

## Model Checkpoint

The pre-trained model checkpoint is available for download from [Google Drive Link]. After downloading, extract it to a directory (e.g., `./ckpts_best_ours`) and specify the path using the `--checkpoint_path` argument in the inference script.

## TODOs

- [x] Release inference code.
- [ ] Update the codebase to be compatible with the latest diffusers and CogVideoX repositories.
- [ ] Release fine-tuning code.
- [ ] Add more features and improvements.

---
## Authors
### **[Daneul Kim](https://carpedkm.github.io/)**, **[Jingxu Zhang](#)**, **[Wonjoon Jin](https://jinwonjoon.github.io/)**, **[Sunghyun Cho](https://www.scho.pe.kr/)**, **[Qi Dai](https://daiqi1989.github.io/)**, **[Jaesik Park](https://jaesik.info)**, **[Chong Luo](https://www.microsoft.com/en-us/research/people/cluo/)**

---
## Acknowledgements
We built our work based on [CogVideoX](https://github.com/THUDM/CogVideo), with dataset from [OminiControl](https://github.com/Yuanshi9815/OminiControl) and [Pexels](https://huggingface.co/datasets/jovianzm/Pexels-400k).



## BibTex
<pre><code>@article{kim2025subject,
  author    = {Kim, Daneul and Zhang, Jingxu and Jin, Wonjoon and Cho, Sunghyun and Dai, Qi and Park, Jaesik and Luo, Chong},
  title     = {Subject-driven Video Generation via Disentangled Identity and Motion},
  journal   = {arXiv},
  year      = {2025},
}
</code></pre>
