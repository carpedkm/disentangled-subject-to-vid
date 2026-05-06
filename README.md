# Learning Zero-Shot Subject-Driven Video Generation Using 1\% Compute

<p align="center">
  <a href="https://arxiv.org/abs/2504.17816"><img src="https://img.shields.io/badge/ArXiv-Disentangled_S2V-red"></a>
  <a href="https://carpedkm.github.io/projects/disentangled_sub/"><img src="https://img.shields.io/badge/Project%20Page-Disentangled_S2V-blue"></a> 
</p>


This repository provides the code for the paper "Learning Zero-Shot Subject-Driven Video Generation Using 1\% Compute". The method enables the generation of high-quality videos based on a subject image and a text prompt, without requiring large annotated video datasets. By leveraging an image customization dataset and a small set of unannotated videos, this approach achieves robust subject consistency and temporal coherence in a zero-shot setting.

**Note:** This repository currently only includes the inference code for version 1 release.
Release for version 3 (Including CogVideoX and Wan) Inference checkpoint and Fine-tuning code is not provided at this time but is planned for a future release.

## 🔥 Latest News
- May 06, 2026 Updated Version (v3) Released to [arXiv](https://arxiv.org/abs/2504.17816v3)!
- Jan 10, 2026 Updated Version (v2) Released to [arXiv](https://arxiv.org/abs/2504.17816v2)!
- Apr 28, 2025 Initial Version (v1) Released to [arXiv](https://arxiv.org/abs/2504.17816v1)!
- Apr 27, 2025 Inference Code Release!

## TODOs

- [x] Release inference code.
- [x] Release v2 arXiv release
- [x] Release v3 arXiv Release
- [ ] Release v3 inference code with Wan and CogVideoX
- [ ] Release fine-tuning code.
- [ ] Add more features and improvements.

## Quick Start

To set up the environment and install dependencies, 
first start with:

   ```bash
   conda create -n disentangled_s2v python=3.12
   conda activate disentangled_s2v
   ```

You can install all the required packages by:
   ```bash
   bash installer.sh
   ```

Or follow these steps:

1. **Install PyTorch:**
   ```bash
   pip install torch==2.4.1+cu121 -f https://download.pytorch.org/whl/torch_stable.html
   ```

2. **Install diffusers in editable mode:**
   ```
   cd diffusers
   pip install -e .
   cd ..
   ```

3. **Install other dependencies:**
   ```bash
   pip install sentencepiece huggingface peft transformers accelerate
   pip install opencv-python imageio ffmpeg imageio-ffmpeg
   ```

## Get Checkpoint 
- **Download the pre-trained model checkpoint:**
   Download the checkpoint from [Google Drive Link](https://drive.google.com/file/d/190lmjS_tPmutSruDmgGthyg_N5wRF3Nl/view?usp=sharing) and extract it to a directory, e.g., `./disentangled_s2v_ckpt`.
   
   Place the ckpt file as follows:
   ```bash
   # Directory structure of the checkpoint folder:
   ./disentangled_s2v_ckpt
   ├── optimizer.bin
   ├── pytorch_lora_weights_transformer.safetensors
   ├── random_states_0.pkl
   └── scheduler.bin
   ```

## Usage
- **Run the inference script:**
   ```bash
   python src/inference.py \
     --reference_image_path <REFERENCE IMAGE PATH>
     --output_dir "./test_output.mp4" \
     --checkpoint_path <CHECKPOINT_PATH> \
     --prompt <PROMPT>
   ```

Alternatively, you can use the provided shell script for quick demo:
```bash
bash s2v_inference_demo.sh
```

## Acknowledgements
We built our work based on [CogVideoX](https://github.com/THUDM/CogVideo), with dataset from [OminiControl](https://github.com/Yuanshi9815/OminiControl) and [Pexels](https://huggingface.co/datasets/jovianzm/Pexels-400k).



## BibTeX
<pre><code>@article{kim2025subject,
  author    = {Kim, Daneul and Zhang, Jingxu and Jin, Wonjoon and Cho, Sunghyun and Dai, Qi and Park, Jaesik and Luo, Chong},
  title     = {Subject-driven Video Generation via Disentangled Identity and Motion},
  journal   = {arXiv},
  year      = {2025},
}
</code></pre>
