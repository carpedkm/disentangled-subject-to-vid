# [Official Implementation] Subject-driven Video Generation via Disentangled Identity and Motion

<p align="center">
  <img src="./assets/s2v_teaser.gif" width="700"/>
</p>

<p align="center">
  <a href="#"><img src="https://img.shields.io/badge/ArXiv_To_Be_Released-Disentangled_S2V-red"></a>
  <a href="https://carpedkm.github.io/projects/disentangled_sub/"><img src="https://img.shields.io/badge/Project%20Page-Disentangled_S2V-blue"></a> 
</p>


This repository provides the code for the paper "Subject-driven Video Generation via Disentangled Identity and Motion." The method enables the generation of high-quality videos based on a subject image and a text prompt, without requiring large annotated video datasets. By leveraging an image customization dataset and a small set of unannotated videos, this approach achieves robust subject consistency and temporal coherence in a zero-shot setting.

**Note:** This repository currently only includes the inference code. Fine-tuning code is not provided at this time but is planned for a future release.

## Preparation

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
   Download the best checkpoint from [Google Drive Link] and extract it to a directory, e.g., `./disentangled_s2v_ckpt`.

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

## TODOs

- [x] Release inference code.
- [ ] Update the codebase to be compatible with the latest diffusers and CogVideoX repositories.
- [ ] Release fine-tuning code.
- [ ] Add more features and improvements.


## Authors
**[Daneul Kim](https://carpedkm.github.io/)**<sup>§</sup>, **[Jingxu Zhang](#)**, **[Wonjoon Jin](https://jinwonjoon.github.io/)**, **[Sunghyun Cho](https://www.scho.pe.kr/)**, **[Qi Dai](https://daiqi1989.github.io/)**, **[Jaesik Park](https://jaesik.info)**, **[Chong Luo](https://www.microsoft.com/en-us/research/people/cluo/)**

§: This work was done while at Microsoft Research Asia.


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
