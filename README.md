# [Official Implementation] Subject-driven Video Generation via Disentangled Identity and Motion

<p align="center">
  <img src="./assets/s2v_teaser.gif" width="700"/>
</p>


## <span style="color:red"><strong> Currently Code is under maintenance -- It would be made available this week. Stay tuned!</strong></span>
---

This repository provides the code for the paper "Subject-driven Video Generation via Disentangled Identity and Motion." The method enables the generation of high-quality videos based on a subject image and a text prompt, without requiring large annotated video datasets. By leveraging an image customization dataset and a small set of unannotated videos, this approach achieves robust subject consistency and temporal coherence in a zero-shot setting.

**Note:** This repository currently only includes the inference code. Fine-tuning code is not provided at this time but is planned for a future release.

## Preparation

To set up the environment and install dependencies, follow these steps:

1. **Install PyTorch:**
   ```bash
   pip install torch==2.4.1+cu121 -f https://download.pytorch.org/whl/torch_stable.html
   ```

2. **Install other dependencies:**
   ```bash
   pip install huggingface transformers numpy peft
   ```

3. **Install diffusers in editable mode:**
   ```
   cd diffusers
   pip install -e .
   cd ..
   ```

4. **Download the pre-trained model checkpoint:**
   Download the best checkpoint from [Google Drive Link] and extract it to a directory, e.g., `./ckpts_best_ours`.

## Usage
- **Run the inference script:**
   ```bash
   python src/inference.py \
     --reference_image_path <REFERENCE IMAGE PATH>
     --output_dir "./test_output.mp4" \
     --checkpoint_path "./ckpts_best_ours" \
     --prompt <PROMPT>
   ```

Alternatively, you can use the provided shell script for quick demo:
```bash
bash src/s2v_inference.sh
```

## TODOs

- [x] Release inference code.
- [ ] Update the codebase to be compatible with the latest diffusers and CogVideoX repositories.
- [ ] Release fine-tuning code.
- [ ] Add more features and improvements.

---
## Authors
**[Daneul Kim](https://carpedkm.github.io/)**<sup>§</sup>, **[Jingxu Zhang](#)**, **[Wonjoon Jin](https://jinwonjoon.github.io/)**, **[Sunghyun Cho](https://www.scho.pe.kr/)**, **[Qi Dai](https://daiqi1989.github.io/)**, **[Jaesik Park](https://jaesik.info)**, **[Chong Luo](https://www.microsoft.com/en-us/research/people/cluo/)**

§: This work was done while at Microsoft Research Asia.

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
