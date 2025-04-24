import argparse
import os
from tqdm import tqdm
from pathlib import Path
from typing import List, Optional, Tuple, Union, Dict, Optional, Any
from PIL import Image
import numpy as np
import torch
from torch.nn import init

from peft import LoraConfig, set_peft_model_state_dict
from torch.utils.data import DataLoader, Dataset, Sampler
from transformers import AutoTokenizer, T5EncoderModel, T5Tokenizer

import torch.nn as nn
import torch.nn.functional as F
from custom_cogvideox_pipe import CustomCogVideoXPipeline

from diffusers import AutoencoderKLCogVideoX, CogVideoXPipeline, CogVideoXTransformer3DModel
from diffusers.models.embeddings import get_3d_rotary_pos_embed
from diffusers.pipelines.cogvideo.pipeline_cogvideox import get_resize_crop_region_for_grid
from diffusers.utils import convert_unet_state_dict_to_peft

from video_generate import inference

os.environ["TOKENIZERS_PARALLELISM"] = "false"

def get_args():
    parser = argparse.ArgumentParser(description="Simple example of a training script for CogVideoX.")
    parser.add_argument("--pretrained_model_name_or_path", type=str, default="THUDM/CogVideoX-5b", help="Path to pretrained model or model identifier from huggingface.co/models.")
    parser.add_argument("--cache_dir", type=str, default="~/.cache", help="The directory where the downloaded models and datasets will be stored.")
    parser.add_argument("--dataset_name", type=str, default="customization", help="The name of the Dataset (from HuggingFace hub) or local path.")
    parser.add_argument("--dataloader_num_workers", type=int, default=16, help="Number of subprocesses to use for data loading. 0 means main process.")
    parser.add_argument('--ref_img_path', type=str, default='', required=True, help='The path of the reference image for validation')
    parser.add_argument("--guidance_scale", type=float, default=6., help="The guidance scale to use while sampling validation videos.")
    parser.add_argument("--use_dynamic_cfg", action="store_true", default=False, help="Use cosine dynamic guidance schedule for validation.")
    parser.add_argument("--seed", type=int, default=2025, help="A seed for reproducible training.")
    parser.add_argument("--rank", type=int, default=128, help="The dimension of the LoRA update matrices.")
    parser.add_argument("--lora_alpha", type=float, default=64, help="Scaling factor for LoRA update (actual is `lora_alpha / rank`).")
    parser.add_argument("--output_dir", type=str, default="cogvideox-lora", help="Output directory for predictions and checkpoints.")
    parser.add_argument("--height", type=int, default=480, help="Resize input videos to this height.")
    parser.add_argument("--width", type=int, default=720, help="Resize input videos to this width.")
    parser.add_argument("--fps", type=int, default=8, help="Input video FPS.")
    parser.add_argument("--max_num_frames", type=int, default=49, help="Max frames used per input video.")
    parser.add_argument("--skip_frames_start", type=int, default=0, help="Frames to skip at the beginning of each video.")
    parser.add_argument("--skip_frames_end", type=int, default=0, help="Frames to skip at the end of each video.")
    parser.add_argument("--resume_from_checkpoint", type=str, default=None, help="Resume training from checkpoint path or 'latest'.")
    parser.add_argument("--enable_slicing", default=True, help="Enable VAE slicing for memory efficiency.")
    parser.add_argument("--enable_tiling", default=True, help="Enable VAE tiling for memory efficiency.")
    parser.add_argument("--hub_model_id", type=str, default=None, help="Repository name to sync with `output_dir`.")
    parser.add_argument('--t5_first', default=True, help='Concatenate T5 encoder prompt before CLIP.')
    parser.add_argument('--local_reference_scale', type=float, default=1.)
    parser.add_argument('--checkpoint_path', type=str, default=None, help='Checkpoint path to load the model')
    parser.add_argument('--prompt', type=str, default='', help='Prompt for inference')
    return parser.parse_args()

# vae_add, inference, validation_ref_image, pos_embed, pos_embed_inf_match, non_shared_pos_embed, add_special, add_specific_loc, layernorm_fix
# wo_shuffle, save_every_timestep, qformer, inference_num_frames, sampling_for_quali, num_of_prompt 


def load_model_hook(models, input_dir):
        """Load LoRA weights for transformer and vision models"""
        transformer_ = None
        
        # Extract models while emptying the list
        while len(models) > 0:
            model = models.pop()
            if isinstance(model, type(transformer)):
                transformer_ = model
            else:
                raise ValueError(f"Unexpected save model: {model.__class__}")
        
        # Load the combined lora state dict
        lora_state_dict = CogVideoXPipeline.lora_state_dict(input_dir)
        
        # Handle transformer model weights
        if transformer_ is not None:
            transformer_state_dict = {
                f'{k.replace("transformer.", "")}': v 
                for k, v in lora_state_dict.items() 
                if k.startswith("transformer.")
            }
            # For transformer, keep the UNet conversion if needed
            transformer_state_dict = convert_unet_state_dict_to_peft(transformer_state_dict)
            
            incompatible_keys = set_peft_model_state_dict(
                transformer_, 
                transformer_state_dict, 
                adapter_name="default"
            )
            if incompatible_keys is not None:
                # check only for unexpected keys
                unexpected_keys = getattr(incompatible_keys, "unexpected_keys", None)
                if unexpected_keys:
                    print(f"Unexpected keys in transformer state dict: {unexpected_keys}")
                    

def _get_t5_prompt_embeds(
    tokenizer: T5Tokenizer,
    text_encoder: T5EncoderModel,
    prompt: Union[str, List[str]],
    num_videos_per_prompt: int = 1,
    max_sequence_length: int = 226,
    device: Optional[torch.device] = None,
    dtype: Optional[torch.dtype] = None,
    text_input_ids=None,
):
    prompt = [prompt] if isinstance(prompt, str) else prompt
    batch_size = len(prompt)

    if tokenizer is not None:
        text_inputs = tokenizer(
            prompt,
            padding="max_length",
            max_length=max_sequence_length,
            truncation=True,
            add_special_tokens=True,
            return_tensors="pt",
        )
        text_input_ids = text_inputs.input_ids
    else:
        if text_input_ids is None:
            raise ValueError("`text_input_ids` must be provided when the tokenizer is not specified.")

    prompt_embeds = text_encoder(text_input_ids.to(device))[0]
    prompt_embeds = prompt_embeds.to(dtype=dtype, device=device)

    # duplicate text embeddings for each generation per prompt, using mps friendly method
    _, seq_len, _ = prompt_embeds.shape
    prompt_embeds = prompt_embeds.repeat(1, num_videos_per_prompt, 1)
    prompt_embeds = prompt_embeds.view(batch_size * num_videos_per_prompt, seq_len, -1)

    return prompt_embeds


def encode_prompt(
    tokenizer: T5Tokenizer,
    text_encoder: T5EncoderModel,
    prompt: Union[str, List[str]],
    num_videos_per_prompt: int = 1,
    max_sequence_length: int = 226,
    device: Optional[torch.device] = None,
    dtype: Optional[torch.dtype] = None,
    text_input_ids=None,
):
    prompt = [prompt] if isinstance(prompt, str) else prompt
    prompt_embeds = _get_t5_prompt_embeds(
        tokenizer,
        text_encoder,
        prompt=prompt,
        num_videos_per_prompt=num_videos_per_prompt,
        max_sequence_length=max_sequence_length,
        device=device,
        dtype=dtype,
        text_input_ids=text_input_ids,
    )

    return prompt_embeds



def main(args):
    print('Start')
    t5_first = args.t5_first
    concatenated_all = args.concatenated_all
    reduce_token = False

    if args.output_dir is not None:
        os.makedirs(args.output_dir, exist_ok=True)

    # Prepare models and scheduler
    tokenizer = AutoTokenizer.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="tokenizer",
    )
    if args.add_special:
        special_token = {"additional_special_tokens": ["<cls>"]}
        tokenizer.add_special_tokens(special_token)

    text_encoder = T5EncoderModel.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="text_encoder",
    )
    if args.add_special:
        text_encoder.resize_token_embeddings(len(tokenizer))
    
    load_dtype = torch.bfloat16 if "5b" in args.pretrained_model_name_or_path.lower() else torch.float16
    transformer = CogVideoXTransformer3DModel.from_pretrained( # DECLARE 
        args.pretrained_model_name_or_path,
        subfolder="transformer",
        torch_dtype=load_dtype,
        customization=True,
        concatenated_all=concatenated_all,
        reduce_token=reduce_token,
        vae_add=args.vae_add,
        local_reference_scale=args.local_reference_scale,
    )
    
    vae = AutoencoderKLCogVideoX.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="vae",
    )
    if args.enable_slicing:
        vae.enable_slicing()
    if args.enable_tiling:
        vae.enable_tiling()

    # weight_dtype = torch.float32
    weight_dtype = torch.bfloat16 if "5b" in args.pretrained_model_name_or_path.lower() else torch.float16

    # for inference, wo accelerate
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    text_encoder.to(device, dtype=weight_dtype)
    transformer.to(device, dtype=weight_dtype)


    transformer_lora_config = LoraConfig(
        r=args.rank,
        lora_alpha=args.lora_alpha,
        init_lora_weights=True,
        target_modules=["to_k", "to_q", "to_v", "to_out.0", "proj", "text_proj","norm1.linear", "norm2.linear", "ff.net.2"],
    )
    
    transformer.add_adapter(transformer_lora_config)
    print(transformer.active_adapters)

    
    checkpoint_dir = os.path.join(args.checkpoint_path, args.resume_from_checkpoint)
    load_model_hook([transformer], checkpoint_dir)
    # Create pipeline
    vae = AutoencoderKLCogVideoX.from_pretrained(
            args.pretrained_model_name_or_path, subfolder="vae",
        )
    if args.enable_slicing:
        vae.enable_slicing()
    if args.enable_tiling:
        vae.enable_tiling()
    vae.requires_grad_(False)
    vae.to(device, dtype=weight_dtype)
    pipe = CustomCogVideoXPipeline.from_pretrained(
        args.pretrained_model_name_or_path,
        transformer=transformer,
        text_encoder=text_encoder,
        vae=vae,
        torch_dtype=weight_dtype,
        customization=True,
        vae_add=args.vae_add,
    )
    ref_img = args.ref_img_path
    validation_prompt = args.prompt
    pipeline_args = {
        "prompt": validation_prompt,
        "guidance_scale": args.guidance_scale,
        "use_dynamic_cfg": args.use_dynamic_cfg,
        "reference_image": ref_img,
        "height": 480,
        "width": 720,
        "eval" : True,
        'vae_add' : args.vae_add,
        'output_dir' : args.output_dir,
    }
    inference(
        pipe=pipe,
        args=args,
        pipeline_args=pipeline_args,
    )
    print('Inference completed')


if __name__ == "__main__":
    print('Code executed')
    args = get_args()
    print('Args:', args)
    main(args)