# Copyright 2024 The CogView team, Tsinghua University & ZhipuAI and The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import argparse
import logging
import math
import os
import shutil
from tqdm import tqdm
from pathlib import Path
from typing import List, Optional, Tuple, Union, Dict, Optional, Any
# from typing import List, Dict, Any, Optional

from PIL import Image
import numpy as np
import torch
from torch.nn import init
import transformers
import random


from huggingface_hub import create_repo, upload_folder
from peft import LoraConfig, get_peft_model_state_dict, set_peft_model_state_dict ,TaskType, get_peft_model
from torch.utils.data import DataLoader, Dataset, Sampler
from torchvision import transforms
from tqdm.auto import tqdm


from transformers import Blip2Processor, Blip2Model
from transformers import AutoTokenizer, T5EncoderModel, T5Tokenizer
# ADDED
import json
from transformers import CLIPProcessor, CLIPTokenizer, CLIPTextModel, CLIPVisionModel
import torch.nn as nn
import torch.nn.functional as F
import multiprocessing
from functools import partial
from safetensors.torch import save_file, load_file
from custom_cogvideox import CustomCogVideoXTransformer3DModel
from custom_cogvideox_pipe import CustomCogVideoXPipeline

import diffusers
from diffusers import AutoencoderKLCogVideoX, CogVideoXDPMScheduler, CogVideoXPipeline, CogVideoXTransformer3DModel
from diffusers.models.embeddings import get_3d_rotary_pos_embed
from diffusers.optimization import get_scheduler
from diffusers.pipelines.cogvideo.pipeline_cogvideox import get_resize_crop_region_for_grid
from diffusers.training_utils import (
    cast_training_params,
    free_memory,
)
from diffusers.utils import check_min_version, convert_unet_state_dict_to_peft, export_to_video, export_to_video_with_frames, is_wandb_available
from diffusers.utils.hub_utils import load_or_create_model_card, populate_model_card
from diffusers.utils.torch_utils import is_compiled_module

from concurrent.futures import ThreadPoolExecutor

if is_wandb_available():
    import wandb
    
    

# Will error if the minimal version of diffusers is not installed. Remove at your own risks.
check_min_version("0.31.0.dev0")



os.environ["TOKENIZERS_PARALLELISM"] = "false"

def get_args():
    parser = argparse.ArgumentParser(description="Simple example of a training script for CogVideoX.")

    # Model information
    parser.add_argument(
        "--pretrained_model_name_or_path",
        type=str,
        default=None,
        required=True,
        help="Path to pretrained model or model identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--revision",
        type=str,
        default=None,
        required=False,
        help="Revision of pretrained model identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--variant",
        type=str,
        default=None,
        help="Variant of the model files of the pretrained model identifier from huggingface.co/models, 'e.g.' fp16",
    )
    parser.add_argument(
        "--cache_dir",
        type=str,
        default=None,
        help="The directory where the downloaded models and datasets will be stored.",
    )

    # Dataset information
    parser.add_argument(
        "--dataset_name",
        type=str,
        default="customization",
        help=(
            "The name of the Dataset (from the HuggingFace hub) containing the training data of instance images (could be your own, possibly private,"
            " dataset). It can also be a path pointing to a local copy of a dataset in your filesystem,"
            " or to a folder containing files that 🤗 Datasets can understand."
        ),
    )
    parser.add_argument(
        "--id_token", type=str, default=None, help="Identifier token appended to the start of each prompt if provided."
    )
    parser.add_argument(
        "--dataloader_num_workers",
        type=int,
        default=16,
        help=(
            "Number of subprocesses to use for data loading. 0 means that the data will be loaded in the main process."
        ),
    )

    # Validation
    parser.add_argument(
        '--validation_reference_image',
        type=str,
        default='val_samples/',
        help='The path of the reference image for validation'
    )
    parser.add_argument(
        "--guidance_scale",
        type=float,
        default=6.,
        help="The guidance scale to use while sampling validation videos.",
    )
    parser.add_argument(
        "--use_dynamic_cfg",
        action="store_true",
        default=False,
        help="Whether or not to use the default cosine dynamic guidance schedule when sampling validation videos.",
    )

    # Training information
    parser.add_argument("--seed", type=int, default=None, help="A seed for reproducible training.")
    parser.add_argument(
        "--rank",
        type=int,
        default=128,
        help=("The dimension of the LoRA update matrices."),
    )
    parser.add_argument(
        "--lora_alpha",
        type=float,
        default=128,
        help=("The scaling factor to scale LoRA weight update. The actual scaling factor is `lora_alpha / rank`"),
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="cogvideox-lora",
        help="The output directory where the model predictions and checkpoints will be written.",
    )
    parser.add_argument(
        "--height",
        type=int,
        default=480,
        help="All input videos are resized to this height.",
    )
    parser.add_argument(
        "--width",
        type=int,
        default=720,
        help="All input videos are resized to this width.",
    )
    parser.add_argument(
        "--height_val",
        type=int,
        default=480,
        help="All input videos are resized to this height.",
    )
    parser.add_argument(
        "--width_val",
        type=int,
        default=720,
        help="All input videos are resized to this width.",
    )
    parser.add_argument("--fps", type=int, default=8, help="All input videos will be used at this FPS.")
    parser.add_argument(
        "--max_num_frames", type=int, default=49, help="All input videos will be truncated to these many frames."
    )
    parser.add_argument(
        "--skip_frames_start",
        type=int,
        default=0,
        help="Number of frames to skip from the beginning of each input video. Useful if training data contains intro sequences.",
    )
    parser.add_argument(
        "--skip_frames_end",
        type=int,
        default=0,
        help="Number of frames to skip from the end of each input video. Useful if training data contains outro sequences.",
    )
    parser.add_argument(
        "--resume_from_checkpoint",
        type=str,
        default=None,
        help=(
            "Whether training should be resumed from a previous checkpoint. Use a path saved by"
            ' `--checkpointing_steps`, or `"latest"` to automatically select the last available checkpoint.'
        ),
    )
    parser.add_argument(
        "--enable_slicing",
        action="store_true",
        default=False,
        help="Whether or not to use VAE slicing for saving memory.",
    )
    parser.add_argument(
        "--enable_tiling",
        action="store_true",
        default=False,
        help="Whether or not to use VAE tiling for saving memory.",
    )
    parser.add_argument(
        "--hub_model_id",
        type=str,
        default=None,
        help="The name of the repository to keep in sync with the local `output_dir`.",
    )
    parser.add_argument(
        '--t5_first',
        action='store_true',
        help='Whether to concatenate the prompt of the t5 encoder with clip encoders first or not'
    )
    parser.add_argument(
        '--concatenated_all',
        action='store_true',
        help='Whether to concatenate encoders and use the concatenated feature itself'
    )
    parser.add_argument(
        '--inference',
        action='store_true',
        help='Whether to use quick poc subset or not'
    )
    parser.add_argument(
        '--vae_add',
        action='store_true',
        help='Whether to use vae latent of reference image directly or not'
    )
    parser.add_argument(
        "--seen_validation",
        action="store_true",
        help="Whether to use seen validation or not"
    )
    parser.add_argument(
        '--pos_embed',
        action='store_true',
        help='Whether to use positional embedding or not on reference image'
    )
    parser.add_argument(
        '--local_reference_scale',
        type=float,
        default=1.,
    )
    parser.add_argument(
        '--add_special',
        action='store_true',
        help='Whether to add special token or not'
    )
    parser.add_argument(
        "--add_multiple_special",
        action="store_true",
        help="Whether to add multiple special tokens or not"
    )
    parser.add_argument(
        '--add_specific_loc',
        action='store_true',
        help='Whether to add special token to speicifc location or not'
    )
    parser.add_argument(
        '--wo_shuffle',
        action='store_true',
        help='Whether to shuffle the dataset or not'
    )
    parser.add_argument(
        '--add_new_split',
        action='store_true',
        help='Whether to add new split or not'
    )
    parser.add_argument(
        '--save_every_timestep',
        action='store_true',
        help='Whether to save every timestep or not'
    )
    parser.add_argument(
        '--qformer',
        action='store_true',
        help='Whether to use qformer or not'
    )
    parser.add_argument(
        '--layernorm_fix',
        action='store_true',
        help='Whether to use fixed layernorm or not'
    )
    parser.add_argument(
        '--pos_embed_inf_match',
        action='store_true',
        help='Whether to use positional embedding for inference match or not'
    )
    parser.add_argument(
        '--non_shared_pos_embed',
        action='store_true',
        help='Whether to use non shared positional embedding or not'
    )
    parser.add_argument(
        '--video_ref_root',
        type=str,
        default=None,
    )
    parser.add_argument(
        '--inference_num_frames',
        type=int,
        default=49,
    )
    parser.add_argument(
        '--phase_name',
        type=str,
        default='validation',
    )
    parser.add_argument(
        '--test_prompt_path',
        type=str,
        default=None,
    )
    parser.add_argument(
        '--sampling_for_quali',
        action='store_true',
        help='Whether to use sampling for quality or not'
    )
    parser.add_argument(
        '--num_of_prompts',
        type=int,
        default=1,
    )
    parser.add_argument(
        '--quali_shard',
        type=int,
        default=0,
    )
    parser.add_argument(
        '--quali_sep_count',
        type=int,
        default=3,
    )
    parser.add_argument(
        '--checkpoint_path',
        type=str,
        default=None,
        help='Checkpoint path to load the model'
    )
    return parser.parse_args()


    
class VideoDataset(Dataset):
    def __init__(
        self,
        height: int = 480,
        width: int = 720,
        fps: int = 8,
        max_num_frames: int = 49,
        id_token: Optional[str] = None,
        seen_validation: bool = False,
        test_prompt_path: Optional[str] = None,
    ) -> None:
            
        self.seen_validation = seen_validation
        self.id_token = id_token or ""

        super().__init__()

        self.prefix = "<cls> "
        self.val_instance_prompt_dict = {}
        if test_prompt_path is not None:
            with open(test_prompt_path, 'r') as f:
                test_prompts = json.load(f)
            for key in test_prompts.keys():
                if key not in self.val_instance_prompt_dict.keys():
                    self.val_instance_prompt_dict[key] = test_prompts[key]

        self.val_instance_prompt_dict = {k: v for k, v in self.val_instance_prompt_dict.items()}
    def __len__(self):
        return len(self.val_instance_prompt_dict)

def log_validation(
    pipe,
    args,
    pipeline_args,
    ckpt_step: int = 0,
    is_final_validation: bool = False,
    phase_name: str = "validation",
    resizing: bool = True,
    vid_id: str = None,
    # prompt: dict = None,
):


    # Move pipeline to device and eval mode
    device = pipe._execution_device
    pipe = pipe.to(device)
    pipe.transformer.eval()
    pipe.vae.eval()
    pipe.text_encoder.eval()

    # Set deterministic generation
    generator = torch.Generator(device=device)
    if args.seed is not None:
        generator.manual_seed(args.seed) # init seed
    num_validation_videos = 1
    videos = []
    for _ in range(num_validation_videos): 
        current_pipeline_args = pipeline_args.copy()
        # seed update
        generator.manual_seed(generator.initial_seed() + 10)
        if args.dataset_name == 'customization':
            if 'validation_reference_image' in pipeline_args:
                try:
                    from PIL import Image
                    ref_image = Image.open(pipeline_args['validation_reference_image']).convert('RGB')
                    # Load and preprocess the reference image
                    if resizing is True:
                        ref_image = ref_image.resize((720, 720)) 
                        width, height = 720, 720
                        target_width, target_height = args.width_val, args.height_val

                        # Calculate coordinates for center crop
                        left = (width - target_width) // 2
                        top = (height - target_height) // 2
                        right = left + target_width
                        bottom = top + target_height

                        # Perform cropping
                        ref_image = ref_image.crop((left, top, right, bottom))
                    ref_image = np.array(ref_image)
                    ref_image = np.expand_dims(ref_image, axis=0)  # Add frame dimension
                    
                    ref_image = torch.from_numpy(ref_image).float() / 255.0 * 2.0 - 1.0
                    ref_image = ref_image.permute(0, 3, 1, 2)  # [B, H, W, C] -> [B, C, H, W]
                    
                    ref_image = ref_image.unsqueeze(0).permute(0, 2, 1, 3, 4).to(device=device, dtype=pipe.transformer.dtype)
                    
                    with torch.no_grad():
                        ref_image_latents = pipe.vae.encode(ref_image).latent_dist
                        ref_image_latents = ref_image_latents.sample() * pipe.vae.config.scaling_factor
                    ref_image_latents = ref_image_latents.permute(0, 2, 1, 3, 4)

                    current_pipeline_args['ref_img_states'] = ref_image_latents
                    current_pipeline_args.pop('validation_reference_image', None)
                    

                except Exception as e:  
                    print(f"Error loading reference image: {e}")
                    raise
            else:
                print("Reference image not found in pipeline arguments.")

        # Generate the video
        try:
            print(f"Generating video with prompt: {pipeline_args['prompt']}")
            
            # Add inference parameters explicitly
            inference_args = {
                'num_inference_steps': 50,
                'output_type': "np",
                'guidance_scale': args.guidance_scale,
                'use_dynamic_cfg': args.use_dynamic_cfg,
                'height': args.height_val,
                'width': args.width_val,
                'num_frames': args.inference_num_frames, #args.max_num_frames,
                'eval': True
            }
            current_pipeline_args.update(inference_args)
            
            # Run inference with torch.no_grad()
            with torch.no_grad():
                output = pipe(**current_pipeline_args)
                video = output.frames[0]
            videos.append(video)
            
            print(f"Generated video with prompt: {pipeline_args['prompt']}")
            
        except Exception as e:
            print(f"Error generating video: {e}")
            raise

    # Log to wandb if enabled
        # phase_name = "test" if is_final_validation else "validation"
        video_filenames = []
        for i, video in enumerate(videos):
            prompt = (
                pipeline_args["prompt"][:30]
                .replace(" ", "_")
                .replace("'", "_")
                .replace('"', "_")
                .replace("/", "_")
            )
            max_num_frames = current_pipeline_args['num_frames']

            filename = os.path.join(args.output_dir, f"ckpt_{ckpt_step}_white_{phase_name}_video_{i}_max_n_f_{max_num_frames}_id_{vid_id}_{prompt}.mp4")
            output_frames_dir = os.path.join(args.output_dir, f"ckpt_{ckpt_step}_white_{phase_name}_video_{i}_max_n_f_{max_num_frames}_id_{vid_id}_{prompt}")

            # export_to_video(video, filename, fps=args.fps)
            export_to_video_with_frames(
                video_frames=video,
                output_video_path=filename,
                output_frames_dir=output_frames_dir,
                fps=args.fps,
            )
            video_filenames.append(filename)
    # Clean up
    free_memory()
    return videos

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
    clip_tokenizer: CLIPTokenizer,
    clip_text_encoder: CLIPTextModel,
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


def compute_prompt_embeddings(
    tokenizer, text_encoder, clip_tokenizer, clip_text_encoder, prompt, max_sequence_length, device, dtype, requires_grad: bool = False
):
    with torch.no_grad():
        prompt_embeds = encode_prompt(
            tokenizer,
            text_encoder,
            clip_tokenizer,
            clip_text_encoder,
            prompt,
            num_videos_per_prompt=1,
            max_sequence_length=max_sequence_length,
            device=device,
            dtype=dtype,
        )

    return prompt_embeds

def prepare_rotary_positional_embeddings(
    height: int,
    width: int,
    num_frames: int,
    vae_scale_factor_spatial: int = 8,
    patch_size: int = 2,
    attention_head_dim: int = 64,
    device: Optional[torch.device] = None,
    base_height: int = 480,
    base_width: int = 720,
) -> Tuple[torch.Tensor, torch.Tensor]:
    grid_height = height // (vae_scale_factor_spatial * patch_size)
    grid_width = width // (vae_scale_factor_spatial * patch_size)
    base_size_width = base_width // (vae_scale_factor_spatial * patch_size)
    base_size_height = base_height // (vae_scale_factor_spatial * patch_size)

    grid_crops_coords = get_resize_crop_region_for_grid((grid_height, grid_width), base_size_width, base_size_height)
    freqs_cos, freqs_sin = get_3d_rotary_pos_embed(
        embed_dim=attention_head_dim,
        crops_coords=grid_crops_coords,
        grid_size=(grid_height, grid_width),
        temporal_size=num_frames,
    )

    freqs_cos = freqs_cos.to(device=device)
    freqs_sin = freqs_sin.to(device=device)
    return freqs_cos, freqs_sin


def main(args):
    print('Start')
    t5_first = args.t5_first
    concatenated_all = args.concatenated_all
    reduce_token = False

    if args.output_dir is not None:
        os.makedirs(args.output_dir, exist_ok=True)

    # Prepare models and scheduler
    tokenizer = AutoTokenizer.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="tokenizer", revision=args.revision
    )
    if args.add_special:
        special_token = {"additional_special_tokens": ["<cls>"]}
        tokenizer.add_special_tokens(special_token)

    text_encoder = T5EncoderModel.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="text_encoder", revision=args.revision
    )
    if args.add_special:
        text_encoder.resize_token_embeddings(len(tokenizer))
    
    load_dtype = torch.bfloat16 if "5b" in args.pretrained_model_name_or_path.lower() else torch.float16
    transformer = CogVideoXTransformer3DModel.from_pretrained( # DECLARE 
        args.pretrained_model_name_or_path,
        subfolder="transformer",
        torch_dtype=load_dtype,
        revision=args.revision,
        variant=args.variant,
        customization=True,
        concatenated_all=concatenated_all,
        reduce_token=reduce_token,
        vae_add=args.vae_add,
        local_reference_scale=args.local_reference_scale,
    )
    
    vae = AutoencoderKLCogVideoX.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="vae", revision=args.revision, variant=args.variant
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
        
    print("Registering save and load hooks")
    # accelerator.register_load_state_pre_hook(load_model_hook)
    checkpoint_dir = os.path.join(args.checkpoint_path, args.resume_from_checkpoint)
    load_model_hook([transformer], checkpoint_dir)

    print("Dataset and DataLoader")
    # Dataset and DataLoader
    video_dataset = VideoDataset(
        height=args.height,
        width=args.width,
        seen_validation=args.seen_validation,
        test_prompt_path=args.test_prompt_path,
    )
    def collate_fn(examples):
        videos = [example['instance_video'] for example in examples]
        prompts = [example['instance_prompt'] for example in examples]
        ref_images = [example['instance_ref_image'] for example in examples]
        # if args.use_latent:
        videos = torch.cat(videos, dim=0)
        ref_images = torch.cat(ref_images, dim=0)
        # else:
        # videos = torch.stack(videos, dim=0)
        videos = videos.to(memory_format=torch.contiguous_format).float()
        batch = {
            "videos": videos,
            "prompts": prompts,
            "ref_images": ref_images,
        }
        return batch
   
    def worker_init_fn(worker_id):
        seed = torch.initial_seed() % 2**32
        np.random.seed(seed)
        random.seed(seed)
    train_dataloader = DataLoader(
        dataset=video_dataset,
        batch_size=2,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=args.dataloader_num_workers,
        prefetch_factor=4,
        worker_init_fn=worker_init_fn,
    )

    path = os.path.basename(args.resume_from_checkpoint)

    if args.inference:
        # Create pipeline
        print('Doing INFERENCE')
        vae = AutoencoderKLCogVideoX.from_pretrained(
                args.pretrained_model_name_or_path, subfolder="vae", revision=args.revision, variant=args.variant
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
            revision=args.revision,
            variant=args.variant,
            torch_dtype=weight_dtype,
            customization=True,
            vae_add=args.vae_add,
        )

        if args.seen_validation:
            args.validation_reference_image = "../seen_samples/omini_right/"
        resizing = True
        if args.sampling_for_quali:
            if True: #args.wo_background_in_inf_sampling:
                args.validation_reference_image = os.path.join(args.validation_reference_image, 'processed_white_720x480')
                resizing = False
            else:
                args.validation_reference_image = os.path.join(args.validation_reference_image, 'processed_bg_720x720')
                resizing = True
        val_len = len(os.listdir(args.validation_reference_image))
        sep_val_len = val_len // args.quali_sep_count
        if args.sampling_for_quali:
            args.output_dir = os.path.join(args.output_dir, f"seed_{args.seed}")
            for i in range(sep_val_len * args.quali_shard, sep_val_len * (args.quali_shard + 1)):
                for cnt in range(args.num_of_prompts):
                    validation_ref_img = os.path.join(args.validation_reference_image, os.listdir(args.validation_reference_image)[i])
                    vid_id = os.listdir(args.validation_reference_image)[i].split('.')[0]
                    validation_prompt = video_dataset.val_instance_prompt_dict[vid_id][cnt]
                    pipeline_args = {
                        "prompt": validation_prompt,
                        "guidance_scale": args.guidance_scale,
                        "use_dynamic_cfg": args.use_dynamic_cfg,
                        "validation_reference_image": validation_ref_img,
                        "height": 480,
                        "width": 720,
                        "eval" : True,
                        "concatenated_all" : concatenated_all,
                        "reduce_token" : reduce_token,
                        'vae_add' : args.vae_add,
                        'pos_embed' : args.pos_embed,
                        'output_dir' : args.output_dir,
                        'layernorm_fix': args.layernorm_fix,
                        'non_shared_pos_embed': args.non_shared_pos_embed,
                    }
                    tmp_idx = 0
                    ckpt_step = int(os.path.basename(args.resume_from_checkpoint).split('-')[1])
                    max_num_frames = args.inference_num_frames
                    phase_name = args.phase_name
                    prompt = (
                        pipeline_args["prompt"][:30]
                        .replace(" ", "_")
                        .replace("'", "_")
                        .replace('"', "_")
                        .replace("/", "_")
                    )
                    if os.path.exists(os.path.join(args.output_dir, f"ckpt_{ckpt_step}_white_{phase_name}_video_{tmp_idx}_max_n_f_{max_num_frames}_id_{vid_id}_{prompt}.mp4")):
                        print('>>> SKIPPING already existing', os.path.join(args.output_dir, f"ckpt_{ckpt_step}_white_{phase_name}_video_{tmp_idx}_max_n_f_{max_num_frames}_id_{vid_id}_{prompt}.mp4"))
                        continue
                    validation_outputs = log_validation(
                        pipe=pipe,
                        args=args,
                        pipeline_args=pipeline_args,
                        ckpt_step=ckpt_step,
                        phase_name=args.phase_name,
                        resizing=resizing,
                        vid_id=vid_id,
                    )
    if args.inference:
        # exit program
        import sys
        sys.exit(0)


if __name__ == "__main__":
    print('Code executed')
    args = get_args()
    print('Args:', args)
    main(args)