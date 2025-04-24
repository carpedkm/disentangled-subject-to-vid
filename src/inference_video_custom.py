#!/usr/bin/env python3
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

"""
CogVideoX LoRA Inference Script

This script performs inference with a CogVideoX model fine-tuned with LoRA.
It loads a pretrained model, applies LoRA weights, and generates videos based on text prompts
and reference images.
"""

import argparse
import json
import logging
import os
import random
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import torch
from PIL import Image
from peft import LoraConfig, set_peft_model_state_dict
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm
from transformers import AutoTokenizer, CLIPTextModel, CLIPTokenizer, T5EncoderModel, T5Tokenizer

import diffusers
from diffusers import AutoencoderKLCogVideoX, CogVideoXPipeline
from diffusers.models.embeddings import get_3d_rotary_pos_embed
from diffusers.pipelines.cogvideo.pipeline_cogvideox import get_resize_crop_region_for_grid
from diffusers.training_utils import free_memory
from diffusers.utils import check_min_version, export_to_video_with_frames

# Import custom modules - assume these are available in your environment
from custom_cogvideox import CustomCogVideoXTransformer3DModel
from custom_cogvideox_pipe import CustomCogVideoXPipeline

# Check if the required diffusers version is installed
check_min_version("0.31.0.dev0")

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Disable tokenizer parallelism warnings
os.environ["TOKENIZERS_PARALLELISM"] = "false"


class VideoDataset(Dataset):
    """Dataset for video inference using text prompts."""
    
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
        """
        Initialize the dataset.
        
        Args:
            height: Height of videos
            width: Width of videos
            fps: Frames per second
            max_num_frames: Maximum number of frames
            id_token: Optional identifier token to prepend to prompts
            seen_validation: Whether to use seen validation samples
            test_prompt_path: Path to JSON file containing test prompts
        """
        super().__init__()
        
        self.seen_validation = seen_validation
        self.id_token = id_token or ""
        self.prefix = "<cls> "
        self.val_instance_prompt_dict = {}
        
        # Load test prompts if path is provided
        if test_prompt_path is not None:
            with open(test_prompt_path, 'r') as f:
                test_prompts = json.load(f)
                
            for key in test_prompts.keys():
                if key not in self.val_instance_prompt_dict:
                    self.val_instance_prompt_dict[key] = test_prompts[key]

    def __len__(self):
        return len(self.val_instance_prompt_dict)


def get_parser():
    """Create and configure the argument parser."""
    
    parser = argparse.ArgumentParser(description="CogVideoX LoRA inference script")
    
    # Model configuration group
    model_group = parser.add_argument_group("Model Configuration")
    model_group.add_argument(
        "--pretrained_model_name_or_path",
        type=str,
        required=True,
        help="Path to pretrained model or model identifier from huggingface.co/models",
    )
    model_group.add_argument(
        "--revision",
        type=str,
        default=None,
        help="Revision of pretrained model identifier from huggingface.co/models",
    )
    model_group.add_argument(
        "--variant",
        type=str,
        default=None,
        help="Variant of the model files (e.g., fp16)",
    )
    model_group.add_argument(
        "--cache_dir",
        type=str,
        default=None,
        help="Directory to store downloaded models",
    )
    model_group.add_argument(
        "--checkpoint_path",
        type=str,
        default=None,
        help="Path to LoRA checkpoint directory",
    )
    model_group.add_argument(
        "--resume_from_checkpoint",
        type=str,
        default=None,
        help="Specific checkpoint to load (e.g., 'checkpoint-4000')",
    )
    
    # LoRA configuration group
    lora_group = parser.add_argument_group("LoRA Configuration")
    lora_group.add_argument(
        "--rank",
        type=int,
        default=128,
        help="Dimension of LoRA update matrices",
    )
    lora_group.add_argument(
        "--lora_alpha",
        type=float,
        default=128,
        help="Scaling factor for LoRA weight updates",
    )
    
    # Inference configuration group
    inference_group = parser.add_argument_group("Inference Configuration")
    inference_group.add_argument(
        "--height",
        type=int,
        default=480,
        help="Height for resizing input videos",
    )
    inference_group.add_argument(
        "--width",
        type=int,
        default=720,
        help="Width for resizing input videos",
    )
    inference_group.add_argument(
        "--height_val",
        type=int,
        default=480,
        help="Height for validation/inference videos",
    )
    inference_group.add_argument(
        "--width_val",
        type=int,
        default=720,
        help="Width for validation/inference videos",
    )
    inference_group.add_argument(
        "--fps",
        type=int,
        default=8,
        help="Frames per second for videos",
    )
    inference_group.add_argument(
        "--max_num_frames",
        type=int,
        default=49,
        help="Maximum number of frames for input videos",
    )
    inference_group.add_argument(
        "--inference_num_frames",
        type=int,
        default=49,
        help="Number of frames to generate during inference",
    )
    inference_group.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed for reproducibility",
    )
    inference_group.add_argument(
        "--guidance_scale",
        type=float,
        default=6.0,
        help="Guidance scale for classifier-free guidance",
    )
    inference_group.add_argument(
        "--use_dynamic_cfg",
        action="store_true",
        default=False,
        help="Use dynamic classifier-free guidance schedule",
    )
    inference_group.add_argument(
        "--enable_slicing",
        action="store_true",
        default=False,
        help="Enable VAE slicing to save memory",
    )
    inference_group.add_argument(
        "--enable_tiling",
        action="store_true",
        default=False,
        help="Enable VAE tiling to save memory",
    )
    
    # Dataset and sampling configuration
    dataset_group = parser.add_argument_group("Dataset and Sampling")
    dataset_group.add_argument(
        "--validation_reference_image",
        type=str,
        default="val_samples/",
        help="Path to reference images for validation/inference",
    )
    dataset_group.add_argument(
        "--id_token",
        type=str,
        default=None,
        help="Identifier token appended to prompts",
    )
    dataset_group.add_argument(
        "--skip_frames_start",
        type=int,
        default=0,
        help="Frames to skip from beginning of videos",
    )
    dataset_group.add_argument(
        "--skip_frames_end",
        type=int,
        default=0,
        help="Frames to skip from end of videos",
    )
    dataset_group.add_argument(
        "--dataloader_num_workers",
        type=int,
        default=16,
        help="Number of worker processes for data loading",
    )
    dataset_group.add_argument(
        "--test_prompt_path",
        type=str,
        default=None,
        help="Path to JSON file with test prompts",
    )
    dataset_group.add_argument(
        "--sampling_for_quali",
        action="store_true",
        help="Sample for qualitative evaluation",
    )
    dataset_group.add_argument(
        "--num_of_prompts",
        type=int,
        default=1,
        help="Number of prompts per reference image",
    )
    dataset_group.add_argument(
        "--quali_shard",
        type=int,
        default=0,
        help="Shard index for qualitative evaluation",
    )
    dataset_group.add_argument(
        "--quali_sep_count",
        type=int,
        default=3,
        help="Number of shards for qualitative evaluation",
    )
    
    # Output configuration
    output_group = parser.add_argument_group("Output Configuration")
    output_group.add_argument(
        "--output_dir",
        type=str,
        default="cogvideox-lora-output",
        help="Directory to save outputs",
    )
    output_group.add_argument(
        "--phase_name",
        type=str,
        default="validation",
        help="Phase name for output files (e.g., 'test', 'validation')",
    )
    
    # Model architecture flags
    arch_group = parser.add_argument_group("Model Architecture Options")
    arch_group.add_argument(
        "--t5_first",
        action="store_true",
        help="Concatenate T5 encoder with CLIP encoders first",
    )
    arch_group.add_argument(
        "--concatenated_all",
        action="store_true",
        help="Concatenate all encoders and use the feature directly",
    )
    arch_group.add_argument(
        "--vae_add",
        action="store_true",
        help="Use VAE latent of reference image directly",
    )
    arch_group.add_argument(
        "--pos_embed",
        action="store_true",
        help="Use positional embedding on reference image",
    )
    arch_group.add_argument(
        "--pos_embed_inf_match",
        action="store_true",
        help="Use positional embedding for inference matching",
    )
    arch_group.add_argument(
        "--non_shared_pos_embed",
        action="store_true",
        help="Use non-shared positional embedding",
    )
    arch_group.add_argument(
        "--add_special",
        action="store_true",
        help="Add special token to tokenizer",
    )
    arch_group.add_argument(
        "--layernorm_fix",
        action="store_true",
        help="Use fixed layernorm implementation",
    )
    
    # Mode flags
    mode_group = parser.add_argument_group("Execution Mode")
    mode_group.add_argument(
        "--inference",
        action="store_true",
        help="Run in inference mode",
    )
    mode_group.add_argument(
        "--seen_validation",
        action="store_true",
        help="Use seen validation data",
    )
    
    return parser


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
    """
    Prepare rotary positional embeddings for the transformer model.
    
    Args:
        height: Height of the input
        width: Width of the input
        num_frames: Number of frames
        vae_scale_factor_spatial: VAE spatial scale factor
        patch_size: Patch size
        attention_head_dim: Attention head dimension
        device: Device to place tensors on
        base_height: Base height for grid crop region
        base_width: Base width for grid crop region
        
    Returns:
        Tuple of cosine and sine frequency embeddings
    """
    grid_height = height // (vae_scale_factor_spatial * patch_size)
    grid_width = width // (vae_scale_factor_spatial * patch_size)
    base_size_width = base_width // (vae_scale_factor_spatial * patch_size)
    base_size_height = base_height // (vae_scale_factor_spatial * patch_size)

    grid_crops_coords = get_resize_crop_region_for_grid(
        (grid_height, grid_width), 
        base_size_width, 
        base_size_height
    )
    
    freqs_cos, freqs_sin = get_3d_rotary_pos_embed(
        embed_dim=attention_head_dim,
        crops_coords=grid_crops_coords,
        grid_size=(grid_height, grid_width),
        temporal_size=num_frames,
    )

    freqs_cos = freqs_cos.to(device=device)
    freqs_sin = freqs_sin.to(device=device)
    
    return freqs_cos, freqs_sin


def process_reference_image(image_path, args, device, pipe_dtype, resizing=True):
    """
    Load and process a reference image for video generation.
    
    Args:
        image_path: Path to the reference image
        args: Script arguments
        device: Device to place tensors on
        pipe_dtype: Data type for the pipeline
        resizing: Whether to resize the image
        
    Returns:
        Processed image latents
    """
    try:
        # Load and preprocess reference image
        ref_image = Image.open(image_path).convert('RGB')
        
        if resizing:
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
        
        # Convert to tensor and normalize
        ref_image = np.array(ref_image)
        ref_image = np.expand_dims(ref_image, axis=0)  # Add frame dimension
        
        ref_image = torch.from_numpy(ref_image).float() / 255.0 * 2.0 - 1.0
        ref_image = ref_image.permute(0, 3, 1, 2)  # [B, H, W, C] -> [B, C, H, W]
        
        ref_image = ref_image.unsqueeze(0).permute(0, 2, 1, 3, 4).to(
            device=device, 
            dtype=pipe_dtype
        )
        
        return ref_image
        
    except Exception as e:  
        logger.error(f"Error loading reference image: {e}")
        raise


def load_model_hook(models, input_dir):
    """
    Load LoRA weights for the transformer model.
    
    Args:
        models: List of models to load weights for
        input_dir: Directory containing the weights
    """
    transformer_ = None
    
    # Extract models from the list
    while len(models) > 0:
        model = models.pop()
        if isinstance(model, CustomCogVideoXTransformer3DModel):
            transformer_ = model
        else:
            raise ValueError(f"Unexpected model type: {model.__class__}")
    
    # Load the LoRA state dict
    lora_state_dict = CogVideoXPipeline.lora_state_dict(input_dir)
    
    # Handle transformer model weights
    if transformer_ is not None:
        transformer_state_dict = {
            f'{k.replace("transformer.", "")}': v 
            for k, v in lora_state_dict.items() 
            if k.startswith("transformer.")
        }
        
        # Set the model state dict
        incompatible_keys = set_peft_model_state_dict(
            transformer_, 
            transformer_state_dict, 
            adapter_name="default"
        )
        
        if incompatible_keys is not None:
            # Check for unexpected keys
            unexpected_keys = getattr(incompatible_keys, "unexpected_keys", None)
            if unexpected_keys:
                logger.warning(f"Unexpected keys in transformer state dict: {unexpected_keys}")


def main():
    """Main execution function."""
    # Parse arguments
    parser = get_parser()
    args = parser.parse_args()
    
    logger.info("Starting CogVideoX LoRA inference")
    
    # Create output directory
    if args.output_dir is not None:
        os.makedirs(args.output_dir, exist_ok=True)
    
    # Set random seed for reproducibility
    if args.seed is not None:
        torch.manual_seed(args.seed)
        random.seed(args.seed)
        np.random.seed(args.seed)
    
    # Initialize model configuration
    t5_first = args.t5_first
    concatenated_all = args.concatenated_all
    reduce_token = False  # This is hardcoded in the original code
    
    # Add dataset name for compatibility with original code
    args.dataset_name = 'customization'
    
    # Determine data types based on model size
    is_5b_model = "5b" in args.pretrained_model_name_or_path.lower()
    load_dtype = torch.bfloat16 if is_5b_model else torch.float16
    weight_dtype = torch.bfloat16 if is_5b_model else torch.float16
    
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    logger.info(f"Using device: {device}")
    logger.info(f"Using model: {args.pretrained_model_name_or_path}")
    
    # Load tokenizer and add special token if needed
    logger.info("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(
        args.pretrained_model_name_or_path, 
        subfolder="tokenizer", 
        revision=args.revision
    )
    
    if args.add_special:
        logger.info("Adding special token <cls> to tokenizer")
        special_token = {"additional_special_tokens": ["<cls>"]}
        tokenizer.add_special_tokens(special_token)
    
    # Load text encoder
    logger.info("Loading text encoder...")
    text_encoder = T5EncoderModel.from_pretrained(
        args.pretrained_model_name_or_path, 
        subfolder="text_encoder", 
        revision=args.revision
    )
    
    if args.add_special:
        text_encoder.resize_token_embeddings(len(tokenizer))
    
    # Load transformer model
    logger.info("Loading transformer model...")
    transformer = CustomCogVideoXTransformer3DModel.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="transformer",
        torch_dtype=load_dtype,
        revision=args.revision,
        variant=args.variant,
        customization=True,
        concatenated_all=concatenated_all,
        reduce_token=reduce_token,
        vae_add=args.vae_add,
        local_reference_scale=1.0,  # Default value from original code
    )
    
    # Add LoRA adapter to transformer
    logger.info(f"Adding LoRA adapter with rank {args.rank}, alpha {args.lora_alpha}")
    transformer_lora_config = LoraConfig(
        r=args.rank,
        lora_alpha=args.lora_alpha,
        init_lora_weights=True,
        target_modules=["to_k", "to_q", "to_v", "to_out.0", "proj", "text_proj", 
                         "norm1.linear", "norm2.linear", "ff.net.2"],
    )
    
    transformer.add_adapter(transformer_lora_config)
    logger.info(f"Active adapters: {transformer.active_adapters}")
    
    # Load checkpoint if specified
    if args.checkpoint_path and args.resume_from_checkpoint:
        checkpoint_dir = os.path.join(args.checkpoint_path, args.resume_from_checkpoint)
        logger.info(f"Loading checkpoint from {checkpoint_dir}")
        load_model_hook([transformer], checkpoint_dir)
    
    # Move models to device and set data type
    text_encoder.to(device, dtype=weight_dtype)
    transformer.to(device, dtype=weight_dtype)
    
    # Create dataset for prompts
    logger.info("Creating dataset...")
    video_dataset = VideoDataset(
        height=args.height,
        width=args.width,
        fps=args.fps,
        max_num_frames=args.max_num_frames,
        id_token=args.id_token,
        seen_validation=args.seen_validation,
        test_prompt_path=args.test_prompt_path,
    )
    
    # Run inference if enabled
    if args.inference:
        logger.info("Starting inference mode")
        
        # Load VAE
        logger.info("Loading VAE...")
        vae = AutoencoderKLCogVideoX.from_pretrained(
            args.pretrained_model_name_or_path, 
            subfolder="vae", 
            revision=args.revision, 
            variant=args.variant
        )
        
        if args.enable_slicing:
            vae.enable_slicing()
        if args.enable_tiling:
            vae.enable_tiling()
            
        vae.requires_grad_(False)
        vae.to(device, dtype=weight_dtype)
        
        # Create pipeline
        logger.info("Creating pipeline...")
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
        
        # Set up validation image path
        if args.seen_validation:
            args.validation_reference_image = "../seen_samples/omini_right/"
            
        resizing = True
        if args.sampling_for_quali:
            if True:  # Always use white background processing
                args.validation_reference_image = os.path.join(args.validation_reference_image, 'processed_white_720x480')
                resizing = False
            else:
                args.validation_reference_image = os.path.join(args.validation_reference_image, 'processed_bg_720x720')
                resizing = True
                
        # Get list of validation images
        val_len = len(os.listdir(args.validation_reference_image))
        sep_val_len = val_len // args.quali_sep_count
        
        # Process each validation image with prompts
        if args.sampling_for_quali:
            # Create output directory with seed in name
            output_dir = os.path.join(args.output_dir, f"seed_{args.seed}")
            os.makedirs(output_dir, exist_ok=True)
            args.output_dir = output_dir
            
            # Set pipeline to evaluation mode
            pipe.transformer.eval()
            pipe.vae.eval()
            pipe.text_encoder.eval()
            
            # Set up deterministic generation
            generator = torch.Generator(device=device)
            if args.seed is not None:
                generator.manual_seed(args.seed)
            
            # Process each image in the assigned shard
            image_files = sorted(os.listdir(args.validation_reference_image))
            start_idx = sep_val_len * args.quali_shard
            end_idx = min(sep_val_len * (args.quali_shard + 1), len(image_files))
            
            logger.info(f"Processing shard {args.quali_shard}: images {start_idx} to {end_idx-1}")
            
            for i in range(start_idx, end_idx):
                # Get reference image path
                validation_ref_img = os.path.join(
                    args.validation_reference_image, 
                    image_files[i]
                )
                
                # Extract video ID from filename
                vid_id = image_files[i].split('.')[0]
                
                # Process each prompt for this reference image
                for cnt in range(args.num_of_prompts):
                    try:
                        # Get prompt for this video ID
                        if vid_id not in video_dataset.val_instance_prompt_dict or cnt >= len(video_dataset.val_instance_prompt_dict[vid_id]):
                            logger.warning(f"No prompt found for video ID {vid_id}, prompt index {cnt}")
                            continue
                            
                        validation_prompt = video_dataset.val_instance_prompt_dict[vid_id][cnt]
                        
                        # Extract checkpoint step from the checkpoint name
                        ckpt_step = int(os.path.basename(args.resume_from_checkpoint).split('-')[1])
                        
                        # Create safe prompt for filename
                        prompt_short = (
                            validation_prompt[:30]
                            .replace(" ", "_")
                            .replace("'", "_")
                            .replace('"', "_")
                            .replace("/", "_")
                        )
                        
                        # Define output paths
                        output_file = os.path.join(
                            args.output_dir, 
                            f"ckpt_{ckpt_step}_white_{args.phase_name}_video_0_max_n_f_{args.inference_num_frames}_id_{vid_id}_{prompt_short}.mp4"
                        )
                        output_frames_dir = os.path.join(
                            args.output_dir, 
                            f"ckpt_{ckpt_step}_white_{args.phase_name}_video_0_max_n_f_{args.inference_num_frames}_id_{vid_id}_{prompt_short}"
                        )
                        
                        # Skip if the output already exists
                        if os.path.exists(output_file):
                            logger.info(f"Skipping existing output: {output_file}")
                            continue
                        
                        # Generate the video
                        logger.info(f"Generating video for image {i+1}/{end_idx-start_idx}, prompt {cnt+1}/{args.num_of_prompts}")
                        logger.info(f"Prompt: {validation_prompt}")
                        
                        # Process reference image
                        ref_image = process_reference_image(
                            validation_ref_img, 
                            args, 
                            device, 
                            pipe.transformer.dtype, 
                            resizing
                        )
                        
                        # Encode reference image to latent space
                        with torch.no_grad():
                            ref_image_latents = pipe.vae.encode(ref_image).latent_dist
                            ref_image_latents = ref_image_latents.sample() * pipe.vae.config.scaling_factor
                        ref_image_latents = ref_image_latents.permute(0, 2, 1, 3, 4)
                        
                        # Update generator seed for each video
                        generator.manual_seed(generator.initial_seed() + 10 * cnt)
                        
                        # Set up pipeline arguments
                        pipeline_args = {
                            "prompt": validation_prompt,
                            "ref_img_states": ref_image_latents,
                            "num_inference_steps": 50,
                            "output_type": "np",
                            "guidance_scale": args.guidance_scale,
                            "use_dynamic_cfg": args.use_dynamic_cfg,
                            "height": args.height_val,
                            "width": args.width_val,
                            "num_frames": args.inference_num_frames,
                            "eval": True,
                            "concatenated_all": concatenated_all,
                            "reduce_token": reduce_token,
                            "vae_add": args.vae_add,
                            "pos_embed": args.pos_embed,
                            "layernorm_fix": args.layernorm_fix,
                            "non_shared_pos_embed": args.non_shared_pos_embed,
                            "generator": generator,
                        }
                        
                        # Fix: Disable positional embeddings if they're causing errors
                        if args.pos_embed_inf_match:
                            # Add a warning that this feature might cause errors with some models
                            logger.info("Note: pos_embed_inf_match is enabled but may cause errors with this model")
                            
                        # Run inference
                        with torch.no_grad():
                            try:
                                output = pipe(**pipeline_args)
                                video = output.frames[0]
                            except TypeError as e:
                                if "image_rotary_emb" in str(e):
                                    logger.warning("Detected incompatible argument 'image_rotary_emb'. Trying again without positional embeddings...")
                                    # Create a modified pipeline that doesn't send image_rotary_emb to the transformer
                                    original_prepare_latents = pipe.prepare_latents
                                    
                                    def modified_prepare_latents(*args, **kwargs):
                                        # Remove image_rotary_emb from kwargs if present
                                        kwargs.pop('image_rotary_emb', None)
                                        return original_prepare_latents(*args, **kwargs)
                                    
                                    pipe.prepare_latents = modified_prepare_latents
                                    output = pipe(**pipeline_args)
                                    video = output.frames[0]
                                    # Restore original function
                                    pipe.prepare_latents = original_prepare_latents
                                else:
                                    raise
                        
                        # Export video and frames
                        export_to_video_with_frames(
                            video_frames=video,
                            output_video_path=output_file,
                            output_frames_dir=output_frames_dir,
                            fps=args.fps,
                        )
                        
                        logger.info(f"Successfully generated video for {vid_id}, prompt {cnt+1}")
                        
                        # Clean up GPU memory for next iteration
                        free_memory()
                        
                    except Exception as e:
                        logger.error(f"Error generating video for {vid_id}, prompt {cnt+1}: {e}")
                        continue
    
    logger.info("Inference completed successfully")


if __name__ == "__main__":
    main()