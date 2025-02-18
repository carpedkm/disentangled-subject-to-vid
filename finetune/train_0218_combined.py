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

from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import DistributedDataParallelKwargs, ProjectConfiguration, set_seed
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
from diffusers.utils import check_min_version, convert_unet_state_dict_to_peft, export_to_video, is_wandb_available
from diffusers.utils.hub_utils import load_or_create_model_card, populate_model_card
from diffusers.utils.torch_utils import is_compiled_module

from concurrent.futures import ThreadPoolExecutor

if is_wandb_available():
    import wandb
    
    

# Will error if the minimal version of diffusers is not installed. Remove at your own risks.
check_min_version("0.31.0.dev0")

logger = get_logger(__name__)

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
        "--dataset_config_name",
        type=str,
        default=None,
        help="The config of the Dataset, leave as None if there's only one config.",
    )
    parser.add_argument(
        "--instance_data_root",
        type=str,
        default=None,
        help=("A folder containing the training ref image data."),
    )
    parser.add_argument(
        "--anno_root",
        type=str,
        default=None,
        help=("A folder containing the training annotation data."),
    )
    # parser.add_argument(
    #     "--video_column",
    #     type=str,
    #     default="video",
    #     help="The column of the dataset containing videos. Or, the name of the file in `--instance_data_root` folder containing the line-separated path to video data.",
    # )
    # parser.add_argument(
    #     "--caption_column",
    #     type=str,
    #     default="text",
    #     help="The column of the dataset containing the instance prompt for each video. Or, the name of the file in `--instance_data_root` folder containing the line-separated instance prompts.",
    # )
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
        "--validation_prompt",
        type=str,
        default='True',
        help="One or more prompt(s) that is used during validation to verify that the model is learning. Multiple validation prompts should be separated by the '--validation_prompt_seperator' string.",
    )
    parser.add_argument(
        "--validation_prompt_separator",
        type=str,
        default=":::",
        help="String that separates multiple validation prompts",
    )
    parser.add_argument(
        "--num_validation_videos",
        type=int,
        default=2,
        help="Number of videos that should be generated during validation per `validation_prompt`.",
    )
    parser.add_argument(
        "--validation_epochs",
        type=int,
        default=1,
        help=(
            "Run validation every X epochs. Validation consists of running the prompt `args.validation_prompt` multiple times: `args.num_validation_videos`."
        ),
    )
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
        "--mixed_precision",
        type=str,
        default=None,
        choices=["no", "fp16", "bf16"],
        help=(
            "Whether to use mixed precision. Choose between fp16 and bf16 (bfloat16). Bf16 requires PyTorch >="
            " 1.10.and an Nvidia Ampere GPU.  Default to the value of accelerate config of the current system or the"
            " flag passed with the `accelerate.launch` command. Use this argument to override the accelerate config."
        ),
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
        "--random_flip",
        action="store_true",
        help="whether to randomly flip videos horizontally",
    )
    parser.add_argument(
        "--train_batch_size", type=int, default=4, help="Batch size (per device) for the training dataloader."
    )
    parser.add_argument("--num_train_epochs", type=int, default=1)
    parser.add_argument(
        "--max_train_steps",
        type=int,
        default=None,
        help="Total number of training steps to perform. If provided, overrides `--num_train_epochs`.",
    )
    parser.add_argument(
        "--checkpointing_steps",
        type=int,
        default=500,
        help=(
            "Save a checkpoint of the training state every X updates. These checkpoints can be used both as final"
            " checkpoints in case they are better than the last checkpoint, and are also suitable for resuming"
            " training using `--resume_from_checkpoint`."
        ),
    )
    parser.add_argument(
        "--checkpoints_total_limit",
        type=int,
        default=None,
        help=("Max number of checkpoints to store."),
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
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )
    parser.add_argument(
        "--gradient_checkpointing",
        action="store_true",
        help="Whether or not to use gradient checkpointing to save memory at the expense of slower backward pass.",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=1e-4,
        help="Initial learning rate (after the potential warmup period) to use.",
    )
    parser.add_argument(
        "--scale_lr",
        action="store_true",
        default=False,
        help="Scale the learning rate by the number of GPUs, gradient accumulation steps, and batch size.",
    )
    parser.add_argument(
        "--lr_scheduler",
        type=str,
        default="constant",
        help=(
            'The scheduler type to use. Choose between ["linear", "cosine", "cosine_with_restarts", "polynomial",'
            ' "constant", "constant_with_warmup"]'
        ),
    )
    parser.add_argument(
        "--lr_warmup_steps", type=int, default=500, help="Number of steps for the warmup in the lr scheduler."
    )
    parser.add_argument(
        "--lr_num_cycles",
        type=int,
        default=1,
        help="Number of hard resets of the lr in cosine_with_restarts scheduler.",
    )
    parser.add_argument("--lr_power", type=float, default=1.0, help="Power factor of the polynomial scheduler.")
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

    # Optimizer
    parser.add_argument(
        "--optimizer",
        type=lambda s: s.lower(),
        default="adam",
        choices=["adam", "adamw", "prodigy"],
        help=("The optimizer type to use."),
    )
    parser.add_argument(
        "--use_8bit_adam",
        action="store_true",
        help="Whether or not to use 8-bit Adam from bitsandbytes. Ignored if optimizer is not set to AdamW",
    )
    parser.add_argument(
        "--adam_beta1", type=float, default=0.9, help="The beta1 parameter for the Adam and Prodigy optimizers."
    )
    parser.add_argument(
        "--adam_beta2", type=float, default=0.95, help="The beta2 parameter for the Adam and Prodigy optimizers."
    )
    parser.add_argument(
        "--prodigy_beta3",
        type=float,
        default=None,
        help="Coefficients for computing the Prodigy optimizer's stepsize using running averages. If set to None, uses the value of square root of beta2.",
    )
    parser.add_argument("--prodigy_decouple", action="store_true", help="Use AdamW style decoupled weight decay")
    parser.add_argument("--adam_weight_decay", type=float, default=1e-04, help="Weight decay to use for unet params")
    parser.add_argument(
        "--adam_epsilon",
        type=float,
        default=1e-08,
        help="Epsilon value for the Adam optimizer and Prodigy optimizers.",
    )
    parser.add_argument("--max_grad_norm", default=1.0, type=float, help="Max gradient norm.")
    parser.add_argument("--prodigy_use_bias_correction", action="store_true", help="Turn on Adam's bias correction.")
    parser.add_argument(
        "--prodigy_safeguard_warmup",
        action="store_true",
        help="Remove lr from the denominator of D estimate to avoid issues during warm-up stage.",
    )

    # Other information
    parser.add_argument("--tracker_name", type=str, default=None, help="Project tracker name")
    parser.add_argument("--push_to_hub", action="store_true", help="Whether or not to push the model to the Hub.")
    parser.add_argument("--hub_token", type=str, default=None, help="The token to use to push to the Model Hub.")
    parser.add_argument(
        "--hub_model_id",
        type=str,
        default=None,
        help="The name of the repository to keep in sync with the local `output_dir`.",
    )
    parser.add_argument(
        "--logging_dir",
        type=str,
        default="logs",
        help="Directory where logs are stored.",
    )
    parser.add_argument(
        "--allow_tf32",
        action="store_true",
        help=(
            "Whether or not to allow TF32 on Ampere GPUs. Can be used to speed up training. For more information, see"
            " https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices"
        ),
    )
    parser.add_argument(
        "--report_to",
        type=str,
        default=None,
        help=(
            'The integration to report the results and logs to. Supported platforms are `"tensorboard"`'
            ' (default), `"wandb"` and `"comet_ml"`. Use `"all"` to report to all integrations.'
        ),
    )
    parser.add_argument(
        '--t5_first',
        action='store_true',
        help='Whether to concatenate the prompt of the t5 encoder with clip encoders first or not'
    )
    parser.add_argument(
        '--subset_cnt',
        type=int,
        default=-1,
        help='Number of videos to use for training'
    )
    parser.add_argument(
        '--concatenated_all',
        action='store_true',
        help='Whether to concatenate encoders and use the concatenated feature itself'
    )
    parser.add_argument(
        '--reduce_token',
        action='store_true',
        help='Whether to reduce the token embeddings'
    )
    parser.add_argument(
        '--add_token',
        action='store_true',
        help='Whether to add the token embeddings'
    )
    
    parser.add_argument(
        '--zero_conv_add',
        action='store_true',
        help='Whether to use zero conv'
    )
    
    parser.add_argument(
        '--cross_pairs',
        action='store_true',
        help='Whether to use cross pairs or not'
    )
    
    parser.add_argument(
        '--sub_driven',
        action='store_true',
        help='Whether to use sub driven or not'
    )
    parser.add_argument(
        '--wo_bg',
        action='store_true',
        help='Whether to use without bg or not'
    )
    parser.add_argument(
        '--use_latent',
        action='store_true',
        help='Whether to use latent directly loaded'
    )
    parser.add_argument(
        '--latent_data_root',
        type=str,
        default=None,
    )
    parser.add_argument(
        '--quick_poc_subset',
        action='store_true',
        help='Whether to use quick poc subset or not'
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
        '--load_to_ram',
        action='store_true',
        help='Whether to load the dataset to RAM or not'
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
        '--cross_attend',
        action='store_true',
        help='Whether to use cross attention or not'
    )
    parser.add_argument(
        '--cross_attend_text',
        action='store_true',
        help='Whether to use cross attention on text or not'
    )
    parser.add_argument(
        '--cross_attn_interval',
        type=int,
        default=2,
        help='Whether to use cross attention interval or not'
    )
    parser.add_argument(
        '--local_reference_scale',
        type=float,
        default=1.,
    )
    parser.add_argument(
        '--cross_attn_dim_head',
        type=int,
        default=128,
    )
    parser.add_argument(
        '--cross_attn_num_head',
        type=int,
        default=16,
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
        '--qk_replace',
        action='store_true',
        help='Whether to replace qk or not'
    )
    parser.add_argument(
        '--input_noise_fix',
        action='store_true',
        help='Whether to use input noise fix or not'
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
        '--text_only_norm_final',
        action='store_true',
        help='Whether to norm final feature or not'
    )
    parser.add_argument(
        '--pos_embed_inf_match',
        action='store_true',
        help='Whether to use positional embedding for inference match or not'
    )
    parser.add_argument(
        '--second_stage',
        action='store_true',
        help='Whether to use second stage video training or not'
    )
    parser.add_argument(
        '--second_stage_ref_image',
        action='store_true',
        help='Whether to use second stage ref image or not'
    )
    parser.add_argument(
        '--video_anno',
        type=str,
        default=None,
    )
    parser.add_argument(
        '--video_instance_root',
        type=str,
        default=None,
    )
    parser.add_argument(
        '--non_shared_pos_embed',
        action='store_true',
        help='Whether to use non shared positional embedding or not'
    )
    parser.add_argument(
        '--random_pos',
        action='store_true',
        help='Whether to use random positional embedding or not'
    )
    parser.add_argument(
        '--video_ref_root',
        type=str,
        default=None,
    )
    parser.add_argument(
        '--joint_train',
        action='store_true',
        help='Whether to use joint training or not'
    )
    parser.add_argument(
        '--prob_sample_video',
        type=float,
        default=0.05,
    )
    return parser.parse_args()



class ZeroConv1D(nn.Module):
    def __init__(self, in_dim=512, out_dim=4096):
        super(ZeroConv1D, self).__init__()
        self.zero_conv = nn.Conv1d(in_channels=in_dim, out_channels=out_dim, kernel_size=1)
        nn.init.zeros_(self.zero_conv.weight)
        nn.init.zeros_(self.zero_conv.bias)
    
    def forward(self, x):
        # x: [batch_size, in_dim, seq_len] -> [batch_size, out_dim, seq_len]
        return self.zero_conv(x)

class SequenceAligner(nn.Module):
    def __init__(self, clip_seq_len=77, t5_seq_len=226):
        super(SequenceAligner, self).__init__()
        self.clip_seq_len = clip_seq_len
        self.t5_seq_len = t5_seq_len

    def forward(self, clip_features):
        """
        Align CLIP sequence length to match T5.
        clip_features: [batch_size, 77, 512]
        Returns: [batch_size, 226, 512]
        """
        batch_size, _, dim = clip_features.shape
        # Interpolate to match T5 sequence length
        clip_features = F.interpolate(clip_features.transpose(1, 2), size=self.t5_seq_len, mode='linear')
        return clip_features.transpose(1, 2)  # Back to [batch_size, seq_len, dim]
class SkipProjectionLayer(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.projection = nn.Linear(in_features, out_features)

    def forward(self, x):
        return x + self.projection(x)

class ReduceProjectionLayer(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.projection = nn.Linear(in_features, out_features)
    
    def forward(self, x):
        return self.projection(x)
    
class ProjectionLayer(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.projection = nn.Linear(in_features, out_features)

    def forward(self, x):
        return self.projection(x)

class PerceiverCrossAttention(nn.Module):
    def __init__(self, dim: int = 3072, dim_head: int = 128, heads: int = 16, kv_dim: int = 2048):
        super().__init__()

        self.scale = dim_head**-0.5
        self.dim_head = dim_head
        self.heads = heads
        inner_dim = dim_head * heads

        # Layer normalization to stabilize training
        self.norm1 = nn.LayerNorm(dim if kv_dim is None else kv_dim)
        self.norm2 = nn.LayerNorm(dim)

        # Linear transformations to produce queries, keys, and values
        self.to_q_= nn.Linear(dim, inner_dim, bias=False)
        self.to_kv_ = nn.Linear(dim if kv_dim is None else kv_dim, inner_dim * 2, bias=False)
        self.to_out_ = nn.Linear(inner_dim, dim, bias=False)

    def forward(self, image_embeds: torch.Tensor, hidden_states: torch.Tensor) -> torch.Tensor:
        # Apply layer normalization to the input image and latent features
        image_embeds = self.norm1(image_embeds)
        hidden_states = self.norm2(hidden_states)

        batch_size, seq_len, _ = hidden_states.shape

        # Compute queries, keys, and values
        query = self.to_q_(hidden_states)
        key, value = self.to_kv_(image_embeds).chunk(2, dim=-1)

        # Reshape tensors to split into attention heads
        query = query.reshape(query.size(0), -1, self.heads, self.dim_head).transpose(1, 2)
        key = key.reshape(key.size(0), -1, self.heads, self.dim_head).transpose(1, 2)
        value = value.reshape(value.size(0), -1, self.heads, self.dim_head).transpose(1, 2)

        # Compute attention weights
        scale = 1 / math.sqrt(math.sqrt(self.dim_head))
        weight = (query * scale) @ (key * scale).transpose(-2, -1)  # More stable scaling than post-division
        weight = torch.softmax(weight.float(), dim=-1).type(weight.dtype)

        # Compute the output via weighted combination of values
        out = weight @ value

        # Reshape and permute to prepare for final linear transformation
        out = out.permute(0, 2, 1, 3).reshape(batch_size, seq_len, -1)

        return self.to_out_(out)   

class QFormerAligner(nn.Module):
    def __init__(self, model_name):
        super().__init__()
        model_name = "Salesforce/blip2-opt-2.7b"
        self.processor = Blip2Processor.from_pretrained(model_name)
        self.model = Blip2Model.from_pretrained(model_name)
        self.qformer = self.model.qformer
        self.fc = nn.Linear(768, 3072)
        
    def forward(self, image_feature):
        pixel_values = self.processor.image_processor(image_feature.float(), return_tensors='pt')["pixel_values"]
        device = next(self.model.parameters()).device
        pixel_values = pixel_values.to(device)
        with torch.no_grad():
            vision_outputs = self.model.vision_model(pixel_values)
            image_embeds = vision_outputs.last_hidden_state
        
        batch_size = image_embeds.shape[0]
        query_tokens = self.model.query_tokens.expand(batch_size, -1, -1)
        
        qformer_outputs = self.qformer(query_embeds=query_tokens,
                                       encoder_hidden_states=image_embeds,
                                       )
        qformer_features = qformer_outputs.last_hidden_state
        qformer_features = self.fc(qformer_features)
        # get it back to bfloat16
        return qformer_features.bfloat16()
    
class VideoDataset(Dataset):
    def __init__(
        self,
        video_instance_root: Optional[str] = None,
        video_anno: Optional[str] = None,
        video_ref_root: Optional[str] = None,
        height: int = 480,
        width: int = 720,
        fps: int = 8,
        max_num_frames: int = 49,
        id_token: Optional[str] = None,
        seen_validation: bool = False,
        joint_train: bool = False,
        image_dataset_len: int = 0,
    ) -> None:
        self.seen_validation = seen_validation
        self.id_token = id_token or ""
        self.joint_train = joint_train
        self.image_dataset_len = image_dataset_len
        super().__init__()
        with open(video_anno, 'r') as f:
            self.video_dict = json.load(f)
        id_mapper = {}
        for id_ in list(self.video_dict.keys()):
            id_splitted = id_.split('/')[-1]
            id_mapper[id_splitted] = self.video_dict[id_]
        self.video_path_dict = {}
        for id_ in list(id_mapper.keys()):
            self.video_path_dict[id_] = os.path.join(video_instance_root, id_ + '_vae_latents.npy')
        self.video_ref_path_dict = {}
        for id_ in list(id_mapper.keys()):
            self.video_ref_path_dict[id_] = os.path.join(video_ref_root, id_ + '_vae_latents.npy')
        self.prompt_dict = {}
        for id_ in list(id_mapper.keys()):
            self.prompt_dict[id_] = self.id_token + id_mapper[id_]['text']
            
        self.prefix = "<cls> "

        self.val_instance_prompt_dict = {
                                    'oranges_omini':"A close up view. A bowl of oranges are placed on a wooden table. The background is a dark room, the TV is on, and the screen is showing a cooking show. ", 
                                    'clock_omini':"In a Bauhaus style room, the clock is placed on a shiny glass table, with a vase of flowers next to it. In the afternoon sun, the shadows of the blinds are cast on the wall.",
                                    'rc_car_omini': "A film style shot. On the moon, toy car goes across the moon surface. The background is that Earth looms large in the foreground.",
                                    'shirt_omini': "On the beach, a lady sits under a beach umbrella. She's wearing hawaiian shirt and has a big smile on her face, with her surfboard hehind her. The sun is setting in the background. The sky is a beautiful shade of orange and purple.",
                                    'cat' : "cat is rollerblading in the park",
                                    'dog' : 'dog is flying in the sky',
                                    'red_toy' : 'red toy is dancing in the room',
                                    'dog_toy' : 'dog toy is walking around the grass',
                                }
        
        if self.seen_validation is True:
            self.val_instance_prompt_dict = {}
            path_for_seen_meta = '../seen_samples/omini_meta/'
            for file in os.listdir(path_for_seen_meta):
                with open(os.path.join(path_for_seen_meta, file), 'r') as f:
                    meta_seen = json.load(f)
                id_ = 'right_' + file.split('_')[1].split('.')[0]
                # id_  = file.split('.')[0]
                tmp_desc = meta_seen['description_0']
                self.val_instance_prompt_dict[id_] = tmp_desc

        self.val_instance_prompt_dict = {k: self.prefix + v for k, v in self.val_instance_prompt_dict.items()}
        

    def __len__(self):
        return len(self.video_path_dict)
 
    def __getitem__(self, index):
        if self.joint_train is True:
            index = index - self.image_dataset_len
        while True:

            try:
                id_key = list(self.video_path_dict.keys())[index]
                video_path_to_load = self.video_path_dict[id_key]
                prompt_loaded = self.prompt_dict[id_key]
                
                np_loaded = torch.from_numpy(np.load(video_path_to_load))
                ref_loaded = torch.from_numpy(np.load(self.video_ref_path_dict[id_key]))
                # what is the shape of np_loaded?
                
                
                return {
                    "instance_prompt": prompt_loaded,
                    "instance_ref_image": ref_loaded,
                    "instance_video": np_loaded,
                }
            except Exception as e:
                print(f"Error loading video {id_key}: {e}")
                index = (index + 1) % len(self.video_path_dict)
            
            
        
class ImageDataset(Dataset):
    def __init__(
        self,
        instance_data_root: Optional[str] = None,
        anno_path: Optional[str] = None,
        dataset_name: Optional[str] = None,
        height: int = 512,
        width: int = 512,
        cache_dir: Optional[str] = None,
        id_token: Optional[str] = None,
        subset_cnt: int = -1,
        cross_pairs: bool = False,
        latent_data_root: str = None,
        use_latent: bool = False,
        wo_bg : bool = False,
        vae_add : bool = False,
        cross_attend : bool = False,
        cross_attend_text: bool = False,
        load_to_ram : bool = False,
        seen_validation : bool = False, 
        add_special: bool = False,
        add_multiple_special: bool = False,
        add_specific_loc: bool = False,
        wo_shuffle: bool = False,
        add_new_split: bool = False,
        qk_replace: bool = False,
        qformer: bool = False,
    ) -> None:
        super().__init__()
        print('Data loader init')
        self.vae_add = vae_add
        self.qk_replace = qk_replace
        self.cross_attend = cross_attend
        self.cross_attend_text = cross_attend_text
        self.qformer = qformer
        self.use_latent = use_latent
        self.instance_data_root = Path(instance_data_root) if instance_data_root is not None else None
        self.seen_validation = seen_validation
        self.height = height
        self.width = width
        self.add_new_split=add_new_split
        self.wo_shuffle=wo_shuffle
        if add_multiple_special:
            self.prefix = "<cls> <a> <b> <c> "
        else:
            self.prefix = "<cls> " 
        # self.val_instance_prompt_dict = {'oranges_omini':"A close up view of the item. It is placed on a wooden table. The background is a dark room, the TV is on, and the screen is showing a cooking show. ", 
        #                                  'clock_omini':"In a Bauhaus style room, the item is placed on a shiny glass table, with a vase of flowers next to it. In the afternoon sun, the shadows of the blinds are cast on the wall.",
        #                                  'rc_car_omini': "A film style shot. On the moon, this item goes across the moon surface. The background is that Earth looms large in the foreground.",
        #                                  'shirt_omini': "On the beach, a lady sits under a beach umbrella with 'Omini' written on it. She's wearing this item and has a big smile on her face, with her surfboard hehind her. The sun is setting in the background. The sky is a beautiful shade of orange and purple.",
        #                                 #  "bag_omini": "A boy is wearing this item inside a beautiful park, walking along the lake."}
        #                                 }
        if add_special and not add_multiple_special:
            self.val_instance_prompt_dict = {
                                            'oranges_omini':"A close up view. A <cls> of oranges are placed on a wooden table. The background is a dark room, the TV is on, and the screen is showing a cooking show. ", 
                                            'clock_omini':"In a Bauhaus style room, the <cls> clock is placed on a shiny glass table, with a vase of flowers next to it. In the afternoon sun, the shadows of the blinds are cast on the wall.",
                                            'rc_car_omini': "A film style shot. On the moon, <cls> toy car goes across the moon surface. The background is that Earth looms large in the foreground.",
                                            'shirt_omini': "On the beach, a lady sits under a beach umbrella. She's wearing <cls> hawaiian shirt and has a big smile on her face, with her surfboard hehind her. The sun is setting in the background. The sky is a beautiful shade of orange and purple.",
                                            'cat' : "<cls> cat is rollerblading in the park",
                                            'dog' : '<cls> dog is flying in the sky',
                                            'red_toy' : '<cls> red toy is dancing in the room',
                                            'dog_toy' : '<cls> dog toy is walking around the grass',
                                            
                                            #  "bag_omini": "A boy is wearing this item inside a beautiful park, walking along the lake."}
                                            }
        self.val_instance_prompt_dict = {
                                    'oranges_omini':"A close up view. A bowl of oranges are placed on a wooden table. The background is a dark room, the TV is on, and the screen is showing a cooking show. ", 
                                    'clock_omini':"In a Bauhaus style room, the clock is placed on a shiny glass table, with a vase of flowers next to it. In the afternoon sun, the shadows of the blinds are cast on the wall.",
                                    'rc_car_omini': "A film style shot. On the moon, toy car goes across the moon surface. The background is that Earth looms large in the foreground.",
                                    'shirt_omini': "On the beach, a lady sits under a beach umbrella. She's wearing hawaiian shirt and has a big smile on her face, with her surfboard hehind her. The sun is setting in the background. The sky is a beautiful shade of orange and purple.",
                                    'cat' : "cat is rollerblading in the park",
                                    'dog' : 'dog is flying in the sky',
                                    'red_toy' : 'red toy is dancing in the room',
                                    'dog_toy' : 'dog toy is walking around the grass',
                                }
        
        if self.seen_validation is True:
            self.val_instance_prompt_dict = {}
            path_for_seen_meta = '../seen_samples/omini_meta/'
            for file in os.listdir(path_for_seen_meta):
                with open(os.path.join(path_for_seen_meta, file), 'r') as f:
                    meta_seen = json.load(f)
                id_ = 'right_' + file.split('_')[1].split('.')[0]
                # id_  = file.split('.')[0]
                tmp_desc = meta_seen['description_0']
                self.val_instance_prompt_dict[id_] = tmp_desc
        if add_special:
            self.val_instance_prompt_dict = {k: self.prefix + v for k, v in self.val_instance_prompt_dict.items()}
        
        self.instance_prompts = []
        self.id_token = id_token or ""

        self.instance_left_pixel_root = os.path.join(str(self.instance_data_root), 'left_images_updated')
        self.instance_right_pixel_root = os.path.join(str(self.instance_data_root), 'right_images_updated')
        
        if self.add_new_split is True:
            print('>> Accessing additional data')
            self.additional_instance_root = os.path.join(str(self.instance_data_root), 'omini200k_720p_new_1024_renamed')
            self.instance_left_pixel_root_additional = os.path.join(self.additional_instance_root, 'left_images_updated')
            self.instance_right_pixel_root_additional = os.path.join(self.additional_instance_root, 'right_images_updated')
            
        self.dataset_name = dataset_name
        
        self.load_to_ram = load_to_ram
        # print('>> Accessing additional data Done')
        print('Get list for image IDs')
        left_ids = os.listdir(self.instance_left_pixel_root)
        left_ids = [int(id.split('_')[1].split('.')[0]) for id in tqdm(left_ids)]

        right_ids = os.listdir(self.instance_right_pixel_root)
        right_ids = [int(id.split('_')[1].split('.')[0]) for id in tqdm(right_ids)]
        
        if self.add_new_split is True:
            additional_left_ids = os.listdir(self.instance_left_pixel_root_additional)
            additional_left_ids = ['add_' + str(int(id.split('_')[2].split('.')[0])) for id in tqdm(additional_left_ids)] # add_0
            additional_right_ids = os.listdir(self.instance_right_pixel_root_additional)
            additional_right_ids = ['add_' + str(int(id.split('_')[2].split('.')[0])) for id in tqdm(additional_right_ids)] # add_0
            
        
        # check if there exists non-existing ids in right_ids
        print("CHECKING MISSING IDS")
        left_ids_set = set(left_ids)
        right_ids_set = set(right_ids)

        if self.add_new_split is True:
            left_ids_set_add = set(additional_left_ids)
            right_ids_set_add = set(additional_right_ids)
        
        # Find ids in left_ids that are not in right_ids
        no_right_ids = left_ids_set - right_ids_set
        for id in no_right_ids:
            print('>> NO RIGHT ID', id)
            left_ids.remove(id)
            
        # additoinal check
        if self.add_new_split is True:
            no_right_ids_add = left_ids_set_add - right_ids_set_add
            for id in no_right_ids_add:
                print('>> NO RIGHT ID ADD', id)
                additional_left_ids.remove(id)

        # Find ids in right_ids that are not in left_ids
        no_left_ids = right_ids_set - left_ids_set
        for id in no_left_ids:
            print('>> NO LEFT ID', id)
            right_ids.remove(id)
    
        # additoinal check
        if self.add_new_split is True:
            no_left_ids_add = right_ids_set_add - left_ids_set_add
            for id in no_left_ids_add:
                print('>> NO LEFT ID ADD', id)
                additional_right_ids.remove(id)
        
        assert set(left_ids) == set(right_ids) # what about now? -> same ids in both left and right 
        if self.add_new_split is True:
            assert set(additional_left_ids) == set(additional_right_ids) # check for additional_ids
            add_ids = additional_left_ids
        else:
            add_ids = []
        ids = left_ids + add_ids
        # add_ids = additional_left_ids
        # self.ids = ids
        # randomly select ids
        if subset_cnt != -1:
            self.ids = random.sample(ids, subset_cnt)
        else:
            self.ids = ids
        self.len_dataset = len(self.ids)
                
        self.instance_left_pixel_root_map_with_id = {}
        self.instance_right_pixel_root_map_with_id = {}
        
        if self.add_new_split is True:
            self.instance_left_pixel_root_map_with_id_add = {}
            self.instance_right_pixel_root_map_with_id_add = {}
        
        print("MAPPING PIXEL IDS")
        for id in tqdm(self.ids):
            if 'add' in str(id):
                self.instance_left_pixel_root_map_with_id[id] = os.path.join(self.instance_left_pixel_root_additional, f'left_{id}.png')
                self.instance_right_pixel_root_map_with_id[id] = os.path.join(self.instance_right_pixel_root_additional, f'right_{id}.png')
            else:
                self.instance_left_pixel_root_map_with_id[id] = os.path.join(self.instance_left_pixel_root, f'left_{id}.png')
                self.instance_right_pixel_root_map_with_id[id] = os.path.join(self.instance_right_pixel_root, f'right_{id}.png')
        

        
        self.instance_left_latent_root = os.path.join(str(self.instance_data_root), 'left_latents_rgb_full')
        self.instance_right_latent_root = os.path.join(str(self.instance_data_root), 'right_latents_rgb_full')
        
        if self.add_new_split is True:
            self.instance_left_latent_root_additional = os.path.join(self.additional_instance_root, 'left_latents')
            self.instance_right_latent_root_additional = os.path.join(self.additional_instance_root, 'right_latents')
        
        self.instance_left_latent_root_map_with_id = {}
        self.instance_right_latent_root_map_with_id = {}
        print("MAPPING LATENT IDS")
        for id in tqdm(self.ids):
            if 'add' in str(id):
                self.instance_left_latent_root_map_with_id[id] = os.path.join(self.instance_left_latent_root_additional, f'left_{id}_vae_latents.npy')
                self.instance_right_latent_root_map_with_id[id] = os.path.join(self.instance_right_latent_root_additional, f'right_{id}_vae_latents.npy')
            else:
                self.instance_left_latent_root_map_with_id[id] = os.path.join(self.instance_left_latent_root, f'left_{id}_vae_latents.npy')
                self.instance_right_latent_root_map_with_id[id] = os.path.join(self.instance_right_latent_root, f'right_{id}_vae_latents.npy')
        
        
        # self.anno_path = os.path.join(str(self.instance_data_root), 'metadata')
        self.anno_path = anno_path
        self.instance_prompts_0 = {}
        self.instance_prompts_1 = {}
        
        print("LOADING METADATA")
        # for id in tqdm(self.ids):
        #     meta_path = os.path.join(self.anno_path, f'meta_{id}.json')
        #     with open(meta_path, 'r') as f:
        #         meta = json.load(f)
        #     self.instance_prompts_0[id] = self.prefix + meta['description_0']
        #     self.instance_prompts_1[id] = self.prefix + meta['description_1']
        with open(self.anno_path, 'r') as f:
            metadata = json.load(f)
        for id in tqdm(self.ids):
            meta = metadata[str(id)]
            if add_special:
                if add_specific_loc:
                    # First find out the 'item' inside the metadata and get the location of the 'item' inside description and then add <cls> in front of the 'item'
                    item_name = meta['item']
                    # find item name regardless of capitalization
                    item_name_lower = item_name.lower()
                    desc_lower = meta['description_0'].lower()

                    if item_name_lower in desc_lower:
                        item_loc = desc_lower.find(item_name_lower)
                        self.instance_prompts_0[id] = (
                            meta['description_0'][:item_loc] + ' ' + self.prefix + meta['description_0'][item_loc:]
                        )
                        # print(f">>>>>> Item name: {item_name}, Item location: {item_loc}, Description: {self.instance_prompts_0[id]}")
                    else:
                        self.instance_prompts_0[id] = self.prefix + meta['description_0']
                    desc_lower = meta['description_1'].lower()
                    # if meta['description_1'].find(item_name) != -1:
                    if item_name_lower in desc_lower:
                        item_loc = desc_lower.find(item_name_lower)
                        self.instance_prompts_1[id] = (
                            meta['description_1'][:item_loc] + ' ' + self.prefix + meta['description_1'][item_loc:]
                        )
                        # print(f">>>>>> Item name: {item_name}, Item location: {item_loc}, Description: {self.instance_prompts_1[id]}")
                    else:
                        self.instance_prompts_1[id] = self.prefix + meta['description_1']
                    
                else:
                    self.instance_prompts_0[id] = self.prefix + meta['description_0']
                    self.instance_prompts_1[id] = self.prefix + meta['description_1']

            else:
                self.instance_prompts_0[id] = meta['description_0']
                self.instance_prompts_1[id] = meta['description_1']
        
        if self.load_to_ram is True:
                self.instance_left_latent_root_map_with_id = {}
                self.instance_right_latent_root_map_with_id = {}
                print("LOAD INTO RAM")
                for id in tqdm(self.ids):
                    self.instance_left_latent_root_map_with_id[id] = np.load(os.path.join(self.instance_left_latent_root, f'left_{id}_vae_latents.npy'))
                    self.instance_right_latent_root_map_with_id[id] = np.load(os.path.join(self.instance_right_latent_root, f'right_{id}_vae_latents.npy'))
        

        
    def __getitem__(self, index):
        while True:
            if self.wo_shuffle:
                target = 1
            else:
                target = index % 2
            index = self.ids[index % self.len_dataset]
            # print(index)
            try:
                if not self.load_to_ram:
                    if target == 0: # left : condition / right : target
                        prompt = self.id_token + self.instance_prompts_1[index]
                        if self.vae_add or self.cross_attend or self.cross_attend_text or self.qk_replace:
                            if not self.load_to_ram:
                                image = torch.from_numpy(np.load(self.instance_left_latent_root_map_with_id[index]))
                            else:
                                image = torch.from_numpy(self.instance_left_latent_root_map_with_id[index])
                        else:
                            image = self._process_single_ref_image(self.instance_left_pixel_root_map_with_id[index])
                        # image = image.unsqueeze(0) # deal it as single-frame video
                        if not self.load_to_ram:
                            latent = torch.from_numpy(np.load(self.instance_right_latent_root_map_with_id[index])) # already expanded //// expand to deal it as single frame video of shape [seq_len, height, width , 3] of seq_len = 1
                        else:
                            latent = torch.from_numpy(self.instance_right_latent_root_map_with_id[index])
                        # latent = latent.unsqueeze(0)
                    else: # left : target / right : condition
                        prompt = self.id_token + self.instance_prompts_0[index]
                        if self.vae_add or self.cross_attend or self.cross_attend_text or self.qk_replace:
                            if not self.load_to_ram:
                                image = torch.from_numpy(np.load(self.instance_right_latent_root_map_with_id[index]))
                            else:
                                image = torch.from_numpy(self.instance_right_latent_root_map_with_id[index])
                        else:
                            image = self._process_single_ref_image(self.instance_right_pixel_root_map_with_id[index])
                        # image = image.unsqueeze(0) # deal it as single-frame video
                        
                        if not self.load_to_ram:
                            latent = torch.from_numpy(np.load(self.instance_left_latent_root_map_with_id[index]))
                        else:
                            latent = torch.from_numpy(self.instance_left_latent_root_map_with_id[index])
                # latent is preprocessed from cv2.imread, so convert BGR to RGB
                # latent = latent[...,::-1]
                    # latent = latent.unsqueeze(0)
                return {
                    "instance_prompt": prompt,
                    "instance_video": latent,
                    "instance_ref_image": image,
                }

            except Exception as e:
                print('>> ERROR', e)
                # change to other random video 
                index = (index + 1) % self.len_dataset
    def __len__(self):
        return self.len_dataset
                
    def _process_single_ref_image(self, ref_image_path):
        try:
            from PIL import Image
            import numpy as np
        except ImportError:
            raise ImportError(
                "The `PIL`, `numpy` package is required for loading the reference images. Install with `pip install PIL numpy`"
            )
        image = Image.open(ref_image_path) # [height, width, 3]
        image = np.array(image.resize((self.width, self.height)))
        # print(f"Image shape after resize: {image.shape}")  # Add this line
        image = torch.from_numpy(image).permute(2, 0, 1).float() 
        # shape : [3, height, width]
        # print(f"Image tensor shape after permute: {image.shape}")  
        return image

class CombinedDataset(Dataset):
    def __init__(self, video_dataset, image_dataset, p=0.1):
        """
        Args:
            video_dataset: The video dataset object.
            image_dataset: The image dataset object.
            p: Probability of selecting from video dataset (default is 0.1).
        """
        self.video_dataset = video_dataset
        self.image_dataset = image_dataset
        self.val_instance_prompt_dict = self.image_dataset.val_instance_prompt_dict
        self.dataset_name = self.image_dataset.dataset_name
    def __len__(self):
        return len(self.video_dataset) + len(self.image_dataset)

    def __getitem__(self, idx):
        if isinstance(idx, list):
            return [self.__get_single_item(i) for i in idx]
        else:
            return self.__get_single_item(idx)

    def __get_single_item(self, idx):
        # get random seed set with batch_idx
        if idx >= len(self.image_dataset):
            return self.video_dataset[idx]
        else:
            # Select from image dataset
            return self.image_dataset[idx % len(self.image_dataset)]


class CustomBatchSampler(Sampler):
    def __init__(self, video_id_len, image_id_len, batch_size, p=0.5):
        """
        Custom batch sampler to ensure each batch contains samples from only one dataset (video or image).
        
        Args:
            video_ids: List of video sample ids.
            image_ids: List of image sample ids.
            batch_size: The batch size for sampling.
            p: Probability of selecting a batch from the video dataset (default is 0.5).
        """
        # random seed set
        self.image_ids = list(range(image_id_len))
        # shuffle the image ids
        random.shuffle(self.image_ids)
        self.video_ids = [idx_  + image_id_len  for idx_ in list(range(video_id_len))]# to not overlap with image ids
        # shuffle the video ids
        random.shuffle(self.video_ids)
        self.batch_size = batch_size
        self.p = p
        self.video_flag = 0
        self.image_flag = 0
        
        self.video_batch_size = 1
        self.image_batch_size = batch_size

    def __iter__(self):
        while True:
            if random.random() < self.p:
                # print('VIDEO BATCH : ', self.video_flag, self.video_flag + self.video_batch_size)
                # Sample from the video dataset
                video_batch = self.video_ids[self.video_flag : self.video_flag + self.video_batch_size]
                self.video_flag = (self.video_flag + self.video_batch_size) % (len(self.video_ids) - self.video_batch_size)
                # print('video flag', self.video_flag)
                yield video_batch
                # # video_batch = random.sample(self.video_ids, self.batch_size)
                # for idx in video_batch:
                #     yield idx
            else:
                # print('IMAGE BATCH : ', self.image_flag, self.image_flag + self.image_batch_size)
                # Sample from the image dataset
                image_batch = self.image_ids[self.image_flag : self.image_flag + self.image_batch_size]
                # image_batch = random.sample(self.image_ids, self.batch_size)
                self.image_flag = (self.image_flag + self.image_batch_size) % (len(self.video_ids) - self.image_batch_size)
                # print('image flag', self.image_flag)
                yield image_batch

    def __len__(self):
        return len(self.video_ids) + len(self.image_ids)

def save_model_card(
    repo_id: str,
    videos=None,
    base_model: str = None,
    validation_prompt=None,
    repo_folder=None,
    fps=8,
):
    widget_dict = []
    if videos is not None:
        for i, video in enumerate(videos):
            export_to_video(video, os.path.join(repo_folder, f"final_video_{i}.mp4", fps=fps))
            widget_dict.append(
                {"text": validation_prompt if validation_prompt else " ", "output": {"url": f"video_{i}.mp4"}}
            )

    model_description = f"""
# CogVideoX LoRA - {repo_id}

<Gallery />

## Model description

These are {repo_id} LoRA weights for {base_model}.

The weights were trained using the [CogVideoX Diffusers trainer](https://github.com/huggingface/diffusers/blob/main/examples/cogvideo/train_cogvideox_lora.py).

Was LoRA for the text encoder enabled? No.

## Download model

[Download the *.safetensors LoRA]({repo_id}/tree/main) in the Files & versions tab.

## Use it with the [🧨 diffusers library](https://github.com/huggingface/diffusers)

```py
from diffusers import CogVideoXPipeline
import torch

pipe = CogVideoXPipeline.from_pretrained("THUDM/CogVideoX-5b", torch_dtype=torch.bfloat16).to("cuda")
pipe.load_lora_weights("{repo_id}", weight_name="pytorch_lora_weights.safetensors", adapter_name=["cogvideox-lora"])

# The LoRA adapter weights are determined by what was used for training.
# In this case, we assume `--lora_alpha` is 32 and `--rank` is 64.
# It can be made lower or higher from what was used in training to decrease or amplify the effect
# of the LoRA upto a tolerance, beyond which one might notice no effect at all or overflows.
pipe.set_adapters(["cogvideox-lora"], [32 / 64])

video = pipe("{validation_prompt}", guidance_scale=6, use_dynamic_cfg=True).frames[0]
```

For more details, including weighting, merging and fusing LoRAs, check the [documentation on loading LoRAs in diffusers](https://huggingface.co/docs/diffusers/main/en/using-diffusers/loading_adapters)

## License

Please adhere to the licensing terms as described [here](https://huggingface.co/THUDM/CogVideoX-5b/blob/main/LICENSE) and [here](https://huggingface.co/THUDM/CogVideoX-2b/blob/main/LICENSE).
"""
    model_card = load_or_create_model_card(
        repo_id_or_path=repo_id,
        from_training=True,
        license="other",
        base_model=base_model,
        prompt=validation_prompt,
        model_description=model_description,
        widget=widget_dict,
    )
    tags = [
        "text-to-video",
        "diffusers-training",
        "diffusers",
        "lora",
        "cogvideox",
        "cogvideox-diffusers",
        "template:sd-lora",
    ]

    model_card = populate_model_card(model_card, tags=tags)
    model_card.save(os.path.join(repo_folder, "README.md"))

def log_validation(
    pipe,
    args,
    accelerator,
    pipeline_args,
    epoch,
    is_final_validation: bool = False,
    # prompt: dict = None,
):
    logger.info(
        f"Running validation... \n Generating {args.num_validation_videos} videos with prompt: {pipeline_args['prompt']}."
    )

    # Move pipeline to device and eval mode
    pipe = pipe.to(accelerator.device)
    pipe.transformer.eval()
    pipe.vae.eval()
    pipe.text_encoder.eval()
    if hasattr(pipe, 'clip_text_encoder'):
        pipe.clip_text_encoder.eval()

    # Initialize CLIP processor if needed
    if args.dataset_name == 'customization':
        from transformers import CLIPProcessor
        clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch16")

    # Set deterministic generation
    # generator = torch.Generator(device=accelerator.device)
    # if args.seed is not None:
    #     generator.manual_seed(args.seed)

    videos = []
    for _ in range(args.num_validation_videos): 
        current_pipeline_args = pipeline_args.copy()

        if args.dataset_name == 'customization':
            if 'validation_reference_image' in pipeline_args:
                try:
                    from PIL import Image
                    import cv2
                    ref_image = Image.open(pipeline_args['validation_reference_image']).convert('RGB')
                    # Load and preprocess the reference image

                    # Process image using CLIP processor
                    if (not args.vae_add) and (not args.cross_attend) and (not args.cross_attend_text) and (not args.qk_replace):
                        # ref_image = Image.open(pipeline_args['validation_reference_image']).convert('RGB')
                        processed_image = clip_processor.image_processor(
                            images=ref_image,
                            return_tensors="pt"
                        )
                        
                        # Move to correct device and dtype
                        pixel_values = processed_image['pixel_values'].to(
                            device=accelerator.device,
                            dtype=pipe.transformer.dtype
                        )
                        
                        # Ensure correct shape using sample_height and sample_width from config
                        target_height = pipe.transformer.config.sample_height
                        target_width = pipe.transformer.config.sample_width
                        
                        if pixel_values.shape[-2:] != (target_height, target_width):
                            from torchvision import transforms
                            resize = transforms.Resize(
                                (224, 224),
                                interpolation=transforms.InterpolationMode.BILINEAR
                            )
                            pixel_values = resize(pixel_values)
                        
                        current_pipeline_args['ref_img_states'] = pixel_values
                        current_pipeline_args.pop('validation_reference_image', None)
                    
                        logger.info(f"Successfully processed reference image with shape: {pixel_values.shape}")
                    else:
                        # Resize and crop to the target resolution
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
                        # ref_image = cv2.cvtColor(np.array(ref_image), cv2.COLOR_RGB2BGR)
                        ref_image = np.array(ref_image)
                        ref_image = np.expand_dims(ref_image, axis=0)  # Add frame dimension
                        
                        ref_image = torch.from_numpy(ref_image).float() / 255.0 * 2.0 - 1.0
                        ref_image = ref_image.permute(0, 3, 1, 2)  # [B, H, W, C] -> [B, C, H, W]
                        
                        ref_image = ref_image.unsqueeze(0).permute(0, 2, 1, 3, 4).to(device=accelerator.device, dtype=pipe.transformer.dtype)
                        
                        with torch.no_grad():
                            ref_image_latents = pipe.vae.encode(ref_image).latent_dist
                            ref_image_latents = ref_image_latents.sample() * pipe.vae.config.scaling_factor
                        ref_image_latents = ref_image_latents.permute(0, 2, 1, 3, 4)

                        current_pipeline_args['ref_img_states'] = ref_image_latents
                        current_pipeline_args.pop('validation_reference_image', None)
                        
                        logger.info(f"Successfully processed reference image & latent with shape: {ref_image_latents.shape}")
                        # Now ref_image_latents has the correct dimensions: [B, C, F, H, W]
                        # print(f">>>>>>>>>>>>>> Ref image latents shape: {ref_image_latents.shape}")
                except Exception as e:
                    logger.error(f"Error processing reference image: {str(e)}")
                    logger.error(f"Available config keys: {pipe.transformer.config.keys()}")
                    raise
            else:
                logger.warning("No reference image provided for customization validation")

        # Generate the video
        try:
            logger.info(f"Generating video with args: {current_pipeline_args}")
            
            # Add inference parameters explicitly
            inference_args = {
                'num_inference_steps': 50,
                # 'generator': generator,
                'output_type': "np",
                'guidance_scale': args.guidance_scale,
                'use_dynamic_cfg': args.use_dynamic_cfg,
                'height': args.height_val,
                'width': args.width_val,
                'num_frames': 25, #args.max_num_frames,
                'eval': True
            }
            current_pipeline_args.update(inference_args)
            
            # Run inference with torch.no_grad()
            with torch.no_grad():
                output = pipe(**current_pipeline_args)
                video = output.frames[0]
            videos.append(video)
            
            logger.info(f"Successfully generated video with shape: {video.shape}")
            
        except Exception as e:
            logger.error(f"Error during video generation: {str(e)}")
            logger.error(f"Pipeline arguments: {current_pipeline_args}")
            raise

    # Log to wandb if enabled
    for tracker in accelerator.trackers:
        phase_name = "test" if is_final_validation else "validation"
        if tracker.name == "wandb":
            video_filenames = []
            for i, video in enumerate(videos):
                prompt = (
                    pipeline_args["prompt"][:25]
                    .replace(" ", "_")
                    .replace("'", "_")
                    .replace('"', "_")
                    .replace("/", "_")
                )
                max_num_frames = current_pipeline_args['num_frames']
                filename = os.path.join(args.output_dir, f"{epoch}_{phase_name}_video_{i}_max_n_f_{max_num_frames}_{prompt}.mp4")
                export_to_video(video, filename, fps=args.fps)
                video_filenames.append(filename)

            tracker.log(
                {
                    phase_name: [
                        wandb.Video(filename, caption=f"{i}: {pipeline_args['prompt']}")
                        for i, filename in enumerate(video_filenames)
                    ],
                    f"{phase_name}_epoch": epoch,
                }
            )

    # Clean up
    free_memory()
    return videos


def _get_clip_prompt_embeds(
    tokenizer,
    text_encoder,
    prompt,
    num_videos_per_prompt=1,
    max_sequence_length=226,
    device=None,
    dtype=None,
    text_input_ids=None,
):
    with torch.no_grad():
        clip_prompt_tokenized = tokenizer(prompt, padding='max_length', max_length=77, truncation=True, return_tensors='pt')
        clip_prompt_tokenized = clip_prompt_tokenized.to(device)
        try:
            clip_prompt_embeds = text_encoder(clip_prompt_tokenized.input_ids)
            if isinstance(clip_prompt_embeds, tuple):
                clip_prompt_embeds = clip_prompt_embeds[0]
            elif isinstance(clip_prompt_embeds, dict):
                clip_prompt_embeds = clip_prompt_embeds['last_hidden_state']
            else:
                raise ValueError(f"Unexpected output type from text_encoder: {type(clip_prompt_embeds)}")
            
            clip_prompt_embeds = clip_prompt_embeds.to(dtype=dtype, device=device)
        except Exception as e:
            print(f"Error in text encoder: {e}")
            print(f"Text encoder output type: {type(clip_prompt_embeds)}")
            print(f"Text encoder output shape: {clip_prompt_embeds.shape if hasattr(clip_prompt_embeds, 'shape') else 'No shape attribute'}")
            raise

    _, seq_len, _ = clip_prompt_embeds.shape
    clip_prompt_embeds = clip_prompt_embeds.repeat(1, num_videos_per_prompt, 1)
    clip_prompt_embeds = clip_prompt_embeds.view(clip_prompt_embeds.shape[0] * num_videos_per_prompt, seq_len, -1)

    return clip_prompt_embeds

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


    clip_prompt_embeds = _get_clip_prompt_embeds(
        clip_tokenizer,
        clip_text_encoder,
        prompt=prompt,
        num_videos_per_prompt=num_videos_per_prompt,
        max_sequence_length=max_sequence_length,
        device=device,
        dtype=dtype,
        text_input_ids=text_input_ids,
    )
    # prompt_embeds = torch.cat([prompt_embeds, clip_prompt_embeds], dim=-1)

    return prompt_embeds, clip_prompt_embeds


def compute_prompt_embeddings(
    tokenizer, text_encoder, clip_tokenizer, clip_text_encoder, prompt, max_sequence_length, device, dtype, requires_grad: bool = False
):
    if requires_grad:
        prompt_embeds, clip_prompt_embeds = encode_prompt(
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
        
    else:
        with torch.no_grad():
            prompt_embeds, clip_prompt_embeds = encode_prompt(
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

    return prompt_embeds, clip_prompt_embeds


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


def get_optimizer(args, params_to_optimize, use_deepspeed: bool = False):
    # Use DeepSpeed optimzer
    if use_deepspeed:
        from accelerate.utils import DummyOptim

        return DummyOptim(
            params_to_optimize,
            lr=args.learning_rate,
            betas=(args.adam_beta1, args.adam_beta2),
            eps=args.adam_epsilon,
            weight_decay=args.adam_weight_decay,
        )

    # Optimizer creation
    supported_optimizers = ["adam", "adamw", "prodigy"]
    if args.optimizer not in supported_optimizers:
        logger.warning(
            f"Unsupported choice of optimizer: {args.optimizer}. Supported optimizers include {supported_optimizers}. Defaulting to AdamW"
        )
        args.optimizer = "adamw"

    if args.use_8bit_adam and not (args.optimizer.lower() not in ["adam", "adamw"]):
        logger.warning(
            f"use_8bit_adam is ignored when optimizer is not set to 'Adam' or 'AdamW'. Optimizer was "
            f"set to {args.optimizer.lower()}"
        )

    if args.optimizer.lower() == "adamw":
        optimizer_class = bnb.optim.AdamW8bit if args.use_8bit_adam else torch.optim.AdamW

        optimizer = optimizer_class(
            params_to_optimize,
            betas=(args.adam_beta1, args.adam_beta2),
            eps=args.adam_epsilon,
            weight_decay=args.adam_weight_decay,
        )
    elif args.optimizer.lower() == "adam":
        optimizer_class = bnb.optim.Adam8bit if args.use_8bit_adam else torch.optim.Adam

        optimizer = optimizer_class(
            params_to_optimize,
            betas=(args.adam_beta1, args.adam_beta2),
            eps=args.adam_epsilon,
            weight_decay=args.adam_weight_decay,
        )
    elif args.optimizer.lower() == "prodigy":
        try:
            import prodigyopt
        except ImportError:
            raise ImportError("To use Prodigy, please install the prodigyopt library: `pip install prodigyopt`")

        optimizer_class = prodigyopt.Prodigy

        if args.learning_rate <= 0.1:
            logger.warning(
                "Learning rate is too low. When using prodigy, it's generally better to set learning rate around 1.0"
            )

        optimizer = optimizer_class(
            params_to_optimize,
            lr=args.learning_rate,
            betas=(args.adam_beta1, args.adam_beta2),
            beta3=args.prodigy_beta3,
            weight_decay=args.adam_weight_decay,
            eps=args.adam_epsilon,
            decouple=args.prodigy_decouple,
            use_bias_correction=args.prodigy_use_bias_correction,
            safeguard_warmup=args.prodigy_safeguard_warmup,
        )

    return optimizer


def main(args):
    print('Start')
    t5_first = args.t5_first
    concatenated_all = args.concatenated_all
    reduce_token = args.reduce_token
    if args.report_to == "wandb" and args.hub_token is not None:
        raise ValueError(
            "You cannot use both --report_to=wandb and --hub_token due to a security risk of exposing your token."
            " Please use `huggingface-cli login` to authenticate with the Hub."
        )

    if torch.backends.mps.is_available() and args.mixed_precision == "bf16":
        # due to pytorch#99272, MPS does not yet support bfloat16.
        raise ValueError(
            "Mixed precision training with bfloat16 is not supported on MPS. Please use fp16 (recommended) or fp32 instead."
        )
    print('Log dir set')
    logging_dir = Path(args.output_dir, args.logging_dir)
    print('Acceleration config set')
    # Accelerator setup
    #FIXME -> wandb name setup
    project_name = "video_customization_consis_id_style"
    experiment_name = os.path.splitext(os.path.basename(args.output_dir))[0]
    os.environ["WANDB_NAME"] = experiment_name
    accelerator_project_config = ProjectConfiguration(project_dir=args.output_dir, logging_dir=logging_dir,)
                                                    #   name=experiment_name)
    # DistributedDataParallelKwargs setup for gradient checkpointing and unused parameters and mixed precision
    # what is distributed dataparallel kwargs here? 
    # find_unused_parameters: If True, find_unused_parameters will be passed to DistributedDataParallel.
    kwargs = DistributedDataParallelKwargs(find_unused_parameters=False)
    # Accelerator setup
    print("Accelerator setup")
    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps, # gradient accumulation steps
        mixed_precision=args.mixed_precision, # mixed precision training
        log_with=args.report_to, # logging to wandb
        project_config=accelerator_project_config, # project configuration
        kwargs_handlers=[kwargs], # DistributedDataParallelKwargs setup
    )
    print('Accelerator setup done')

    # Disable AMP for MPS.
    if torch.backends.mps.is_available():
        accelerator.native_amp = False
    
    if args.report_to == "wandb":
        print('Wandb setup')
        if not is_wandb_available():
            raise ImportError("Make sure to install wandb if you want to use it for logging during training.")

    # Make one log on every process with the configuration for debugging.
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state, main_process_only=False)
    if accelerator.is_local_main_process:
        transformers.utils.logging.set_verbosity_warning()
        diffusers.utils.logging.set_verbosity_info()
    else:
        transformers.utils.logging.set_verbosity_error()
        diffusers.utils.logging.set_verbosity_error()

    # If passed along, set the training seed now.
    if args.seed is not None:
        set_seed(args.seed)

    # Handle the repository creation
    if accelerator.is_main_process: # main process
        if args.output_dir is not None:
            os.makedirs(args.output_dir, exist_ok=True)

        if args.push_to_hub:
            repo_id = create_repo(
                repo_id=args.hub_model_id or Path(args.output_dir).name,
                exist_ok=True,
            ).repo_id

    # Prepare models and scheduler
    tokenizer = AutoTokenizer.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="tokenizer", revision=args.revision
    )
    if args.add_special:
        if args.add_multiple_special:
            special_token = {"additional_special_tokens": ["<cls>", "<a>", "<b>", "<c>"]}
        else:
            special_token = {"additional_special_tokens": ["<cls>"]}
        tokenizer.add_special_tokens(special_token)

    text_encoder = T5EncoderModel.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="text_encoder", revision=args.revision
    )
    if args.add_special:
        text_encoder.resize_token_embeddings(len(tokenizer))
    
    # Prepare additional model and scheduler (CLIP) for prompt encoding in customization
    clip_tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-base-patch16")  # You can change to another version if needed
    clip_text_encoder = CLIPTextModel.from_pretrained("openai/clip-vit-base-patch16")
    clip_text_encoder.to(accelerator.device, dtype=torch.float16)

    # SHOULD BE MOVED TO THE COGVIDEOX MODEL PIPELINE FOR DEEPSPEED / ACCELERATOR PREPARE AT ONCE FOR FINETUNING VISION MODEL OF CLIP
    clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch16")
    # clip_vision_model = CLIPVisionModelWithLoRA # CLIPVisionModel.from_pretrained("openai/clip-vit-base-patch16")
    # clip_vision_base_model = CLIPVisionModel.from_pretrained("openai/clip-vit-base-patch16")
    # clip_vision_model = CLIPVisionModelWithLoRA(clip_vision_base_model.config)
    # clip_vision_model.to("cuda")  # If GPU is available
    ### SHOULD BE MOVED UP TO HERE FOR DEEPSPEED / ACCELERATOR PREPARE AT ONCE FOR FINETUNING CLIP VISION MODEL
    
    # SHOULD BE MOVED TO THE COGVIDEOX MODEL PIPELINE FOR DEEPSPEED / ACCELERATOR PREPARE AT ONCE FOR FINETUNING TEXT AND VISION PROJECTION LAYERS
    # T5ProjectionLayer = ProjectionLayer(in_features=4096, out_features=4096).to(dtype=torch.bfloat16)
    # CLIPTextProjectionLayer = ProjectionLayer(in_features=512, out_features=4096).to(dtype=torch.bfloat16)
    # CLIPVisionProjectionLayer = ProjectionLayer(in_features=768, out_features=4096).to(dtype=torch.bfloat16)
    
    # T5ProjectionLayer.to("cuda")
    # CLIPTextProjectionLayer.to("cuda")
    # CLIPVisionProjectionLayer.to("cuda")
    ### SHOULD BE MOVED UP TO HERE FOR DEEPSPEED / ACCELERATOR PREPARE AT ONCE FOR FINETUNING TEXT AND VISION PROJECTION LAYERS

    # CogVideoX-2b weights are stored in float16
    # CogVideoX-5b and CogVideoX-5b-I2V weights are stored in bfloat16
    print("Loading CogVideoX Transformer model")
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
        zero_conv_add=args.zero_conv_add,
        vae_add=args.vae_add,
        cross_attend=args.cross_attend,
        cross_attend_text=args.cross_attend_text,
        cross_attn_interval=args.cross_attn_interval,
        local_reference_scale=args.local_reference_scale,
        # cross_inner_dim=args.cross_inner_dim,
        cross_attn_dim_head=args.cross_attn_dim_head,
        cross_attn_num_heads=args.cross_attn_num_head,
        qk_replace=args.qk_replace,
        qformer=args.qformer,
        second_stage=args.second_stage,
        # cross_attn_kv_dim=args.cross_attn_kv_dim,
    )
    print("Done - CogVideoX Transformer model loaded")
    # Initialize submodules
    if args.second_stage:
        pass
    else:
        if (not args.vae_add) and (not args.cross_attend) and (not args.cross_attend_text) and (not args.qk_replace) and (not args.qformer):
            transformer.reference_vision_encoder = CLIPVisionModel.from_pretrained("openai/clip-vit-base-patch16")
        if args.qformer is True:
            transformer.QformerAligner = QFormerAligner(model_name="Salesforce/blip2-opt-2.7b")
            transformer.QformerAligner.qformer.requires_grad_(True)
            transformer.QformerAligner.fc.requires_grad_(True)
        else:
            if (args.vae_add or args.qk_replace) and (not args.cross_attend) and ( not args.cross_attend):
                # transformer.CLIPTextProjectionLayer = None
                # transformer.CLIPVisionProjectionLayer = None
                # transformer.CLIPTextProjectionLayer2 = None
                # transformer.CLIPVisionProjectionLayer2 = None
                # transformer.T5ProjectionLayer = None
                pass
            elif (not args.vae_add) and (not args.cross_attend) and (not args.cross_attend_text) and (not args.qk_replace):
                if args.zero_conv_add:
                    transformer.text_sequence_aligner = SequenceAligner(77, 226)
                    transformer.vision_sequence_aligner = SequenceAligner(197, 226)
                    
                    transformer.CLIPTextProjectionLayer = ZeroConv1D(in_dim=512, out_dim=4096)
                    transformer.CLIPVisionProjectionLayer = ZeroConv1D(in_dim=768, out_dim=4096)
                    transformer.CLIPTextProjectionLayer2 = ZeroConv1D(in_dim=4096, out_dim=4096)
                    transformer.CLIPVisionProjectionLayer2 = ZeroConv1D(in_dim=4096, out_dim=4096)
                    transformer.T5ProjectionLayer = SkipProjectionLayer(4096, 4096)
                    with torch.no_grad():
                        transformer.T5ProjectionLayer.projection.weight.fill_(0.0)
                        if transformer.T5ProjectionLayer.projection.bias is not None:
                            transformer.T5ProjectionLayer.projection.bias.fill_(0.0)
                    
                    # Learnable single parameter
                    # transformer.alpha = torch.nn.Parameter(torch.tensor(0.0, dtype=torch.float16))
                    # transformer.beta = torch.nn.Parameter(torch.tensor(0.0, dtype=torch.float16))
                    
                    # requries grad True
                    transformer.CLIPTextProjectionLayer.requires_grad_(True)
                    transformer.CLIPVisionProjectionLayer.requires_grad_(True)
                    transformer.CLIPTextProjectionLayer2.requires_grad_(True)
                    transformer.CLIPVisionProjectionLayer2.requires_grad_(True)
                    transformer.reference_vision_encoder.requires_grad_(True)
                    transformer.T5ProjectionLayer.requires_grad_(True)
                    
                    # transformer.alpha.requires_grad_(True)
                    # transformer.beta.requires_grad_(True)
                    
                else:
                    if args.add_token is True:
                        transformer.T5ProjectionLayer = ProjectionLayer(in_features=4096, out_features=4096)
                        with torch.no_grad():
                            transformer.T5ProjectionLayer.projection.weight.fill_(1.0)
                            if transformer.T5ProjectionLayer.projection.bias is not None:
                                transformer.T5ProjectionLayer.projection.bias.fill_(1.0)
                        transformer.CLIPTextProjectionLayer = ProjectionLayer(in_features=77, out_features=226)
                        with torch.no_grad():
                            transformer.CLIPTextProjectionLayer.projection.weight.fill_(0.0)
                            if transformer.CLIPTextProjectionLayer.projection.bias is not None:
                                transformer.CLIPTextProjectionLayer.projection.bias.fill_(0.0)
                        transformer.CLIPVisionProjectionLayer = ProjectionLayer(in_features=197, out_features=226)
                        with torch.no_grad():
                            transformer.CLIPVisionProjectionLayer.projection.weight.fill_(0.0)
                            if transformer.CLIPVisionProjectionLayer.projection.bias is not None:
                                transformer.CLIPVisionProjectionLayer.projection.bias.fill_(0.0)
                        # Requires grad true
                        transformer.reference_vision_encoder.requires_grad_(True)
                        transformer.T5ProjectionLayer.requires_grad_(True)
                        transformer.CLIPTextProjectionLayer.requires_grad_(True)
                        transformer.CLIPVisionProjectionLayer.requires_grad_(True)
                    else:    
                        if reduce_token is not True:
                            transformer.T5ProjectionLayer = SkipProjectionLayer(in_features=4096, out_features=4096)
                            with torch.no_grad():
                                transformer.T5ProjectionLayer.projection.weight.fill_(0.0)
                                if transformer.T5ProjectionLayer.projection.bias is not None:
                                    transformer.T5ProjectionLayer.projection.bias.fill_(0.0)
                            # Requires grad true
                            transformer.reference_vision_encoder.requires_grad_(True)
                            transformer.T5ProjectionLayer.requires_grad_(True)
                            
                            # if concatenated_all:
                            transformer.CLIPTextProjectionLayer = ProjectionLayer(in_features=512, out_features=4096)
                            with torch.no_grad():
                                transformer.CLIPTextProjectionLayer.projection.weight.fill_(0.0)
                                if transformer.CLIPTextProjectionLayer.projection.bias is not None:
                                    transformer.CLIPTextProjectionLayer.projection.bias.fill_(0.0)
                            transformer.CLIPVisionProjectionLayer = ProjectionLayer(in_features=768, out_features=4096)
                            with torch.no_grad():
                                transformer.CLIPVisionProjectionLayer.projection.weight.fill_(0.0)
                                if transformer.CLIPVisionProjectionLayer.projection.bias is not None:
                                    transformer.CLIPVisionProjectionLayer.projection.bias.fill_(0.0)
                            # Requires grad true
                            transformer.CLIPTextProjectionLayer.requires_grad_(True)
                            transformer.CLIPVisionProjectionLayer.requires_grad_(True)
                        else:
                            transformer.reference_vision_encoder = CLIPVisionModel.from_pretrained("openai/clip-vit-base-patch16")
                            transformer.T5ProjectionLayer = ReduceProjectionLayer(in_features=500, out_features=226)
                            with torch.no_grad():
                                transformer.T5ProjectionLayer.projection.weight.fill_(0.0)
                                if transformer.T5ProjectionLayer.projection.bias is not None:
                                    transformer.T5ProjectionLayer.projection.bias.fill_(0.0)
                            # Requires grad true
                            transformer.reference_vision_encoder.requires_grad_(True)
                            transformer.T5ProjectionLayer.requires_grad_(True)
            elif args.cross_attend or args.cross_attend_text:
                    if args.cross_attend:
                        print('Loading PERCEIVER CROSS ATTENTION')
                        perceiver_cross_attention = None
                        local_reference_scale = args.local_reference_scale
                        num_cross_attn = 42 // args.cross_attn_interval
                        cross_inner_dim = 3072
                        cross_attn_dim_head = args.cross_attn_dim_head
                        cross_attn_num_head = args.cross_attn_num_head
                        # cross_attn_kv_dim =  int(cross_inner_dim / 3 * 2)
                        cross_attn_kv_dim = 3072
                        transformer.perceiver_cross_attention = nn.ModuleList(
                            [
                                PerceiverCrossAttention(
                                    dim=cross_inner_dim,
                                    dim_head=cross_attn_dim_head,
                                    heads=cross_attn_num_head,
                                    kv_dim=cross_attn_kv_dim,
                                ).to(dtype=torch.bfloat16)
                                for _ in range(num_cross_attn)
                            ]
                        )
                        transformer.perceiver_cross_attention.requires_grad_(True)
                    if args.cross_attend_text:
                        print('Loading PERCEIVER CROSS ATTENTION FOR TEXT')
                        perceiver_cross_attention_text = None
                        local_reference_scale = args.local_reference_scale
                        num_cross_attn = 42 // args.cross_attn_interval
                        cross_inner_dim = 3072
                        cross_attn_dim_head = args.cross_attn_dim_head
                        cross_attn_num_head = args.cross_attn_num_head
                        cross_attn_kv_dim = 3072
                        transformer.perceiver_cross_attention_text = nn.ModuleList(
                            [
                                PerceiverCrossAttention(
                                    dim=cross_inner_dim,
                                    dim_head=cross_attn_dim_head,
                                    heads=cross_attn_num_head,
                                    kv_dim=cross_attn_kv_dim,
                                ).to(dtype=torch.bfloat16)
                                for _ in range(num_cross_attn)
                            ]
                        )
    # Print out parameter names to verify # FOR DEBUGGING
    # for name, param in transformer.named_parameters():
    #     if param.requires_grad:
    #         print(f"{name}: {param.size()}")

    ## ADCDITIONAL projection l
    if args.use_latent is True:
        print("Not loading CogVideoX VAE model --> using VAE Latents directly!..")
    else:
        print("Loading CogVideoX VAE model")
        vae = AutoencoderKLCogVideoX.from_pretrained(
            args.pretrained_model_name_or_path, subfolder="vae", revision=args.revision, variant=args.variant
        )
        if args.enable_slicing:
            vae.enable_slicing()
        if args.enable_tiling:
            vae.enable_tiling()
        vae.requires_grad_(False)
        print("Done - CogVideoX VAE model loaded")
    print("Loading CogVideoX Scheduler model")
    scheduler = CogVideoXDPMScheduler.from_pretrained(args.pretrained_model_name_or_path, subfolder="scheduler")
    


    # We only train the additional adapter LoRA layers
    text_encoder.requires_grad_(False)
    transformer.requires_grad_(False)
    
    # Do Not Train CLIP text encoder
    clip_text_encoder.requires_grad_(False)
    # Train CLIP image encoder
    # SHOULD BE MOVED TO THE COGVIDEOX MODEL PIPELINE FOR DEEPSPEED / ACCELERATOR PREPARE AT ONCE FOR FINETUNING VISION MODEL OF CLIP
    # clip_vision_model.requires_grad_(True)
    ### SHOULD BE MOVED UP TO HERE FOR DEEPSPEED / ACCELERATOR PREPARE AT ONCE FOR FINETUNING VISION MODEL OF CLIP


    # For mixed precision training we cast all non-trainable weights (vae, text_encoder and transformer) to half-precision
    # as these weights are only used for inference, keeping weights in full precision is not required.
    weight_dtype = torch.float32
    if accelerator.state.deepspeed_plugin:
        # DeepSpeed is handling precision, use what's in the DeepSpeed config
        if (
            "fp16" in accelerator.state.deepspeed_plugin.deepspeed_config
            and accelerator.state.deepspeed_plugin.deepspeed_config["fp16"]["enabled"]
        ):
            weight_dtype = torch.float16
        if (
            "bf16" in accelerator.state.deepspeed_plugin.deepspeed_config
            and accelerator.state.deepspeed_plugin.deepspeed_config["bf16"]["enabled"]
        ):
            weight_dtype = torch.float16
    else:
        if accelerator.mixed_precision == "fp16":
            weight_dtype = torch.float16
        elif accelerator.mixed_precision == "bf16":
            weight_dtype = torch.bfloat16

    if torch.backends.mps.is_available() and weight_dtype == torch.bfloat16:
        # due to pytorch#99272, MPS does not yet support bfloat16.
        raise ValueError(
            "Mixed precision training with bfloat16 is not supported on MPS. Please use fp16 (recommended) or fp32 instead."
        )

    text_encoder.to(accelerator.device, dtype=weight_dtype)
    transformer.to(accelerator.device, dtype=weight_dtype)
    if not args.use_latent:
        vae.to(accelerator.device, dtype=weight_dtype)

    if args.gradient_checkpointing:
        transformer.enable_gradient_checkpointing()

    # now we will add new LoRA weights to the attention layers
    # transformer_lora_config = LoraConfig(
    #     r=args.rank,
    #     lora_alpha=args.lora_alpha,
    #     init_lora_weights=True,
    #     target_modules=["to_k", "to_q", "to_v", "to_out.0"],
    # )
    unfreeze_modules = ["perceiver_cross_attention", "perceiver_cross_attention_text", "QformerAligner.qformer", "QformerAligner.fc"]
    transformer_lora_config = LoraConfig(
        r=args.rank,
        lora_alpha=args.lora_alpha,
        init_lora_weights=True,
        target_modules=["to_k", "to_q", "to_v", "to_out.0", "proj", "text_proj","norm1.linear", "norm2.linear", "ff.net.2"], # "time_embedding.linear_1", "time_embedding.linear_2"],
        exclude_modules=unfreeze_modules
    )
    # need to also train cross_attention layer separately wholely, not with LoRA
    
    
    transformer.add_adapter(transformer_lora_config)
    print(transformer.active_adapters)
    def unwrap_model(model):
        model = accelerator.unwrap_model(model)
        model = model._orig_mod if is_compiled_module(model) else model
        return model

    # create custom saving & loading hooks so that `accelerator.save_state(...)` serializes in a nice format
    def save_model_hook(models, weights, output_dir):
        if accelerator.is_main_process:

            transformer_lora_layers_to_save = None
            vision_model_lora_layers_to_save = None

            for model in models:
                # Use 'unwrap_model' to handle any wrapped model cases
                unwrapped_model = unwrap_model(model)
                print(f"Unwrapped model class: {unwrapped_model.__class__}")  # Add this line for debugging
                # Check for CogVideoX transformer
                if isinstance(unwrapped_model, type(unwrap_model(transformer))):
                    transformer_lora_layers_to_save = get_peft_model_state_dict(model)
                    
                    if (not args.vae_add) and (not args.cross_attend) and (not args.cross_attend_text) and (not args.qk_replace) and (not args.qformer):
                        # Save ProjectionLayer state_dicts
                        projection_layers_state_dict = {
                            "T5ProjectionLayer": unwrapped_model.T5ProjectionLayer.state_dict(),
                            "CLIPTextProjectionLayer": unwrapped_model.CLIPTextProjectionLayer.state_dict(),
                            "CLIPVisionProjectionLayer": unwrapped_model.CLIPVisionProjectionLayer.state_dict(),
                        }
                        if args.zero_conv_add:
                            projection_layers_state_dict["CLIPTextProjectionLayer2"] = unwrapped_model.CLIPTextProjectionLayer2.state_dict()
                            projection_layers_state_dict["CLIPVisionProjectionLayer2"] = unwrapped_model.CLIPVisionProjectionLayer2.state_dict()
                        # Save CLIPVisionModel state_dict
                        vision_model_state_dict = unwrapped_model.reference_vision_encoder.state_dict()
                    elif args.qformer is True:
                        qformer_state = unwrapped_model.QformerAligner.state_dict()
                    if args.cross_attend or args.cross_attend_text:
                        if args.cross_attend:
                            cross_attention_layer_state_dict = unwrapped_model.perceiver_cross_attention.state_dict()
                        if args.cross_attend_text:
                            cross_attention_layer_state_dict_text = unwrapped_model.perceiver_cross_attention_text.state_dict()
                
                # Raise an error for unexpected models
                else:
                    raise ValueError(f"Unexpected save model: {unwrapped_model.__class__}")

                # Ensure to pop weight so that the corresponding model is not saved again
                weights.pop()

            # Save LoRA weights for CogVideoX
            # if args.second_stage:
            #     CogVideoXPipeline.save_lora_weights(
            #         output_dir,
            #         weight_name="pytorch_lora_weights_transformer_second_stage.safetensors",
            #         transformer_lora_layers=transformer_lora_layers_to_save,
            #     )
            # else:
            if transformer_lora_layers_to_save:
                CogVideoXPipeline.save_lora_weights(
                    output_dir,
                    weight_name="pytorch_lora_weights_transformer.safetensors",
                    transformer_lora_layers=transformer_lora_layers_to_save,
                )
            if (not args.vae_add) and (not args.cross_attend) and (not args.cross_attend_text) and (not args.qk_replace) and (not args.qformer):
                # Save Projection Layer weights
                for name, state_dict in projection_layers_state_dict.items():
                    save_path = os.path.join(output_dir, f"{name}.pth")
                    torch.save(state_dict, save_path)
                
                # Save CLIPVisionModel weights
                if vision_model_state_dict is not None:
                    save_path = os.path.join(output_dir, "pytorch_clip_vision_model.bin")
                    torch.save(vision_model_state_dict, save_path)
            elif args.qformer is True:
                # Save QFormer weights
                save_path = os.path.join(output_dir, "QformerAligner.pth")
                torch.save(qformer_state, save_path)
            if args.cross_attend or args.cross_attend_text:
                if args.cross_attend:
                    save_path = os.path.join(output_dir, "perceiver_cross_attention.pth")
                    torch.save(cross_attention_layer_state_dict, save_path)
                if args.cross_attend_text:
                    save_path = os.path.join(output_dir, "perceiver_cross_attention_text.pth")
                    torch.save(cross_attention_layer_state_dict_text, save_path)


    def load_model_hook(models, input_dir):
        """Load LoRA weights for transformer and vision models"""
        transformer_ = None
        
        # Extract models while emptying the list
        while len(models) > 0:
            model = models.pop()
            if isinstance(model, type(unwrap_model(transformer))):
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
            if (not args.vae_add) and (not args.cross_attend) and (not args.cross_attend_text) and (not args.qk_replace) and (not args.qformer):
                # Load ProjectionLayer weights
                transformer_.T5ProjectionLayer.load_state_dict(torch.load(os.path.join(input_dir, "T5ProjectionLayer.pth")))
                transformer_.CLIPTextProjectionLayer.load_state_dict(torch.load(os.path.join(input_dir, "CLIPTextProjectionLayer.pth")))
                transformer_.CLIPVisionProjectionLayer.load_state_dict(torch.load(os.path.join(input_dir, "CLIPVisionProjectionLayer.pth")))
                
                # Load CLIPVisionModel weights
                vision_model_state_dict = torch.load(os.path.join(input_dir, "pytorch_clip_vision_model.bin"))
                transformer_.reference_vision_encoder.load_state_dict(vision_model_state_dict)
            elif args.qformer:
                # Load QFormer weights
                qformer_state = torch.load(os.path.join(input_dir, "QformerAligner.pth"))
                transformer.QformerAligner.load_state_dict(qformer_state)
            if args.cross_attend or args.cross_attend_text:
                if args.cross_attend:
                    cross_attention_layer_state_dict = torch.load(os.path.join(input_dir, "perceiver_cross_attention.pth"))
                    transformer_.perceiver_cross_attention.load_state_dict(cross_attention_layer_state_dict)
                if args.cross_attend_text:
                    cross_attention_layer_state_dict_text = torch.load(os.path.join(input_dir, "perceiver_cross_attention_text.pth"))
                    transformer_.perceiver_cross_attention_text.load_state_dict(cross_attention_layer_state_dict_text)
            if incompatible_keys is not None:
                # check only for unexpected keys
                unexpected_keys = getattr(incompatible_keys, "unexpected_keys", None)
                if unexpected_keys:
                    logger.warning(
                        f"Loading adapter weights from state_dict led to unexpected keys not found in the model: "
                        f" {unexpected_keys}. "
                    )
        
    print("Registering save and load hooks")
    accelerator.register_save_state_pre_hook(save_model_hook)
    accelerator.register_load_state_pre_hook(load_model_hook)

    # Enable TF32 for faster training on Ampere GPUs,
    # cf https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices
    if args.allow_tf32 and torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True

    if args.scale_lr:
        args.learning_rate = (
            args.learning_rate * args.gradient_accumulation_steps * args.train_batch_size * accelerator.num_processes
        )

    # Make sure the trainable params are in float32.
    if args.mixed_precision == "fp16":
        # only upcast trainable parameters (LoRA) into fp32
        cast_training_params([transformer], dtype=torch.float32)

    transformer_lora_parameters = list(filter(lambda p: p.requires_grad, transformer.parameters()))
    transformer_parameters_with_lr = {"params": transformer_lora_parameters, "lr": args.learning_rate}

    # Add the parameters of the projection layers with their learning rates
    if (not args.vae_add) and (not args.cross_attend) and (not args.cross_attend_text) and (not args.qk_replace) and (not args.qformer):
        if concatenated_all is True:
            projection_parameters = [
                {"params": transformer.T5ProjectionLayer.parameters(), "lr": args.learning_rate},
            ]
        else:
            projection_parameters = [
                {"params": transformer.T5ProjectionLayer.parameters(), "lr": args.learning_rate},
                {"params": transformer.CLIPTextProjectionLayer.parameters(), "lr": args.learning_rate},
                {"params": transformer.CLIPVisionProjectionLayer.parameters(), "lr": args.learning_rate}
            ]
        # Add the parameters of the CLIP Vision Model
        clip_vision_parameters_with_lr = {
            "params": transformer.reference_vision_encoder.parameters(),
            "lr": args.learning_rate  # You might consider using a smaller LR here
        }
        
        if args.zero_conv_add:
            # learnable_weights = [transformer.alpha, transformer.beta]
            learnable_weights = [
                {"params": transformer.CLIPTextProjectionLayer2.parameters(), "lr": args.learning_rate},
                {"params": transformer.CLIPVisionProjectionLayer2.parameters(), "lr": args.learning_rate},
            ]

        # Combine all parameters to optimize
        params_to_optimize = [transformer_parameters_with_lr, clip_vision_parameters_with_lr] + projection_parameters
        if args.zero_conv_add:
            # params_to_optimize.append({"params": learnable_weights, "lr": args.learning_rate})
            params_to_optimize = params_to_optimize + learnable_weights
    elif args.cross_attend or args.cross_attend_text:
        cross_attention_parameters = []
        if args.cross_attend:
            cross_attention_parameters += [
                {"params": transformer.perceiver_cross_attention.parameters(), "lr": args.learning_rate},
            ]
        if args.cross_attend_text:
            cross_attention_parameters += [
                {"params": transformer.perceiver_cross_attention_text.parameters(), "lr": args.learning_rate},
            ]
        params_to_optimize = [transformer_parameters_with_lr] + cross_attention_parameters
    elif args.qformer :
        qformer_parameters = [
            {"params": transformer.QformerAligner.parameters(), "lr": args.learning_rate},
        ]
        params_to_optimize = [transformer_parameters_with_lr] + qformer_parameters
    else:
        params_to_optimize = [transformer_parameters_with_lr]

    # Check for DeepSpeed optimizer and scheduler configuration
    use_deepspeed_optimizer = (
        accelerator.state.deepspeed_plugin is not None
        and "optimizer" in accelerator.state.deepspeed_plugin.deepspeed_config
    )
    use_deepspeed_scheduler = (
        accelerator.state.deepspeed_plugin is not None
        and "scheduler" not in accelerator.state.deepspeed_plugin.deepspeed_config
    )

    # Create the optimizer
    print("Getting optimizer")
    optimizer = get_optimizer(args, params_to_optimize, use_deepspeed=use_deepspeed_optimizer)


    print("Dataset and DataLoader")
    # Dataset and DataLoader
    if args.joint_train is True: #FIXME - add argument for joint training
            image_dataset = ImageDataset(
                instance_data_root=args.instance_data_root,
                dataset_name=args.dataset_name,
                anno_path=args.anno_root,
                cache_dir=args.cache_dir,
                id_token=args.id_token,
                subset_cnt=args.subset_cnt,
                use_latent=args.use_latent,
                vae_add=args.vae_add,
                cross_attend=args.cross_attend,
                cross_attend_text=args.cross_attend_text,
                seen_validation=args.seen_validation,
                add_special=args.add_special,
                add_multiple_special=args.add_multiple_special,
                add_specific_loc=args.add_specific_loc,
                wo_shuffle=args.wo_shuffle,
                add_new_split=args.add_new_split,
                qk_replace=args.qk_replace,
                qformer=args.qformer,
            )
            video_dataset = VideoDataset(
                video_instance_root=args.video_instance_root,
                video_anno=args.video_anno,
                video_ref_root = args.video_ref_root,
                height=args.height,
                width=args.width,
                seen_validation=args.seen_validation,
                joint_train=args.joint_train,
                image_dataset_len=len(image_dataset),
            )
            # SHOULD ALSO UPDATE COMBINED_DATASET-> We do not need prob samp video iside the dataset, as we have it inside custom batch sampler
            train_dataset = CombinedDataset(video_dataset, image_dataset, args.prob_sample_video) #FIXME - add argument for prob_sample_video
            def collate_fn(examples):
                examples = examples[0]
                videos = [example['instance_video'] for example in examples]
                prompts = [example['instance_prompt'] for example in examples]
                ref_images = [example['instance_ref_image'] for example in examples]
                # if args.use_latent:
                videos = torch.cat(videos, dim=0)
                ref_images = torch.cat(ref_images, dim=0).to(memory_format=torch.contiguous_format).float()
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
                train_dataset,
                shuffle=False,
                sampler=CustomBatchSampler(len(video_dataset), len(image_dataset), batch_size=args.train_batch_size, p=args.prob_sample_video),
                collate_fn=collate_fn,
                num_workers=args.dataloader_num_workers,
                prefetch_factor=4,
                worker_init_fn=worker_init_fn,
            )
    else:
        if args.second_stage is True:
            print('Loading VideoDataset for Second Stage')
            train_dataset = VideoDataset(
                video_instance_root=args.video_instance_root,
                video_anno=args.video_anno,
                video_ref_root = args.video_ref_root,
                height=args.height,
                width=args.width,
                seen_validation=args.seen_validation,
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
        else:
            train_dataset = ImageDataset(
                instance_data_root=args.instance_data_root,
                dataset_name=args.dataset_name,
                anno_path=args.anno_root,
                cache_dir=args.cache_dir,
                id_token=args.id_token,
                subset_cnt=args.subset_cnt,
                use_latent=args.use_latent,
                vae_add=args.vae_add,
                cross_attend=args.cross_attend,
                cross_attend_text=args.cross_attend_text,
                seen_validation=args.seen_validation,
                add_special=args.add_special,
                add_multiple_special=args.add_multiple_special,
                add_specific_loc=args.add_specific_loc,
                wo_shuffle=args.wo_shuffle,
                add_new_split=args.add_new_split,
                qk_replace=args.qk_replace,
                qformer=args.qformer,
            )
            def collate_fn(examples):
                videos = [example["instance_video"] for example in examples]
                prompts = [example["instance_prompt"] for example in examples]
                if 'instance_ref_image' in examples[0]:
                    ref_images = [example["instance_ref_image"] for example in examples]
                else:
                    ref_images = None

                # Stack the videos
                if args.use_latent:
                    videos = torch.cat(videos, dim=0)
                else:
                    videos = torch.stack(videos)
                videos = videos.to(memory_format=torch.contiguous_format).float()

                batch = {
                    "videos": videos,
                    "prompts": prompts,
                }
                if ref_images is not None:
                    if (not args.vae_add) and (not args.cross_attend) and (not args.cross_attend_text) and (not args.qk_replace):
                        batch["ref_images"] = ref_images
                    else:
                        batch["ref_images"] = torch.cat(ref_images, dim=0).to(memory_format=torch.contiguous_format).float()
                return batch
   
        def worker_init_fn(worker_id):
            seed = torch.initial_seed() % 2**32
            np.random.seed(seed)
            random.seed(seed)
        train_dataloader = DataLoader(
            train_dataset,
            batch_size=args.train_batch_size,
            shuffle=True,
            # collate_fn=collate_fn_with_args,
            collate_fn=collate_fn,
            num_workers=args.dataloader_num_workers,
            prefetch_factor=4,
            worker_init_fn=worker_init_fn,
        )

    # Scheduler and math around the number of training steps.
    overrode_max_train_steps = False
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if args.max_train_steps is None:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
        overrode_max_train_steps = True

    if use_deepspeed_scheduler:
        from accelerate.utils import DummyScheduler

        lr_scheduler = DummyScheduler(
            name=args.lr_scheduler,
            optimizer=optimizer,
            total_num_steps=args.max_train_steps * accelerator.num_processes,
            num_warmup_steps=args.lr_warmup_steps * accelerator.num_processes,
        )
    else:
        lr_scheduler = get_scheduler(
            args.lr_scheduler,
            optimizer=optimizer,
            num_warmup_steps=args.lr_warmup_steps * accelerator.num_processes,
            num_training_steps=args.max_train_steps * accelerator.num_processes,
            num_cycles=args.lr_num_cycles,
            power=args.lr_power,
        )

    # Prepare everything with our `accelerator`.
    transformer, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
        transformer, optimizer, train_dataloader, lr_scheduler
    )

    # We need to recalculate our total training steps as the size of the training dataloader may have changed.
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if overrode_max_train_steps:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
    # Afterwards we recalculate our number of training epochs
    args.num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)

    # We need to initialize the trackers we use, and also store our configuration.
    # The trackers initializes automatically on the main process.
    if accelerator.is_main_process:
        tracker_name = args.tracker_name or "cogvideox-lora"
        accelerator.init_trackers(tracker_name, config=vars(args))

    # Train!
    total_batch_size = args.train_batch_size * accelerator.num_processes * args.gradient_accumulation_steps
    num_trainable_parameters = sum(param.numel() for model in params_to_optimize for param in model["params"])

    logger.info("***** Running training *****")
    logger.info(f"  Num trainable parameters = {num_trainable_parameters}")
    logger.info(f"  Num examples = {len(train_dataset)}")
    logger.info(f"  Num batches each epoch = {len(train_dataloader)}")
    logger.info(f"  Num epochs = {args.num_train_epochs}")
    logger.info(f"  Instantaneous batch size per device = {args.train_batch_size}")
    logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
    logger.info(f"  Gradient accumulation steps = {args.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {args.max_train_steps}")
    print("***** Running training *****")
    print(f"  Num trainable parameters = {num_trainable_parameters}")
    print(f"  Num examples = {len(train_dataset)}")
    print(f"  Num batches each epoch = {len(train_dataloader)}")
    print(f"  Num epochs = {args.num_train_epochs}")
    print(f"  Instantaneous batch size per device = {args.train_batch_size}")
    print(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
    print(f"  Gradient accumulation steps = {args.gradient_accumulation_steps}")
    global_step = 0
    first_epoch = 0

    # Potentially load in the weights and states from a previous save
    if not args.resume_from_checkpoint:
        initial_global_step = 0
    else:
        if (not args.second_stage):
            if args.resume_from_checkpoint != "latest":
                path = os.path.basename(args.resume_from_checkpoint)
            else:
                # Get the mos recent checkpoint
                dirs = os.listdir(args.output_dir)
                dirs = [d for d in dirs if d.startswith("checkpoint")]
                dirs = sorted(dirs, key=lambda x: int(x.split("-")[1]))
                path = dirs[-1] if len(dirs) > 0 else None

            if path is None:
                accelerator.print(
                    f"Checkpoint '{args.resume_from_checkpoint}' does not exist. Starting a new training run."
                )
                args.resume_from_checkpoint = None
                initial_global_step = 0
            else:
                accelerator.print(f"Resuming from checkpoint {path}")
                accelerator.load_state(os.path.join(args.output_dir, path))
                global_step = int(path.split("-")[1])

                initial_global_step = global_step
                first_epoch = global_step // num_update_steps_per_epoch
        else:
            accelerator.print(
                f"Resuming from checkpoint {args.resume_from_checkpoint} for the second stage/joint training"
            )
            accelerator.load_state(args.resume_from_checkpoint)
            global_step = 0
            initial_global_step = 0
            first_epoch = 0

    progress_bar = tqdm(
        range(0, args.max_train_steps),
        initial=initial_global_step,
        desc="Steps",
        # Only show the progress bar once on each machine.
        disable=not accelerator.is_local_main_process,
    )
    if not args.use_latent:  
        vae_scale_factor_spatial = 2 ** (len(vae.config.block_out_channels) - 1)
    else:
        vae_scale_factor_spatial = 8 # 5B

    # For DeepSpeed training
    model_config = transformer.module.config if hasattr(transformer, "module") else transformer.config
    # from tqdm import tqdm
    for epoch in range(first_epoch, args.num_train_epochs):
        transformer.train()
        # update random seed
        # set_seed(args.seed + epoch)
        for step, batch in enumerate(train_dataloader):
            if (epoch == first_epoch and step == 0)and args.inference:
                break
            # set_seed(args.seed + epoch)
            models_to_accumulate = [transformer]

            with accelerator.accumulate(models_to_accumulate):
                # videos = batch["videos"].to(accelerator.device, dtype=vae.dtype)
                # print('Batch shape: ', batch["videos"].shape)
                videos = batch["videos"].to(accelerator.device, dtype=weight_dtype)
                videos = videos.permute(0, 2, 1, 3, 4).to(dtype=weight_dtype)  # [B, F, C, H, W]
                if not args.use_latent: # if use videos directly in end-to-end manner
                    vae.eval()
                    with torch.no_grad():
                        latent_dist = vae.encode(videos).latent_dist
                    model_input = latent_dist.sample() * vae.config.scaling_factor
                    model_input = model_input.permute(0, 2, 1, 3, 4)
                else: # if use latent vectors directly loaded with pre-extracted numpy files
                    if args.second_stage:
                        model_input = videos
                    else:
                        latent_dist = videos * 0.7
                        model_input = latent_dist
                
                
                prompts = batch["prompts"]
                if args.second_stage:
                    if args.second_stage_ref_image:
                        images = batch["ref_images"].permute(0, 2, 1, 3, 4).to(dtype=weight_dtype)  # [B, F, C, H, W]
                    else:
                        pass
                else:
                    if train_dataset.dataset_name == 'customization':
                        if (not args.vae_add) and (not args.cross_attend) and (not args.cross_attend_text) and (not args.qk_replace) and (not args.qformer):
                            image_processor = clip_processor.image_processor
                            images = image_processor(batch["ref_images"], return_tensors="pt").to(accelerator.device)
                        elif args.qformer:
                            images = torch.stack(batch['ref_images'], dim=0).to(accelerator.device, dtype=weight_dtype)
                        else:
                            images = batch["ref_images"].permute(0, 2, 1, 3, 4).to(dtype=weight_dtype)  # [B, F, C, H, W]
                            images = images * 0.7
                            # print(images.shape)
                        
                # encode prompts
                prompt_embeds, clip_prompt_embeds = compute_prompt_embeddings(
                    tokenizer,
                    text_encoder,
                    clip_tokenizer,
                    clip_text_encoder,
                    prompts,  # Now correctly passing 'prompts'
                    model_config.max_text_seq_length,
                    accelerator.device,
                    weight_dtype,
                    requires_grad=False,
                )
                # Process images
                if args.second_stage and (not args.joint_train):
                    if args.second_stage_ref_image:
                        image_input = images
                    else:
                        image_input = None
                else:
                    if (not args.vae_add) and (not args.cross_attend) and (not args.cross_attend_text) and (not args.qk_replace) and (not args.qformer):
                        if images is not None and 'pixel_values' in images:
                            image_input = images['pixel_values'].to(device=accelerator.device, dtype=weight_dtype)
                    else:
                        image_input = images
                    # # Projection through the linear layer to match dimension
                    # prompt_embeds = torch.cat([prompt_embeds, clip_prompt_embeds, image_embeds], dim=1) # FIXME (Learn additional separate projection layer to match the dimension)

                # Sample noise that will be added to the latents
                noise = torch.randn_like(model_input)
                batch_size, num_frames, num_channels, height, width = model_input.shape

                # Sample a random timestep for each image
                timesteps = torch.randint(
                    0, scheduler.config.num_train_timesteps, (batch_size,), device=model_input.device
                )
                timesteps = timesteps.long()
                if args.joint_train:
                    image_rotary_emb_src = (
                        prepare_rotary_positional_embeddings(
                            height=args.height,
                            width=args.width,
                            num_frames=num_frames + 1,
                            vae_scale_factor_spatial=vae_scale_factor_spatial,
                            patch_size=model_config.patch_size,
                            attention_head_dim=model_config.attention_head_dim,
                            device=accelerator.device,
                        )
                    )
                    ref_image_rotary_emb = (image_rotary_emb_src[0][:1350,...], image_rotary_emb_src[1][:1350,...])
                    image_rotary_emb = (image_rotary_emb_src[0][1350:,...], image_rotary_emb_src[1][1350:,...])
                else:
                    if args.pos_embed_inf_match and (not args.second_stage):
                        if args.non_shared_pos_embed:
                            image_rotary_emb_src = (
                                prepare_rotary_positional_embeddings(
                                    height=args.height,
                                    width=args.width,
                                    num_frames=14,
                                    vae_scale_factor_spatial=vae_scale_factor_spatial,
                                    patch_size=model_config.patch_size,
                                    attention_head_dim=model_config.attention_head_dim,
                                    device=accelerator.device,
                                )
                                if model_config.use_rotary_positional_embeddings
                                else None
                            )
                            if args.random_pos:
                                # Randomly select a location for the positional embeddings between 0 and 49 (inclusive)
                                random_loc = random.randint(0, 12)
                                ref_image_rotary_emb = (image_rotary_emb_src[0][1350 * random_loc:1350 * (random_loc + 1),...], image_rotary_emb_src[1][1350 * random_loc:1350 * (random_loc + 1),...])
                                image_rotary_emb = (image_rotary_emb_src[0][1350 * (random_loc + 1):1350 * (random_loc + 2),...], image_rotary_emb_src[1][1350 * (random_loc + 1):1350 * (random_loc + 2),...])
                            else:
                                image_rotary_emb = (image_rotary_emb_src[0][1350:2700,...], image_rotary_emb_src[1][1350:2700,...])
                                ref_image_rotary_emb = (image_rotary_emb_src[0][:1350,...], image_rotary_emb_src[1][:1350,...])
                        else:
                            # Prepare rotary embeds
                            image_rotary_emb = (
                                prepare_rotary_positional_embeddings(
                                    height=args.height,
                                    width=args.width,
                                    num_frames=13,
                                    vae_scale_factor_spatial=vae_scale_factor_spatial,
                                    patch_size=model_config.patch_size,
                                    attention_head_dim=model_config.attention_head_dim,
                                    device=accelerator.device,
                                )
                                if model_config.use_rotary_positional_embeddings
                                else None
                            )
                            # Get the first one only for training
                            image_rotary_emb = (image_rotary_emb[0][:1350,...], image_rotary_emb[1][:1350,...])
                    else:
                        if args.second_stage_ref_image:
                            image_rotary_emb = (
                                prepare_rotary_positional_embeddings(
                                    height=args.height,
                                    width=args.width,
                                    num_frames=args.max_num_frames // 4 + 2,
                                    vae_scale_factor_spatial=vae_scale_factor_spatial,
                                    patch_size=model_config.patch_size,
                                    attention_head_dim=model_config.attention_head_dim,
                                    device=accelerator.device,
                                )
                                if model_config.use_rotary_positional_embeddings
                                else None
                            )
                            ref_image_rotary_emb = (image_rotary_emb[0][:1350,...], image_rotary_emb[1][:1350,...])
                            image_rotary_emb = (image_rotary_emb[0][1350:,...], image_rotary_emb[1][1350:,...])
                            
                        else:
                            # Prepare rotary embeds
                            image_rotary_emb = (
                                prepare_rotary_positional_embeddings(
                                    height=args.height,
                                    width=args.width,
                                    num_frames=args.max_num_frames // 4 + 1,
                                    vae_scale_factor_spatial=vae_scale_factor_spatial,
                                    patch_size=model_config.patch_size,
                                    attention_head_dim=model_config.attention_head_dim,
                                    device=accelerator.device,
                                )
                                if model_config.use_rotary_positional_embeddings
                                else None
                            )
                    # Add noise to the model input according to the noise magnitude at each timestep
                    # (this is the forward diffusion process)
                
                if args.input_noise_fix:
                    print('Shape of model_input: ', model_input.shape)
                    print('Shape of image_input: ', image_input.shape)
                    # Concatenate model_input and image_input along the batch dimension
                    concat_input = torch.cat([model_input, image_input], dim=0)

                    # Generate noise for the concatenated input
                    concat_noise = torch.randn_like(concat_input)

                    # Apply noise to the concatenated input
                    noisy_concat_input = scheduler.add_noise(concat_input, concat_noise, timesteps)

                    # Split back into model_input and image_input
                    noisy_model_input, noisy_image_input = torch.chunk(noisy_concat_input, chunks=2, dim=0)
                else:
                    noisy_model_input = scheduler.add_noise(model_input, noise, timesteps)
                # Predict the noise residual
                # print('SHAPE OF NOISY MODEL INPUT >>>', noisy_model_input.shape)
                # print('SHAPE OF IMAGE ROTARY EMB >>>', image_rotary_emb[0].shape)
                if args.joint_train:
                    pass
                else:
                    if args.second_stage:
                        if not args.second_stage_ref_image:
                            ref_image_rotary_emb = None
                        else:
                            pass
                    else:
                        if args.non_shared_pos_embed:
                            ref_image_rotary_emb = ref_image_rotary_emb
                        else:
                            ref_image_rotary_emb = image_rotary_emb
                model_output = transformer(
                    hidden_states=noisy_model_input,
                    ref_img_states=noisy_image_input if args.input_noise_fix else image_input,
                    encoder_hidden_states=prompt_embeds,
                    clip_prompt_embeds=clip_prompt_embeds,
                    timestep=timesteps,
                    image_rotary_emb=image_rotary_emb,
                    ref_image_rotary_emb=ref_image_rotary_emb,
                    return_dict=False,
                    customization=True,
                    t5_first=t5_first,
                    concatenated_all=concatenated_all,
                    reduce_token=reduce_token,
                    add_token=args.add_token,
                    zero_conv_add=args.zero_conv_add,
                    vae_add=args.vae_add,
                    pos_embed=args.pos_embed,
                    cross_attend=args.cross_attend,
                    cross_attend_text=args.cross_attend_text,
                    layernorm_fix=args.layernorm_fix,
                    text_only_norm_final=args.text_only_norm_final,
                    second_stage_ref_image=args.second_stage_ref_image,
                    joint_train=args.joint_train,
                )[0]
                model_pred = scheduler.get_velocity(model_output, noisy_model_input, timesteps)

                alphas_cumprod = scheduler.alphas_cumprod[timesteps]
                weights = 1 / (1 - alphas_cumprod)
                while len(weights.shape) < len(model_pred.shape):
                    weights = weights.unsqueeze(-1)

                target = model_input

                loss = torch.mean((weights * (model_pred - target) ** 2).reshape(batch_size, -1), dim=1)
                loss = loss.mean()
                accelerator.backward(loss)

                if accelerator.sync_gradients:
                    params_to_clip = transformer.parameters()
                    accelerator.clip_grad_norm_(params_to_clip, args.max_grad_norm)

                if accelerator.state.deepspeed_plugin is None:
                    optimizer.step()
                    optimizer.zero_grad()

                lr_scheduler.step()

        # Checks if the accelerator has performed an optimization step behind the scenes
            if accelerator.sync_gradients:
                progress_bar.update(1)
                global_step += 1

                if accelerator.is_main_process:
                    if global_step % args.checkpointing_steps == 0:
                        # _before_ saving state, check if this save would set us over the `checkpoints_total_limit`
                        if args.checkpoints_total_limit is not None:
                            checkpoints = os.listdir(args.output_dir)
                            checkpoints = [d for d in checkpoints if d.startswith("checkpoint")]
                            checkpoints = sorted(checkpoints, key=lambda x: int(x.split("-")[1]))

                            # before we save the new checkpoint, we need to have at _most_ `checkpoints_total_limit - 1` checkpoints
                            if len(checkpoints) >= args.checkpoints_total_limit:
                                num_to_remove = len(checkpoints) - args.checkpoints_total_limit + 1
                                removing_checkpoints = checkpoints[0:num_to_remove]

                                logger.info(
                                    f"{len(checkpoints)} checkpoints already exist, removing {len(removing_checkpoints)} checkpoints"
                                )
                                logger.info(f"Removing checkpoints: {', '.join(removing_checkpoints)}")

                                for removing_checkpoint in removing_checkpoints:
                                    removing_checkpoint = os.path.join(args.output_dir, removing_checkpoint)
                                    shutil.rmtree(removing_checkpoint)

                        save_path = os.path.join(args.output_dir, f"checkpoint-{global_step}")
                        accelerator.save_state(save_path)
                        logger.info(f"Saved state to {save_path}")

            logs = {"loss": loss.detach().item(), "lr": lr_scheduler.get_last_lr()[0]}
            progress_bar.set_postfix(**logs)
            accelerator.log(logs, step=global_step)

            if global_step >= args.max_train_steps: 
                break

        if accelerator.is_main_process: 
                # print('saving now')
                # epoch = 1
            
            if args.inference or (args.validation_prompt is not None and (epoch + 1) % args.validation_epochs == 0):
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
                vae.to(accelerator.device, dtype=weight_dtype)
                pipe = CustomCogVideoXPipeline.from_pretrained(
                    args.pretrained_model_name_or_path,
                    transformer=unwrap_model(transformer),
                    text_encoder=unwrap_model(text_encoder),
                    clip_tokenizer=clip_tokenizer,
                    clip_text_encoder=unwrap_model(clip_text_encoder),
                    vae=unwrap_model(vae),
                    scheduler=scheduler,
                    revision=args.revision,
                    variant=args.variant,
                    torch_dtype=weight_dtype,
                    customization=True,
                    zero_conv_add=args.zero_conv_add,
                    vae_add=args.vae_add,
                    cross_attend=args.cross_attend,
                    cross_attend_text=args.cross_attend_text,
                    qk_replace=args.qk_replace
                )

                # validation_prompts = args.validation_prompt.split(args.validation_prompt_separator)
                # for validation_prompt in validation_prompts:
                if args.seen_validation:
                    args.validation_reference_image = "../seen_samples/omini_right/"
                val_len = len(os.listdir(args.validation_reference_image))
                for i in range(val_len):
                    validation_ref_img = os.path.join(args.validation_reference_image, os.listdir(args.validation_reference_image)[i])
                    vid_id = os.listdir(args.validation_reference_image)[i].split('.')[0]
                    validation_prompt = train_dataset.val_instance_prompt_dict[vid_id]
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
                        "add_token": args.add_token,
                        'zero_conv_add': args.zero_conv_add,
                        'vae_add' : args.vae_add,
                        'pos_embed' : args.pos_embed if args.second_stage is not True else True,
                        'cross_attend' : args.cross_attend,
                        'cross_attend_text' : args.cross_attend_text,
                        'input_noise_fix' : args.input_noise_fix,
                        'output_dir' : args.output_dir,
                        'save_every_timestep' : args.save_every_timestep,
                        'layernorm_fix': args.layernorm_fix if args.second_stage is not True else True,
                        'text_only_norm_final': args.text_only_norm_final,
                        'non_shared_pos_embed': args.non_shared_pos_embed,
                    }

                    validation_outputs = log_validation(
                        pipe=pipe,
                        args=args,
                        accelerator=accelerator,
                        pipeline_args=pipeline_args,
                        epoch=epoch,
                    )
        if args.inference: # do inference_only, so no training
            break
    if args.inference:
        # exit program
        import sys
        sys.exit(0)



    # Save the lora layers
    accelerator.wait_for_everyone()
    # Modified final inference section
    if accelerator.is_main_process:
        transformer = unwrap_model(transformer)
        dtype = (
            torch.float16
            if args.mixed_precision == "fp16"
            else torch.bfloat16
            if args.mixed_precision == "bf16"
            else torch.float32
        )
        transformer = transformer.to(dtype)
        transformer_lora_layers = get_peft_model_state_dict(transformer)

        # Save LoRA weights and additional components
        CogVideoXPipeline.save_lora_weights(
            save_directory=args.output_dir,
            transformer_lora_layers=transformer_lora_layers,
        )

        # Save projection layers and vision encoder separately
        torch.save(transformer.T5ProjectionLayer.state_dict(), 
                os.path.join(args.output_dir, "T5ProjectionLayer.safetensors"))
        torch.save(transformer.CLIPTextProjectionLayer.state_dict(), 
                os.path.join(args.output_dir, "CLIPTextProjectionLayer.safetensors"))
        torch.save(transformer.CLIPVisionProjectionLayer.state_dict(), 
                os.path.join(args.output_dir, "CLIPVisionProjectionLayer.safetensors"))
        torch.save(transformer.reference_vision_encoder.state_dict(), 
                os.path.join(args.output_dir, "reference_vision_encoder.safetensors"))

        # Final test inference
        pipe = CustomCogVideoXPipeline.from_pretrained(
            args.pretrained_model_name_or_path,
            revision=args.revision,
            variant=args.variant,
            torch_dtype=weight_dtype,
            customization=True
        )
        pipe.scheduler = CogVideoXDPMScheduler.from_config(pipe.scheduler.config)

        if args.enable_slicing:
            pipe.vae.enable_slicing()
        if args.enable_tiling:
            pipe.vae.enable_tiling()

        # Load LoRA weights
        lora_scaling = args.lora_alpha / args.rank
        pipe.load_lora_weights(args.output_dir, adapter_name="cogvideox-lora")
        pipe.set_adapters(["cogvideox-lora"], [lora_scaling])

        # Load additional components
        pipe.transformer.T5ProjectionLayer.load_state_dict(
            torch.load(os.path.join(args.output_dir, "T5ProjectionLayer.pth")))
        pipe.transformer.CLIPTextProjectionLayer.load_state_dict(
            torch.load(os.path.join(args.output_dir, "CLIPTextProjectionLayer.pth")))
        pipe.transformer.CLIPVisionProjectionLayer.load_state_dict(
            torch.load(os.path.join(args.output_dir, "CLIPVisionProjectionLayer.pth")))
        pipe.transformer.reference_vision_encoder.load_state_dict(
            torch.load(os.path.join(args.output_dir, "reference_vision_encoder.pth")))

        # Run inference
        validation_outputs = []
        if args.validation_prompt and args.num_validation_videos > 0:
            validation_prompts = args.validation_prompt.split(args.validation_prompt_separator)
            for validation_prompt in validation_prompts:
                pipeline_args = {
                    "prompt": validation_prompt,
                    "guidance_scale": args.guidance_scale,
                    "use_dynamic_cfg": args.use_dynamic_cfg,
                    "height": args.height,
                    "width": args.width,
                }
                
                # Add reference image path for customization
                if args.dataset_name == 'customization' and hasattr(args, 'validation_reference_image'):
                    pipeline_args['reference_image_path'] = args.validation_reference_image

                video = log_validation(
                    pipe=pipe,
                    args=args,
                    accelerator=accelerator,
                    pipeline_args=pipeline_args,
                    epoch=epoch,
                    is_final_validation=True,
                )
                validation_outputs.extend(video)

        if args.push_to_hub:
            save_model_card(
                repo_id,
                videos=validation_outputs,
                base_model=args.pretrained_model_name_or_path,
                validation_prompt=args.validation_prompt,
                repo_folder=args.output_dir,
                fps=args.fps,
            )
            upload_folder(
                repo_id=repo_id,
                folder_path=args.output_dir,
                commit_message="End of training",
                ignore_patterns=["step_*", "epoch_*"],
            )

    accelerator.end_training()


if __name__ == "__main__":
    print('Code executed')
    args = get_args()
    print('Args:', args)
    main(args)