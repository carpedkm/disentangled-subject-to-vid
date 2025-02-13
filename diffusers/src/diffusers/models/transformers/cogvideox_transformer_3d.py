# Copyright 2024 The CogVideoX team, Tsinghua University & ZhipuAI and The HuggingFace Team.
# All rights reserved.
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

from typing import Any, Dict, Optional, Tuple, Union

import math
import torch
from torch import nn

import torch.nn.functional as F

from ...configuration_utils import ConfigMixin, register_to_config
from ...loaders import PeftAdapterMixin
from ...utils import USE_PEFT_BACKEND, is_torch_version, logging, scale_lora_layers, unscale_lora_layers
from ...utils.torch_utils import maybe_allow_in_graph
from ..attention import Attention, FeedForward
from ..attention_processor import AttentionProcessor, CogVideoXAttnProcessor2_0, FusedCogVideoXAttnProcessor2_0
from ..embeddings import CogVideoXPatchEmbed, TimestepEmbedding, Timesteps
from ..modeling_outputs import Transformer2DModelOutput
from ..modeling_utils import ModelMixin
from ..normalization import AdaLayerNorm, CogVideoXLayerNormZero

from transformers import CLIPVisionConfig, CLIPProcessor, CLIPTokenizer, CLIPTextModel, CLIPVisionModel

logger = logging.get_logger(__name__)  # pylint: disable=invalid-name


@maybe_allow_in_graph
class CogVideoXBlock(nn.Module):
    r"""
    Transformer block used in [CogVideoX](https://github.com/THUDM/CogVideo) model.

    Parameters:
        dim (`int`):
            The number of channels in the input and output.
        num_attention_heads (`int`):
            The number of heads to use for multi-head attention.
        attention_head_dim (`int`):
            The number of channels in each head.
        time_embed_dim (`int`):
            The number of channels in timestep embedding.
        dropout (`float`, defaults to `0.0`):
            The dropout probability to use.
        activation_fn (`str`, defaults to `"gelu-approximate"`):
            Activation function to be used in feed-forward.
        attention_bias (`bool`, defaults to `False`):
            Whether or not to use bias in attention projection layers.
        qk_norm (`bool`, defaults to `True`):
            Whether or not to use normalization after query and key projections in Attention.
        norm_elementwise_affine (`bool`, defaults to `True`):
            Whether to use learnable elementwise affine parameters for normalization.
        norm_eps (`float`, defaults to `1e-5`):
            Epsilon value for normalization layers.
        final_dropout (`bool` defaults to `False`):
            Whether to apply a final dropout after the last feed-forward layer.
        ff_inner_dim (`int`, *optional*, defaults to `None`):
            Custom hidden dimension of Feed-forward layer. If not provided, `4 * dim` is used.
        ff_bias (`bool`, defaults to `True`):
            Whether or not to use bias in Feed-forward layer.
        attention_out_bias (`bool`, defaults to `True`):
            Whether or not to use bias in Attention output projection layer.
    """

    def __init__(
        self,
        dim: int,
        num_attention_heads: int,
        attention_head_dim: int,
        time_embed_dim: int,
        dropout: float = 0.0,
        activation_fn: str = "gelu-approximate",
        attention_bias: bool = False,
        qk_norm: bool = True,
        norm_elementwise_affine: bool = True,
        norm_eps: float = 1e-5,
        final_dropout: bool = True,
        ff_inner_dim: Optional[int] = None,
        ff_bias: bool = True,
        attention_out_bias: bool = True,
    ):
        super().__init__()

        # 1. Self Attention
        self.norm1 = CogVideoXLayerNormZero(time_embed_dim, dim, norm_elementwise_affine, norm_eps, bias=True)

        self.attn1 = Attention(
            query_dim=dim,
            dim_head=attention_head_dim,
            heads=num_attention_heads,
            qk_norm="layer_norm" if qk_norm else None,
            eps=1e-6,
            bias=attention_bias,
            out_bias=attention_out_bias,
            processor=CogVideoXAttnProcessor2_0(),
        )

        # 2. Feed Forward
        self.norm2 = CogVideoXLayerNormZero(time_embed_dim, dim, norm_elementwise_affine, norm_eps, bias=True)

        self.ff = FeedForward(
            dim,
            dropout=dropout,
            activation_fn=activation_fn,
            final_dropout=final_dropout,
            inner_dim=ff_inner_dim,
            bias=ff_bias,
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        temb: torch.Tensor,
        enc_hidden_states1: Optional[torch.Tensor] = None,
        image_rotary_emb: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        embed_ref_img: bool = False,
        ref_img_seq_start: Optional[int] = None,
        ref_img_seq_end: Optional[int] = None,
        position_delta: Optional[torch.Tensor] = None,
        timestep: Optional[int] = None,
        layer : Optional[int] = None,
        ref_image_rotary_emb : Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> torch.Tensor:
        text_seq_length = encoder_hidden_states.size(1)

        # norm & modulate
        if enc_hidden_states1 is not None:
            norm_hidden_states, norm_encoder_hidden_states, norm_cond_hidden_states, gate_msa, enc_gate_msa, cond_gate_msa = self.norm1(
                hidden_states=hidden_states, 
                encoder_hidden_states=encoder_hidden_states, 
                cond_hidden_states=enc_hidden_states1,
                temb=temb,
            )
        else:
            norm_cond_hidden_states = None
            norm_hidden_states, norm_encoder_hidden_states, _, gate_msa, enc_gate_msa, _ = self.norm1(
                hidden_states=hidden_states, 
                encoder_hidden_states=encoder_hidden_states, 
                cond_hidden_states=None,
                temb=temb,
            )
        
        if norm_cond_hidden_states is not None:
            norm_encoder_hidden_states = torch.cat([norm_encoder_hidden_states, norm_cond_hidden_states], dim=1)
        # attention
        attn_hidden_states, attn_encoder_hidden_states = self.attn1(
            hidden_states=norm_hidden_states,
            encoder_hidden_states=norm_encoder_hidden_states,
            image_rotary_emb=image_rotary_emb,
            embed_ref_img=embed_ref_img,
            ref_img_seq_start=ref_img_seq_start,
            ref_img_seq_end=ref_img_seq_end,
            position_delta=position_delta,
            timestep=timestep,
            layer=layer,
            ref_image_rotary_emb=ref_image_rotary_emb,
        )
        if enc_hidden_states1 is not None:
            attn_cond_hidden_states = attn_encoder_hidden_states[:,text_seq_length:]
            attn_encoder_hidden_states = attn_encoder_hidden_states[:,:text_seq_length]
            
        hidden_states = hidden_states + gate_msa * attn_hidden_states
        encoder_hidden_states = encoder_hidden_states + enc_gate_msa * attn_encoder_hidden_states
        if enc_hidden_states1 is not None:
            enc_hidden_states1 = enc_hidden_states1 + cond_gate_msa * attn_cond_hidden_states

        # norm & modulate
        if enc_hidden_states1 is not None:
            norm_hidden_states, norm_encoder_hidden_states, norm_cond_hidden_states, gate_ff, enc_gate_ff, cond_gate_ff = self.norm2(
                hidden_states=hidden_states, 
                encoder_hidden_states=encoder_hidden_states, 
                cond_hidden_states=enc_hidden_states1,
                temb=temb,
            )
        else:
            norm_cond_hidden_states = None
            norm_hidden_states, norm_encoder_hidden_states, _, gate_ff, enc_gate_ff, _ = self.norm2(
                hidden_states=hidden_states, 
                encoder_hidden_states=encoder_hidden_states, 
                cond_hidden_states=None,
                temb=temb,
            )
            

        if norm_cond_hidden_states is not None:
            seq_len_temp = torch.cat([norm_encoder_hidden_states, norm_cond_hidden_states], dim=1).shape[1]
            # feed-forward
            norm_hidden_states = torch.cat([norm_encoder_hidden_states, norm_cond_hidden_states, norm_hidden_states], dim=1)
        else:
            norm_hidden_states = torch.cat([norm_encoder_hidden_states, norm_hidden_states], dim=1)
        ff_output = self.ff(norm_hidden_states)

        if norm_cond_hidden_states is not None:
            hidden_states = hidden_states + gate_ff * ff_output[:, seq_len_temp:]
            encoder_hidden_states = encoder_hidden_states + enc_gate_ff * ff_output[:, :text_seq_length]
            enc_hidden_states1 = enc_hidden_states1 + cond_gate_ff * ff_output[:, text_seq_length:seq_len_temp]
            # print(hidden_states.shape, encoder_hidden_states.shape, enc_hidden_states1.shape)
            return hidden_states, encoder_hidden_states, enc_hidden_states1
        else:
            hidden_states = hidden_states + gate_ff * ff_output[:, text_seq_length:]
            encoder_hidden_states = encoder_hidden_states + enc_gate_ff * ff_output[:, :text_seq_length]

            return hidden_states, encoder_hidden_states
        
def get_sinusoidal_positional_embeddings(seq_length, embed_dim, device):
    position = torch.arange(seq_length, dtype=torch.bfloat16, device=device).unsqueeze(1)
    div_term = torch.exp(torch.arange(0, embed_dim, 2, device=device) * -(math.log(10000.0) / embed_dim))
    pos_embedding = torch.zeros(seq_length, embed_dim, device=device)
    pos_embedding[:, 0::2] = torch.sin(position * div_term)
    pos_embedding[:, 1::2] = torch.cos(position * div_term)
    return pos_embedding

class ProjectionLayer(nn.Module):
    def __init__(self, in_features, out_features):
        super(ProjectionLayer, self).__init__()
        self.linear = nn.Linear(in_features, out_features)

    def forward(self, x):
        return self.linear(x)
    


class CogVideoXTransformer3DModel(ModelMixin, ConfigMixin, PeftAdapterMixin):
    """
    A Transformer model for video-like data in [CogVideoX](https://github.com/THUDM/CogVideo).

    Parameters:
        num_attention_heads (`int`, defaults to `30`):
            The number of heads to use for multi-head attention.
        attention_head_dim (`int`, defaults to `64`):
            The number of channels in each head.
        in_channels (`int`, defaults to `16`):
            The number of channels in the input.
        out_channels (`int`, *optional*, defaults to `16`):
            The number of channels in the output.
        flip_sin_to_cos (`bool`, defaults to `True`):
            Whether to flip the sin to cos in the time embedding.
        time_embed_dim (`int`, defaults to `512`):
            Output dimension of timestep embeddings.
        text_embed_dim (`int`, defaults to `4096`):
            Input dimension of text embeddings from the text encoder.
        num_layers (`int`, defaults to `30`):
            The number of layers of Transformer blocks to use.
        dropout (`float`, defaults to `0.0`):
            The dropout probability to use.
        attention_bias (`bool`, defaults to `True`):
            Whether or not to use bias in the attention projection layers.
        sample_width (`int`, defaults to `90`):
            The width of the input latents.
        sample_height (`int`, defaults to `60`):
            The height of the input latents.
        sample_frames (`int`, defaults to `49`):
            The number of frames in the input latents. Note that this parameter was incorrectly initialized to 49
            instead of 13 because CogVideoX processed 13 latent frames at once in its default and recommended settings,
            but cannot be changed to the correct value to ensure backwards compatibility. To create a transformer with
            K latent frames, the correct value to pass here would be: ((K - 1) * temporal_compression_ratio + 1).
        patch_size (`int`, defaults to `2`):
            The size of the patches to use in the patch embedding layer.
        temporal_compression_ratio (`int`, defaults to `4`):
            The compression ratio across the temporal dimension. See documentation for `sample_frames`.
        max_text_seq_length (`int`, defaults to `226`):
            The maximum sequence length of the input text embeddings.
        activation_fn (`str`, defaults to `"gelu-approximate"`):
            Activation function to use in feed-forward.
        timestep_activation_fn (`str`, defaults to `"silu"`):
            Activation function to use when generating the timestep embeddings.
        norm_elementwise_affine (`bool`, defaults to `True`):
            Whether or not to use elementwise affine in normalization layers.
        norm_eps (`float`, defaults to `1e-5`):
            The epsilon value to use in normalization layers.
        spatial_interpolation_scale (`float`, defaults to `1.875`):
            Scaling factor to apply in 3D positional embeddings across spatial dimensions.
        temporal_interpolation_scale (`float`, defaults to `1.0`):
            Scaling factor to apply in 3D positional embeddings across temporal dimensions.
    """

    _supports_gradient_checkpointing = True

    @register_to_config
    def __init__(
        self,
        num_attention_heads: int = 30,
        attention_head_dim: int = 64,
        in_channels: int = 16,
        out_channels: Optional[int] = 16,
        flip_sin_to_cos: bool = True,
        freq_shift: int = 0,
        time_embed_dim: int = 512,
        text_embed_dim: int = 4096,
        num_layers: int = 30,
        dropout: float = 0.0,
        attention_bias: bool = True,
        sample_width: int = 90,
        sample_height: int = 60,
        sample_frames: int = 49,
        patch_size: int = 2,
        temporal_compression_ratio: int = 4,
        max_text_seq_length: int = 226,
        activation_fn: str = "gelu-approximate",
        timestep_activation_fn: str = "silu",
        norm_elementwise_affine: bool = True,
        norm_eps: float = 1e-5,
        spatial_interpolation_scale: float = 1.875,
        temporal_interpolation_scale: float = 1.0,
        use_rotary_positional_embeddings: bool = False,
        use_learned_positional_embeddings: bool = False,
        customization: bool = False,
        concatenated_all: bool = False,
        reduce_token: bool = False,
        zero_conv_add : bool = False,
        vae_add: bool = False,
        cross_attend : bool = False,
        cross_attend_text: bool = False,
        cross_attn_interval: int = 1,
        local_reference_scale: float = 1.,
        cross_attn_dim_head: int = 128,
        cross_attn_num_heads: int = 16,
        qk_replace: bool = False,
        qformer: bool = False,
        # cross_attn_kv_dim: int = 2048,
    ):
        
        self.qk_replace = qk_replace
        self.qformer = qformer
        super().__init__()
        inner_dim = num_attention_heads * attention_head_dim

        if not use_rotary_positional_embeddings and use_learned_positional_embeddings:
            raise ValueError(
                "There are no CogVideoX checkpoints available with disable rotary embeddings and learned positional "
                "embeddings. If you're using a custom model and/or believe this should be supported, please open an "
                "issue at https://github.com/huggingface/diffusers/issues."
            )
        # if customization:
        clip_vision_config = None # CLIPVisionConfig.from_pretrained("openai/clip-vit-base-patch16")
        if not vae_add and not cross_attend and not qformer:
            if zero_conv_add:
                self.text_sequence_aligner = None
                self.vision_sequence_aligner = None
                self.CLIPTextProjectionLayer2 = None
                self.CLIPVisionProjectionLayer2 = None
                # self.alpha = None
                # self.beta = None
            if concatenated_all is not True:
                self.reference_vision_encoder = None # CLIPVisionModel(clip_vision_config)
                self.T5ProjectionLayer = None # ProjectionLayer(in_features=4096, out_features=4096).to(dtype=torch.bfloat16)
                self.CLIPTextProjectionLayer = None # ProjectionLayer(in_features=512, out_features=4096).to(dtype=torch.bfloat16)
                self.CLIPVisionProjectionLayer = None # ProjectionLayer(in_features=768, out_features=4096).to(dtype=torch.bfloat16)
            else:
                self.reference_vision_encoder = None
                self.T5ProjectionLayer = None
        elif qformer:
            self.QformerAligner = None
        else:
            pass


            
    
            # all should be trainable # requires grad = True
            # self.reference_vision_encoder.requires_grad = True
            # self.T5ProjectionLayer.requires_grad = True
            # self.CLIPTextProjectionLayer.requires_grad = True
            # self.CLIPVisionProjectionLayer.requires_grad = True 
    
        # 1. Patch embedding
        self.patch_embed = CogVideoXPatchEmbed(
            patch_size=patch_size,
            in_channels=in_channels,
            embed_dim=inner_dim,
            text_embed_dim=text_embed_dim,
            bias=True,
            sample_width=sample_width,
            sample_height=sample_height,
            sample_frames=sample_frames,
            temporal_compression_ratio=temporal_compression_ratio,
            max_text_seq_length=max_text_seq_length,
            spatial_interpolation_scale=spatial_interpolation_scale,
            temporal_interpolation_scale=temporal_interpolation_scale,
            use_positional_embeddings=not use_rotary_positional_embeddings,
            use_learned_positional_embeddings=use_learned_positional_embeddings,
        ).to(dtype=torch.bfloat16)
        self.embedding_dropout = nn.Dropout(dropout)

        # 2. Time embeddings
        self.time_proj = Timesteps(inner_dim, flip_sin_to_cos, freq_shift)
        self.time_embedding = TimestepEmbedding(inner_dim, time_embed_dim, timestep_activation_fn)
        
        # 3. Define spatio-temporal transformers blocks
        self.transformer_blocks = nn.ModuleList(
            [
                CogVideoXBlock(
                    dim=inner_dim,
                    num_attention_heads=num_attention_heads,
                    attention_head_dim=attention_head_dim,
                    time_embed_dim=time_embed_dim,
                    dropout=dropout,
                    activation_fn=activation_fn,
                    attention_bias=attention_bias,
                    norm_elementwise_affine=norm_elementwise_affine,
                    norm_eps=norm_eps,
                )
                for _ in range(num_layers)
            ]
        )
        self.norm_final = nn.LayerNorm(inner_dim, norm_eps, norm_elementwise_affine)

        # 4. Output blocks
        self.norm_out = AdaLayerNorm(
            embedding_dim=time_embed_dim,
            output_dim=2 * inner_dim,
            norm_elementwise_affine=norm_elementwise_affine,
            norm_eps=norm_eps,
            chunk_dim=1,
        )
        self.proj_out = nn.Linear(inner_dim, patch_size * patch_size * out_channels)

        self.gradient_checkpointing = False
        self.cross_attend = cross_attend
        self.cross_attend_text = cross_attend_text
        # self.num_cross_att
        if self.cross_attend:
            self.perceiver_cross_attention = None
        if self.cross_attend_text:
            self.perceiver_cross_attention_text = None
        self.num_layers = num_layers
        # Define modality embeddings
        # num_modalities = 3  # T5, CLIP text, CLIP vision
        # embed_dim = text_embed_dim  # Should match the embedding dimension
        # self.modality_embeddings = nn.Embedding(num_modalities, embed_dim)
        # nn.init.normal_(self.modality_embeddings.weight, mean=0.0, std=0.02)  # Initialize embeddings

    def _set_gradient_checkpointing(self, module, value=False):
        self.gradient_checkpointing = value

    @property
    # Copied from diffusers.models.unets.unet_2d_condition.UNet2DConditionModel.attn_processors
    def attn_processors(self) -> Dict[str, AttentionProcessor]:
        r"""
        Returns:
            `dict` of attention processors: A dictionary containing all attention processors used in the model with
            indexed by its weight name.
        """
        # set recursively
        processors = {}

        def fn_recursive_add_processors(name: str, module: torch.nn.Module, processors: Dict[str, AttentionProcessor]):
            if hasattr(module, "get_processor"):
                processors[f"{name}.processor"] = module.get_processor()

            for sub_name, child in module.named_children():
                fn_recursive_add_processors(f"{name}.{sub_name}", child, processors)

            return processors

        for name, module in self.named_children():
            fn_recursive_add_processors(name, module, processors)

        return processors

    # Copied from diffusers.models.unets.unet_2d_condition.UNet2DConditionModel.set_attn_processor
    def set_attn_processor(self, processor: Union[AttentionProcessor, Dict[str, AttentionProcessor]]):
        r"""
        Sets the attention processor to use to compute attention.

        Parameters:
            processor (`dict` of `AttentionProcessor` or only `AttentionProcessor`):
                The instantiated processor class or a dictionary of processor classes that will be set as the processor
                for **all** `Attention` layers.

                If `processor` is a dict, the key needs to define the path to the corresponding cross attention
                processor. This is strongly recommended when setting trainable attention processors.

        """
        count = len(self.attn_processors.keys())

        if isinstance(processor, dict) and len(processor) != count:
            raise ValueError(
                f"A dict of processors was passed, but the number of processors {len(processor)} does not match the"
                f" number of attention layers: {count}. Please make sure to pass {count} processor classes."
            )

        def fn_recursive_attn_processor(name: str, module: torch.nn.Module, processor):
            if hasattr(module, "set_processor"):
                if not isinstance(processor, dict):
                    module.set_processor(processor)
                else:
                    module.set_processor(processor.pop(f"{name}.processor"))

            for sub_name, child in module.named_children():
                fn_recursive_attn_processor(f"{name}.{sub_name}", child, processor)

        for name, module in self.named_children():
            fn_recursive_attn_processor(name, module, processor)

    # Copied from diffusers.models.unets.unet_2d_condition.UNet2DConditionModel.fuse_qkv_projections with FusedAttnProcessor2_0->FusedCogVideoXAttnProcessor2_0
    def fuse_qkv_projections(self):
        """
        Enables fused QKV projections. For self-attention modules, all projection matrices (i.e., query, key, value)
        are fused. For cross-attention modules, key and value projection matrices are fused.

        <Tip warning={true}>

        This API is 🧪 experimental.

        </Tip>
        """
        self.original_attn_processors = None

        for _, attn_processor in self.attn_processors.items():
            if "Added" in str(attn_processor.__class__.__name__):
                raise ValueError("`fuse_qkv_projections()` is not supported for models having added KV projections.")

        self.original_attn_processors = self.attn_processors

        for module in self.modules():
            if isinstance(module, Attention):
                module.fuse_projections(fuse=True)

        self.set_attn_processor(FusedCogVideoXAttnProcessor2_0())

    # Copied from diffusers.models.unets.unet_2d_condition.UNet2DConditionModel.unfuse_qkv_projections
    def unfuse_qkv_projections(self):
        """Disables the fused QKV projection if enabled.

        <Tip warning={true}>

        This API is 🧪 experimental.

        </Tip>

        """
        if self.original_attn_processors is not None:
            self.set_attn_processor(self.original_attn_processors)

    def forward(
        self,
        hidden_states: torch.Tensor,
        ref_img_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        clip_prompt_embeds: torch.Tensor,
        timestep: Union[int, float, torch.LongTensor],
        timestep_cond: Optional[torch.Tensor] = None,
        image_rotary_emb: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        ref_image_rotary_emb: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        attention_kwargs: Optional[Dict[str, Any]] = None,
        return_dict: bool = True,
        customization: bool = False,
        t5_first: bool = True,
        eval: bool = False,
        concatenated_all: bool = False,
        reduce_token: bool = False,
        add_token: bool = False,
        zero_conv_add: bool = False,
        vae_add: bool = False,
        pos_embed: bool = False,
        cross_attend: bool = False,
        cross_attend_text: bool = False,
        layernorm_fix: bool = False,
        text_only_norm_final: bool = False, 
        # qk_replace: bool = False,
    ):
        qk_replace = self.qk_replace
        qformer = self.qformer
        if customization:
            if not qformer:
                if not qk_replace:
                    if (not cross_attend) and (not cross_attend_text):
                        if not vae_add:
                            if add_token:
                                ref_img_states = self.reference_vision_encoder(ref_img_states).last_hidden_state
                                ref_img_states = torch.cat([ref_img_states, torch.zeros((ref_img_states.shape[0], ref_img_states.shape[1], 4096 - ref_img_states.shape[2])).to(ref_img_states.device)], dim=2)
                                clip_text_states = torch.cat([clip_prompt_embeds, torch.zeros((clip_prompt_embeds.shape[0], clip_prompt_embeds.shape[1], 4096 - clip_prompt_embeds.shape[2])).to(clip_prompt_embeds.device)], dim=2)
                                ref_img_states = self.CLIPVisionProjectionLayer(ref_img_states.to(dtype=torch.bfloat16).transpose(1,2)).transpose(1,2)
                                clip_text_states = self.CLIPTextProjectionLayer(clip_text_states.to(dtype=torch.bfloat16).transpose(1,2)).transpose(1,2)
                                # encoder_hidden_states = (encoder_hidden_states + ref_img_states + clip_text_states) / 3
                                encoder_hidden_states = (encoder_hidden_states + ref_img_states + clip_text_states) / (1 + torch.mean(torch.abs(ref_img_states), dim=(0,1,2)) + torch.mean(torch.abs(clip_text_states), dim=(0,1,2)))
                                encoder_hidden_states = self.T5ProjectionLayer(encoder_hidden_states)
                            elif zero_conv_add:
                                ref_img_states = self.reference_vision_encoder(ref_img_states).last_hidden_state
                                ref_img_states = self.vision_sequence_aligner(ref_img_states)
                                clip_text_states = self.text_sequence_aligner(clip_prompt_embeds)
                                ref_img_states = self.CLIPVisionProjectionLayer(ref_img_states.to(dtype=torch.bfloat16).transpose(1, 2)).transpose(1, 2)     
                                clip_text_states = self.CLIPTextProjectionLayer(clip_text_states.to(dtype=torch.bfloat16).transpose(1, 2)).transpose(1, 2)
                                ref_img_states = self.CLIPVisionProjectionLayer2(ref_img_states.transpose(1, 2)).transpose(1, 2)
                                clip_text_states = self.CLIPTextProjectionLayer2(clip_text_states.transpose(1, 2)).transpose(1, 2)
                                encoder_hidden_states = encoder_hidden_states + ref_img_states + clip_text_states
                                # encoder_hidden_states = self.T5ProjectionLayer(encoder_hidden_states)
                            else:
                                if concatenated_all:
                                    ref_img_states = self.reference_vision_encoder(ref_img_states).last_hidden_state
                                    #  match the feature dimension 4096 by padding zeros
                                    ref_img_states = torch.cat([ref_img_states, torch.zeros((ref_img_states.shape[0], ref_img_states.shape[1], 4096 - ref_img_states.shape[2])).to(ref_img_states.device)], dim=2)
                                    clip_text_states = torch.cat([clip_prompt_embeds, torch.zeros((clip_prompt_embeds.shape[0], clip_prompt_embeds.shape[1], 4096 - clip_prompt_embeds.shape[2])).to(clip_prompt_embeds.device)], dim=2)
                                    if eval:
                                        ref_img_states = torch.cat([ref_img_states, ref_img_states], dim=0)
                                    encoder_hidden_states = torch.cat([encoder_hidden_states, clip_text_states, ref_img_states], dim=1)
                                    if reduce_token is True: # reduce 500 -> 226
                                        encoder_hidden_states = self.T5ProjectionLayer(encoder_hidden_states.transpose(1,2)).transpose(1,2)
                                    else: # mix 4096 -> 4096
                                        encoder_hidden_states = self.T5ProjectionLayer(encoder_hidden_states)
                                else:
                                    ref_img_states = self.reference_vision_encoder(ref_img_states).last_hidden_state
                                    enc_hidden_states0 = self.T5ProjectionLayer(encoder_hidden_states.to(dtype=torch.bfloat16))
                                    enc_hidden_states1 = self.CLIPTextProjectionLayer(clip_prompt_embeds.to(dtype=torch.bfloat16))
                                    enc_hidden_states2 = self.CLIPVisionProjectionLayer(ref_img_states.to(dtype=torch.bfloat16))
                                    if eval:
                                        enc_hidden_states2 = torch.cat([enc_hidden_states2, enc_hidden_states2], dim=0)
                                    if t5_first is True:
                                        encoder_hidden_states = torch.cat([enc_hidden_states0, enc_hidden_states1, enc_hidden_states2], dim=1)
                                    else:
                                        encoder_hidden_states = torch.cat([enc_hidden_states2, enc_hidden_states1, enc_hidden_states0], dim=1)
                        else:
                            enc_hidden_states0 = encoder_hidden_states
                            enc_hidden_states1 = ref_img_states
                    elif vae_add is False and (cross_attend is True or cross_attend_text is True):
                        enc_hidden_states0 = encoder_hidden_states
                        ref_img_emb = ref_img_states
                        # print('>>>> REF IMG STATES SHAPE : ', ref_img_states.shape)
                    elif vae_add is True and (cross_attend is True or cross_attend_text is True):
                        enc_hidden_states0 = encoder_hidden_states
                        enc_hidden_states1 = ref_img_states
                        ref_img_emb = ref_img_states
                    else:
                        print('Something is Wrong >>>>>>>> RE CHECK')
                else:
                    # encoder_hidden_states = encoder_hidden_states
                    enc_hidden_states0 = encoder_hidden_states
                    enc_hidden_states1 = ref_img_states
            else:
                enc_hidden_states0 = encoder_hidden_states
                enc_hidden_states1 = ref_img_states
                enc_hidden_states1 = self.QformerAligner(ref_img_states)
        else:
            # encoder_hidden_states = self.T5ProjectionLayer(encoder_hidden_states)
            pass

        if attention_kwargs is not None:
            attention_kwargs = attention_kwargs.copy()
            lora_scale = attention_kwargs.pop("scale", 1.0)
        else:
            lora_scale = 1.0

        if USE_PEFT_BACKEND:
            # weight the lora layers by setting `lora_scale` for each PEFT layer
            scale_lora_layers(self, lora_scale)
        else:
            if attention_kwargs is not None and attention_kwargs.get("scale", None) is not None:
                logger.warning(
                    "Passing `scale` via `attention_kwargs` when not using the PEFT backend is ineffective."
                )

        batch_size, num_frames, channels, height, width = hidden_states.shape

        # 1. Time embedding
        timesteps = timestep
        t_emb = self.time_proj(timesteps)

        # timesteps does not contain any weights and will always return f32 tensors
        # but time_embedding might actually be running in fp16. so we need to cast here.
        # there might be better ways to encapsulate this.
        t_emb = t_emb.to(dtype=hidden_states.dtype)
        emb = self.time_embedding(t_emb, timestep_cond)


        
        # 2. Patch embedding
        # if not cross_attend: # ADDME
        if not qk_replace:
            if vae_add :
                enc_hidden_states0 = self.patch_embed.text_proj(enc_hidden_states0)
                # text_seq_length = enc_hidden_states0.shape[1]
                b_1, nf_1, c_1, h_1, w_1 = enc_hidden_states1.shape
                enc_hidden_states1 = enc_hidden_states1.reshape(-1, c_1, h_1, w_1)
                enc_hidden_states1 = self.patch_embed.proj(enc_hidden_states1)
                enc_hidden_states1 = enc_hidden_states1.view(b_1, nf_1, *enc_hidden_states1.shape[1:])
                enc_hidden_states1 = enc_hidden_states1.flatten(3).transpose(2, 3)  # [batch, num_frames, height x width, channels]
                enc_hidden_states1 = enc_hidden_states1.flatten(1, 2)  # [batch, num_frames x height x width, channels]
                if eval:
                    enc_hidden_states1 = torch.cat([enc_hidden_states1, enc_hidden_states1], dim=0)
                if layernorm_fix:
                    pass # pass through separate layers
                else:
                    encoder_hidden_states_temp = torch.cat([enc_hidden_states1, enc_hidden_states0], dim=1)
            else:
                pass
            if cross_attend or cross_attend_text:
                b_1, nf_1, c_1, h_1, w_1 = ref_img_emb.shape
                ref_img_emb = ref_img_emb.reshape(-1, c_1, h_1, w_1)
                ref_img_emb = self.patch_embed.proj(ref_img_emb)
                ref_img_emb = ref_img_emb.view(b_1, nf_1, *ref_img_emb.shape[1:])
                ref_img_emb = ref_img_emb.flatten(3).transpose(2, 3)  # [batch, num_frames, height x width, channels]
                ref_img_emb = ref_img_emb.flatten(1, 2)  # [batch, num_frames x height x width, channels]
                if eval:
                    ref_img_emb = torch.cat([ref_img_emb, ref_img_emb], dim=0)
            else:
                pass
        else:
            enc_hidden_states0 = self.patch_embed.text_proj(enc_hidden_states0)
            # enc_hidden_states0 = F.pad(enc_hidden_states0, (0, 1350 - enc_hidden_states0.shape[1]))
            # print('ENC_HIDDEN_STATES0.shape before padded', enc_hidden_states0.shape) 
            

            # print('ENC_HIDDEN_STATES0.shape padded', enc_hidden_states0.shape) 
            b_1, nf_1, c_1, h_1, w_1 = enc_hidden_states1.shape
            enc_hidden_states1 = enc_hidden_states1.reshape(-1, c_1, h_1, w_1)
            enc_hidden_states1 = self.patch_embed.proj(enc_hidden_states1)
            enc_hidden_states1 = enc_hidden_states1.view(b_1, nf_1, *enc_hidden_states1.shape[1:])
            enc_hidden_states1 = enc_hidden_states1.flatten(3).transpose(2, 3)  # [batch, num_frames, height x width, channels]
            enc_hidden_states1 = enc_hidden_states1.flatten(1, 2)  # [batch, num_frames x height x width, channels]
            if eval:
                enc_hidden_states1 = torch.cat([enc_hidden_states1, enc_hidden_states1], dim=0)
            # encoder_hidden_states = enc_hidden_states0
            
        
        hidden_states = self.patch_embed(encoder_hidden_states, hidden_states)
        hidden_states = self.embedding_dropout(hidden_states)
        
        if not qk_replace:
            if vae_add is True and pos_embed is True:
                embed_ref_img = True
                ref_img_seq_start = enc_hidden_states0.shape[1]
                ref_img_seq_end = enc_hidden_states0.shape[1] + enc_hidden_states1.shape[1]
                position_delta = 0
            else:
                embed_ref_img = False
                ref_img_seq_start = 0
                ref_img_seq_end = 0
                position_delta = 0
        else:
            embed_ref_img = False
            ref_img_seq_start = 0
            ref_img_seq_end = enc_hidden_states1.shape[1]
            position_delta = 0
            

        
        # if (not vae_add) or cross_attend:
        if not qk_replace:
            if vae_add is False and not qformer:
                text_seq_length = encoder_hidden_states.shape[1]
                encoder_hidden_states = hidden_states[:, :text_seq_length]
            elif qformer:
                text_seq_length = encoder_hidden_states.shape[1]
                encoder_hidden_states = hidden_states[:, :text_seq_length]
                hidden_states = hidden_states[:, text_seq_length:]
                if eval:
                    enc_hidden_states1 = torch.cat([enc_hidden_states1, enc_hidden_states1], dim=0)
                encoder_hidden_states = torch.cat([enc_hidden_states1, encoder_hidden_states], dim=1)
                text_seq_length = encoder_hidden_states.shape[1]
            else:
                text_seq_length = encoder_hidden_states.shape[1] # to handle the hidden_states after
                if layernorm_fix:
                    pass
                else:
                    text_seq_length_temp = encoder_hidden_states_temp.shape[1]
                    encoder_hidden_states = encoder_hidden_states_temp # replace the encoder_hidden_states with the concatenated tensor
        else:
            # enc_hidden_states0 = F.pad(enc_hidden_states0, (0, 0, 0, 1350 - 226))
            text_seq_length = enc_hidden_states0.shape[1]
            # encoder_hidden_states = enc_hidden_states0
            # encoder_hidden_states = encoder_hidden_states0
            hidden_states = hidden_states[:, text_seq_length:]
            enc_hidden_states0 = F.pad(enc_hidden_states0, (0, 0, 0, 1350 - 226))
            text_seq_length = enc_hidden_states0.shape[1]
            
        if vae_add:
            hidden_states = hidden_states[:, text_seq_length:]
            if layernorm_fix:
                pass
            else:
                text_seq_length = text_seq_length_temp
        
        # 3. Transformer blocks
        ca_idx = 0
        for i, block in enumerate(self.transformer_blocks):
            # if qk_replace is True and i % 2 == 1: # v2
            # if qk_replace is True and i >= 21: # v1
            if qk_replace is True and i < 21: # v3
                encoder_hidden_states = enc_hidden_states1
                embed_ref_img = True
                # print('QK CHECK 1')
            # elif qk_replace is True and i % 2 == 0:
            # elif qk_replace is True and i < 21:
            elif qk_replace is True and i >= 21:
                encoder_hidden_states = enc_hidden_states0
                embed_ref_img = False
            #     print('QK CHECK 0')
            # print('ENCODER HIDDEN_STATES shape', encoder_hidden_states.shape)
            
            if self.training and self.gradient_checkpointing:
                def create_custom_forward(module):
                    def custom_forward(*inputs):
                        return module(*inputs)

                    return custom_forward
                if layernorm_fix:
                    ckpt_kwargs: Dict[str, Any] = {"use_reentrant": False} if is_torch_version(">=", "1.11.0") else {}
                    hidden_states, enc_hidden_states0, enc_hidden_states1 = torch.utils.checkpoint.checkpoint(
                        create_custom_forward(block),
                        hidden_states,
                        enc_hidden_states0,
                        emb,
                        enc_hidden_states1,
                        image_rotary_emb,
                        embed_ref_img,
                        ref_img_seq_start,
                        ref_img_seq_end,
                        position_delta,
                        ref_image_rotary_emb,
                        **ckpt_kwargs,
                    )
                else:
                    ckpt_kwargs: Dict[str, Any] = {"use_reentrant": False} if is_torch_version(">=", "1.11.0") else {}
                    hidden_states, encoder_hidden_states = torch.utils.checkpoint.checkpoint(
                        create_custom_forward(block),
                        hidden_states,
                        encoder_hidden_states,
                        emb,
                        image_rotary_emb,
                        embed_ref_img,
                        ref_img_seq_start,
                        ref_img_seq_end,
                        position_delta,
                        ref_image_rotary_emb,
                        **ckpt_kwargs,
                    )
            else:
                if layernorm_fix:
                    hidden_states, enc_hidden_states0, enc_hidden_states1 = block(
                        hidden_states=hidden_states,
                        encoder_hidden_states=enc_hidden_states0,
                        temb=emb,
                        enc_hidden_states1=enc_hidden_states1,
                        image_rotary_emb=image_rotary_emb,
                        embed_ref_img=embed_ref_img,
                        ref_img_seq_start=ref_img_seq_start,
                        ref_img_seq_end=ref_img_seq_end,
                        position_delta=position_delta,
                        timestep=timestep,
                        ref_image_rotary_emb=ref_image_rotary_emb,
                        layer=i,
                    )
                else:
                    hidden_states, encoder_hidden_states = block(
                        hidden_states=hidden_states,
                        encoder_hidden_states=encoder_hidden_states,
                        temb=emb,
                        image_rotary_emb=image_rotary_emb,
                        ref_image_rotary_emb=ref_image_rotary_emb,
                        embed_ref_img=embed_ref_img,
                        ref_img_seq_start=ref_img_seq_start,
                        ref_img_seq_end=ref_img_seq_end,
                        position_delta=position_delta,
                        timestep=timestep,
                        layer=i,
                    )
            if self.cross_attend or self.cross_attend_text:
                if ca_idx % self.cross_attn_interval == 0 and ca_idx <= self.num_layers:
                    if self.cross_attend_text:
                        encoder_hidden_states = encoder_hidden_states + self.local_reference_scale * self.perceiver_cross_attention_text[ca_idx](
                            ref_img_emb, encoder_hidden_states
                        )
                    if self.cross_attend:
                        hidden_states = hidden_states + self.local_reference_scale * self.perceiver_cross_attention[ca_idx](
                            ref_img_emb, hidden_states
                        )
                    ca_idx += 1

        if not self.config.use_rotary_positional_embeddings:
            # CogVideoX-2B
            hidden_states = self.norm_final(hidden_states)
        else:
            # CogVideoX-5B
            if layernorm_fix:
                if text_only_norm_final is True:
                    hidden_states = torch.cat([enc_hidden_states0, hidden_states], dim=1)
                    hidden_states = self.norm_final(hidden_states)
                    text_seq_length = enc_hidden_states0.shape[1]
                    hidden_states = hidden_states[:, text_seq_length:]
                else:
                    hidden_states = torch.cat([enc_hidden_states0, enc_hidden_states1, hidden_states], dim=1)
                    hidden_states = self.norm_final(hidden_states)
                    text_seq_length = enc_hidden_states0.shape[1] + enc_hidden_states1.shape[1]
                    hidden_states = hidden_states[:, text_seq_length:]
            else:
                hidden_states = torch.cat([encoder_hidden_states, hidden_states], dim=1)
                hidden_states = self.norm_final(hidden_states)
                text_seq_length = encoder_hidden_states.shape[1]
                hidden_states = hidden_states[:, text_seq_length:]

        # 4. Final block
        hidden_states = self.norm_out(hidden_states, temb=emb)
        hidden_states = self.proj_out(hidden_states)

        # 5. Unpatchify
        # Note: we use `-1` instead of `channels`:
        #   - It is okay to `channels` use for CogVideoX-2b and CogVideoX-5b (number of input channels is equal to output channels)
        #   - However, for CogVideoX-5b-I2V also takes concatenated input image latents (number of input channels is twice the output channels)
        p = self.config.patch_size
        output = hidden_states.reshape(batch_size, num_frames, height // p, width // p, -1, p, p)
        output = output.permute(0, 1, 4, 2, 5, 3, 6).flatten(5, 6).flatten(3, 4)

        if USE_PEFT_BACKEND:
            # remove `lora_scale` from each PEFT layer
            unscale_lora_layers(self, lora_scale)

        if not return_dict:
            return (output,)
        return Transformer2DModelOutput(sample=output)
