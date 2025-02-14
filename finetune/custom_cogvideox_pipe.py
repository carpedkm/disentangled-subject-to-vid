from typing import Optional, Union, List, Dict, Any, Tuple, Callable
import math
import torch

from diffusers import CogVideoXPipeline
from diffusers.pipelines.cogvideo.pipeline_cogvideox import CogVideoXPipelineOutput

from diffusers import (
    CogVideoXPipeline,
    CogVideoXDDIMScheduler,
    CogVideoXDPMScheduler,
    CogVideoXImageToVideoPipeline,
    CogVideoXVideoToVideoPipeline,
)

from transformers import AutoTokenizer, CLIPTextModel
from diffusers.utils import check_min_version, convert_unet_state_dict_to_peft, export_to_video, is_wandb_available

class CustomCogVideoXPipeline(CogVideoXPipeline):
    def __init__(
        self,
        tokenizer,
        text_encoder,
        transformer,
        vae,
        scheduler,
        clip_tokenizer=None,
        clip_text_encoder=None,
        customization=False,
        vae_add=False,
        cross_attend=False,
        cross_attend_text=False,
        pos_embed=False,
        qk_replace=False,
    ):
        # Call the base class __init__ without the 'customization' argument
        super().__init__(tokenizer, text_encoder, vae, transformer, scheduler)
        # Initialize additional attributes
        self.customization = customization
        transformer.qk_replace = qk_replace
        if customization is True:
            # Initialize CLIP tokenizer and CLIP text encoder
            if clip_tokenizer is None:
                self.clip_tokenizer = AutoTokenizer.from_pretrained("openai/clip-vit-base-patch16")
            else:
                self.clip_tokenizer = clip_tokenizer

            if clip_text_encoder is None:
                self.clip_text_encoder = CLIPTextModel.from_pretrained("openai/clip-vit-base-patch16")
            else:
                self.clip_text_encoder = clip_text_encoder
            # Move CLIP text encoder to the same device as text_encoder
            self.clip_text_encoder.to(self.text_encoder.device)


    
    def _get_clip_prompt_embeds(
        self,
        prompt: Union[str, List[str]] = None,
        num_videos_per_prompt: int = 1,
        max_sequence_length: int = 77,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ):
        # device=device or self._execution_device
        # dtype=dtype or self._execution_dtype
        
        prompt = [prompt] if isinstance(prompt, str) else prompt
        batch_size = len(prompt)
        
        clip_text_inputs = self.clip_tokenizer(
                prompt,
                padding="max_length",
                max_length=max_sequence_length,
                truncation=True,
                return_tensors="pt",
            )
        clip_text_input_ids = clip_text_inputs.input_ids.to(device)
        
        clip_prompt_embeds = self.clip_text_encoder(clip_text_input_ids)
        if isinstance(clip_prompt_embeds, tuple):
            clip_prompt_embeds = clip_prompt_embeds[0]
        elif isinstance(clip_prompt_embeds, dict):
            clip_prompt_embeds = clip_prompt_embeds["last_hidden_state"]
        else:
            raise ValueError("Unexpected output type from CLIP text encoder.")
        clip_prompt_embeds = clip_prompt_embeds.to(device=device)

        _, seq_len, _ = clip_prompt_embeds.shape
        clip_prompt_embeds = clip_prompt_embeds.repeat(1, num_videos_per_prompt, 1)
        clip_prompt_embeds = clip_prompt_embeds.view(clip_prompt_embeds.shape[0] * num_videos_per_prompt, seq_len, -1)
        return clip_prompt_embeds
    
    def encode_prompt(
        self,
        prompt: Union[str, List[str]],
        negative_prompt: Optional[Union[str, List[str]]] = None,
        do_classifier_free_guidance: bool = True,
        num_videos_per_prompt: int = 1,
        prompt_embeds: Optional[torch.Tensor] = None,
        negative_prompt_embeds: Optional[torch.Tensor] = None,
        negative_clip_prompt_embeds: Optional[torch.Tensor] = None,
        max_sequence_length: int = 226,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ):
        r"""
        Encodes the prompt into text encoder hidden states.

        Args:
            prompt (`str` or `List[str]`, *optional*):
                prompt to be encoded
            negative_prompt (`str` or `List[str]`, *optional*):
                The prompt or prompts not to guide the image generation. If not defined, one has to pass
                `negative_prompt_embeds` instead. Ignored when not using guidance (i.e., ignored if `guidance_scale` is
                less than `1`).
            do_classifier_free_guidance (`bool`, *optional*, defaults to `True`):
                Whether to use classifier free guidance or not.
            num_videos_per_prompt (`int`, *optional*, defaults to 1):
                Number of videos that should be generated per prompt. torch device to place the resulting embeddings on
            prompt_embeds (`torch.Tensor`, *optional*):
                Pre-generated text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt weighting. If not
                provided, text embeddings will be generated from `prompt` input argument.
            negative_prompt_embeds (`torch.Tensor`, *optional*):
                Pre-generated negative text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt
                weighting. If not provided, negative_prompt_embeds will be generated from `negative_prompt` input
                argument.
            device: (`torch.device`, *optional*):
                torch device
            dtype: (`torch.dtype`, *optional*):
                torch dtype
        """
        # device = device or self._execution_device

        prompt = [prompt] if isinstance(prompt, str) else prompt
        if prompt is not None:
            batch_size = len(prompt)
        else:
            batch_size = prompt_embeds.shape[0]

        if prompt_embeds is None:
            prompt_embeds = self._get_t5_prompt_embeds(
                prompt=prompt,
                num_videos_per_prompt=num_videos_per_prompt,
                max_sequence_length=max_sequence_length,
                device=device,
                dtype=dtype,
            )
            if self.customization is True:
                clip_prompt_embeds = self._get_clip_prompt_embeds(
                    prompt=prompt,
                    num_videos_per_prompt=num_videos_per_prompt,
                    max_sequence_length=77,
                    device=device,
                    dtype=dtype,
                )

        if do_classifier_free_guidance and negative_prompt_embeds is None:
            negative_prompt = negative_prompt or ""
            negative_prompt = batch_size * [negative_prompt] if isinstance(negative_prompt, str) else negative_prompt

            if prompt is not None and type(prompt) is not type(negative_prompt):
                raise TypeError(
                    f"`negative_prompt` should be the same type to `prompt`, but got {type(negative_prompt)} !="
                    f" {type(prompt)}."
                )
            elif batch_size != len(negative_prompt):
                raise ValueError(
                    f"`negative_prompt`: {negative_prompt} has batch size {len(negative_prompt)}, but `prompt`:"
                    f" {prompt} has batch size {batch_size}. Please make sure that passed `negative_prompt` matches"
                    " the batch size of `prompt`."
                )

            negative_prompt_embeds = self._get_t5_prompt_embeds(
                prompt=negative_prompt,
                num_videos_per_prompt=num_videos_per_prompt,
                max_sequence_length=max_sequence_length,
                device=device,
                dtype=dtype,
            )
            if self.customization is True:
                negative_clip_prompt_embeds = self._get_clip_prompt_embeds(
                    prompt=negative_prompt,
                    num_videos_per_prompt=num_videos_per_prompt,
                    max_sequence_length=77,
                    device=device,
                    dtype=dtype,
                )
        if self.customization is True:
            return prompt_embeds, negative_prompt_embeds, clip_prompt_embeds, negative_clip_prompt_embeds
        else:
            return prompt_embeds, negative_prompt_embeds, None, None
    # def encode_prompt(
    #     self,
    #     prompt,
    #     negative_prompt=None,
    #     do_classifier_free_guidance=True,
    #     num_videos_per_prompt=1,
    #     prompt_embeds=None,
    #     negative_prompt_embeds=None,
    #     max_sequence_length=77,
    #     device=None,
    # ):
    #     # device = device or self._execution_device
    #     prompt = [prompt] if isinstance(prompt, str) else prompt
    #     batch_size = len(prompt)

    #     if prompt_embeds is not None:
    #         # Assume clip_prompt_embeds are also provided if prompt_embeds are
    #         return prompt_embeds, None

    #     # Compute prompt embeddings using the text_encoder
    #     text_inputs = self.tokenizer(
    #         prompt,
    #         padding="max_length",
    #         max_length=max_sequence_length,
    #         truncation=True,
    #         return_tensors="pt",
    #     )
    #     # text_input_ids = text_inputs.input_ids.to(device)
    #     # prompt_embeds = self.text_encoder(text_input_ids.to(device))[0]
    #         # Move inputs to the device
    #     ## CUSTOMIZATION ## # FIXME
    #     text_input_ids = text_inputs.input_ids.to(device)
    #     attention_mask = text_inputs.attention_mask.to(device)

    #     # Add print statements to check devices
    #     print("Device of text_input_ids:", text_input_ids.device)
    #     print("Device of attention_mask:", attention_mask.device)
    #     print("Device of text_encoder:", next(self.text_encoder.parameters()).device)

    #     # Pass both input_ids and attention_mask to the text_encoder
    #     prompt_embeds = self.text_encoder(
    #         input_ids=text_input_ids, attention_mask=attention_mask
    #     )[0]
    #     ### CUSOMIZATION ### END
    #     # duplicate text embeddings for each generation per prompt, using mps friendly method
    #     _, seq_len, _ = prompt_embeds.shape
    #     prompt_embeds = prompt_embeds.repeat(1, num_videos_per_prompt, 1)
    #     prompt_embeds = prompt_embeds.view(batch_size * num_videos_per_prompt, seq_len, -1)

    #     if self.customization is True:
    #         # Compute CLIP prompt embeddings using the CLIP text encoder
            


    #     if self.customization is True:
    #         return prompt_embeds, clip_prompt_embeds
    #     else:
    #         return prompt_embeds, None

    @torch.no_grad()
    def __call__(
        self,
        prompt: Optional[Union[str, List[str]]] = None,
        ref_img_states: Optional[torch.FloatTensor] = None,
        negative_prompt: Optional[Union[str, List[str]]] = None,
        height: int = 480,
        width: int = 720,
        num_frames: int = 49,
        num_inference_steps: int = 50,
        timesteps: Optional[List[int]] = None,
        guidance_scale: float = 6,
        use_dynamic_cfg: bool = False,
        num_videos_per_prompt: int = 1,
        eta: float = 0.0,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        latents: Optional[torch.FloatTensor] = None,
        prompt_embeds: Optional[torch.FloatTensor] = None,
        negative_prompt_embeds: Optional[torch.FloatTensor] = None,
        output_type: str = "pil",
        return_dict: bool = True,
        attention_kwargs: Optional[Dict[str, Any]] = None,
        callback_on_step_end: Optional[Callable[[int, int, Dict], None]] = None,
        callback_on_step_end_tensor_inputs: List[str] = ["latents"],
        max_sequence_length: int = 226,
        eval: bool = False,
        t5_first: bool = True,
        concatenated_all: bool = False,
        reduce_token: bool = False,
        add_token: bool = False,
        zero_conv_add: bool = False,
        vae_add: bool = False,
        pos_embed: bool = False,
        cross_attend: bool = False,
        cross_attend_text: bool = False,
        input_noise_fix: bool = False,
        output_dir : str = None,
        save_every_timestep : bool = False, 
        layernorm_fix: bool = False,
        text_only_norm_final: bool = False,
    ) -> Union[CogVideoXPipelineOutput, Tuple]:
        if num_frames > 49:
            raise ValueError(
                "The number of frames must be less than or equal to 49 due to static positional embeddings."
            )

        # if ref_img_states is None:
        #     raise ValueError("ref_img_states must be provided when using t2v mode with a reference image.")

        # 1. Check inputs
        self.check_inputs(
            prompt,
            height,
            width,
            negative_prompt,
            callback_on_step_end_tensor_inputs,
            prompt_embeds,
            negative_prompt_embeds,
        )

        # 2. Default call parameters
        device = self._execution_device
        print('>> PROMPT: ', prompt)
        print('>> PROMPT TYPE: ', type(prompt))
        # Determine batch size
        if prompt is not None and isinstance(prompt, str):
            batch_size = 1
        elif prompt is not None and isinstance(prompt, list):
            batch_size = len(prompt)
        else:
            batch_size = prompt_embeds.shape[0]

        # Check if guidance is needed
        do_classifier_free_guidance = guidance_scale > 1.0
        # print('>> PROMPT: ', prompt)
        # 3. Encode prompt and CLIP prompt embeddings
        prompt_embeds, negative_prompt_embeds, clip_prompt_embeds, negative_clip_prompt_embeds = self.encode_prompt( # FIXME Do negative prompt embed
            prompt,
            negative_prompt=negative_prompt,
            do_classifier_free_guidance=do_classifier_free_guidance,
            num_videos_per_prompt=num_videos_per_prompt,
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
            max_sequence_length=max_sequence_length,
            device=device,
        )
        if do_classifier_free_guidance:
            # prompt_embeds = torch.cat([prompt_embeds, prompt_embeds], dim=0)
            # if self.customization is True:
            #     clip_prompt_embeds = torch.cat([clip_prompt_embeds, clip_prompt_embeds], dim=0)
            prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds], dim=0)
            if self.customization is True:
                clip_prompt_embeds = torch.cat([negative_clip_prompt_embeds, clip_prompt_embeds], dim=0)
            

        # 4. Prepare timesteps
        if timesteps is None:
            self.scheduler.set_timesteps(num_inference_steps, device=device)
            timesteps = self.scheduler.timesteps
        else:
            self.scheduler.timesteps = torch.tensor(timesteps, device=device)
        self._num_timesteps = len(timesteps)

        # 5. Prepare latents
        latent_channels = self.transformer.config.in_channels
        latents = self.prepare_latents(
            batch_size * num_videos_per_prompt,
            latent_channels,
            num_frames,
            height,
            width,
            prompt_embeds.dtype,
            device,
            generator,
            latents,
        )

        # 6. Prepare extra step kwargs
        extra_step_kwargs = self.prepare_extra_step_kwargs(generator, eta)

        # 7. Create rotary embeddings if required
        # ref_image_rotary_emb = (
        #     self._prepare_rotary_positional_embeddings(height, width, ref_img_states.size(1), device)
        #     if getattr(self.transformer.config, "use_rotary_positional_embeddings", False)
        #     else None
        # )
        image_rotary_emb = (
            self._prepare_rotary_positional_embeddings(height, width, latents.size(1), device)
            if getattr(self.transformer.config, "use_rotary_positional_embeddings", False)
            else None
        )
        ref_image_rotary_emb = (
            image_rotary_emb[0][:1350,...],
            image_rotary_emb[1][:1350,...]
        )

        # 8. Denoising loop
        num_warmup_steps = max(len(timesteps) - num_inference_steps * self.scheduler.order, 0)

        with self.progress_bar(total=num_inference_steps) as progress_bar:
            old_pred_original_sample = None
            for i, t in enumerate(timesteps):
                if getattr(self, 'interrupt', False):
                    break

                # Expand latents for classifier-free guidance
                latent_model_input = (
                    torch.cat([latents] * 2) if do_classifier_free_guidance else latents
                )
                
                if input_noise_fix:
                    print('latent_model_input: ', latent_model_input.shape)
                    print('ref_img_states: ', ref_img_states.shape)
                    # Concatenate latent_model_input and ref_img_states along the batch dimension
                    concat_input = torch.cat([latent_model_input, ref_img_states], dim=0)

                    # Apply scale_model_input to the concatenated input
                    scaled_concat_input = self.scheduler.scale_model_input(concat_input, t)

                    # Split back into latent_model_input and noisy_ref_img_states
                    latent_model_input, noisy_ref_img_states = torch.chunk(scaled_concat_input, chunks=2, dim=0)
                else:
                    latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)
                # Broadcast timestep
                timestep = t.expand(latent_model_input.shape[0])

                # Predict the noise residual
                noise_pred = self.transformer(
                    hidden_states=latent_model_input,
                    encoder_hidden_states=prompt_embeds.to(latent_model_input.dtype),
                    clip_prompt_embeds=clip_prompt_embeds.to(latent_model_input.dtype) if clip_prompt_embeds is not None else None,
                    ref_img_states=noisy_ref_img_states if input_noise_fix else ref_img_states,
                    timestep=timestep,
                    image_rotary_emb=image_rotary_emb,
                    ref_image_rotary_emb=ref_image_rotary_emb,
                    attention_kwargs=attention_kwargs,
                    customization=self.customization,
                    return_dict=False,
                    eval=True,
                    t5_first=t5_first,
                    concatenated_all=concatenated_all,
                    reduce_token=reduce_token,
                    add_token=add_token,
                    zero_conv_add=zero_conv_add,
                    vae_add=vae_add,
                    pos_embed=pos_embed,
                    cross_attend=cross_attend,
                    cross_attend_text=cross_attend_text,
                    layernorm_fix=layernorm_fix,
                    text_only_norm_final=text_only_norm_final,
                )[0]
                noise_pred = noise_pred.float()

                # Perform dynamic classifier-free guidance
                if use_dynamic_cfg:
                    self._guidance_scale = 1 + guidance_scale * (
                        (1 - math.cos(math.pi * ((num_inference_steps - i) / num_inference_steps) ** 5.0)) / 2
                    )
                else:
                    self._guidance_scale = guidance_scale

                # Apply classifier-free guidance
                if do_classifier_free_guidance:
                    noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                    noise_pred = noise_pred_uncond + self._guidance_scale * (noise_pred_text - noise_pred_uncond)

                # Compute the previous noisy sample x_t -> x_t-1
                if not isinstance(self.scheduler, CogVideoXDPMScheduler):
                    latents = self.scheduler.step(
                        noise_pred, t, latents, **extra_step_kwargs, return_dict=False
                    )[0]
                else:
                    latents, old_pred_original_sample = self.scheduler.step(
                        noise_pred,
                        old_pred_original_sample,
                        t,
                        timesteps[i - 1] if i > 0 else None,
                        latents,
                        **extra_step_kwargs,
                        return_dict=False,
                    )
                latents = latents.to(prompt_embeds.dtype)

                # Call the callback, if provided
                if callback_on_step_end is not None:
                    callback_kwargs = {k: locals()[k] for k in callback_on_step_end_tensor_inputs}
                    callback_outputs = callback_on_step_end(self, i, t, callback_kwargs)

                    latents = callback_outputs.get("latents", latents)
                    prompt_embeds = callback_outputs.get("prompt_embeds", prompt_embeds)
                    negative_prompt_embeds = callback_outputs.get("negative_prompt_embeds", negative_prompt_embeds)
                    clip_prompt_embeds = callback_outputs.get("clip_prompt_embeds", clip_prompt_embeds)
                if save_every_timestep:
                    # 8.1 Decode latents
                    import os
                    video = self.decode_latents(latents)
                    video = self.video_processor.postprocess_video(video=video, output_type=output_type)
                    tmp_prompt_str_with__ = prompt[:20].replace(" ", "_")
                    os.makedirs(f'{output_dir}/timesteps/video_{tmp_prompt_str_with__}', exist_ok=True)
                    timestep_output_video_path = f"{output_dir}/timesteps/video_{tmp_prompt_str_with__}/video_{i:05d}.mp4"
                    # 8.2 Save video
                    if output_type == "latent":
                        video = latents
                    video_output = CogVideoXPipelineOutput(video).frames[0]
                    export_to_video(video_output, timestep_output_video_path)
                    # export_to_gif(video, output_gif_path)
                    print(f"Video saved to {timestep_output_video_path}")
                    
                # Update progress bar
                if i == len(timesteps) - 1 or (
                    (i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0
                ):
                    progress_bar.update()

        # 9. Decode latents
        if output_type != "latent":
            video = self.decode_latents(latents)
            video = self.video_processor.postprocess_video(video=video, output_type=output_type)
        else:
            video = latents

        # 10. Offload all models
        self.maybe_free_model_hooks()

        if not return_dict:
            return (video,)

        return CogVideoXPipelineOutput(frames=video)

# custom_cogvideox_pipe.py

# from typing import Optional, Union, List, Dict, Any, Tuple, Callable
# import math
# import torch
# import torch.nn as nn

# from diffusers import CogVideoXPipeline
# from diffusers.pipelines.cogvideo.pipeline_cogvideox import CogVideoXPipelineOutput

# from transformers import AutoTokenizer, CLIPTextModel

# class SkipProjectionLayer(nn.Module):
#     def __init__(self, in_features, out_features):
#         super().__init__()
#         self.projection = nn.Linear(in_features, out_features)

#     def forward(self, x):
#         return x + self.projection(x)
    
# class ProjectionLayer(nn.Module):
#     def __init__(self, in_features, out_features):
#         super().__init__()
#         self.projection = nn.Linear(in_features, out_features)

#     def forward(self, x):
#         return self.projection(x)

# class CustomCogVideoXPipeline(CogVideoXPipeline):
#     def __init__(
#         self,
#         tokenizer,
#         text_encoder,
#         transformer,
#         vae,
#         scheduler,
#         clip_tokenizer=None,
#         clip_text_encoder=None,
#         customization=False,
#     ):
#         super().__init__(tokenizer, text_encoder, vae, transformer, scheduler)
#         # Initialize additional attributes
#         self.customization = customization
#         if customization:
#             # Initialize CLIP tokenizer and CLIP text encoder
#             if clip_tokenizer is None:
#                 self.clip_tokenizer = AutoTokenizer.from_pretrained("openai/clip-vit-base-patch16")
#             else:
#                 self.clip_tokenizer = clip_tokenizer

#             if clip_text_encoder is None:
#                 self.clip_text_encoder = CLIPTextModel.from_pretrained("openai/clip-vit-base-patch16")
#             else:
#                 self.clip_text_encoder = clip_text_encoder

#             # Move CLIP text encoder to the same device as text_encoder
#             self.clip_text_encoder.to(self.text_encoder.device)

#     # def encode_prompt(
#     #     self,
#     #     prompt,
#     #     negative_prompt=None,
#     #     do_classifier_free_guidance=True,
#     #     num_videos_per_prompt=1,
#     #     prompt_embeds=None,
#     #     negative_prompt_embeds=None,
#     #     max_sequence_length=77,
#     #     device=None,
#     # ):
#     #     prompt = [prompt] if isinstance(prompt, str) else prompt
#     #     batch_size = len(prompt)

#     #     if prompt_embeds is not None:
#     #         # Assume clip_prompt_embeds are also provided if prompt_embeds are
#     #         return prompt_embeds, None

#     #     # Compute prompt embeddings using the text_encoder
#     #     text_inputs = self.tokenizer(
#     #         prompt,
#     #         padding="max_length",
#     #         max_length=max_sequence_length,
#     #         truncation=True,
#     #         return_tensors="pt",
#     #     )
#     #     text_input_ids = text_inputs.input_ids.to(device)
#     #     prompt_embeds = self.text_encoder(text_input_ids)[0]

#     #     # Compute CLIP prompt embeddings using the CLIP text encoder
#     #     clip_text_inputs = self.clip_tokenizer(
#     #         prompt,
#     #         padding="max_length",
#     #         max_length=77,
#     #         truncation=True,
#     #         return_tensors="pt",
#     #     )
#     #     clip_text_input_ids = clip_text_inputs.input_ids.to(device)
#     #     clip_prompt_embeds = self.clip_text_encoder(clip_text_input_ids)[0]

#     #     # Duplicate embeddings for each num_videos_per_prompt
#     #     prompt_embeds = prompt_embeds.repeat_interleave(num_videos_per_prompt, dim=0)
#     #     clip_prompt_embeds = clip_prompt_embeds.repeat_interleave(num_videos_per_prompt, dim=0)

#     #     # Handle negative prompts if classifier-free guidance is used
#     #     if do_classifier_free_guidance:
#     #         if negative_prompt is None:
#     #             negative_prompt = [""] * batch_size
#     #         elif isinstance(negative_prompt, str):
#     #             negative_prompt = [negative_prompt]

#     #         # Compute negative prompt embeddings
#     #         negative_text_inputs = self.tokenizer(
#     #             negative_prompt,
#     #             padding="max_length",
#     #             max_length=max_sequence_length,
#     #             truncation=True,
#     #             return_tensors="pt",
#     #         )
#     #         negative_text_input_ids = negative_text_inputs.input_ids.to(device)
#     #         negative_prompt_embeds = self.text_encoder(negative_text_input_ids)[0]

#     #         negative_clip_text_inputs = self.clip_tokenizer(
#     #             negative_prompt,
#     #             padding="max_length",
#     #             max_length=77,
#     #             truncation=True,
#     #             return_tensors="pt",
#     #         )
#     #         negative_clip_text_input_ids = negative_clip_text_inputs.input_ids.to(device)
#     #         negative_clip_prompt_embeds = self.clip_text_encoder(negative_clip_text_input_ids)[0]

#     #         # Duplicate negative embeddings
#     #         negative_prompt_embeds = negative_prompt_embeds.repeat_interleave(num_videos_per_prompt, dim=0)
#     #         negative_clip_prompt_embeds = negative_clip_prompt_embeds.repeat_interleave(num_videos_per_prompt, dim=0)

#     #         # Concatenate embeddings
#     #         prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds], dim=0)
#     #         clip_prompt_embeds = torch.cat([negative_clip_prompt_embeds, clip_prompt_embeds], dim=0)

#     def encode_prompt(
#         self,
#         prompt,
#         negative_prompt=None,
#         do_classifier_free_guidance=True,
#         num_videos_per_prompt=1,
#         prompt_embeds=None,
#         negative_prompt_embeds=None,
#         max_sequence_length=77,
#         device=None,
#     ):
#         if device is None:
#             device = self._execution_device

#         # Move tokenizers to the same device if they have device attributes
#         if hasattr(self.tokenizer, "to"):
#             self.tokenizer.to(device)
#         if hasattr(self.clip_tokenizer, "to"):
#             self.clip_tokenizer.to(device)

#         if prompt_embeds is not None:
#             if isinstance(prompt_embeds, (list, tuple)):
#                 prompt_embeds = torch.cat(prompt_embeds, dim=0)
#             return prompt_embeds.to(device), None

#         if isinstance(prompt, str):
#             prompt = [prompt]
        
#         # Process T5 embeddings
#         text_inputs = self.tokenizer(
#             prompt,
#             padding="max_length",
#             max_length=max_sequence_length,
#             truncation=True,
#             return_tensors="pt",
#         )
#         text_input_ids = text_inputs.input_ids.to(device)
        
#         with torch.no_grad():
#             prompt_embeds = self.text_encoder(text_input_ids)[0].to(device)

#         # Process CLIP embeddings
#         clip_text_inputs = self.clip_tokenizer(
#             prompt,
#             padding="max_length",
#             max_length=77,
#             truncation=True,
#             return_tensors="pt",
#         )
#         clip_text_input_ids = clip_text_inputs.input_ids.to(device)
        
#         with torch.no_grad():
#             clip_prompt_embeds = self.clip_text_encoder(clip_text_input_ids)[0].to(device)

#         bs_embed = prompt_embeds.shape[0]

#         # Duplicate embeddings for each num_videos_per_prompt
#         prompt_embeds = prompt_embeds.repeat(1, num_videos_per_prompt, 1)
#         prompt_embeds = prompt_embeds.view(bs_embed * num_videos_per_prompt, -1, prompt_embeds.shape[-1])
        
#         clip_prompt_embeds = clip_prompt_embeds.repeat(1, num_videos_per_prompt, 1)
#         clip_prompt_embeds = clip_prompt_embeds.view(bs_embed * num_videos_per_prompt, -1, clip_prompt_embeds.shape[-1])

#         # Handle negative prompts for classifier-free guidance
#         if do_classifier_free_guidance:
#             if negative_prompt is None:
#                 negative_prompt = [""] * len(prompt)
#             elif isinstance(negative_prompt, str):
#                 negative_prompt = [negative_prompt] * len(prompt)

#             # Process negative T5 embeddings
#             uncond_tokens = self.tokenizer(
#                 negative_prompt,
#                 padding="max_length",
#                 max_length=max_sequence_length,
#                 truncation=True,
#                 return_tensors="pt",
#             )
#             negative_prompt_embeds = self.text_encoder(uncond_tokens.input_ids.to(device))[0]

#             # Process negative CLIP embeddings
#             uncond_clip_tokens = self.clip_tokenizer(
#                 negative_prompt,
#                 padding="max_length",
#                 max_length=77,
#                 truncation=True,
#                 return_tensors="pt",
#             )
#             negative_clip_prompt_embeds = self.clip_text_encoder(uncond_clip_tokens.input_ids.to(device))[0]

#             # Duplicate negative embeddings for each num_videos_per_prompt
#             negative_prompt_embeds = negative_prompt_embeds.repeat(1, num_videos_per_prompt, 1)
#             negative_prompt_embeds = negative_prompt_embeds.view(bs_embed * num_videos_per_prompt, -1, negative_prompt_embeds.shape[-1])
            
#             negative_clip_prompt_embeds = negative_clip_prompt_embeds.repeat(1, num_videos_per_prompt, 1)
#             negative_clip_prompt_embeds = negative_clip_prompt_embeds.view(bs_embed * num_videos_per_prompt, -1, negative_clip_prompt_embeds.shape[-1])

#             # Concatenate negative and positive embeddings
#             prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds], dim=0)
#             clip_prompt_embeds = torch.cat([negative_clip_prompt_embeds, clip_prompt_embeds], dim=0)

#         return prompt_embeds, clip_prompt_embeds.to(prompt_embeds.dtype)#     return prompt_embeds, clip_prompt_embeds
        

#     @torch.no_grad()
#     def __call__(
#         self,
#         prompt: Optional[Union[str, List[str]]] = None,
#         ref_img_states: Optional[torch.FloatTensor] = None,
#         negative_prompt: Optional[Union[str, List[str]]] = None,
#         height: int = 480,
#         width: int = 720,
#         num_frames: int = 49,
#         num_inference_steps: int = 50,
#         timesteps: Optional[List[int]] = None,
#         guidance_scale: float = 6,
#         use_dynamic_cfg: bool = False,
#         num_videos_per_prompt: int = 1,
#         eta: float = 0.0,
#         generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
#         latents: Optional[torch.FloatTensor] = None,
#         prompt_embeds: Optional[torch.FloatTensor] = None,
#         negative_prompt_embeds: Optional[torch.FloatTensor] = None,
#         output_type: str = "pil",
#         return_dict: bool = True,
#         attention_kwargs: Optional[Dict[str, Any]] = None,
#         callback_on_step_end: Optional[Callable[[int, int, Dict], None]] = None,
#         callback_on_step_end_tensor_inputs: List[str] = ["latents"],
#         max_sequence_length: int = 226,
#         eval: bool = False,
#     ) -> Union[CogVideoXPipelineOutput, Tuple]:
#         if num_frames > 49:
#             raise ValueError(
#                 "The number of frames must be less than or equal to 49 due to static positional embeddings."
#             )

#         # 1. Check inputs
#         self.check_inputs(
#             prompt,
#             height,
#             width,
#             negative_prompt,
#             callback_on_step_end_tensor_inputs,
#             prompt_embeds,
#             negative_prompt_embeds,
#         )

#         # 2. Default call parameters
#         device = self._execution_device

#         # Determine batch size
#         if prompt is not None and isinstance(prompt, str):
#             batch_size = 1
#         elif prompt is not None and isinstance(prompt, list):
#             batch_size = len(prompt)
#         else:
#             batch_size = prompt_embeds.shape[0]

#         # Check if guidance is needed
#         do_classifier_free_guidance = guidance_scale > 1.0

#         # 3. Encode prompt and CLIP prompt embeddings
#         prompt_embeds, clip_prompt_embeds = self.encode_prompt(
#             prompt,
#             negative_prompt=negative_prompt,
#             do_classifier_free_guidance=do_classifier_free_guidance,
#             num_videos_per_prompt=num_videos_per_prompt,
#             prompt_embeds=prompt_embeds,
#             negative_prompt_embeds=negative_prompt_embeds,
#             max_sequence_length=max_sequence_length,
#             device=device,
#         )

#         # 4. Prepare timesteps
#         if timesteps is None:
#             self.scheduler.set_timesteps(num_inference_steps, device=device)
#             timesteps = self.scheduler.timesteps
#         else:
#             self.scheduler.timesteps = torch.tensor(timesteps, device=device)
#         self._num_timesteps = len(timesteps)

#         # 5. Prepare latents
#         latent_channels = self.transformer.config.in_channels
#         latents = self.prepare_latents(
#             batch_size * num_videos_per_prompt,
#             latent_channels,
#             num_frames,
#             height,
#             width,
#             prompt_embeds.dtype,
#             device,
#             generator,
#             latents,
#         )

#         # 6. Prepare extra step kwargs
#         extra_step_kwargs = self.prepare_extra_step_kwargs(generator, eta)

#         # 7. Create rotary embeddings if required
#         image_rotary_emb = (
#             self._prepare_rotary_positional_embeddings(height, width, latents.size(1), device)
#             if getattr(self.transformer.config, "use_rotary_positional_embeddings", False)
#             else None
#         )

#         # 8. Denoising loop
#         num_warmup_steps = max(len(timesteps) - num_inference_steps * self.scheduler.order, 0)

#         with self.progress_bar(total=num_inference_steps) as progress_bar:
#             old_pred_original_sample = None
#             for i, t in enumerate(timesteps):
#                 if getattr(self, 'interrupt', False):
#                     break

#                 # Expand latents for classifier-free guidance
#                 latent_model_input = (
#                     torch.cat([latents] * 2) if do_classifier_free_guidance else latents
#                 )
#                 latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)

#                 # Broadcast timestep
#                 timestep = t.expand(latent_model_input.shape[0])

#                 # Predict the noise residual
#                 noise_pred = self.transformer(
#                     hidden_states=latent_model_input,
#                     encoder_hidden_states=prompt_embeds.to(latent_model_input.dtype),
#                     clip_prompt_embeds=clip_prompt_embeds.to(latent_model_input.dtype) if clip_prompt_embeds is not None else None,
#                     ref_img_states=ref_img_states.to(latent_model_input.dtype) if ref_img_states is not None else None,
#                     timestep=timestep,
#                     image_rotary_emb=image_rotary_emb,
#                     attention_kwargs=attention_kwargs,
#                     customization=self.customization,
#                     return_dict=False,
#                     eval=True,
#                 )[0]
#                 noise_pred = noise_pred.float()

#                 # Perform dynamic classifier-free guidance
#                 if use_dynamic_cfg:
#                     self._guidance_scale = 1 + guidance_scale * (
#                         (1 - math.cos(math.pi * ((num_inference_steps - i) / num_inference_steps) ** 5.0)) / 2
#                     )
#                 else:
#                     self._guidance_scale = guidance_scale

#                 # Apply classifier-free guidance
#                 if do_classifier_free_guidance:
#                     noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
#                     noise_pred = noise_pred_uncond + self._guidance_scale * (noise_pred_text - noise_pred_uncond)

#                 # Compute the previous noisy sample x_t -> x_t-1
#                 if not isinstance(self.scheduler, CogVideoXDPMScheduler):
#                     latents = self.scheduler.step(
#                         noise_pred, t, latents, **extra_step_kwargs, return_dict=False
#                     )[0]
#                 else:
#                     latents, old_pred_original_sample = self.scheduler.step(
#                         noise_pred,
#                         old_pred_original_sample,
#                         t,
#                         timesteps[i - 1] if i > 0 else None,
#                         latents,
#                         **extra_step_kwargs,
#                         return_dict=False,
#                     )
#                 latents = latents.to(prompt_embeds.dtype)

#                 # Update progress bar
#                 if i == len(timesteps) - 1 or (
#                     (i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0
#                 ):
#                     progress_bar.update()

#         # 9. Decode latents
#         if output_type != "latent":
#             video = self.decode_latents(latents)
#             video = self.video_processor.postprocess_video(video=video, output_type=output_type)
#         else:
#             video = latents

#         # 10. Offload all models
#         self.maybe_free_model_hooks()

#         if not return_dict:
#             return (video,)

#         return CogVideoXPipelineOutput(frames=video)
