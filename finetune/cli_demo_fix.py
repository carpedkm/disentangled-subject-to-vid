import argparse
from typing import Literal, Optional

import torch
from transformers import (
    AutoTokenizer,
    T5EncoderModel,
    CLIPProcessor,
    CLIPVisionModel,
)
from diffusers import (
    AutoencoderKLCogVideoX,
    CogVideoXDPMScheduler,
)
from diffusers.utils import export_to_video, load_image, load_video
from peft import LoraConfig, get_peft_model, set_peft_model_state_dict

# Import your modified transformer model
from custom_cogvideox import CustomCogVideoXTransformer3DModel  # Adjust the import as necessary
from custom_cogvideox_pipe import CustomCogVideoXPipeline  # Adjust the import as necessary

from safetensors.torch import load_file as safetensors_load_file

# Define the ProjectionLayer class
import torch.nn as nn

class ProjectionLayer(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.projection = nn.Linear(in_features, out_features)

    def forward(self, x):
        return self.projection(x)

def generate_video(
    prompt: str,
    model_path: str,
    lora_path: Optional[str] = None,
    lora_rank: int = 128,
    lora_alpha: int = 64,
    output_path: str = "./output.mp4",
    image_or_video_path: Optional[str] = None,
    reference_image_path: Optional[str] = None,
    num_inference_steps: int = 50,
    guidance_scale: float = 6.0,
    num_videos_per_prompt: int = 1,
    dtype: torch.dtype = torch.bfloat16,
    generate_type: Literal["t2v", "i2v", "v2v"] = "t2v",
    seed: int = 42,
):
    # Initialize variables
    image = None
    video = None
    reference_image = None

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load components
    tokenizer = AutoTokenizer.from_pretrained(
        model_path, subfolder="tokenizer"
    )
    text_encoder = T5EncoderModel.from_pretrained(
        model_path, subfolder="text_encoder", torch_dtype=dtype
    ).to(device)
    vae = AutoencoderKLCogVideoX.from_pretrained(
        model_path, subfolder="vae", torch_dtype=dtype
    ).to(device)
    print(">>> VAE CONFIG <<<")
    print(vae.config)
    # vae.config = dict(vae.config)
    # Extracting the block_out_channels from the VAE config
    # block_out_channels = list(vae.config['block_out_channels'])
    # block_out_channels = vae.config.get('block_out_channels', [128, 256, 256, 512])
   

    # Load the base transformer model
    transformer = CustomCogVideoXTransformer3DModel.from_pretrained(
        model_path,
        subfolder="transformer",
        torch_dtype=dtype,
        customization=False,
    ).to(device)

    # Initialize custom components if not initialized in from_pretrained
    transformer.reference_vision_encoder = CLIPVisionModel.from_pretrained(
        "openai/clip-vit-base-patch16",
        torch_dtype=dtype,
    ).to(device)

    transformer.T5ProjectionLayer = ProjectionLayer(in_features=4096, out_features=4096).to(device=device, dtype=dtype)
    transformer.CLIPTextProjectionLayer = ProjectionLayer(in_features=512, out_features=4096).to(device=device, dtype=dtype)
    transformer.CLIPVisionProjectionLayer = ProjectionLayer(in_features=768, out_features=4096).to(device=device, dtype=dtype)

    # Load weights for custom components
    # Load weights for custom components (projection layers) using torch.load
    state_dict = torch.load(f"{lora_path}/T5ProjectionLayer.safetensors", map_location=device)
    transformer.T5ProjectionLayer.load_state_dict(state_dict)

    state_dict = torch.load(f"{lora_path}/CLIPTextProjectionLayer.safetensors", map_location=device)
    transformer.CLIPTextProjectionLayer.load_state_dict(state_dict)

    state_dict = torch.load(f"{lora_path}/CLIPVisionProjectionLayer.safetensors", map_location=device)
    transformer.CLIPVisionProjectionLayer.load_state_dict(state_dict)

    # Load weights for reference_vision_encoder (clip_vision_model)
    state_dict = torch.load(f"{lora_path}/pytorch_clip_vision_model.bin", map_location=device)
    transformer.reference_vision_encoder.load_state_dict(state_dict)

    # Add LoRA layers using PEFT
    # lora_config = LoraConfig(
    #     r=lora_rank,
    #     lora_alpha=lora_alpha,
    #     target_modules=["to_k", "to_q", "to_v", "to_out.0"],
    #     bias="none",
    #     task_type="UNET3D",  # Adjust based on your model
    # )

    # transformer = get_peft_model(transformer, lora_config)
    

    # Load LoRA weights, if provided
    # Load LoRA weights using safetensors_load_file

        # lora_state_dict = safetensors_load_file(f"{lora_path}/pytorch_lora_weights_transformer.safetensors")
        # lora_state_dict = {k: v.to(device) for k, v in lora_state_dict.items()}
        # set_peft_model_state_dict(transformer, lora_state_dict)

    scheduler = CogVideoXDPMScheduler.from_pretrained(model_path, subfolder="scheduler")
    # Create the pipeline
    pipe = CustomCogVideoXPipeline(
        tokenizer=tokenizer,
        text_encoder=text_encoder,
        transformer=transformer,
        vae=vae,
        scheduler=scheduler,
        customization=False,
        # dynamic_cfg=True,
        # guidance_scale=guidance_scale,
    ).to(device)
    
    if lora_path:
        lora_scaling =1. # FIXME
        pipe.load_lora_weights(args.lora_path, weight_name='pytorch_lora_weights_transformer.safetensors', adapter_name="cogvideox-lora")
        pipe.set_adapters(["cogvideox-lora"], [lora_scaling])
    # pipe.scheduler = CogVideoXDPMScheduler.from_config(pipe.scheduler.config, timestep_spacing="trailing")
    # pipe.dtype = dtype  # Set the dtype for the pipeline

    # Prepare the reference image
    image_input = None
    if generate_type == "t2v":
        if reference_image_path is not None:
            reference_image = load_image(reference_image_path)
            # Process the reference image
            clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch16")
            images = clip_processor(images=reference_image, return_tensors="pt")
            image_input = images['pixel_values'].to(device).to(dtype)
        # else:
            # raise ValueError("A reference image must be provided for t2v generation.")

    # Generate the video
    generator = torch.Generator(device=device).manual_seed(seed)
    if generate_type == "t2v":
        video_generate = pipe(
            prompt=prompt,
            num_videos_per_prompt=num_videos_per_prompt,
            num_inference_steps=num_inference_steps,
            num_frames=49,
            use_dynamic_cfg=True,
            guidance_scale=guidance_scale,
            generator=generator,
            ref_img_states=image_input,  # Pass the reference image embeddings
        ).frames[0]
    elif generate_type == "i2v":
        # Load the image for image-to-video generation
        image = load_image(image_or_video_path)
        video_generate = pipe(
            prompt=prompt,
            image=image,
            num_videos_per_prompt=num_videos_per_prompt,
            num_inference_steps=num_inference_steps,
            num_frames=49,
            use_dynamic_cfg=True,
            guidance_scale=guidance_scale,
            generator=generator,
        ).frames[0]
    elif generate_type == "v2v":
        # Load the video for video-to-video generation
        video = load_video(image_or_video_path)
        video_generate = pipe(
            prompt=prompt,
            video=video,
            num_videos_per_prompt=num_videos_per_prompt,
            num_inference_steps=num_inference_steps,
            use_dynamic_cfg=True,
            guidance_scale=guidance_scale,
            generator=generator,
        ).frames[0]
    else:
        raise ValueError("Invalid generate_type. Choose from 't2v', 'i2v', or 'v2v'.")

    # Export the generated frames to a video file
    export_to_video(video_generate, output_path, fps=8)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate a video from a text prompt using CogVideoX"
    )
    parser.add_argument(
        "--prompt", type=str, required=True, help="The description of the video to be generated"
    )
    parser.add_argument(
        "--image_or_video_path",
        type=str,
        default=None,
        help="The path of the image or video to be used (for i2v and v2v)",
    )
    parser.add_argument(
        "--reference_image_path",
        type=str,
        default=None,
        help="The path of the reference image to be used (for t2v)",
    )
    parser.add_argument(
        "--model_path",
        type=str,
        default="THUDM/CogVideoX-5b",
        help="The path of the pre-trained model to be used",
    )
    parser.add_argument(
        "--lora_path", type=str, default=None, help="The path of the LoRA weights to be used"
    )
    parser.add_argument(
        "--lora_rank", type=int, default=128, help="The rank of the LoRA weights"
    )
    parser.add_argument(
        "--lora_alpha", type=int, default=64, help="The lora_alpha used during finetuning"
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default="./output2.mp4",
        help="The path where the generated video will be saved",
    )
    parser.add_argument(
        "--guidance_scale",
        type=float,
        default=6.0,
        help="The scale for classifier-free guidance",
    )
    parser.add_argument(
        "--num_inference_steps",
        type=int,
        default=50,
        help="Number of steps for the inference process",
    )
    parser.add_argument(
        "--num_videos_per_prompt",
        type=int,
        default=1,
        help="Number of videos to generate per prompt",
    )
    parser.add_argument(
        "--generate_type",
        type=str,
        default="t2v",
        choices=["t2v", "i2v", "v2v"],
        help="The type of video generation",
    )
    parser.add_argument(
        "--dtype",
        type=str,
        default="bfloat16",
        choices=["float16", "bfloat16"],
        help="The data type for computation",
    )
    parser.add_argument("--seed", type=int, default=42, help="The seed for reproducibility")

    args = parser.parse_args()
    dtype = torch.float16 if args.dtype == "float16" else torch.bfloat16

    generate_video(
        prompt=args.prompt,
        model_path=args.model_path,
        lora_path=args.lora_path,
        lora_rank=args.lora_rank,
        lora_alpha=args.lora_alpha,
        output_path=args.output_path,
        image_or_video_path=args.image_or_video_path,
        reference_image_path=args.reference_image_path,
        num_inference_steps=args.num_inference_steps,
        guidance_scale=args.guidance_scale,
        num_videos_per_prompt=args.num_videos_per_prompt,
        dtype=dtype,
        generate_type=args.generate_type,
        seed=args.seed,
    )
