import os
import argparse
from typing import Literal, Optional

import torch
from transformers import (
    AutoTokenizer,
    T5EncoderModel,
    CLIPProcessor,
    CLIPTokenizer,
    CLIPTextModel,
    CLIPVisionModel,
)
from diffusers import (
    AutoencoderKLCogVideoX,
    CogVideoXDPMScheduler,
)
from diffusers.utils import export_to_video, load_image, load_video
from safetensors.torch import load_file as safetensors_load_file

# Import your modified transformer model
from custom_cogvideox import CustomCogVideoXTransformer3DModel  # Adjust the import as necessary
from custom_cogvideox_pipe import CustomCogVideoXPipeline  # Adjust the import as necessary

# Define the custom layers
import torch.nn as nn

class SkipProjectionLayer(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.projection = nn.Linear(in_features, out_features)

    def forward(self, x):
        return x + self.projection(x)
        
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
    reference_image_path: Optional[str] = None,
    num_inference_steps: int = 50,
    guidance_scale: float = 6.0,
    num_videos_per_prompt: int = 1,
    dtype: torch.dtype = torch.bfloat16,  # Default to torch.float32 for compatibility
    generate_type: Literal["t2v"] = "t2v",  # Only "t2v" is supported in this script
    seed: int = 42,
    negative_prompt: Optional[str] = None,
):
    # Initialize variables
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    # Load models and processors
    print("Loading models and processors...")
    tokenizer = AutoTokenizer.from_pretrained(model_path, subfolder="tokenizer")
    text_encoder = T5EncoderModel.from_pretrained(
        model_path, subfolder="text_encoder", torch_dtype=dtype
    ).to(device)
    vae = AutoencoderKLCogVideoX.from_pretrained(
        model_path, subfolder="vae", torch_dtype=dtype
    ).to(device)
    scheduler = CogVideoXDPMScheduler.from_pretrained(model_path, subfolder="scheduler")
    clip_tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-base-patch16")
    clip_text_encoder = CLIPTextModel.from_pretrained("openai/clip-vit-base-patch16").to(device)
    clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch16")

    print("Loading transformer...")
    transformer = CustomCogVideoXTransformer3DModel.from_pretrained(
        model_path,
        subfolder="transformer",
        torch_dtype=dtype,
        customization=True,
    ).to(device)

    print("Initializing additional components...")
    # Initialize additional components before loading their state dictionaries
    transformer.T5ProjectionLayer = SkipProjectionLayer(4096, 4096).to(device) 
    transformer.CLIPTextProjectionLayer = ProjectionLayer(512, 4096).to(device)
    transformer.CLIPVisionProjectionLayer = ProjectionLayer(768, 4096).to(device)
    with torch.no_grad():
        transformer.CLIPTextProjectionLayer.projection.weight.fill_(0.0)
        if transformer.CLIPTextProjectionLayer.projection.bias is not None:
            transformer.CLIPTextProjectionLayer.projection.bias.fill_(0.0)
    transformer.CLIPVisionProjectionLayer = ProjectionLayer(in_features=768, out_features=4096)
    with torch.no_grad():
        transformer.CLIPVisionProjectionLayer.projection.weight.fill_(0.0)
        if transformer.CLIPVisionProjectionLayer.projection.bias is not None:
            transformer.CLIPVisionProjectionLayer.projection.bias.fill_(0.0)
    transformer.reference_vision_encoder = CLIPVisionModel.from_pretrained(
        "openai/clip-vit-base-patch16"
    ).to(device)
    # dtype to bfloat16
    # transformer.T5ProjectionLayer.projection.weight = transformer.T5ProjectionLayer.projection.weight.to(dtype=dtype)
    # transformer.T5ProjectionLayer.projection.bias = transformer.T5ProjectionLayer.projection.bias.to(dtype=dtype)
    # transformer.CLIPTextProjectionLayer.projection.weight = transformer.CLIPTextProjectionLayer.projection.weight.to(dtype=dtype)
    # transformer.CLIPTextProjectionLayer.projection.bias = transformer.CLIPTextProjectionLayer.projection.bias.to(dtype=dtype)
    # transformer.CLIPVisionProjectionLayer.projection.weight = transformer.CLIPVisionProjectionLayer.projection.weight.to(dtype=dtype)
    # transformer.CLIPVisionProjectionLayer.projection.bias = transformer.CLIPVisionProjectionLayer.projection.bias.to(dtype=dtype)


    print("Loading additional components...")
    # Load additional components
    component_files = {
        "T5ProjectionLayer": ["T5ProjectionLayer.pth", "T5ProjectionLayer.safetensors"],
        # "CLIPTextProjectionLayer": [
        #     "CLIPTextProjectionLayer.pth",
        #     "CLIPTextProjectionLayer.safetensors",
        # ],
        # "CLIPVisionProjectionLayer": [
        #     "CLIPVisionProjectionLayer.pth",
        #     "CLIPVisionProjectionLayer.safetensors",
        # ],
        "reference_vision_encoder": [
            "pytorch_clip_vision_model.bin",
            "reference_vision_encoder.safetensors",
        ],
    }

    for component_name, filenames in component_files.items():
        loaded = False
        for filename in filenames:
            filepath = os.path.join(lora_path, filename)
            if os.path.exists(filepath):
                try:
                    if filename.endswith(".safetensors"):
                        state_dict = safetensors_load_file(filepath)
                    else:
                        state_dict = torch.load(filepath, map_location=device)
                    component = getattr(transformer, component_name)
                    component.load_state_dict(state_dict)
                    component.to(dtype=dtype)
                    print(f"Successfully loaded {component_name} from {filename}")
                    loaded = True
                    break
                except Exception as e:
                    print(f"Error loading {component_name} from {filename}: {e}")
        if not loaded:
            print(f"Warning: Could not load {component_name} from any of the attempted files")

    # Load LoRA weights, if provided
    if lora_path:
        lora_weights_path = os.path.join(lora_path, "pytorch_lora_weights_transformer.safetensors")
        if os.path.exists(lora_weights_path):
            state_dict = safetensors_load_file(lora_weights_path)
            transformer.load_state_dict(state_dict, strict=False)
            print(f"Successfully loaded LoRA weights from {lora_weights_path}")
        else:
            print(f"Warning: Could not find LoRA weights in {lora_path}")
            print("Available files:", os.listdir(lora_path))
            raise FileNotFoundError("LoRA weights not found")
    
    print("Creating pipeline...")
    # Create the pipeline
    pipe = CustomCogVideoXPipeline(
        tokenizer=tokenizer,
        text_encoder=text_encoder,
        transformer=transformer,
        vae=vae,
        scheduler=scheduler,
        clip_tokenizer=clip_tokenizer,
        clip_text_encoder=clip_text_encoder,
        customization=True,
    ).to(device)

    # Move pipeline components to device and set data types
    pipe.transformer.to(device=device)
    pipe.text_encoder.to(device=device)
    pipe.vae.to(device=device) 
    pipe.clip_text_encoder.to(device=device)
    pipe.transformer.reference_vision_encoder.to(device=device)
    pipe.transformer.T5ProjectionLayer.to(device=device)
    pipe.transformer.CLIPTextProjectionLayer.to(device=device)
    pipe.transformer.CLIPVisionProjectionLayer.to(device=device)
    # # Apply dtype setting to each component after initialization
    pipe.transformer.T5ProjectionLayer.projection.weight.data = transformer.T5ProjectionLayer.projection.weight.data.to(dtype=dtype)
    pipe.transformer.T5ProjectionLayer.projection.bias.data = transformer.T5ProjectionLayer.projection.bias.data.to(dtype=dtype)
    pipe.transformer.CLIPTextProjectionLayer.projection.weight.data = transformer.CLIPTextProjectionLayer.projection.weight.data.to(dtype=dtype)
    pipe.transformer.CLIPTextProjectionLayer.projection.bias.data = transformer.CLIPTextProjectionLayer.projection.bias.data.to(dtype=dtype)
    pipe.transformer.CLIPVisionProjectionLayer.projection.weight.data = transformer.CLIPVisionProjectionLayer.projection.weight.data.to(dtype=dtype)
    pipe.transformer.CLIPVisionProjectionLayer.projection.bias.data = transformer.CLIPVisionProjectionLayer.projection.bias.data.to(dtype=dtype)

    # Set models to eval mode
    pipe.transformer.eval()
    pipe.text_encoder.eval()
    pipe.vae.eval()
    pipe.clip_text_encoder.eval()
    pipe.transformer.reference_vision_encoder.eval()

    # Prepare the reference image
    if generate_type == "t2v":
        if reference_image_path is not None:
            reference_image = load_image(reference_image_path)
            # Process the reference image
            processed_image = clip_processor(images=reference_image, return_tensors="pt")
            image_input = processed_image["pixel_values"].to(device=device)
        else:
            raise ValueError("A reference image must be provided for t2v generation.")
    else:
        raise ValueError("Only 't2v' generation type is supported in this code.")

    # Generate the video
    generator = torch.Generator(device=device).manual_seed(seed)

    with torch.no_grad():
        # Exclude text_encoder from autocast if necessary
        # text_input_ids = tokenizer(prompt, return_tensors="pt", padding="max_length", max_length=77).input_ids.to(device)
        # text_embeds = text_encoder(text_input_ids).last_hidden_state

        # negative_text_embeds = None
        # if negative_prompt is not None:
        #     negative_input_ids = tokenizer(negative_prompt, return_tensors="pt", padding="max_length", max_length=77).input_ids.to(device)
        #     negative_text_embeds = text_encoder(negative_input_ids).last_hidden_state

        # with torch.cuda.amp.autocast(device_type="cuda", dtype=dtype):
        output = pipe(
                prompt=prompt,
                negative_prompt=negative_prompt,
                ref_img_states=image_input,
                height=480,
                width=720,
                num_frames=49,
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale,
                use_dynamic_cfg=True,
                generator=generator,
                output_type="pil",
            )

    video_generate = output.frames[0]

    # Export the generated frames to a video file
    export_to_video(video_generate, output_path, fps=8)
    print(f"Video saved to {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate a video from a text prompt using CogVideoX"
    )
    parser.add_argument(
        "--prompt", type=str, required=True, help="The description of the video to be generated"
    )
    parser.add_argument(
        "--negative_prompt", type=str, default=None, help="Negative prompt to guide the generation"
    )
    parser.add_argument(
        "--reference_image_path",
        type=str,
        required=True,
        help="The path of the reference image to be used (for t2v)",
    )
    parser.add_argument(
        "--model_path",
        type=str,
        default="THUDM/CogVideoX-5b",
        help="The path of the pre-trained model to be used",
    )
    parser.add_argument(
        "--lora_path", type=str, required=True, help="The path of the LoRA weights to be used"
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default="./output_debug_checkingg.mp4",
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
        "--dtype",
        type=str,
        default="bfloat16",
        choices=["float16", "float32", "bfloat16"],
        help="The data type for computation",
    )
    parser.add_argument("--seed", type=int, default=42, help="The seed for reproducibility")

    args = parser.parse_args()
    if args.dtype == "float16":
        dtype = torch.float16
    elif args.dtype == "bfloat16":
        dtype = torch.bfloat16
    else:
        dtype = torch.float32

    generate_video(
        prompt=args.prompt,
        model_path=args.model_path,
        lora_path=args.lora_path,
        output_path=args.output_path,
        reference_image_path=args.reference_image_path,
        num_inference_steps=args.num_inference_steps,
        guidance_scale=args.guidance_scale,
        dtype=dtype,
        seed=args.seed,
        negative_prompt=args.negative_prompt,
    )
