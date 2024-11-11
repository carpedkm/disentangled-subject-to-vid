import torch
from diffusers import AutoencoderKLCogVideoX, CogVideoXDPMScheduler
from transformers import T5Tokenizer, T5EncoderModel, CLIPProcessor, CLIPTokenizer, CLIPTextModel, CLIPVisionModel
from custom_cogvideox_pipe import CustomCogVideoXPipeline
from custom_cogvideox import CustomCogVideoXTransformer3DModel
from PIL import Image
import os
from safetensors.torch import load_file

# Define the custom layers if not already defined
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

def main():
    # Model paths and parameters
    pretrained_model_name_or_path = "THUDM/CogVideoX-5b"
    output_dir = "/mnt/carpedkm_data/finetune_result/finetune4000_custom_zero_init_t5_full_custom_with_clip/checkpoint-800"
    prompt = "Two dogs one with a black and tan coat and another with a black and white coat appear to be playing on a lush green lawn with trees and a building in the background"
    negative_prompt = "Low quality, bad image, artifacts"
    reference_image_path = "/root/daneul/projects/refactored/CogVideo/finetune/val_samples/854179_background_boxes.jpg"

    # Device and dtype setup
    device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
    dtype = torch.float32  # Use torch.float32 for better compatibility

    print("Using device:", device)

    # Load models and processors
    print("Loading models and processors...")
    tokenizer = T5Tokenizer.from_pretrained(pretrained_model_name_or_path, subfolder="tokenizer")
    text_encoder = T5EncoderModel.from_pretrained(pretrained_model_name_or_path, subfolder="text_encoder")
    vae = AutoencoderKLCogVideoX.from_pretrained(pretrained_model_name_or_path, subfolder="vae")
    scheduler = CogVideoXDPMScheduler.from_pretrained(pretrained_model_name_or_path, subfolder="scheduler")
    
    clip_tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-base-patch16")
    clip_text_encoder = CLIPTextModel.from_pretrained("openai/clip-vit-base-patch16")
    clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch16")

    print("Loading transformer...")
    transformer = CustomCogVideoXTransformer3DModel.from_pretrained(
        pretrained_model_name_or_path,
        subfolder="transformer",
        torch_dtype=dtype,
        customization=True,
    )

    print("Creating pipeline...")
    pipe = CustomCogVideoXPipeline(
        tokenizer=tokenizer,
        text_encoder=text_encoder,
        transformer=transformer,
        vae=vae,
        scheduler=scheduler,
        clip_tokenizer=clip_tokenizer,
        clip_text_encoder=clip_text_encoder,
        customization=True,
    )

    print("Loading LoRA weights...")
    # Load LoRA weights manually
    lora_path = os.path.join(output_dir, "pytorch_lora_weights_transformer.safetensors")
    if not os.path.exists(lora_path):
        lora_path = os.path.join(output_dir, "pytorch_lora_weights.safetensors")
    
    if os.path.exists(lora_path):
        state_dict = load_file(lora_path)
        pipe.transformer.load_state_dict(state_dict, strict=False)
        print(f"Successfully loaded LoRA weights from {lora_path}")
    else:
        print(f"Warning: Could not find LoRA weights in {output_dir}")
        print("Available files:", os.listdir(output_dir))
        raise FileNotFoundError("LoRA weights not found")

    print("Initializing additional components...")
    # Initialize additional components before loading their state dictionaries
    pipe.transformer.T5ProjectionLayer = SkipProjectionLayer(4096, 4096)
    pipe.transformer.CLIPTextProjectionLayer = ProjectionLayer(512, 4096)
    pipe.transformer.CLIPVisionProjectionLayer = ProjectionLayer(768, 4096)
    pipe.transformer.reference_vision_encoder = CLIPVisionModel.from_pretrained("openai/clip-vit-base-patch16")

    print("Loading additional components...")
    # Load additional components
    component_files = {
        "T5ProjectionLayer": ["T5ProjectionLayer.pth", "T5ProjectionLayer.safetensors"],
        "CLIPTextProjectionLayer": ["CLIPTextProjectionLayer.pth", "CLIPTextProjectionLayer.safetensors"],
        "CLIPVisionProjectionLayer": ["CLIPVisionProjectionLayer.pth", "CLIPVisionProjectionLayer.safetensors"],
        "reference_vision_encoder": ["pytorch_clip_vision_model.bin", "reference_vision_encoder.safetensors"]
    }

    for component_name, filenames in component_files.items():
        loaded = False
        for filename in filenames:
            filepath = os.path.join(output_dir, filename)
            if os.path.exists(filepath):
                try:
                    if filename.endswith('.safetensors'):
                        state_dict = load_file(filepath)
                    else:
                        state_dict = torch.load(filepath, map_location=device)
                    getattr(pipe.transformer, component_name).load_state_dict(state_dict)
                    print(f"Successfully loaded {component_name} from {filename}")
                    loaded = True
                    break
                except Exception as e:
                    print(f"Error loading {component_name} from {filename}: {e}")
        
        if not loaded:
            print(f"Warning: Could not load {component_name} from any of the attempted files")

    # Move pipeline components to device and set data types
    pipe.transformer.to(device=device, dtype=dtype)
    pipe.text_encoder.to(device=device, dtype=dtype)
    pipe.vae.to(device=device, dtype=dtype)
    pipe.clip_text_encoder.to(device=device, dtype=dtype)
    pipe.transformer.reference_vision_encoder.to(device=device, dtype=dtype)
    pipe.transformer.T5ProjectionLayer.to(device=device, dtype=dtype)
    pipe.transformer.CLIPTextProjectionLayer.to(device=device, dtype=dtype)
    pipe.transformer.CLIPVisionProjectionLayer.to(device=device, dtype=dtype)

    # Set models to eval mode
    pipe.transformer.eval()
    pipe.text_encoder.eval()
    pipe.vae.eval()
    pipe.clip_text_encoder.eval()
    pipe.transformer.reference_vision_encoder.eval()

    print("Processing reference image...")
    ref_image = Image.open(reference_image_path).convert('RGB')
    processed_image = clip_processor(
        images=ref_image,
        return_tensors="pt"
    )
    # Move the pixel_values tensor to device and dtype
    pixel_values = processed_image['pixel_values'].to(device=device, dtype=dtype)

    # Generate video
    print("Generating video...")
    generator = torch.Generator(device=device).manual_seed(42)
    
    with torch.no_grad():
        output = pipe(
            prompt=prompt,
            negative_prompt=negative_prompt,
            ref_img_states=pixel_values,
            height=480,
            width=720,
            num_frames=49,
            num_inference_steps=50,
            guidance_scale=6.0,
            use_dynamic_cfg=False,
            generator=generator,
            output_type="pil",
        )

    # Save the output video
    output_path = "output_video2.mp4"
    from diffusers.utils import export_to_video
    export_to_video(output.frames[0], output_path, fps=8)
    print(f"Video saved to {output_path}")

if __name__ == "__main__":
    main()
