# %%
import torch
device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
x = torch.tensor([1.0], device=device)
y = x * 2
print(y)


# %%
import os
import torch
from PIL import Image
from diffusers import CogVideoXDPMScheduler
from diffusers.utils import export_to_video
from custom_cogvideox_pipe import CustomCogVideoXPipeline
from transformers import CLIPProcessor, CLIPTokenizer, CLIPTextModel, CLIPVisionModel


# Move the pipeline to the appropriate device (GPU or CPU)
device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")

# %%
# Define the paths to your pretrained model and the output directory where your checkpoints are saved
pretrained_model_name_or_path = "THUDM/CogVideoX-5b"  # Replace with your pretrained model path or name
output_dir = "/mnt/carpedkm_data/finetune_result/finetune4000_custom_zero_init_t5_full_custom_with_clip/checkpoint-800"  # Replace with your output directory where the checkpoints are saved

# Prepare the input prompt and reference image
prompt = "Two dogs one with a black and tan coat and another with a black and white coat appear to be playing on a lush green lawn with trees and a building in the background"  # Replace with your desired prompt
reference_image_path = "/root/daneul/projects/refactored/CogVideo/finetune/val_samples/854179_background_boxes.jpg"  # Replace with the path to your reference image
ref_image = Image.open(reference_image_path).convert('RGB')

# %%
# Set the LoRA parameters (use the same values as during training)
lora_alpha = 128  # Replace with your value if different
rank = 128        # Replace with your value if different
lora_scaling = lora_alpha / rank

# %%
# Load the pipeline
pipe = CustomCogVideoXPipeline.from_pretrained(
    pretrained_model_name_or_path,
    customization=True  # Ensure this is set to True for customization
)
pipe.scheduler = CogVideoXDPMScheduler.from_config(pipe.scheduler.config)

# %%
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

# %%

pipe = pipe.to(device)

# Load the LoRA weights directly from the local file
lora_weights_path = os.path.join(output_dir, "pytorch_lora_weights_transformer.safetensors")
if not os.path.exists(lora_weights_path):
    raise FileNotFoundError(f"LoRA weights not found at {lora_weights_path}")

# Load the LoRA state dictionary
from safetensors.torch import load_file
lora_state_dict = load_file(lora_weights_path)

# Load the LoRA weights into the pipeline
pipe.load_lora_weights(
    pretrained_model_name_or_path_or_dict=lora_state_dict,
    adapter_name="cogvideox-lora"
)
pipe.set_adapters(["cogvideox-lora"], [lora_scaling])

# Load additional components (projection layers and reference vision encoder)
# Ensure the paths are correct and the files exist
projection_layers = {
    "T5ProjectionLayer": "T5ProjectionLayer.pth",
    "CLIPTextProjectionLayer": "CLIPTextProjectionLayer.pth",
    "CLIPVisionProjectionLayer": "CLIPVisionProjectionLayer.pth",
    "reference_vision_encoder": "pytorch_clip_vision_model.bin"
}

for layer_name, filename in projection_layers.items():
    layer_path = os.path.join(output_dir, filename)
    if not os.path.exists(layer_path):
        raise FileNotFoundError(f"{layer_name} weights not found at {layer_path}")
pipe.transformer.T5ProjectionLayer = SkipProjectionLayer(4096, 4096)
pipe.transformer.CLIPTextProjectionLayer = ProjectionLayer(512, 4096)
pipe.transformer.CLIPVisionProjectionLayer = ProjectionLayer(768, 4096)
# pipe.transformer.reference_vision_encoder = CLIPVisionEncoder()
# Correctly initialize the CLIPVisionModel using from_pretrained
pipe.transformer.reference_vision_encoder = CLIPVisionModel.from_pretrained("openai/clip-vit-base-patch16")

# Move the reference vision encoder to the appropriate device
pipe.transformer.reference_vision_encoder = pipe.transformer.reference_vision_encoder.to(device)

# Load the projection layers and vision encoder
pipe.transformer.T5ProjectionLayer.load_state_dict(
    torch.load(os.path.join(output_dir, "T5ProjectionLayer.pth"), map_location=device)
)
pipe.transformer.CLIPTextProjectionLayer.load_state_dict(
    torch.load(os.path.join(output_dir, "CLIPTextProjectionLayer.pth"), map_location=device)
)
pipe.transformer.CLIPVisionProjectionLayer.load_state_dict(
    torch.load(os.path.join(output_dir, "CLIPVisionProjectionLayer.pth"), map_location=device)
)
pipe.transformer.reference_vision_encoder.load_state_dict(
    torch.load(os.path.join(output_dir, "pytorch_clip_vision_model.bin"), map_location=device)
)

# %%
# Move models to device
pipe.transformer.to(device)
pipe.text_encoder.to(device)
pipe.clip_text_encoder.to(device)
pipe.vae.to(device)
pipe.transformer.reference_vision_encoder.to(device)
pipe.transformer.T5ProjectionLayer.to(device)
pipe.transformer.CLIPTextProjectionLayer.to(device)
pipe.transformer.CLIPVisionProjectionLayer.to(device)
# dtype = torch.float16  # or torch.float32, depending on your setup
dtype = torch.float32

pipe.transformer.to(dtype=dtype)
pipe.text_encoder.to(dtype=dtype)
pipe.clip_text_encoder.to(dtype=dtype)
pipe.vae.to(dtype=dtype)
pipe.transformer.reference_vision_encoder.to(dtype=dtype)
pipe.transformer.T5ProjectionLayer.to(dtype=dtype)
pipe.transformer.CLIPTextProjectionLayer.to(dtype=dtype)
pipe.transformer.CLIPVisionProjectionLayer.to(dtype=dtype)

# %%
# Generate the video
# Process the reference image
clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch16")
processed_image = clip_processor(images=ref_image, return_tensors="pt")
ref_img_states = processed_image['pixel_values'].to(device)
print("ref_img_states shape:", ref_img_states.shape)
print("ref_img_states device:", ref_img_states.device)
ref_img_states = ref_img_states.to(dtype=dtype)

with torch.no_grad():
    video_output = pipe(
        prompt=prompt,
        ref_img_states=ref_img_states,
        guidance_scale=6,          # Adjust guidance scale if needed
        use_dynamic_cfg=True,      # Set to True if you want to use dynamic CFG
        num_frames=49,             # Adjust the number of frames if needed
        height=480,                # Set the desired video height
        width=720,                 # Set the desired video width
        num_inference_steps=50,    # Adjust the number of inference steps if needed
        output_type='np',          # Output as numpy array
        eval=True                  # Set to True for evaluation mode
    )

# %%
# Extract the video frames from the output
video_frames = video_output.frames[0]  # Assuming frames is a list of videos

# Save the video to a file
output_video_path = "output_video.mp4"  # Specify the output video file path
export_to_video(video_frames, output_video_path, fps=8)  # Adjust FPS if needed

print(f"Video saved to {output_video_path}")

# %%


# %%



