import json
import os

from tqdm import tqdm
from PIL import Image
import torch
from diffusers import CogVideoXImageToVideoPipeline
from diffusers.utils import export_to_video, load_image, export_to_video_with_frames

prompt_path = "/root/daneul/projects/refactored/CogVideo/Pexels_subset_100K_fps8_flow-25-50_sample500/small/metadata.jsonl"
type_to_eval = "small" # medium, large
first_frame_path = "/root/daneul/projects/refactored/CogVideo/Pexels_subset_100K_fps8_flow-25-50_sample500/small/first_frame"
video_save_path = "/root/daneul/projects/refactored/CogVideo/Baselines/I2V_baseline/Temporal_eval"
sampling_count = 100

pipe = CogVideoXImageToVideoPipeline.from_pretrained(
    "THUDM/CogVideoX-5b-I2V",
    torch_dtype=torch.bfloat16
)

pipe.enable_sequential_cpu_offload()
pipe.vae.enable_tiling()
pipe.vae.enable_slicing()

# load path jsonl
with open(prompt_path, "r") as f:
    lines = f.readlines()
    meta_list = [line for line in lines]
# make meta dict 
meta_dict = {}
meta_list = []
with open(prompt_path, 'r') as f:
    for line in f:
        try:
            meta_list.append(json.loads(line))
        except:
            print('Error in loading json')
meta_dict = {}
for meta in meta_list:
    vid_id = str(meta['video_latent_path'].split('/')[-1].split('.')[0])
    meta_dict[vid_id] = meta
# get input image list
image_list = os.listdir(first_frame_path)

for i in tqdm(range(sampling_count)):
    vid_id = str(image_list[i].split(".")[0])
    input_image = os.path.join(first_frame_path, image_list[i])
    prompt = meta_dict[vid_id]["prompt"]
    input_image = Image.open(input_image)
    # pass through I2V
    video = pipe(
        prompt=prompt,
        image=input_image,
        num_videos_per_prompt=1,
        num_inference_steps=50,
        num_frames=49,
        guidance_scale=6,
        generator=torch.Generator(device="cuda").manual_seed(42),
    ).frames[0]
    save_path = os.path.join(video_save_path, f"{i}.mp4")
    frames_save_path = os.path.join(video_save_path, type_to_eval, "video_frames")
    export_to_video_with_frames(video, save_path, frames_save_path, fps=8, eval_mode=True)
    