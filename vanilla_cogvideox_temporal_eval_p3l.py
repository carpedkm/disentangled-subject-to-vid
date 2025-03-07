import os
import torch
from diffusers import CogVideoXPipeline
from diffusers.utils import export_to_video, export_to_video_with_frames
import json

save_dir = "/mnt/carpedkm_data/temporal_eval_result/t2v_vanilla_final"
prompt_path="/mnt/carpedkm_data/image_gen_ds/Pexels_subset_100K_fps8_flow-25-50_sample500/medium/metadata.jsonl"
temporal_eval_first_frame="/mnt/carpedkm_data/image_gen_ds/Pexels_subset_100K_fps8_flow-25-50_sample500/medium/first_frame"
temporal_eval_type = "large"
temporal_eval_use_amount = 300
temporal_eval_shard = 3

pipe = CogVideoXPipeline.from_pretrained(
    "THUDM/CogVideoX-5b",
    torch_dtype=torch.bfloat16
)

pipe.enable_model_cpu_offload()
pipe.vae.enable_tiling()

os.makedirs(save_dir, exist_ok=True)
output_dir = os.path.join(save_dir, temporal_eval_type)
os.makedirs(output_dir, exist_ok=True)
resizing=False
meta_path = prompt_path
meta_list = []
with open(meta_path, 'r') as f:
    for line in f:
        try:
            meta_list.append(json.loads(line))
        except:
            print('Error in loading json')
# make dictionary to parse the video id : the other info
meta_dict = {}
for meta in meta_list:
    vid_id = str(meta['video_latent_path'].split('/')[-1].split('.')[0])
    meta_dict[vid_id] = meta
input_image_path = temporal_eval_first_frame # prepend 'small', 'medium', 'large'
input_image_list = sorted(os.listdir(input_image_path))

shard_amount = temporal_eval_use_amount // 4

for i in range(temporal_eval_shard * shard_amount, (temporal_eval_shard + 1) * shard_amount):
    print(f"Processing {i}th video")
    input_image = os.path.join(input_image_path, input_image_list[i])
    vid_id = str(input_image_list[i].split('.')[0])
    if os.path.exists(os.path.join(output_dir, 'video_frames', vid_id)):
        print('Already exists: ', os.path.join(output_dir, 'video_frames', vid_id))
        continue
    if vid_id in meta_dict.keys():
        prompt = meta_dict[vid_id]['prompt']
    else:
        print('No prompt found for vid_id: ', vid_id)
        break
    video = pipe(
        prompt=prompt,
        num_videos_per_prompt=1,
        num_inference_steps=50,
        num_frames=49,
        guidance_scale=6,
        generator=torch.Generator(device=f"cuda").manual_seed(42),
    ).frames[0]
    vid_save_dir = os.path.join(output_dir, 'videos')
    os.makedirs(vid_save_dir, exist_ok=True)
    frames_save_dir = os.path.join(output_dir, 'video_frames', vid_id)
    os.makedirs(frames_save_dir, exist_ok=True)
    # export_to_video(video, "output.mp4", fps=8)
    export_to_video_with_frames(
        video_frames=video,
        output_video_path=os.path.join(vid_save_dir, f"{vid_id}.mp4"),
        output_frames_dir=frames_save_dir,
        fps=8,
        eval_mode=True,
    )