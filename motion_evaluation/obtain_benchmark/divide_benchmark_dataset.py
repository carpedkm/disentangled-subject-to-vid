import os
import math
import time
import inspect
import argparse
import datetime
import subprocess
import json
import pdb
import random
from pathlib import Path
from tqdm import tqdm
from matplotlib import pyplot as plt

import copy
import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms
from einops import rearrange, repeat

from decord import VideoReader
from data.utils import preprocess_video_with_resize_and_filtering, save_video, save_video_fast

from modules.utils.util import save_videos_grid

from diffusers import (
    AutoencoderKLCogVideoX,
    CogVideoXDPMScheduler,
    CogVideoXImageToVideoPipeline,
    CogVideoXTransformer3DModel,
)

random.seed(7777)


class VideoSampler:
    def __init__(
        self,
        root_path,
        sample_fps=16,
        sample_n_frames=49,
        sample_size=[480, 720],
        index_first=30,
    ):
        self.root_path = Path(root_path)
        self.sample_fps = sample_fps
        self.sample_n_frames = sample_n_frames
        self.index_first = index_first

        self.sample_n_frames, self.height, self.width = sample_n_frames, sample_size[0], sample_size[1]
        
        self.__frame_transforms = transforms.Compose([transforms.Lambda(lambda x: x / 255.0 * 2.0 - 1.0)])
        self.__image_transforms = self.__frame_transforms

    def video_transform(self, frames: torch.Tensor) -> torch.Tensor:
        return torch.stack([self.__frame_transforms(f) for f in frames], dim=0)

    def preprocess(self, video_path: Path) -> torch.Tensor:
        return preprocess_video_with_resize_and_filtering(
            video_path,
            self.sample_n_frames,
            self.height,
            self.width,
            self.sample_fps,
            index_first = self.index_first,
        )

    def __call__(self, video_path):
        path = self.root_path / video_path
        frames = self.preprocess(path)
        frames = self.video_transform(frames).contiguous()
        first_frame = frames[0:1].detach().clone() # [C, H, W]

        return {
            'video': frames,
            'first_frame': first_frame,
        }

        



# def save_video(video_path, save_path, video_name, current_sample_stride):
#     os.makedirs(os.path.join(save_path, 'selected_clips'), exist_ok=True)
#     os.makedirs(os.path.join(save_path, 'selected_images'), exist_ok=True)
    
#     pixel_values, _ = get_batch(video_path, current_sample_stride)
    
#     mp4_path = os.path.join(save_path, 'selected_clips', video_name)
#     gt_video = rearrange(pixel_values, 'f c h w -> c f h w').to('cpu')  # [3, f 2h, w]
#     save_videos_grid(gt_video[None, ...], mp4_path, rescale=False)
    
#     for idx ,image in enumerate(pixel_values):
#         os.makedirs(os.path.join(save_path, 'selected_images',  video_name.split(".mp4")[0]), exist_ok=True)
#         image_path = os.path.join(save_path, 'selected_images',  video_name.split(".mp4")[0], f"{idx:02d}.jpg")
#         torchvision.utils.save_image(image, image_path)

def main(
    json_path,
    data_root_path,
    save_path,
    num,
    sample_fps,
    ):

    with open(json_path, "r") as f:
        flow_statistics = json.load(f)
    
    metadata = []
    for key, value in flow_statistics.items():
        tmp = {}
        tmp['video_path'] = key
        tmp['fg_mean'] = value['fg_mean']
        tmp['bg_mean'] = value['bg_mean']
        tmp['prompt'] = value['prompt']
        metadata.append(tmp)

    
    
    video_sampler = VideoSampler(data_root_path, sample_fps=sample_fps)
    # data_dict = video_sampler(metadata[0]['video_path']) # FCHW, 1CHW
    # pdb.set_trace()

    # Filtering dataset
    metadata_filtered_list = [x for x in metadata if x['bg_mean'] <= 10 and x['bg_mean'] != 0 and x['fg_mean'] >= 0.1]

    metadata_filtered_fg_small = [x for x in metadata_filtered_list if x['fg_mean'] >= 0.1 and x['fg_mean'] < 25.]
    metadata_filtered_fg_medium = [x for x in metadata_filtered_list if x['fg_mean'] >= 25 and x['fg_mean'] < 50.]
    metadata_filtered_fg_large = [x for x in metadata_filtered_list if x['fg_mean'] >= 50 and x['fg_mean'] < 500.]

    
    # Get Statistics
    bg_mean_list = [x['bg_mean'] for x in metadata]
    fg_mean_list = [x['fg_mean'] for x in metadata]
    fg_mean_bg_filtered_list = [x['fg_mean'] for x in metadata_filtered_list]

    # stat_dir = "./statistics"
    # os.makedirs(stat_dir, exist_ok=True)
    # counts, bins = np.histogram(bg_mean_list, bins=50)
    # counts = counts / counts.sum()
    # plt.bar(bins[:-1], counts, width=np.diff(bins), color='green', edgecolor='black', alpha=0.7)
    # plt.show()
    # plt.savefig(os.path.join(stat_dir, 'bg_mean_10K_fps8.png'))

    # plt.clf()
    # counts, bins = np.histogram(fg_mean_list, bins=50)
    # counts = counts / counts.sum()
    # plt.bar(bins[:-1], counts, width=np.diff(bins), color='green', edgecolor='black', alpha=0.7)
    # plt.show()
    # plt.savefig(os.path.join(stat_dir, 'fg_mean_10K_fps8.png'))

    # plt.clf()
    # counts, bins = np.histogram(fg_mean_bg_filtered_list, bins=50)
    # counts = counts / counts.sum()
    # plt.bar(bins[:-1], counts, width=np.diff(bins), color='green', edgecolor='black', alpha=0.7)
    # plt.show()
    # plt.savefig(os.path.join(stat_dir, 'fg_mean_bg_filtered_10K_fps8.png'))
    # pdb.set_trace()

    # Sampling data
    assert len(metadata_filtered_fg_small) >= int(num) and len(metadata_filtered_fg_medium) >= int(num) and len(metadata_filtered_fg_large) >= int(num), f"Filtered dataset size is less than {int(num)}"
    metadata_filtered_fg_small = random.sample(metadata_filtered_fg_small, int(num))
    metadata_filtered_fg_medium = random.sample(metadata_filtered_fg_medium, int(num))
    metadata_filtered_fg_large = random.sample(metadata_filtered_fg_large, int(num))

    
    save_path_small = os.path.join(save_path, 'small')
    save_path_medium = os.path.join(save_path, 'medium')
    save_path_large = os.path.join(save_path, 'large')
    os.makedirs(save_path_small, exist_ok=True)
    os.makedirs(save_path_medium, exist_ok=True)
    os.makedirs(save_path_large, exist_ok=True)

    # For Visualization
    # for _metadata in tqdm(metadata_filtered_fg_small):
    #     data_dict = video_sampler(_metadata['video_path'])
    #     frames = data_dict['video'] # F C H W

    #     video_name = _metadata['video_path'].split('/')[-1]
    #     save_video(rearrange(frames, 'f c h w -> c f h w'), os.path.join(save_path_small, video_name), fps=16)
    

    # for _metadata in tqdm(metadata_filtered_fg_medium):
    #     data_dict = video_sampler(_metadata['video_path'])
    #     frames = data_dict['video'] # F C H W

    #     video_name = _metadata['video_path'].split('/')[-1]
    #     save_video(rearrange(frames, 'f c h w -> c f h w'), os.path.join(save_path_medium, video_name), fps=16)
    

    # for _metadata in tqdm(metadata_filtered_fg_large):
    #     data_dict = video_sampler(_metadata['video_path'])
    #     frames = data_dict['video'] # F C H W

    #     video_name = _metadata['video_path'].split('/')[-1]
    #     save_video(rearrange(frames, 'f c h w -> c f h w'), os.path.join(save_path_large, video_name), fps=16)
    

    # For Constructing benchmark dataset

    # prepare model
    MODEL_PATH="THUDM/CogVideoX-5b-I2V"
    dtype=torch.bfloat16
    pipe = CogVideoXImageToVideoPipeline.from_pretrained(MODEL_PATH, torch_dtype=dtype)
    vae = copy.deepcopy(pipe.vae.to('cuda').eval())
    del pipe
    # vae = AutoencoderKLCogVideoX.from_pretrained(MODEL_PATH, torch_dtype=dtype).to('cuda').eval()


    #---------------------------------------------------------------------------------------------------------
    # SMALL
    metadata_small_final = []
    metadata_save_path = os.path.join(save_path_small, 'metadata.jsonl')

    video_save_path = os.path.join(save_path_small, 'video')
    video_frames_save_path = os.path.join(save_path_small, 'video_frames')
    video_latent_save_path = os.path.join(save_path_small, 'video_latent')
    first_frame_save_path = os.path.join(save_path_small, 'first_frame')
    first_frame_latent_save_path = os.path.join(save_path_small, 'first_frame_latent')

    os.makedirs(video_save_path, exist_ok=True)
    os.makedirs(video_frames_save_path, exist_ok=True)
    os.makedirs(video_latent_save_path, exist_ok=True)
    os.makedirs(first_frame_save_path, exist_ok=True)
    os.makedirs(first_frame_latent_save_path, exist_ok=True)

    for _metadata in tqdm(metadata_filtered_fg_small):
        video_name = _metadata['video_path'].split('/')[-1].split('.mp4')[0]
        data_dict = video_sampler(_metadata['video_path'])
        frames = data_dict['video'] # F C H W
        prompt = _metadata['prompt']

        metadata_final = {
            'video_latent_path': os.path.join('video_latent', video_name+'.npy'),
            'first_frame_latent_path': os.path.join('first_frame_latent', video_name+'.npy'),
            'prompt': prompt,
        }
        metadata_small_final.append(metadata_final)

        # save video
        save_video_fast(rearrange(frames, 'f c h w -> c f h w'), os.path.join(video_save_path, video_name+'.mp4'), fps=sample_fps)

        # save video frames
        save_path = os.path.join(video_frames_save_path, f'{video_name}')
        os.makedirs(save_path, exist_ok=True)
        for idx, frame in enumerate(frames):
            torchvision.utils.save_image(((frames[0:1]+1)/2).detach().cpu(), os.path.join(save_path, f'{idx:02d}.png'))

        # save video_latent
        with torch.cuda.amp.autocast(enabled=True, dtype=dtype):
            with torch.no_grad():
                video_latent = encode_video(rearrange(frames, 'f c h w -> c f h w').unsqueeze(0), vae)
        np.save(os.path.join(video_latent_save_path, video_name+'.npy'), video_latent.float().detach().cpu().numpy()) # BCFHW

        # save first_frame
        torchvision.utils.save_image(((frames[0:1]+1)/2).detach().cpu(), os.path.join(first_frame_save_path, video_name+'.png'))

        # save first_frame_latent
        with torch.cuda.amp.autocast(enabled=True, dtype=dtype):
            with torch.no_grad():
                first_frame_latent = encode_video(rearrange(frames[0:1], 'f c h w -> c f h w').unsqueeze(0), vae)
        np.save(os.path.join(first_frame_latent_save_path, video_name+'.npy'), rearrange(first_frame_latent, 'b c f h w -> b (c f) h w').float().detach().cpu().numpy()) # BCHW

    
    # save metadata
    with open(metadata_save_path, "a", encoding="utf-8") as f:
        for line in metadata_small_final:
            json.dump(line, f)
            f.write("\n")
    

    # MEDIUM
    metadata_medium_final = []
    metadata_save_path = os.path.join(save_path_medium, 'metadata.jsonl')

    video_save_path = os.path.join(save_path_medium, 'video')
    video_frames_save_path = os.path.join(save_path_medium, 'video_frames')
    video_latent_save_path = os.path.join(save_path_medium, 'video_latent')
    first_frame_save_path = os.path.join(save_path_medium, 'first_frame')
    first_frame_latent_save_path = os.path.join(save_path_medium, 'first_frame_latent')

    os.makedirs(video_save_path, exist_ok=True)
    os.makedirs(video_frames_save_path, exist_ok=True)
    os.makedirs(video_latent_save_path, exist_ok=True)
    os.makedirs(first_frame_save_path, exist_ok=True)
    os.makedirs(first_frame_latent_save_path, exist_ok=True)

    for _metadata in tqdm(metadata_filtered_fg_medium):
        video_name = _metadata['video_path'].split('/')[-1].split('.mp4')[0]
        data_dict = video_sampler(_metadata['video_path'])
        frames = data_dict['video'] # F C H W
        prompt = _metadata['prompt']

        metadata_final = {
            'video_latent_path': os.path.join('video_latent', video_name+'.npy'),
            'first_frame_latent_path': os.path.join('first_frame_latent', video_name+'.npy'),
            'prompt': prompt,
        }
        metadata_medium_final.append(metadata_final)

        # save video
        save_video_fast(rearrange(frames, 'f c h w -> c f h w'), os.path.join(video_save_path, video_name+'.mp4'), fps=sample_fps)

        # save video frames
        save_path = os.path.join(video_frames_save_path, f'{video_name}')
        os.makedirs(save_path, exist_ok=True)
        for idx, frame in enumerate(frames):
            torchvision.utils.save_image(((frames[0:1]+1)/2).detach().cpu(), os.path.join(save_path, f'{idx:02d}.png'))

        # save video_latent
        with torch.cuda.amp.autocast(enabled=True, dtype=dtype):
            with torch.no_grad():
                video_latent = encode_video(rearrange(frames, 'f c h w -> c f h w').unsqueeze(0), vae)
        np.save(os.path.join(video_latent_save_path, video_name+'.npy'), video_latent.float().detach().cpu().numpy()) # BCFHW

        # save first_frame
        torchvision.utils.save_image(((frames[0:1]+1)/2).detach().cpu(), os.path.join(first_frame_save_path, video_name+'.png'))

        # save first_frame_latent
        with torch.cuda.amp.autocast(enabled=True, dtype=dtype):
            with torch.no_grad():
                first_frame_latent = encode_video(rearrange(frames[0:1], 'f c h w -> c f h w').unsqueeze(0), vae)
        np.save(os.path.join(first_frame_latent_save_path, video_name+'.npy'), rearrange(first_frame_latent, 'b c f h w -> b (c f) h w').float().detach().cpu().numpy()) # BCHW

    
    # save metadata
    with open(metadata_save_path, "a", encoding="utf-8") as f:
        for line in metadata_medium_final:
            json.dump(line, f)
            f.write("\n")
    

    # Large
    metadata_large_final = []
    metadata_save_path = os.path.join(save_path_large, 'metadata.jsonl')

    video_save_path = os.path.join(save_path_large, 'video')
    video_frames_save_path = os.path.join(save_path_large, 'video_frames')
    video_latent_save_path = os.path.join(save_path_large, 'video_latent')
    first_frame_save_path = os.path.join(save_path_large, 'first_frame')
    first_frame_latent_save_path = os.path.join(save_path_large, 'first_frame_latent')

    os.makedirs(video_save_path, exist_ok=True)
    os.makedirs(video_frames_save_path, exist_ok=True)
    os.makedirs(video_latent_save_path, exist_ok=True)
    os.makedirs(first_frame_save_path, exist_ok=True)
    os.makedirs(first_frame_latent_save_path, exist_ok=True)

    for _metadata in tqdm(metadata_filtered_fg_large):
        video_name = _metadata['video_path'].split('/')[-1].split('.mp4')[0]
        data_dict = video_sampler(_metadata['video_path'])
        frames = data_dict['video'] # F C H W
        prompt = _metadata['prompt']

        metadata_final = {
            'video_latent_path': os.path.join('video_latent', video_name+'.npy'),
            'first_frame_latent_path': os.path.join('first_frame_latent', video_name+'.npy'),
            'prompt': prompt,
        }
        metadata_large_final.append(metadata_final)

        # save video
        save_video_fast(rearrange(frames, 'f c h w -> c f h w'), os.path.join(video_save_path, video_name+'.mp4'), fps=sample_fps)

        # save video frames
        save_path = os.path.join(video_frames_save_path, f'{video_name}')
        os.makedirs(save_path, exist_ok=True)
        for idx, frame in enumerate(frames):
            torchvision.utils.save_image(((frames[0:1]+1)/2).detach().cpu(), os.path.join(save_path, f'{idx:02d}.png'))

        # save video_latent
        with torch.cuda.amp.autocast(enabled=True, dtype=dtype):
            with torch.no_grad():
                video_latent = encode_video(rearrange(frames, 'f c h w -> c f h w').unsqueeze(0), vae)
        np.save(os.path.join(video_latent_save_path, video_name+'.npy'), video_latent.float().detach().cpu().numpy()) # BCFHW

        # save first_frame
        torchvision.utils.save_image(((frames[0:1]+1)/2).detach().cpu(), os.path.join(first_frame_save_path, video_name+'.png'))

        # save first_frame_latent
        with torch.cuda.amp.autocast(enabled=True, dtype=dtype):
            with torch.no_grad():
                first_frame_latent = encode_video(rearrange(frames[0:1], 'f c h w -> c f h w').unsqueeze(0), vae)
        np.save(os.path.join(first_frame_latent_save_path, video_name+'.npy'), rearrange(first_frame_latent, 'b c f h w -> b (c f) h w').float().detach().cpu().numpy()) # BCHW

    
    # save metadata
    with open(metadata_save_path, "a", encoding="utf-8") as f:
        for line in metadata_large_final:
            json.dump(line, f)
            f.write("\n")



def encode_video(video: torch.Tensor, vae) -> torch.Tensor:
    # shape of input video: [B, C, F, H, W]
    video = video.to(vae.device, dtype=vae.dtype)
    latent_dist = vae.encode(video).latent_dist
    latent = latent_dist.sample() * vae.config.scaling_factor
    return latent


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--json_path", type=str, required=True)
    parser.add_argument("--num", type=int, required=True)
    parser.add_argument("--sample_fps", type=int, required=True)
    parser.add_argument("--data_root_path", type=str, required=False, default='/video_data/')
    parser.add_argument("--save_path", type=str, required=False, default='./sample_video')
    args = parser.parse_args()

    main(json_path=args.json_path, num=args.num, sample_fps=args.sample_fps, data_root_path=args.data_root_path, save_path=args.save_path)