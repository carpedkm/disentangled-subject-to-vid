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
from tqdm import tqdm

from decord import VideoReader
import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms
from einops import rearrange, repeat

from modules.utils.util import save_videos_grid

random.seed(7777)

sample_size=[480, 720]
pixel_transforms = transforms.Resize(sample_size)

def load_video_reader(video_path):
    video_reader = VideoReader(video_path)
    return video_reader

def get_batch(video_path, current_sample_stride=1):
    sample_n_frames=14
    
    # scene_path = self.scene_path_list[idx]
    video_reader = load_video_reader(video_path)
    
    total_frames = len(video_reader)
    
    assert total_frames >= sample_n_frames * current_sample_stride * 2
    
    cropped_length = (sample_n_frames+1) * current_sample_stride
    start_frame_ind = total_frames // 2 # middle frame
    end_frame_ind = min(start_frame_ind + cropped_length, total_frames)
    assert end_frame_ind - start_frame_ind >= (sample_n_frames+1)
    frame_indices_total = np.linspace(start_frame_ind, end_frame_ind - 1, (sample_n_frames+1), dtype=int)
    
    # end_frame_ind = start_frame_ind + sample_n_frames - 1
    # frame_indices_total = np.linspace(start_frame_ind, end_frame_ind, (sample_n_frames+1), dtype=int)
    
    condition_image_ind = frame_indices_total[:1]
    frame_indices = frame_indices_total[1:]
    
    image_tensor_all = torch.from_numpy(video_reader.get_batch(frame_indices_total).asnumpy()).permute(0, 3, 1, 2).contiguous() / 255.
    condition_image = image_tensor_all[0:1]
    pixel_values = image_tensor_all[1:]
    
    pixel_values = pixel_transforms(pixel_values)
    condition_image = pixel_transforms(condition_image)

    return pixel_values, condition_image

# def save_video(video_path, save_path, current_sample_stride):
#     pixel_values, condition_image = get_batch(video_path, current_sample_stride)
    
#     gt_video = rearrange(pixel_values, 'f c h w -> c f h w').to('cpu')  # [3, f 2h, w]
#     save_videos_grid(gt_video[None, ...], save_path, rescale=False)

def save_video(video_path, save_path, video_name, current_sample_stride):
    os.makedirs(os.path.join(save_path, 'selected_clips'), exist_ok=True)
    os.makedirs(os.path.join(save_path, 'selected_images'), exist_ok=True)
    
    pixel_values, _ = get_batch(video_path, current_sample_stride)
    
    mp4_path = os.path.join(save_path, 'selected_clips', video_name)
    gt_video = rearrange(pixel_values, 'f c h w -> c f h w').to('cpu')  # [3, f 2h, w]
    save_videos_grid(gt_video[None, ...], mp4_path, rescale=False)
    
    for idx ,image in enumerate(pixel_values):
        os.makedirs(os.path.join(save_path, 'selected_images',  video_name.split(".mp4")[0]), exist_ok=True)
        image_path = os.path.join(save_path, 'selected_images',  video_name.split(".mp4")[0], f"{idx:02d}.jpg")
        torchvision.utils.save_image(image, image_path)

def main(
    json_path,
    data_root_path,
    save_path,
    num,
    current_sample_stride,
    ):
    
    os.makedirs(save_path, exist_ok=True)
    
    with open(json_path, "r") as st_json:
        flow_statistics = json.load(st_json)
    
    
    # criterion 기준 내림차순 저장.
    criterion = 'bg_mean'
    # criterion = 'fg_mean'
    sorted_data_list = sorted(flow_statistics.items(), key=lambda x: x[1][criterion], reverse=True)
    sorted_data_dict = dict(sorted_data_list)
    
    # Filtering dataset..
    # FG, BG 둘 다 사용하는 것이 좋아보임.
    
    filtered_data_list = [x for x in sorted_data_dict.items() if x[1]['bg_mean'] != 0]
    
    # randomly sample N videos
    filtered_data_set = random.sample(filtered_data_list, int(num))
    
    for idx, video_item in enumerate(tqdm(filtered_data_set, desc='set')):
        video_path = os.path.join(data_root_path, video_item[0])
        video_name = video_path.split('/')[-1]

        video_save_path = os.path.join(save_path, 'randomly_sampled')
        try:
            save_video(video_path, video_save_path, video_name, current_sample_stride)
        except:
            print(f"pass {video_name}")
    
    

    
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--json_path", type=str, required=True)
    parser.add_argument("--num", type=int, required=True)
    parser.add_argument("--current_sample_stride", type=int, required=True)
    parser.add_argument("--data_root_path", type=str, required=False, default='/video_data/')
    parser.add_argument("--save_path", type=str, required=False, default='./sample_video')
    args = parser.parse_args()

    main(json_path=args.json_path, num=args.num, current_sample_stride=args.current_sample_stride, data_root_path=args.data_root_path, save_path=args.save_path)