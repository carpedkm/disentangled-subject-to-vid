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
import torchvision.transforms as transforms
from einops import rearrange, repeat

from modules.utils.util import save_videos_grid

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



def save_video(video_path, save_path, current_sample_stride):
    pixel_values, condition_image = get_batch(video_path, current_sample_stride)
    
    gt_video = rearrange(pixel_values, 'f c h w -> c f h w').to('cpu')  # [3, f 2h, w]
    save_videos_grid(gt_video[None, ...], save_path, rescale=False)


def main(
    json_path,
    data_root_path,
    save_path,
    num,
    current_sample_stride,
    ):
    
    
    with open(json_path, "r") as st_json:
        flow_statistics = json.load(st_json)


    os.makedirs(save_path, exist_ok=True)

    # criterion 기준 내림차순 저장.
    criterion = 'bg_mean'
    # criterion = 'fg_mean'
    sorted_data_list = sorted(flow_statistics.items(), key=lambda x: x[1][criterion], reverse=True)
    sorted_data_dict = dict(sorted_data_list)
    
    # Filtering dataset..
    # FG, BG 둘 다 사용하는 것이 좋아보임.
    filtered_data_list = [x for x in sorted_data_dict.items() if x[1]['bg_mean'] != 0 and x[1]['bg_mean'] <= 5.0 and x[1]['fg_mean'] >= 0.1]
    
    
    # 0 < fg < 20
    # filtered_data_list = [x for x in filtered_data_list if x[1]['fg_mean'] < 20]
    # fg > 40
    filtered_data_list = [x for x in filtered_data_list if x[1]['fg_mean'] >= 20 and x[1]['fg_mean'] < 40]
    # fg > 40
    # filtered_data_list = [x for x in filtered_data_list if x[1]['fg_mean'] >= 40]
    
    pdb.set_trace()
    
    # filtered_data_list = [x for x in sorted_data_dict.items() if x[1]['bg_mean'] >= 0.1 and x[1]['bg_mean'] <= 1.0 and x[1]['fg_mean'] >= 0.1 and x[1]['fg_mean'] < 25.0]
    # filtered_data_list = [x for x in sorted_data_dict.items() if x[1]['bg_mean'] >= 0.1 and x[1]['bg_mean'] <= 1.0 and x[1]['fg_mean'] >= 25.0 and x[1]['fg_mean'] < 50.0]
    # filtered_data_list = [x for x in sorted_data_dict.items() if x[1]['bg_mean'] >= 0.1 and x[1]['bg_mean'] <= 1.0 and x[1]['fg_mean'] >= 50.0 and x[1]['fg_mean'] < 75.0]
    # filtered_data_list = [x for x in sorted_data_dict.items() if x[1]['bg_mean'] >= 0.1 and x[1]['bg_mean'] <= 1.0 and x[1]['fg_mean'] >= 75.0 and x[1]['fg_mean'] < 100.0]
    # filtered_data_list = [x for x in sorted_data_dict.items() if x[1]['bg_mean'] >= 0.1 and x[1]['bg_mean'] <= 1.0 and x[1]['fg_mean'] >= 100.0 and x[1]['fg_mean'] < 300]
    
    # filtered_data_list = [x for x in sorted_data_dict.items() if x[1]['bg_mean'] != 0]
    # filtered_data_dict = dict(filtered_data_list)
    
    # filtered_data_list = random.sample(filtered_data_list, num*3)
    
    
    idx_list = np.linspace(0,num-1,num, dtype=np.int64)
    idx_list = idx_list.tolist() + (-(idx_list+1)).tolist()
    # idx_list = idx_list.tolist()

    for idx in tqdm(idx_list):
        
        # For the filtered videos
        video_path = os.path.join(data_root_path, filtered_data_list[idx][0])
        video_save_path = os.path.join(save_path, f'pexels_bg5.0_fg0-20', f"video_bg_mean-({filtered_data_list[idx][1]['bg_mean']:04f})_fg_mean-({filtered_data_list[idx][1]['fg_mean']:04f}).mp4")
        try:
            save_video(video_path, video_save_path, current_sample_stride=current_sample_stride)
        except:
            print(f'fail to save {video_save_path} for current stride {current_sample_stride}')
            pass
        
        # # For the filtered videos
        # video_path = os.path.join(data_root_path, filtered_data_list[idx][0])
        # video_save_path = os.path.join(save_path, f'af_filter_stride-{current_sample_stride}', f"video_bg_mean-({filtered_data_list[idx][1]['bg_mean']:04f})_fg_mean-({filtered_data_list[idx][1]['fg_mean']:04f}).mp4")
        # try:
        #     save_video(video_path, video_save_path, current_sample_stride=current_sample_stride)
        # except:
        #     print(f'fail to save {video_save_path} for current stride {current_sample_stride}')
        #     pass

    
    # for idx in tqdm(range(len(filtered_data_list))):
        
    #     # For the filtered videos
    #     video_path = os.path.join(data_root_path, filtered_data_list[idx][0])
    #     video_save_path = os.path.join(save_path, f'shutterstock-fg-100-300_ver2', f"video_bg_mean-({filtered_data_list[idx][1]['bg_mean']:04f})_fg_mean-({filtered_data_list[idx][1]['fg_mean']:04f}).mp4")
    #     try:
    #         save_video(video_path, video_save_path, current_sample_stride=current_sample_stride)
    #     except:
    #         print(f'fail to save {video_save_path} for current stride {current_sample_stride}')
    #         pass
        
    #     # # For the filtered videos
    #     # video_path = os.path.join(data_root_path, filtered_data_list[idx][0])
    #     # video_save_path = os.path.join(save_path, f'af_filter_stride-{current_sample_stride}', f"video_bg_mean-({filtered_data_list[idx][1]['bg_mean']:04f})_fg_mean-({filtered_data_list[idx][1]['fg_mean']:04f}).mp4")
    #     # try:
    #     #     save_video(video_path, video_save_path, current_sample_stride=current_sample_stride)
    #     # except:
    #     #     print(f'fail to save {video_save_path} for current stride {current_sample_stride}')
    #     #     pass
    
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--json_path", type=str, required=True)
    parser.add_argument("--num", type=int, required=True)
    parser.add_argument("--current_sample_stride", type=int, required=True)
    parser.add_argument("--data_root_path", type=str, required=False, default='/video_data/')
    parser.add_argument("--save_path", type=str, required=False, default='./sample_video')
    args = parser.parse_args()

    main(json_path=args.json_path, num=args.num, current_sample_stride=args.current_sample_stride, data_root_path=args.data_root_path, save_path=args.save_path)