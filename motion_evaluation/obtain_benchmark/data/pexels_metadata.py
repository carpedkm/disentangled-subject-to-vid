import os
import random
import json
import torch
import csv
from pathlib import Path

import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.transforms.functional as F
import numpy as np
import glob
from PIL import Image

from torch.utils.data.dataset import Dataset
from packaging import version as pver

from decord import VideoReader
from .utils import preprocess_video_with_resize_and_filtering

import pdb


def check_frame_validity(index_first, video_fps, video_num_frames, sample_fps, sample_num_frames):
    frame_interval = int(round(video_fps / sample_fps))
    index_last = index_first + frame_interval * (sample_num_frames-1)
    if index_last > video_num_frames:
        return False
    else:
        return True

class PexelsDataset(Dataset):
    def __init__(
            self,
            root_path,
            metadata_path,
            sample_fps=16,
            sample_n_frames=49,
            sample_size=[480, 720],
            index_first=30,
            exclude_list=None,
            sample_num=None,
    ):
        self.root_path = root_path
        self.sample_fps = sample_fps
        self.sample_n_frames = sample_n_frames
        self.index_first = index_first

        self.sample_n_frames, self.height, self.width = sample_n_frames, sample_size[0], sample_size[1]
        
        # Read new metadata
        f = open(metadata_path, "r")
        reader = csv.reader(f)
        
        
        metadata = list(reader)[1:]
        f.close()
        
        # Use sample only if satisfying conditions below.
        # if h > w. Only use h < w cases
        # if h > h_target or w > w_target
        metadata = [x for x in metadata if (float(x[4]) < float(x[5])) and (float(x[4]) > float(sample_size[0])) and (float(x[5]) > float(sample_size[1]))]

        # if max_num_frames > video frame count
        metadata = [x for x in metadata if check_frame_validity(index_first, float(x[2]), float(x[3]), sample_fps, sample_n_frames)]

        # if no text prompt, discard.
        metadata = [x for x in metadata if len(x[1]) != 0]


        if exclude_list is not None:
            print(f"Filtering exclusion list from {exclude_list}")
            if os.path.splitext(exclude_list)[-1] == '.txt':
                f = open(exclude_list, 'r')
                lines = f.readlines()
                f.close()

                exclude_scene_list = [x.split('\n')[0] for x in lines]
                metadata = [x for x in metadata if x[0].split('/')[-1] not in exclude_scene_list]
            else:
                raise NotImplementedError("Now, just use txt format for metadata")

        if sample_num is not None:
            metadata = metadata[:sample_num]

        # Gather video paths
        scene_list = [x[0] + '.mp4' for x in metadata]
        scene_path_list = sorted([os.path.join(self.root_path, path) for path in scene_list])
        
        self.scene_path_list = scene_path_list
        self.prompts = [x[1] if (x[1][0] != ' ') else x[1][1:] for x in metadata]
        
        self.length = len(self.scene_path_list)

        sample_size = tuple(sample_size) if not isinstance(sample_size, int) else (sample_size, sample_size)
        self.sample_size = sample_size
        

        
        self.__frame_transforms = transforms.Compose([transforms.Lambda(lambda x: x / 255.0 * 2.0 - 1.0)])
        self.__image_transforms = self.__frame_transforms
        
        # for _ in range(100):
        #     self.__getitem__(random.randint(0, self.length-1))
        

    def __len__(self):
        return self.length

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

    def __getitem__(self, index):
        try:
            video_path = self.scene_path_list[index]
            prompt = self.prompts[index]
            frames = self.preprocess(video_path)
        except:
            video_path = 'null'
            prompt = 'null'
            frames = 'null'
            first_frame = 'null'

        # video_path = self.scene_path_list[index]
        # prompt = self.prompts[index]
        # frames = self.preprocess(video_path)

        if frames != "null":
            # Current shape of frames: [F, C, H, W]
            frames = self.video_transform(frames).contiguous()
            first_frame = frames[0:1].detach().clone() # [C, H, W]



        return {
            'video': frames,
            'prompt': prompt,
            'first_frame': first_frame,
            'video_path': str(video_path),
            }
