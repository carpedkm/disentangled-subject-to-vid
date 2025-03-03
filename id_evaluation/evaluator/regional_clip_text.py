import os
import cv2
import glob
import torch
import numpy as np
from tqdm import tqdm
from omegaconf import OmegaConf
import decord
import random
from PIL import Image
import torch.nn.functional as F
import json
import re

from transformers import CLIPProcessor, CLIPModel

from .distributed import (
    get_world_size,
    get_rank,
    all_gather,
    barrier,
    distribute_list_to_rank,
    distribute_dict_to_rank,
    gather_list_of_dict,
)

decord.bridge.set_bridge('torch')

class RegionalCLIPText:
    def __init__(self, device):
        self.device = device
        self.load_model()
    
    def load_model(self):
        self.model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(self.device)
        self.processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
        return
    
    def get_frames(self, video_path, n_frames='full', mode='default'):
        frame_list = []
        dirname, basename = os.path.split(video_path)
        videoname = basename.rsplit('.', 1)[0]
        video_frames = [f for f in os.listdir(dirname) if re.match(f'{videoname}_\d+\.png', f)]
        len_video = len(video_frames)
        if n_frames == 'full':
            frame_list = list(range(len_video))
        else:
            if mode == 'default': # default extract center frames
                start_idx = max(len_video // 2 - n_frames // 2, 0)
                end_idx = min(start_idx + n_frames, len_video)
                frame_list = list(range(start_idx, end_idx))
            elif mode == 'first':
                end_idx = min(n_frames, len_video)
                frame_list = list(range(end_idx))
            elif mode == 'random': # random uniform sampling
                n_frames = min(n_frames, len_video)
                start_idx = random.randint(0, len_video - n_frames)
                frame_list = list(range(start_idx, start_idx + n_frames))
            else:
                raise NotImplementedError(f"Unsupported mode: {mode}")
        frame_list = [Image.open(os.path.join(dirname, f'{videoname}_{i}.png')).convert('RGB') for i in frame_list]

        return frame_list
    
    def preprocess_frames(self, frames):
        frames = self.processor(images=frames, return_tensors="pt").get('pixel_values')
        frames = frames.to(self.device)
        return frames
    
    def encode_image(self, images):
        image_features = self.model.get_image_features(pixel_values=images)
        return image_features
    
    def encode_text(self, text):
        if isinstance(text, str):
            text = [text]
        inputs = self.processor(text=text, return_tensors="pt", padding=True).get('input_ids')
        with torch.no_grad():
            text_features = self.model.get_text_features(input_ids=inputs.to(self.device))
        return text_features
    
    def compute_similarity(self, video_dict, prompt_json, n_frames='full', mode='default'):
        sim = []
        video_results = []
        for video_path, image_path in tqdm(video_dict.items()):
            frames = self.get_frames(video_path, n_frames=n_frames, mode=mode)
            frames = self.preprocess_frames(frames)
            video_name = os.path.basename(video_path)
            text = prompt_json[video_name]
            with torch.no_grad():
                text_features = self.encode_text(text)
                video_features = self.encode_image(frames)
            if text_features.shape[0] != 1:
                text_features = text_features.unsqueeze(0)
            similarity = F.cosine_similarity(text_features, video_features).detach().to('cpu')
            # video_results.append({'video_path': video_path, 'text': text, 'video_results': similarity.tolist()}) # set this if you want to record score of each frame
            video_results.append({'video_path': video_path, 'text': text, 'video_results': float(similarity.mean())})
            sim.append(similarity.mean())
        avg_score = np.mean(sim)
        return avg_score, video_results


def compute_regional_clip_text(video_list, device, submodules_list, **kwargs):
    if isinstance(video_list, list):
        raise TypeError("video_list should be a dictionary for clip text")
    video_dir = kwargs.get('video_dir', '')
    image_dir = kwargs.get('image_dir', '')

    regional_suffix_video = kwargs.get('regional_suffix_video', 'regional')
    regional_suffix_image = kwargs.get('regional_suffix_image', 'clip_image')
    video_dir = os.path.join(video_dir, regional_suffix_video)
    image_dir = os.path.join(image_dir, regional_suffix_image)

    video_dict = {os.path.join(video_dir, k): os.path.join(image_dir, v) for k,v in video_list.items()}
    video_dict = distribute_dict_to_rank(video_dict)
    n_frames = kwargs.get('n_frames', 'full')
    mode = kwargs.get('mode', 'default')
    prompt_json = kwargs.get('prompt_json', None)
    if prompt_json is None:
        raise ValueError("prompt_json should be provided for clip text")
    with open(prompt_json, 'r') as f:
        prompt_json = json.load(f)
    regional_clip_text = RegionalCLIPText(device)
    all_results, video_results = regional_clip_text.compute_similarity(video_dict, prompt_json, n_frames=n_frames, mode=mode)
    if get_world_size() > 1:
        video_results = gather_list_of_dict(video_results)
        all_results = sum([np.mean(np.array(d['video_results'])) for d in video_results]) / len(video_results)
    all_results = float(all_results.item())
        
    return all_results, video_results

