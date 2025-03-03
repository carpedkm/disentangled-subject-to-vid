import omegaconf.listconfig
import os
import math
import time
import inspect
import argparse
import datetime
import subprocess
import json
import multiprocessing

from pathlib import Path
from omegaconf import OmegaConf
from typing import Dict, Tuple
from datetime import timedelta
import numpy as np
from PIL import Image
from einops import rearrange, repeat
from tqdm import tqdm

import torch
import torchvision
import torch.distributed as dist
import torch.nn.functional as F
import torch.nn as nn
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP

from modules.utils.util import save_videos_grid
from modules.utils.flow_viz import flow_to_image
# from data.shutterstock import ShutterstockDataset
from data.pexels import PexelsDataset
from modules.Segmentation.segmentation_wrapper import Segmentation_wrapper
from modules.utils.filter_flow_cycle_consistency import run_filtering

from torchvision.models.optical_flow import raft_large, Raft_Large_Weights

from diffusers.utils import check_min_version

import pdb

# os.environ["TORCH_DISTRIBUTED_DEBUG"] = "DETAIL"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

def init_dist(launcher="slurm", backend='nccl', port=29500, **kwargs):
    """Initializes distributed environment."""
    if launcher == 'pytorch':
        rank = int(os.environ['RANK'])
        num_gpus = torch.cuda.device_count()
        local_rank = rank % num_gpus
        torch.cuda.set_device(local_rank)
        dist.init_process_group(backend=backend, timeout=timedelta(minutes=30), **kwargs)

    elif launcher == 'slurm':
        proc_id = int(os.environ['SLURM_PROCID'])
        ntasks = int(os.environ['SLURM_NTASKS'])
        node_list = os.environ['SLURM_NODELIST']
        num_gpus = torch.cuda.device_count()
        local_rank = proc_id % num_gpus
        torch.cuda.set_device(local_rank)
        addr = subprocess.getoutput(
            f'scontrol show hostname {node_list} | head -n1')
        os.environ['MASTER_ADDR'] = addr
        os.environ['WORLD_SIZE'] = str(ntasks)
        os.environ['RANK'] = str(proc_id)
        port = os.environ.get('PORT', port)
        os.environ['MASTER_PORT'] = str(port)
        dist.init_process_group(backend=backend, timeout=timedelta(minutes=30))

    else:
        raise NotImplementedError(f'Not implemented launcher type: `{launcher}`!')
    # https://github.com/pytorch/pytorch/issues/98763
    # torch.cuda.set_device(local_rank)

    return local_rank


def flow_vis(flow, clip_flow=None):
    flow_np = flow.permute(0,2,3,1).detach().cpu().numpy()
    flow_ = torch.stack([torch.tensor(flow_to_image(x, clip_flow=clip_flow)) for x in flow_np], dim=0).permute(0,3,1,2)
    flow_ = flow_ / 255 * 2 - 1
    return flow_


def dilate_mask_func(mask, kernel_size=3):
    """
    이진 마스크에 대한 팽창 연산을 수행하는 함수

    Args:
        mask (torch.Tensor): 이진 마스크 (shape: [batch_size, 1, height, width])
        kernel_size (int): 팽창 커널의 크기

    Returns:
        torch.Tensor: 팽창된 마스크
    """
    # 커널 크기를 설정하고 패딩 크기를 계산
    if kernel_size % 2 == 0:
        kernel_size += 1
    padding = kernel_size // 2
    
    # 최대 풀링을 통해 팽창 연산 수행
    dilated_mask = F.max_pool2d(mask, kernel_size=kernel_size, stride=1, padding=padding)
    
    return dilated_mask


def save_dict(condition_image, pixel_values, flow_f, observed_mask, object_mask, object_mask_dilated, output_dir, step):
    # condition_image: b1chw
    # pixel_values: bfchw
    # flow_f: (bf)chw
    # object_mask: (bf)chw
    # object_mask_dilated: (bf)chw
    
    save_path_root = f"{output_dir}/visualization/"
    os.makedirs(save_path_root, exist_ok=True)
    
    # save GT video    
    gt_video = rearrange((pixel_values[0] + 1) / 2, 'f c h w -> c f h w').to('cpu')  # [3, f 2h, w]
    save_videos_grid(gt_video[None, ...], os.path.join(save_path_root, f"{step:03d}-video.mp4"), rescale=False, fps=16)
    
    # save condition image and pixel values
    conditioning_image = condition_image[0][0] / 2. + 0.5  # 3 h w
    pixel_values = (rearrange(pixel_values, "b f c h w -> (b f) c h w") / 2. + 0.5).clip(0., 1.)
    pixel_values_grid = torchvision.utils.make_grid(pixel_values, nrow=14)
    
    torchvision.utils.save_image(conditioning_image, os.path.join(save_path_root, f'{step:03d}-condition_image.png'))
    torchvision.utils.save_image(pixel_values_grid, os.path.join(save_path_root, f'{step:03d}-pixel_values.png'))
    
    # save masks
    observed_mask_grid = torchvision.utils.make_grid(observed_mask.cpu().clip(0.,1.), nrow=14)
    object_mask_grid = torchvision.utils.make_grid(object_mask.cpu().clip(0.,1.), nrow=14)
    object_mask_dilated_grid = torchvision.utils.make_grid(object_mask_dilated.cpu().clip(0.,1.), nrow=14)
    
    torchvision.utils.save_image(observed_mask_grid, os.path.join(save_path_root, f'{step:03d}-observed_mask.png'))
    torchvision.utils.save_image(object_mask_grid, os.path.join(save_path_root, f'{step:03d}-object_mask.png'))
    torchvision.utils.save_image(object_mask_dilated_grid, os.path.join(save_path_root, f'{step:03d}-object_mask_dilated.png'))
    
    # Save flow map
    flow_map = flow_vis(flow_f, clip_flow=max(flow_f.shape[-2:]))
    flow_map = rearrange(flow_map, 'f c h w -> c f h w')
    save_videos_grid(flow_map[None, ...], os.path.join(save_path_root, f"{step:03d}-flow.mp4"), rescale=False, fps=16)
     

class Curator(nn.Module):
    def __init__(self,
                 raft,
                 segmentation_wrapper,
                 chunk=2):
        super().__init__()
        
        self.raft = raft
        self.segmentation_wrapper = segmentation_wrapper
        self.chunk = 2

    def forward(self, condition_image, pixel_values):
        bsz, video_length = pixel_values.shape[:2]
        chunk = self.chunk * bsz
        chunk_size = ((video_length) // self.chunk) + 1
        
        # # RAFT
        # image_ctxt_raft = repeat(condition_image, "b 1 c h w -> b f c h w", f=video_length) # [b, 1, c, h, w] -> [b, f, c, h, w]
        # image_ctxt_raft = rearrange(image_ctxt_raft, "b f c h w -> (b f) c h w")
        # image_trgt_raft = rearrange(pixel_values, "b f c h w -> (b f) c h w")
        # flow_f = self.raft(image_ctxt_raft, image_trgt_raft, num_flow_updates=20)[-1]
        # flow_b = self.raft(image_trgt_raft, image_ctxt_raft, num_flow_updates=20)[-1]
        
        # RAFT
        image_ctxt_raft = repeat(condition_image, "b 1 c h w -> b f c h w", f=video_length) # [b, 1, c, h, w] -> [b, f, c, h, w]
        image_ctxt_raft = rearrange(image_ctxt_raft, "b f c h w -> (b f) c h w")
        image_trgt_raft = rearrange(pixel_values, "b f c h w -> (b f) c h w")
        
        #----------------------------------------------------------------------------------
        # ctxt_raft = torch.cat([image_ctxt_raft, image_trgt_raft], dim=0)
        # trgt_raft = torch.cat([image_trgt_raft, image_ctxt_raft], dim=0)
        
        # flow_all = self.raft(ctxt_raft, trgt_raft, num_flow_updates=20)[-1]
        # flow_f = flow_all[:(bsz*video_length)]
        # flow_b = flow_all[(bsz*video_length):]
        #----------------------------------------------------------------------------------
        flow_f_list, flow_b_list = [], []
        for i in range(chunk):
            start = chunk_size * i
            end = chunk_size * (i+1)

            flow_f = self.raft(image_ctxt_raft[start:end], image_trgt_raft[start:end], num_flow_updates=20)[-1]
            flow_b = self.raft(image_trgt_raft[start:end], image_ctxt_raft[start:end], num_flow_updates=20)[-1]

            flow_f_list.append(flow_f)
            flow_b_list.append(flow_b)
        
        flow_f = torch.cat(flow_f_list)
        flow_b = torch.cat(flow_b_list)
        #----------------------------------------------------------------------------------
        
        # Grounded-SAM2
        condition_image_sam = rearrange(condition_image, 'b 1 c h w -> b c h w')
        
        static_mask, _ = self.segmentation_wrapper(condition_image=condition_image_sam, inference_type='image')
        
        return flow_f, flow_b, static_mask
        

def main(name: str,
         launcher: str,
         port: int,
         data_type: str,

         output_dir: str,
         data_kwargs: Dict,
         segmentation_wrapper_kwargs: Dict,
         
         num_workers: int,
         batch_size: int,
         
         global_seed: int = 42,
         ):
    check_min_version("0.10.0.dev0")
    
    assert batch_size == 1, "Use batch size 1"

    # Initialize distributed training
    local_rank = init_dist(launcher=launcher, port=port)
    global_rank = dist.get_rank()
    num_processes = dist.get_world_size()
    is_main_process = global_rank == 0

    seed = global_seed + global_rank
    torch.manual_seed(seed)

    lock = multiprocessing.Lock()

    # Logging folder
    # folder_name = name + datetime.datetime.now().strftime("-%Y-%m-%dT%H-%M-%S")
    folder_name = name
    output_dir = os.path.join(output_dir, folder_name)

    # Handle the output folder creation
    if is_main_process:
        os.makedirs(output_dir, exist_ok=True)
    global_step = 0

    # Prepare dataset
    if data_type=='shutterstock':
        dataset = ShutterstockDataset(**data_kwargs)
    elif data_type=='pexels':
        dataset = PexelsDataset(**data_kwargs)
    else:
        raise NotImplementedError("Choose dataset type between shutterstock and pexels!")
    root_path = dataset.root_path
    
    distributed_sampler = DistributedSampler(
        dataset,
        num_replicas=num_processes,
        rank=global_rank,
        shuffle=True,
        seed=global_seed,
    )
    
    # DataLoaders creation:
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        sampler=distributed_sampler,
        num_workers=num_workers,
        pin_memory=False, # True
        drop_last=False,
        multiprocessing_context='spawn'
    )


    # Set model
    # RAFT
    raft = raft_large(weights=Raft_Large_Weights.DEFAULT, progress=False).eval()  
    
    # GroundedSAM2
    segmentation_wrapper = Segmentation_wrapper(**segmentation_wrapper_kwargs).eval()  
    
    raft.to(local_rank)
    segmentation_wrapper.to(local_rank)
    
    # Curator
    curator = Curator(raft=raft, segmentation_wrapper=segmentation_wrapper)
    
    # DDP wrapper
    # if is_main_process:
    #     print("Building DDP")
    
    curator.to(local_rank)
    # curator = DDP(curator, device_ids=[local_rank], output_device=local_rank)
    
    dist.barrier()

    
    # Inference
    curator.eval()
    dataloader.sampler.set_epoch(1)

    
    flow_statistics_dict_all = {}
    json_save_path = os.path.join(output_dir, "flow_statistics.jsonl")

    # FILE_EXIST_LIST=[]
    # if Path(json_save_path).is_file():
    #     FILE_EXIST_FLAG=True
    #     metadata_EXIST = []
    #     with open(json_save_path, "r") as f:
    #         for line in f:
    #             metadata_EXIST.append( json.loads(line) )
    #     FILE_EXIST_LIST = [os.path.join(root_path, x['video_path']) for x in metadata_EXIST]

        

    data_iter = iter(dataloader)
    for step in tqdm(range(0, len(dataloader))):

        iter_start_time = time.time()
        batch = next(data_iter)
        data_end_time = time.time()
        

        pixel_values = batch["video"]     # [b, f, c, h, w]
        condition_image = batch['first_frame'] # [b 1 c h w]
        prompt = batch["prompt"] # [b] list
        video_path_list = batch["video_path"]

            

        # if pixel_values[0] == 'null' or video_path_list[0] in FILE_EXIST_LIST:
        #     print(f"pass - {video_path_list[0]}")

        with torch.cuda.amp.autocast(enabled=True, dtype=torch.bfloat16):
            with torch.no_grad():
                pixel_values = pixel_values.to(local_rank)
                condition_image = condition_image.to(local_rank)

                flow_f, flow_b, static_mask = curator(condition_image, pixel_values)
                observed_mask = run_filtering(flow_f, flow_b)
                
                object_mask = 1. - static_mask
                object_mask_dilated = dilate_mask_func(object_mask, kernel_size=5)
                static_mask_dilated = 1. - object_mask_dilated
        
        # Save an example..
        # dist.barrier()
        # if step == 0:
        #     if is_main_process:
        #         save_dict(condition_image, pixel_values, flow_f, observed_mask, object_mask, object_mask_dilated, output_dir, step)    
        #         print(f"save vis to {output_dir}")
        
        
        
        # Compute flow statistics
        video_path_list = [x.split(root_path)[-1] for x in video_path_list]
        flow_map = rearrange(flow_f, '(b f) c h w -> b f c h w', f=pixel_values.shape[1]).abs()
        observed_mask = rearrange(observed_mask, '(b f) c h w -> b f c h w', f=pixel_values.shape[1])
        object_mask = repeat(object_mask, "b c h w -> b f c h w", f=pixel_values.shape[1])
        object_mask_dilated = repeat(object_mask_dilated, "b c h w -> b f c h w", f=pixel_values.shape[1])

        
        # 1) total
        mask = observed_mask
        total_flow_mean, total_flow_max, total_flow_min = get_statistics(flow_map, mask)
        
        # 2) foreground
        mask = observed_mask * object_mask_dilated
        fg_flow_mean, fg_flow_max, fg_flow_min = get_statistics(flow_map, mask)
        
        # 3) background
        mask = observed_mask * (1. - object_mask_dilated)
        bg_flow_mean, bg_flow_max, bg_flow_min = get_statistics(flow_map, mask)
        
        #-----------------------------------------------------------------
        # Save..
        flow_statistics_dict_rank = {}
        for idx in range(len(video_path_list)):
            dict_tmp = {
                'total_mean': total_flow_mean[idx].item(),
                'fg_mean': fg_flow_mean[idx].item(),
                'bg_mean': bg_flow_mean[idx].item(),
                'prompt': prompt[0]
            }
            
            flow_statistics_dict_rank[video_path_list[idx]] = dict_tmp
        
        gathered_results = [None] * num_processes
        
        dist.all_gather_object(gathered_results, flow_statistics_dict_rank)
        
        for results in gathered_results:
            flow_statistics_dict_all.update(results)

        # Save metadata
        # metadata = {
        #     "video_path": video_path_list[0],
        #     "prompt": prompt[0],
        #     "total_mean": total_flow_mean[0].item(),
        #     "fg_mean": fg_flow_mean[0].item(),
        #     "bg_mean": bg_flow_mean[0].item(),
        # }

        # with lock:
        #     with open(json_save_path, "a", encoding="utf-8") as f:
        #         json.dump(metadata, f)
        #         f.write("\n")

        
        dist.barrier()
        global_step += 1



    if is_main_process:
        json_save_path = os.path.join(output_dir, "flow_statistics.json")
        with open(json_save_path, "w") as json_file:
            json.dump(flow_statistics_dict_all, json_file, indent = 4, sort_keys = True)
        print(f"Save json file in {json_save_path}")
    
    dist.destroy_process_group()

def get_statistics(flow_map, mask):
    flow = flow_map * mask
    
    flow_mean = flow.sum(dim=(1,2,3,4))/(mask.sum(dim=(1,2,3,4)) + 1e-7)
    flow_max = flow.amax(dim=(1,2,3,4))
    flow_min = flow.amin(dim=(1,2,3,4))
    
    return flow_mean, flow_max, flow_min
    
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--data_type", type=str, default='pexels')
    parser.add_argument("--launcher", type=str, choices=["pytorch", "slurm"], default="pytorch")
    parser.add_argument("--port", type=int)
    parser.add_argument("--local_rank", type=int, help="Local rank. Necessary for using the torch.distributed.launch utility.")
    args = parser.parse_args()

    name = Path(args.config).stem
    config = OmegaConf.load(args.config)

    main(name=name, launcher=args.launcher, port=args.port, data_type=args.data_type, **config)
