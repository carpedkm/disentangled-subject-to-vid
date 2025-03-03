import os
import cv2
import torch
import argparse
import numpy as np
import supervision as sv
from PIL import Image
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor
from transformers import AutoProcessor, AutoModelForCausalLM
from utils.supervision_utils import CUSTOM_COLOR_MAP
import json
import pycocotools.mask as mask_util
from pathlib import Path
from torchvision.ops import box_convert
from grounding_dino.groundingdino.util.inference import load_model, load_image, predict
import pandas as pd
import decord
import tqdm
import time
import grounding_dino.groundingdino.datasets.transforms as T
import traceback
import torch.distributed as dist
import random
from datetime import timedelta

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--json_path', type=str, required=True)
    parser.add_argument('--dataroot', type=str, required=True)
    parser.add_argument('--datasave', type=str, default=None, required=False)
    parser.add_argument('--process_image', action='store_true')
    parser.add_argument('--use_sam', action='store_true')
    parser.add_argument('--save_suffix', type=str, default='regional', required=False)
    args = parser.parse_args()
    return args

def single_mask_to_rle(mask):
    rle = mask_util.encode(np.array(mask[:, :, None], order="F", dtype="uint8"))[0]
    rle["counts"] = rle["counts"].decode("utf-8")
    return rle

def load_image(image_input):
    transform = T.Compose(
        [
            T.RandomResize([800], max_size=1333),
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )
    image_source = image_input
    image = np.asarray(image_source)
    image_transformed, _ = transform(image_source, None)
    return image, image_transformed

def get_world_size():
    return torch.distributed.get_world_size() if torch.distributed.is_initialized() else 1

def get_rank():
    return torch.distributed.get_rank() if torch.distributed.is_initialized() else 0

def distribute_list_to_rank(data_list):
    data_list = data_list[get_rank()::get_world_size()]
    return data_list

def check_and_pad(cropped_image, target_size=(224, 224), padding_color=(127, 127, 127)):
    h, w = cropped_image.shape[:2]
    
    if h < target_size[0] or w < target_size[1]:
        if h < target_size[0]:
            pad_h = target_size[0] - h
            top = pad_h // 2
            bottom = pad_h - top
        else:
            top = bottom = 0
        
        if w < target_size[1]:
            pad_w = target_size[1] - w
            left = pad_w // 2
            right = pad_w - left
        else:
            left = right = 0
        
        cropped_image = cv2.copyMakeBorder(cropped_image, top, bottom, left, right, cv2.BORDER_CONSTANT, value=padding_color)
    
    return cropped_image


if __name__ == '__main__':
    args = parse_args()
    json_path = args.json_path
    dataroot = args.dataroot
    datasave = args.datasave
    if datasave is None:
        datasave = dataroot
    process_image = args.process_image
    use_sam = args.use_sam
    save_suffix = args.save_suffix
    padding_color = (127, 127, 127)

    os.makedirs(datasave, exist_ok=True)
    os.makedirs(os.path.join(datasave, save_suffix), exist_ok=True)
    os.makedirs(os.path.join(datasave, f'{save_suffix}_sam'), exist_ok=True)
    os.makedirs(os.path.join(datasave,'error'), exist_ok=True)


    torch.set_grad_enabled(False)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Setup DDP:
    global_seed=42
    torch.manual_seed(global_seed)

    dist.init_process_group("nccl", timeout=timedelta(seconds=7200000))
    rank = dist.get_rank()
    device = rank % torch.cuda.device_count()
    seed = global_seed * dist.get_world_size() + rank
    torch.manual_seed(seed)
    torch.cuda.set_device(device)
    print(f"Starting rank={rank}, seed={seed}, world_size={dist.get_world_size()}.")

    SAM2_CHECKPOINT = "./checkpoints/sam2.1_hiera_large.pt"
    SAM2_MODEL_CONFIG = "configs/sam2.1/sam2.1_hiera_l.yaml"
    GROUNDING_DINO_CONFIG = "grounding_dino/groundingdino/config/GroundingDINO_SwinT_OGC.py"
    GROUNDING_DINO_CHECKPOINT = "gdino_checkpoints/groundingdino_swint_ogc.pth"
    BOX_THRESHOLD = 0.35
    TEXT_THRESHOLD = 0.25
    
    
    sam2_checkpoint = SAM2_CHECKPOINT
    model_cfg = SAM2_MODEL_CONFIG
    sam2_model = build_sam2(model_cfg, sam2_checkpoint, device=device)
    sam2_predictor = SAM2ImagePredictor(sam2_model)

    # build grounding dino model
    grounding_model = load_model(
        model_config_path=GROUNDING_DINO_CONFIG, 
        model_checkpoint_path=GROUNDING_DINO_CHECKPOINT,
        device=device
    )

    with open(json_path, 'r') as f:
        data = json.load(f)
    
    image_paths = list(data.keys())
    image_paths = distribute_list_to_rank(image_paths)

    error_files = []
    for image_path in image_paths:
        image_fp = os.path.join(dataroot, image_path)
        caption = data[image_path] # class_name

        
        if process_image:
            mask_image_save_path = os.path.join(datasave, save_suffix, image_path)
            masked_image_full_save_path = os.path.join(datasave, f'{save_suffix}_sam', image_path)
            image = Image.open(image_fp).convert("RGB")
            image_source, image = load_image(image)
            boxes, confidences, labels = predict(
                            model=grounding_model,
                            image=image,
                            caption=caption,
                            box_threshold=BOX_THRESHOLD,
                            text_threshold=TEXT_THRESHOLD,
                        )
            h, w, _ = image_source.shape
            boxes = boxes * torch.Tensor([w, h, w, h])
            input_boxes = box_convert(boxes=boxes, in_fmt="cxcywh", out_fmt="xyxy").numpy()
                
            if len(input_boxes) == 0: # use full image if result is not detected
                error_files.append(image_path)
                image = Image.open(image_fp).convert("RGB")
                image.save(mask_image_save_path)
                image.save(masked_image_full_save_path)
                mask = np.zeros((h, w), dtype=np.uint8)
            elif len(input_boxes) > 1:
                #Top 1 result.
                top_idx = np.argmax(confidences)
                input_boxes, confidences = input_boxes[top_idx:top_idx+1], confidences[top_idx:top_idx+1]
            
            if use_sam:
                sam2_predictor.set_image(image_source)
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    masks, scores, logits = sam2_predictor.predict(
                        point_coords=None,
                        point_labels=None,
                        box=input_boxes,
                        multimask_output=False,
                    )
                if masks.ndim == 4:
                    masks = masks.squeeze(1)
                binary_mask = masks[0]
                img_annotate = image_source
                masked_image = np.where(binary_mask[:, :, None], img_annotate, np.array(padding_color, dtype=img_annotate.dtype))            
            #save image
            img_annotate = image_source
            x1, y1, x2, y2 = input_boxes[0]
            x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])
            cropped = img_annotate[y1:y2, x1:x2].copy()
            cropped = check_and_pad(cropped, padding_color=padding_color)
            Image.fromarray(cropped.astype(np.uint8)).save(mask_image_save_path)
            if use_sam:
                img_annotate = masked_image
                cropped = img_annotate[y1:y2, x1:x2].copy()
                cropped = check_and_pad(cropped, padding_color=padding_color)
                Image.fromarray(cropped.astype(np.uint8)).save(masked_image_full_save_path)
        else:
            vr = decord.VideoReader(image_fp)
            for i, frame in enumerate(tqdm.tqdm(vr)):
                videoname = os.path.splitext(image_path)[0].rsplit('.',1)[0]
                mask_image_save_path = os.path.join(datasave, save_suffix, f'{videoname}_{i}.png')
                masked_image_full_save_path = os.path.join(datasave, f'{save_suffix}_sam', f'{videoname}_{i}.png')
                frame = frame.asnumpy()
                frame = Image.fromarray(frame)
                image_source, image = load_image(frame)
                boxes, confidences, labels = predict(
                                model=grounding_model,
                                image=image,
                                caption=caption,
                                box_threshold=BOX_THRESHOLD,
                                text_threshold=TEXT_THRESHOLD,
                            )
                h, w, _ = image_source.shape
                boxes = boxes * torch.Tensor([w, h, w, h])
                input_boxes = box_convert(boxes=boxes, in_fmt="cxcywh", out_fmt="xyxy").numpy()
                    
                if len(input_boxes) == 0: # use full image if result is not detected
                    error_files.append(image_path)
                    image = frame
                    image.save(mask_image_save_path)
                    image.save(masked_image_full_save_path)
                    mask = np.zeros((h, w), dtype=np.uint8)
                elif len(input_boxes) > 1:
                    #Top 1 result.
                    top_idx = np.argmax(confidences)
                    input_boxes, confidences = input_boxes[top_idx:top_idx+1], confidences[top_idx:top_idx+1]
                
                if use_sam:
                    sam2_predictor.set_image(image_source)
                    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                        masks, scores, logits = sam2_predictor.predict(
                            point_coords=None,
                            point_labels=None,
                            box=input_boxes,
                            multimask_output=False,
                        )
                    if masks.ndim == 4:
                        masks = masks.squeeze(1)
                    binary_mask = masks[0]
                    img_annotate = image_source
                    masked_image = np.where(binary_mask[:, :, None], img_annotate, np.array(padding_color, dtype=img_annotate.dtype))            
                #save image
                img_annotate = image_source
                x1, y1, x2, y2 = input_boxes[0]
                x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])
                cropped = img_annotate[y1:y2, x1:x2].copy()
                cropped = check_and_pad(cropped, padding_color=padding_color)
                Image.fromarray(cropped.astype(np.uint8)).save(mask_image_save_path)
                if use_sam:
                    img_annotate = masked_image
                    cropped = img_annotate[y1:y2, x1:x2].copy()
                    cropped = check_and_pad(cropped, padding_color=padding_color)
                    Image.fromarray(cropped.astype(np.uint8)).save(masked_image_full_save_path)
                


        
    with open(os.path.join(datasave, 'error', f'error_{rank}.txt'), 'w') as f:
        for error_file in error_files:
            f.write(error_file + '\n')