from pathlib import Path
from typing import List, Tuple

import math
import torch
import torch.nn.functional as F

# Must import after torch because this can sometimes lead to a nasty segmentation fault, or stack smashing error
# Very few bug reports but it happens. Look in decord Github issues for more relevant information.
import decord  # isort:skip

decord.bridge.set_bridge("torch")

from PIL import Image
import imageio
import numpy as np
from einops import rearrange, repeat

import pdb
import av

def preprocess_video_with_resize_and_filtering(
    video_path: Path | str,
    max_num_frames: int,
    height: int,
    width: int,
    target_fps: int = 16,
    index_first: int = 30, # To avoid fade-in effect in the video
) -> torch.Tensor:
    """
    Loads and resizes a single video.

    The function processes the video through these steps:
      1. If video height and width are smaller than 480 and 720, discard the video
      2. If video aspect ratio (H/W) > 1, discard the video.
      3. If video frame count < max_num_frames, discard the video.
      4. If video fps does not match target_fps, match the fps.
      5. If video dimensions don't match (height, width), resize frames

    Args:
        video_path: Path to the video file.
        max_num_frames: Maximum number of frames to keep.
        height: Target height for resizing.
        width: Target width for resizing.
        target_fps: Target fps
        index_first: first index for the whole video frame, default=10

    Returns:
        A torch.Tensor with shape [F, C, H, W] where:
          F = number of frames
          C = number of channels (3 for RGB)
          H = height
          W = width
    """

    # 짧으면 버리기. None return. 코드에서는 frames == None 일 경우, continue.
    # continue 시, dist.barrier 잘 고려해서 사용하기.
    # fps는 decord VideoReader로 맞추기.

    if isinstance(video_path, str):
        video_path = Path(video_path)
    video_reader = decord.VideoReader(uri=video_path.as_posix())

    
    H_orig, W_orig = video_reader[0].shape[:2]

    # If video height and width are smaller than 480 and 720, discard the video
    if H_orig < 480 or W_orig < 720:
        return 'null'

    # If video aspect ratio (H/W) <= 1, discard the video.
    if H_orig/W_orig > 1: 
        return 'null'

    # If video frame count < max_num_frames, discard the video.
    video_num_frames = len(video_reader)
    original_fps = video_reader.get_avg_fps()
    frame_interval = int(round(original_fps / target_fps))
    index_last = index_first + frame_interval * (max_num_frames-1)
    if index_last > video_num_frames:
        return 'null'
    
    # If video fps does not match target_fps, match the fps.
    indices = list(range(index_first, index_last+1, frame_interval))
    frames = video_reader.get_batch(indices).float()
    frames = frames.permute(0, 3, 1, 2).contiguous()
    
    # If video dimensions don't match (height, width), resize frames
    frames_resize = resize_and_center_crop(frames, height, width)

    # import torchvision
    # torchvision.utils.save_image(frames[0]/255, "img_orig.png")
    # torchvision.utils.save_image(frames_resize[0]/255, "img_resize.png")

    return frames_resize



def resize_and_center_crop(
    img: torch.Tensor,
    target_h: int = 480,
    target_w: int = 720
) -> torch.Tensor:
    """
    Args:
        img: (F, C, H, W) 형태의 PyTorch Tensor (예: RGB라면 C=3)
        target_h, target_w: 목표로 하는 해상도

    Returns:
        aspect ratio를 유지하면서 (target_h x target_w)로 center crop된 PyTorch Tensor
    """
    # 1) 현재 이미지의 세로(H), 가로(W) 크기
    _, C, H, W = img.shape

    # 2) 'cover' 방식으로 스케일 결정
    #    -> 최종적으로 (target_h, target_w)를 완전히 덮어야 함
    scale_h = target_h / H
    scale_w = target_w / W
    scale = max(scale_h, scale_w)  # 둘 중 더 큰 값을 택해야, 한 쪽이라도 모자라지 않음

    # 3) 새로 리사이즈될 크기
    new_h = int(round(H * scale))
    new_w = int(round(W * scale))

    # 4) PyTorch의 interpolate 사용
    # mode='bilinear', 'bicubic' 등 선택 가능
    img_resized = F.interpolate(img, size=(new_h, new_w), mode='bilinear', align_corners=False, antialias=True)

    # 5) Center Crop 진행
    #    - 만약 new_h > target_h 이거나 new_w > target_w이면 중앙 부분만 잘라냄
    top = (new_h - target_h) // 2
    left = (new_w - target_w) // 2
    # PyTorch의 텐서 슬라이싱: [채널, 세로범위, 가로범위]
    img_cropped = img_resized[:, :, top:top + target_h, left:left + target_w]

    return img_cropped


def save_video(videos, path, fps):
    # videos: CFHW, scale ~ [-1,1]
    assert videos.ndim == 4, "Video has a shape of CFHW"
    videos = rearrange(videos, "c f h w -> f h w c")
    num_frame, H, W, C = videos.shape

    videos = np.clip((videos.float().cpu().numpy() + 1) / 2 * 255, 0., 255.).astype(np.uint8)
    # pdb.set_trace()
    container = av.open(str(path), mode="w")
    stream = container.add_stream("h264", rate=fps)
    stream.width = W
    stream.height = H
    stream.pix_fmt = "yuv420p"

    for frame in videos:
        av_frame = av.VideoFrame.from_ndarray(frame, format="rgb24")
        packet = stream.encode(av_frame)
        container.mux(packet)

    container.close()


def save_video_fast(videos, path, fps):
    # videos: CFHW, scale ~ [-1,1]
    assert videos.ndim == 4, "Video has a shape of CFHW"
    videos = rearrange(videos, "c f h w -> f h w c")
    num_frame, H, W, C = videos.shape

    videos = np.clip((videos.float().cpu().numpy() + 1) / 2 * 255, 0., 255.).astype(np.uint8)
    # pdb.set_trace()
    imageio.mimwrite(str(path), videos)