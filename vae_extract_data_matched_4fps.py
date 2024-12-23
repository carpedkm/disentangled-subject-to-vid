# %%
# Import necessary packages
import os
import json
import cv2
import torch
import numpy as np
from tqdm import tqdm
from pathlib import Path
from diffusers import AutoencoderKLCogVideoX
import decord
from decord import VideoReader
from multiprocessing import Process, Queue, Value


# %%
def process_video(queue, progress_queue, vae_model_path, max_frames, width, height, gpu_id, output_dir, fps):
    """
    Process videos assigned to a specific GPU.
    """
    device = f"cuda:{gpu_id}"

    # Load the VAE model
    vae = AutoencoderKLCogVideoX.from_pretrained(vae_model_path, subfolder="vae")
    vae.to(device)
    vae.eval()

    while True:
        video_path = queue.get()
        if video_path is None:  # End signal
            break

        try:
            # Load video using Decord
            decord.bridge.set_bridge("native")
            vr = VideoReader(video_path, ctx=decord.cpu(0))

            # Calculate frame interval
            original_fps = 8
            frame_interval = int(original_fps / fps)

            # Extract frames
            # frames = vr.get_batch(range(0, min(len(vr), max_frames * frame_interval), frame_interval)).asnumpy()
            frames = vr.get_batch(range(0, min(len(vr), max_frames * frame_interval), frame_interval)).asnumpy()

            # Ensure exact number of frames
            if frames.shape[0] < max_frames:
                pad_frames = max_frames - frames.shape[0]
                print('>> shorter than max_frames : doing padding')
                frames = np.pad(frames, ((0, pad_frames), (0, 0), (0, 0), (0, 0)), mode="constant")
            elif frames.shape[0] > max_frames:
                frames = frames[:max_frames]
            # Resize frames using OpenCV
            frames = np.array([cv2.resize(frame, (width, height), interpolation=cv2.INTER_LINEAR) for frame in frames])

            # Convert to torch tensor and preprocess
            frames = torch.from_numpy(frames).float() / 255.0 * 2.0 - 1.0  # Normalize [-1, 1]
            frames = frames.permute(0, 3, 1, 2)  # [F, H, W, C] -> [F, C, H, W]
            
            # Add batch dimension and permute for VAE
            frames = frames.unsqueeze(0).permute(0, 2, 1, 3, 4).to(device)  # [B, C, F, H, W]

            # Encode video to latent space
            with torch.no_grad():
                latent_dist = vae.encode(frames).latent_dist
                latents = latent_dist.sample() * vae.config.scaling_factor

            # Save latents
            output_path = os.path.join(output_dir, Path(video_path).stem + "_vae_latents.npy")
            np.save(output_path, latents.cpu().numpy())

        except Exception as e:
            print(f"Error processing video {video_path}: {e}")

        # Clear GPU memory
        torch.cuda.empty_cache()

        # Notify progress
        progress_queue.put(1)


def extract_vae_latents(
    video_paths, vae_model_path, output_dir, height=480, width=720, max_frames=49, fps=8
):
    """
    Extract VAE latents using multiple GPUs with controlled processes.
    """
    os.makedirs(output_dir, exist_ok=True)

    # Detect available GPUs
    available_gpus = list(range(torch.cuda.device_count()))
    if not available_gpus:
        raise RuntimeError("No GPUs are available!")

    print(f"Using GPUs: {available_gpus}")

    # Create a process for each GPU
    queues = [Queue() for _ in available_gpus]
    progress_queue = Queue()
    processes = []

    for gpu_id, queue in zip(available_gpus, queues):
        process = Process(target=process_video, args=(queue, progress_queue, vae_model_path, max_frames, width, height, gpu_id, output_dir, fps))
        process.start()
        processes.append(process)

    # Distribute videos to queues
    for i, video_path in enumerate(video_paths):
        queues[i % len(available_gpus)].put(video_path)

    # Send termination signals
    for queue in queues:
        queue.put(None)

    # Track progress using tqdm
    with tqdm(total=len(video_paths), desc="Extracting VAE latents") as pbar:
        completed = 0
        while completed < len(video_paths):
            progress_queue.get()  # Wait for progress notification
            completed += 1
            pbar.update(1)

    # Wait for all processes to finish
    for process in processes:
        process.join()


if __name__ == '__main__':
    import multiprocessing
    multiprocessing.set_start_method('spawn', True)

    json_dir = "/root/daneul/projects/refactored/CogVideo/annotation/video_dict_foreground_subset4000.json"

    with open(json_dir, "r") as f:
        video_dict = json.load(f)

    video_dict

    video_keys = list(video_dict.keys())

    video_dir = '/mnt/carpedkm_data/pexels_8fps'
    vae_model_path = "THUDM/CogVideoX-5b"
    output_dir = "/mnt/carpedkm_data/pexels_4k_updatd_vae_latents_4fps"
    csv_path = "/mnt/carpedkm_data/pexels_4k_updatd_vae_latents.csv"
    video_paths = [str(Path(video_dir) / f"{video_key}.mp4") for video_key in video_keys]

    extract_vae_latents(
            video_paths,
            vae_model_path,
            output_dir,
            height=480,
            width=720,
            max_frames=25,
            fps=4
        )
