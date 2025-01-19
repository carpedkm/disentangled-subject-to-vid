import os
import json
import cv2
import torch
import numpy as np
from tqdm import tqdm
from pathlib import Path
from diffusers import AutoencoderKLCogVideoX
# import decord
# from decord import VideoReader
from multiprocessing import Process, Queue, Value

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
            frames = cv2.imread(video_path) 
            frames = cv2.cvtColor(frames, cv2.COLOR_BGR2RGB)
            frames = cv2.resize(frames, (width, height), interpolation=cv2.INTER_LINEAR)
            frames = np.expand_dims(frames, axis=0)  # Add frame dimension# single frame so open with cv2
            # frames = np.array([cv2.resize(frames, (width, height), interpolation=cv2.INTER_LINEAR)])
            frames = np.array(frames)
            # Convert to torch tensor and preprocess
            frames = torch.from_numpy(frames).float() / 255.0 * 2.0 - 1.0  # Normalize [-1, 1]
            frames = frames.permute(0, 3, 1, 2)  # [F, H, W, C] -> [F, C, H, W]
            
            # Add batch dimension and permute for VAE
            frames = frames.unsqueeze(0).permute(0, 2, 1, 3, 4).to(device)  # [B, C, F, H, W]

            # Encode video to latent space
            with torch.no_grad():
                latent_dist = vae.encode(frames).latent_dist
                latents = latent_dist.sample()

            # Save latents
            output_path = os.path.join(output_dir, Path(video_path).stem + "_vae_latents.npy")
            np.save(output_path, latents.cpu().numpy())
            # shape of latent : # (1, 4, 49, 32, 32)
 
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
        
video_dir1 = "/mnt/carpedkm_data/image_gen_ds/omini200k_720p_full/right_images_updated"
# video_dir2 = "output/right_images"
video_paths = sorted([os.path.join(video_dir1, f) for f in os.listdir(video_dir1) if f.endswith(".png")]) # single frame video (image)
total_cnt = len(video_paths)
# half of the videos
video_paths = video_paths[:total_cnt//2]
print(f"Total video paths: {len(video_paths)}")
# video_paths += [os.path.join(video_dir2, f) for f in os.listdir(video_dir2) if f.endswith(".png")] # single frame video (image)

# Extract VAE latents
extract_vae_latents(
    video_paths,
    vae_model_path="THUDM/CogVideoX-5b",
    output_dir="/scratch/amlt_code/videocustom/p3",
    # output_dir = "/dev/shm/vae_latents",
    height=480,
    width=720,
    max_frames=1,
    # fps=8,
)