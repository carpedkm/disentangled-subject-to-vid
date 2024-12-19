import os
import cv2
import torch
import numpy as np
from tqdm import tqdm
from pathlib import Path
from diffusers import AutoencoderKLCogVideoX
import decord
from decord import VideoReader

def extract_vae_latents(
    video_paths, vae_model_path, output_dir, height=480, width=720, max_frames=49, device="cuda", fps=8
):
    """
    Extract VAE latents for a list of video paths and save them as .npy files.

    Args:
        video_paths (list): List of video file paths.
        vae_model_path (str): Path to the pretrained 3D VAE model.
        output_dir (str): Directory to save the latent files.
        height (int): Video frame height after resizing.
        width (int): Video frame width after resizing.
        max_frames (int): Maximum number of frames per video.
        device (str): Device to use for processing.

    Returns:
        None
    """
    # Load the VAE model
    print("Loading VAE model...")
    vae = AutoencoderKLCogVideoX.from_pretrained(vae_model_path, subfolder="vae")
    vae.to(device)
    vae.eval()
    print("VAE model loaded.")

    # Create output directory if it does not exist
    os.makedirs(output_dir, exist_ok=True)

    def process_video(video_path, fps):  # default fps is 8
        try:
            # Load video using Decord
            decord.bridge.set_bridge("native")
            vr = VideoReader(video_path, ctx=decord.cpu(0))

            # Calculate the frame interval based on the desired fps
            original_fps = 8
            frame_interval = int(original_fps / fps)

            # Extract frames
            frames = vr.get_batch(range(0, min(len(vr), max_frames * frame_interval), frame_interval)).asnumpy()

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

            return latents.cpu().numpy()
        except Exception as e:
            print(f"Error processing video {video_path}: {e}")
            return None

    # Process all videos
    for video_path in tqdm(video_paths, desc="Extracting VAE latents"):
        latents = process_video(video_path, fps=fps)
        if latents is not None:
            output_path = os.path.join(output_dir, Path(video_path).stem + "_vae_latents.npy")
            np.save(output_path, latents)
        else:
            print(f"Skipping video {video_path}")

if __name__ == "__main__":
    # Example usage
    video_dir = "/mnt/carpedkm_data/pexels_8fps"
    vae_model_path = "THUDM/CogVideoX-5b"
    output_dir = "/mnt/carpedkm_data/pexels_8fps_latents_1600"
    csv_path = '/mnt/video_data/pexels/metadata/results_400k_train_rfcap_pexels_reformat_clean.csv'
    video_paths = [str(p) for p in Path(video_dir).glob("*.mp4")]
    # utilize the given csv file to filter out the videos in portrait aspect ratio
    with open(csv_path, 'r') as f:
        lines = f.readlines()
    # Resolution distribution (height)
    res_height = [line.split(',')[-2] for line in lines[1:]]
    res_height = [int(h) for h in res_height]
    # Resolution distribution (width)
    res_width = [line.split(',')[-1].strip() for line in lines[1:]]
    res_width = [int(w) for w in res_width]
    # Aspect ratio distribution (width/height)
    res_ratio = [w/h for w, h in zip(res_width, res_height)]
    # video id : aspect ratio dict
    path_aspect = {}
    for line in lines[1:]:
        video_id = line.split(',')[0].split('/')[-1]
        aspect_ratio = float(line.split(',')[-1].strip())/float(line.split(',')[-2])
        # print(video_id, aspect_ratio)
        path_aspect[video_id] = aspect_ratio
    # if the aspect ratio is greater than 1, then the video is in landscape mode
    # video_paths = [p for p in video_paths if path_aspect[Path(p).stem] > 1] # FIXME
    # Filter video paths to include only those in landscape mode and present in the path_aspect dictionary
    video_paths = [
        p for p in video_paths
        if Path(p).stem in path_aspect and path_aspect[Path(p).stem] > 1
    ]
    # Get 80 videos
    video_paths = video_paths[:1600]
    # Extract VAE latents
    extract_vae_latents(
        video_paths,
        vae_model_path,
        output_dir,
        height=480,
        width=720,
        max_frames=49,
        device="cuda",
        fps=8
    )
    # # Extract VAE latents in 4 FPS
    # extract_vae_latents(
    #     video_paths,
    #     vae_model_path,
    #     output_dir,
    #     height=480,
    #     width=720,
    #     max_frames=49,
    #     device="cuda",
    #     fps=4
    # )
