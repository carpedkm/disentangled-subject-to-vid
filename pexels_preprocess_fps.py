import os
import subprocess
from multiprocessing import Pool, cpu_count
from tqdm import tqdm

# Input and output directories
INPUT_DIR = "/mnt/video_data/pexels/videos-popular"
OUTPUT_DIR = "/mnt/carpedkm_data/pexels_8fps/"
TARGET_FPS = 8

os.makedirs(OUTPUT_DIR, exist_ok=True)

def process_video(file_name):
    input_path = os.path.join(INPUT_DIR, file_name)
    output_path = os.path.join(OUTPUT_DIR, file_name)
    if not os.path.exists(output_path):
        if not file_name.endswith(".mp4"):
            return

        try:
            cmd = [
                "ffmpeg", "-y", "-i", input_path,
                "-vf", f"fps={TARGET_FPS}",
                "-c:v", "libx264", "-crf", "23", "-preset", "fast",
                "-c:a", "copy", output_path
            ]
            # Run FFmpeg command and suppress output to prevent hang
            subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
            # print(f"Success: {file_name}")
        except subprocess.CalledProcessError as e:
            print(f"Error processing {file_name}: {e}")
    else:
        print(f"Skipping {file_name}")

def main():
    # List all video files
    video_files = [f for f in sorted(os.listdir(INPUT_DIR)) if f.endswith(".mp4")]
    print(f"Starting processing with {int(cpu_count() * 0.4)} workers...")

    # Use multiprocessing with tqdm
    with Pool(int(cpu_count() * 0.4)) as pool:  # Limit workers to 8
        list(tqdm(pool.imap(process_video, video_files), total=len(video_files), desc="Processing Videos"))

if __name__ == "__main__":
    main()