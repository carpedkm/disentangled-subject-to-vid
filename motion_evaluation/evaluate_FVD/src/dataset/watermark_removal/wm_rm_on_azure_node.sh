#!/bin/bash

# install requirements
sudo apt-get update
sudo apt-get install ffmpeg libsm6 libxext6  -y
sudo apt-get install python3 python3-pip -y
pip install decord opencv-python Pillow numpy av matplotlib tqdm pandas 

# mount blob
wget https://packages.microsoft.com/config/ubuntu/20.04/packages-microsoft-prod.deb
sudo dpkg -i packages-microsoft-prod.deb
sudo apt-get update
sudo apt-get install blobfuse

sudo mkdir /mnt/resource/blobfusetmp -p
sudo chown wmrm$1 /mnt/resource/blobfusetmp

chmod 600 ~/t2vg/src/dataset/t2vg_data.cfg

mkdir ~/mount

blobfuse ~/mount --tmp-path=/mnt/resource/blobfusetmp  --config-file=/home/wmrm$1/t2vg/src/dataset/t2vg_data.cfg -o attr_timeout=240 -o entry_timeout=240 -o negative_timeout=120



# run wm rm python cmd
python3 video_watermark_removal.py --video_dir ~/mount/videos --output_dir ~/mount/videos_rmwm --multiprocessing --save_fail --meta_dir ~/mount/webvid10m_meta --split_n 8 --split_i $1
