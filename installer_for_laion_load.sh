#!/bin/bash

# Update and upgrade the system
echo ">>>>>> Updating and upgrading system..."
sudo apt update && sudo apt upgrade -y

# Install Python 3 and pip if not already installed
echo ">>>>>> Installing Python 3 and pip..."
sudo apt install -y python3 python3-pip

# Install required Python packages
echo ">>>>>> Installing required Python packages..."
pip3 install pandas pillow tqdm

# Install base64utils (if needed, part of standard library in most cases)
# Ensure multiprocessing and CPU utilities are present
echo ">>>>>> Installing additional Python packages (if necessary)..."
pip3 install base64

# Confirm installation of all packages
echo ">>>>>> Installation complete. Confirming versions..."
python3 --version
pip3 --version
pip3 list | grep -E "pandas|Pillow|tqdm"

# Script usage message
echo ">>>>>> All dependencies are installed. Run your Python script as needed!"

echo ">>>>>> BLOBFUSE 2 installation start"
sudo apt update && sudo apt install -y fuse
curl -sSL https://packages.microsoft.com/keys/microsoft.asc | sudo apt-key add -
sudo apt-add-repository https://packages.microsoft.com/ubuntu/20.04/prod
sudo apt update -y
sudo apt install blobfuse2

echo ">>>>>> BLOBFUSE 2 installation complete"

az login
echo ">>>>>> AZ login complete"

cd /mnt
sudo chattr -i /mnt/DATALOSS_WARNING_README.txt
echo ">>>>>> README in MNT complete"

cd ..
sudo chmod -R 777 /mnt
cd mnt
mkdir carpedkm_data
mkdir video_data

mv /home/v-daneulkim/t2vgusw2_videos_sc.yaml /mnt/
mv /home/v-daneulkim/t2vgusw2_videos_sc_shutterstock.yaml /mnt/
blobfuse2 mount carpedkm_data --config-file=t2vgusw2_videos_sc.yaml --foreground=false
blobfuse2 mount video_data --config-file=t2vgusw2_videos_sc_shutterstock.yaml --foreground=false

echo ">>>>>> BLOBFUSE 2 mount complete"
echo "All dependencies are installed. Run your Python script as needed!"

python3 -m pip install pandas tqdm pillow
sudo ln -s /usr/bin/python3 /usr/bin/python