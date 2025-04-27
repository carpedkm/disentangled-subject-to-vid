pip install torch==2.4.1 torchvision==0.19.1 torchaudio==2.4.1 --index-url https://download.pytorch.org/whl/cu121
cd diffusers
pip install -e .
cd ..
pip install sentencepiece huggingface peft transformers numpy accelerate
pip install opencv-python imageio ffmpeg imageio-ffmpeg