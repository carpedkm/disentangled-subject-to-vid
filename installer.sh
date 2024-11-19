pip install torch==2.4.1 torchvision==0.19.1 torchaudio==2.4.1 --index-url https://download.pytorch.org/whl/cu121

# cd videocustom
git checkout develop
cd ..
mv annotation videocustom/
cd videocustom
cd diffusers
pip install -e .
cd ..
pip install -U accelerate deepspeed sentencepiece
pip install opencv-python imageio
