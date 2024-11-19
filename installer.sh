git clone https://github.com/carpedkm/videocustom.git
echo "carpedkm"
echo "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"

cd videocustom
git checkout develop
cd ..
mv annotation videocustom/
cd videocustom
cd diffusers
pip install -e .
cd ..
pip install -U accelerate deepspeed sentencepiece
pip install opencv-python imageio
