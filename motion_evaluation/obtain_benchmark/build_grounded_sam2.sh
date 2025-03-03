#!/bin/bash

cd modules/Segmentation/GroundedSAM2
export CUDA_HOME=/usr/local/cuda-12.4/
pip install -e .
pip install --no-build-isolation -e grounding_dino
cd ../../../