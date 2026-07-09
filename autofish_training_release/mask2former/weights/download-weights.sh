#!/bin/bash

# Downloads weights from: https://github.com/facebookresearch/Mask2Former/blob/main/MODEL_ZOO.md

# R50 weights
echo "Downloading R50 weights"
wget https://dl.fbaipublicfiles.com/maskformer/mask2former/coco/instance/maskformer2_R50_bs16_50ep/model_final_3c8ec9.pkl -q --show-progress -O r50.pkl

# SwinB weights
echo "Downloading SwinB weights"
wget https://dl.fbaipublicfiles.com/maskformer/mask2former/coco/instance/maskformer2_swin_base_384_bs16_50ep/model_final_f6e0f6.pkl -q --show-progress -O swinb.pkl
