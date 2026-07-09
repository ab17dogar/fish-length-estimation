#!/bin/bash

# Path the autofish_dataset
DATASET_PATH=$1

# Find path of autofish_training folder relative to location of the run script
SHARED_FOLDER=$(dirname $(readlink -f $0) | rev | cut -d'/' -f2- | rev)

# Spin up container
xhost +local:docker
docker run --gpus all -it --rm \
    -e DISPLAY=$DISPLAY \
    -v $SHARED_FOLDER:/workspace/autofish_training \
    -v $DATASET_PATH:/workspace/autofish_dataset \
    --shm-size 8G \
    mask2former_container
xhost -local:docker
