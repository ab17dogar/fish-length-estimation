# Use an official NVIDIA CUDA base image
FROM nvidia/cuda:12.2.0-base-ubuntu22.04

RUN apt-get update && apt-get install -y software-properties-common git nvidia-cuda-toolkit python3-pip nano

WORKDIR /workspace
COPY std_function.h /workspace/std_function.h
RUN mv std_function.h /usr/include/c++/11/bits/

# Install python modules needed for detectron2 and Mask2Former
RUN pip3 install --upgrade pip  && \
    pip install opencv-python==4.8.0.76 && \
    pip install torch==2.0.1+cu117 torchvision==0.15.2+cu117 torchaudio==2.0.2 --index-url https://download.pytorch.org/whl/cu117 && \
    ln -s /usr/bin/python3.10 /usr/bin/python && \
    pip install git+https://github.com/cocodataset/panopticapi.git && \
    pip install git+https://github.com/mcordts/cityscapesScripts.git 

# Clone the repos
RUN git clone https://github.com/facebookresearch/Mask2Former.git && \
    git clone https://github.com/facebookresearch/detectron2.git 
# Install detectron2
RUN python -m pip install -e detectron2
# Install requirements for Mask2Former
RUN pip3 install -r Mask2Former/requirements.txt

# Fix various errors regarding version mismatch
RUN pip3 install seaborn && \
    pip3 uninstall -y matplotlib && \
    pip3 uninstall -y shapely && \
    pip3 install matplotlib==3.7.3 && \
    pip3 install shapely==1.8.0

# Needed for camera calibration scripts    
RUN pip3 install pyransac3d

# Cache for training length estimation network
RUN pip3 install stocaching
    
RUN apt-get autoremove -y && \
    apt-get autoclean -y && \
    rm -rf /var/lib/apt/lists/*
