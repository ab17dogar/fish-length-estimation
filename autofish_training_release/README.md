# AutoFish #

---

## Dataset
[Download the dataset](https://vap.aau.dk/autofish/)

---

## Installation

### 1) Install Docker
Install [Docker](https://docs.docker.com/engine/install/) and install [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html)

### 2) (optional) Build Docker image
You can build the Docker image locally or just pull a pre-built image from Dockerhub. It is recommended that you do the later and just skip this step.

- Clone the repo and then build the Docker image:
```
cd autofish_training/mask2former/docker
bash build_docker.sh
```

- Run a Docker container from the newly built image:
```
cd autofish_training/mask2former/docker
bash run_docker.sh /your/path/to/autofish_dataset/autofish_groups/ mask2former_container
```

- Compile the CUDA kernel for MSDeformAttn:
```
cd autofish_training/mask2former/docker
bash run_docker.sh /your/path/to/autofish_dataset/autofish_groups/ mask2former_container
```

- Open a new terminal and commit the changes to the Docker image after compiling the CUDA kernel. Start by identifying the ID of the currenly running Docker container:
```
docker ps
```

- Then commit the changes to the Docker image:
```
docker commit 87dbc95e93bb mask2former_container
```


### 3) Run Docker
- You can run a Docker container using the pre-built image from Dockerhub:
```
cd autofish_training/mask2former/docker
bash run_docker.sh /your/path/to/autofish_dataset/autofish_groups/ shbe/mask2former_container
```
__NOTE__ remember to specify the path to the AutoFish dataset!

- Or you can run a Docker container using the locally built image (requires step 2):
```
cd autofish_training/mask2former/docker
bash run_docker.sh /your/path/to/autofish_dataset/autofish_groups/ mask2former_container
```

---

## [Mask2Former (Instance segmentation and classification)](https://github.com/ab17dogar/fish-length-estimation/tree/main/autofish_training_release/mask2former)


---

## [Length estimation](https://github.com/ab17dogar/fish-length-estimation/tree/main/autofish_training_release/length_estimation)


