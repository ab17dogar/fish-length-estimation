import torch
from torchvision import models
from torch import optim, cuda
import torch.nn as nn


class DINOv2Encoder(nn.Module):
    """Vision Foundation Model encoder (DINOv2 ViT) used as a drop-in backbone.

    This mirrors the role of the MobileNetV2 backbone in the baseline: it maps
    an RGB image to a single global feature vector. DINOv2 returns the
    CLS-token embedding, analogous to MobileNetV2's globally-pooled feature.

    The rest of the pipeline (FishLengthDataset, train.py, eval) is unchanged,
    so images still arrive as raw [0, 1] tensors. DINOv2 was pre-trained with
    ImageNet normalization, so that normalization is applied here, inside the
    encoder, keeping every VFM-specific detail local to the encoder swap.
    """

    # CLS-token embedding dimension per DINOv2 variant.
    _DIMS = {
        "dinov2_vits14": 384,
        "dinov2_vitb14": 768,
        "dinov2_vitl14": 1024,
        "dinov2_vitg14": 1536,
    }
    _IMAGENET_MEAN = [0.485, 0.456, 0.406]
    _IMAGENET_STD = [0.229, 0.224, 0.225]
    # Pin the DINOv2 hub ref to a fixed commit so the architecture (and hence the
    # saved state_dict keys) is identical across the train container, the eval
    # container, and future runs — unlike an unpinned 'main' which can drift.
    _HUB_REF = "facebookresearch/dinov2:7764ea0f912e53c92e82eb78a2a1631e92725fc8"

    def __init__(self, model_name, freeze=True):
        super().__init__()
        if model_name not in self._DIMS:
            raise ValueError(
                f"Unknown DINOv2 variant '{model_name}'. "
                f"Expected one of {list(self._DIMS)}.")
        # Official DINOv2 weights via torch.hub (needs internet on first run,
        # just like torchvision downloading the MobileNetV2 ImageNet weights).
        # trust_repo=True avoids the interactive trust prompt hanging a
        # non-interactive/`docker run` session.
        self.backbone = torch.hub.load(self._HUB_REF, model_name, trust_repo=True)
        self.num_features = self._DIMS[model_name]

        self.frozen = freeze
        if freeze:
            for p in self.backbone.parameters():
                p.requires_grad = False

        self.register_buffer(
            "mean", torch.tensor(self._IMAGENET_MEAN).view(1, 3, 1, 1))
        self.register_buffer(
            "std", torch.tensor(self._IMAGENET_STD).view(1, 3, 1, 1))

    def train(self, mode=True):
        # Keep a frozen encoder in eval mode so its features stay deterministic
        # (no stochastic depth / drop-path) even though train.py calls .train().
        super().train(mode)
        if self.frozen:
            self.backbone.eval()
        return self

    def forward(self, x):
        x = (x - self.mean) / self.std
        return self.backbone(x)  # -> [B, num_features] CLS-token embedding


class Model(nn.Module):
    def __init__(self, bbox_input=False, plane_input=False, freeze_backend=True,
                 model_size=None, model_name="mobilenet_v2"):
        super().__init__()

        # Load feature extractor
        if model_name.startswith("dinov2"):
            # ----- Vision Foundation Model encoder (drop-in replacement) -----
            # Produces a single global feature vector, just like the pooled
            # MobileNetV2 feature below; the MLP head and inputs are identical.
            self.features = DINOv2Encoder(model_name, freeze=freeze_backend)
            n_inputs = self.features.num_features
        else:
            model_func = getattr(models, model_name)
            self.features = model_func(weights=models.MobileNet_V2_Weights.IMAGENET1K_V1)

            # Freeze early layers
            if(freeze_backend):
                for param in self.features.parameters():
                    param.requires_grad = False

            # Add on classifier
            n_inputs = self.features.classifier[-1].in_features

            # Strip aways last layers in classifier
            self.features.classifier = self.features.classifier[:-1]


        if(bbox_input):
            n_inputs = n_inputs + 4

        if(plane_input):
            n_inputs = n_inputs + 4

        #self.classifier =  nn.Sequential(
        #    nn.Linear(n_inputs, 256), nn.ReLU(), nn.Dropout(0.2),
        #    nn.Linear(256, 1))

        # Inspired by: https://arxiv.org/pdf/1708.05628
        # Note: missing batchnorm: nn.Linear(n_inputs, 4096), nn.BatchNorm1D(4096), nn.ReLU(),
        # self.classifier =  nn.Sequential(
        #     nn.Linear(n_inputs, 4096), nn.ReLU(),
        #     nn.Linear(4096, 500), nn.ReLU(),
        #     nn.Linear(500, 1))

        # old model (nov 29) - default
        self.classifier =  nn.Sequential(
            nn.Linear(n_inputs, 1000), nn.BatchNorm1d(1000), nn.ReLU(),
            nn.Linear(1000, 500), nn.BatchNorm1d(500), nn.ReLU(),
            nn.Linear(500, 1))

        # new model
        if(model_size == "big"):
            self.classifier =  nn.Sequential(
                nn.Linear(n_inputs, 1000), nn.BatchNorm1d(1000), nn.ReLU(),
                nn.Linear(1000, 1000), nn.BatchNorm1d(1000), nn.ReLU(),
                nn.Linear(1000, 1000), nn.BatchNorm1d(1000), nn.ReLU(),
                nn.Linear(1000, 500), nn.BatchNorm1d(500), nn.ReLU(),
                nn.Linear(500, 500), nn.BatchNorm1d(500), nn.ReLU(),            
                nn.Linear(500, 1))
        

        #self.forward = self.forward_img_only

        # if(bbox_input):
        #     self.forward = self.forward_bbox_input

        # if(plane_input):
        #     self.forward = self.forward_plane_input


    # ordering of input in 'x' is always
    # image
    # bbox (if available)
    # plane (if availble)
    def forward(self, x):
        x1 = self.features(x[0])
        feature_in = torch.cat(([x1]+x[1:]),1)
        y = self.classifier(feature_in)
        return y            

    # def forward_img_only(self, x): #, bbox, plane):
    #     img = x[0]
    #     x1 = self.features(img)
    #     y = self.classifier(x1)
    #     return y

    # def forward_bbox_input(self, x): #img, bbox, plane):
    #     img = x[0]
    #     bbox = x[1]
    #     x1 = self.features(img)
    #     feature_in = torch.cat((x1,bbox),1)
    #     y = self.classifier(feature_in)
    #     return y

    # def forward_plane_input(self, img, bbox, plane):
    #     x1 = self.features(img)
    #     feature_in = torch.cat((x1,bbox,plane),1)
    #     y = self.classifier(feature_in)
    #     return y
