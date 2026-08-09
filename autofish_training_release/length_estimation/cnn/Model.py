import torch
from torchvision import models
from torch import optim, cuda
import torch.nn as nn


class DINOv2Encoder(nn.Module):
    """Vision Foundation Model encoder (DINOv2 ViT) used as a drop-in backbone.

    This mirrors the role of the MobileNetV2 backbone in the baseline: it maps
    an RGB image to a global feature vector fed to the regression head.

    Readout options (how the ViT token grid is pooled into that vector):
      * "cls"       — the CLS-token embedding only (D-dim). The original,
                      minimal analogue of MobileNetV2's pooled feature.
      * "clspatch"  — concat[CLS, mean(patch tokens), max(patch tokens)] (3D-dim).
                      The patch tokens form a 16x16 spatial grid that localises
                      the fish within the crop, which carries the geometric /
                      extent information a *length* regressor needs but that a
                      single semantic CLS summary tends to discard. This is the
                      standard strong readout for dense/geometric downstream
                      tasks and is our main lever for a competitive VFM.

    Images arrive as raw [0, 1] tensors from FishLengthDataset; DINOv2 was
    pre-trained with ImageNet normalization, applied here inside the encoder.
    """

    # CLS-token embedding dimension per DINOv2 variant.
    _DIMS = {
        "dinov2_vits14": 384,
        "dinov2_vitb14": 768,
        "dinov2_vitl14": 1024,
        "dinov2_vitg14": 1536,
    }
    _READOUT_MULT = {"cls": 1, "clspatch": 3}
    _IMAGENET_MEAN = [0.485, 0.456, 0.406]
    _IMAGENET_STD = [0.229, 0.224, 0.225]
    # Pin the DINOv2 hub ref to a fixed commit so the architecture (and hence the
    # saved state_dict keys) is identical across the train container, the eval
    # container, and future runs — unlike an unpinned 'main' which can drift.
    _HUB_REF = "facebookresearch/dinov2:7764ea0f912e53c92e82eb78a2a1631e92725fc8"

    def __init__(self, model_name, freeze=True, readout="cls"):
        super().__init__()
        if model_name not in self._DIMS:
            raise ValueError(
                f"Unknown DINOv2 variant '{model_name}'. "
                f"Expected one of {list(self._DIMS)}.")
        if readout not in self._READOUT_MULT:
            raise ValueError(
                f"Unknown readout '{readout}'. "
                f"Expected one of {list(self._READOUT_MULT)}.")
        # Official DINOv2 weights via torch.hub (needs internet on first run,
        # just like torchvision downloading the MobileNetV2 ImageNet weights).
        # trust_repo=True avoids the interactive trust prompt hanging a
        # non-interactive/`docker run` session.
        self.backbone = torch.hub.load(self._HUB_REF, model_name, trust_repo=True)
        self.readout = readout
        self.embed_dim = self._DIMS[model_name]
        self.num_features = self.embed_dim * self._READOUT_MULT[readout]

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
        if self.readout == "cls":
            return self.backbone(x)  # -> [B, D] CLS-token embedding
        # "clspatch": pool the patch-token grid alongside the CLS token.
        feats = self.backbone.forward_features(x)
        cls = feats["x_norm_clstoken"]                 # [B, D]
        patches = feats["x_norm_patchtokens"]          # [B, N, D]
        return torch.cat(
            [cls, patches.mean(dim=1), patches.amax(dim=1)], dim=1)  # [B, 3D]


class DINOv3Encoder(nn.Module):
    """DINOv3 ViT encoder (Meta, 2025) — the same role as DINOv2Encoder.

    DINOv3's headline advance over DINOv2 is the quality of its *dense* (patch)
    features (its "Gram-anchoring" training keeps patch features from degrading),
    which is exactly what the "clspatch" readout here exploits for the geometric
    length-estimation signal — so DINOv3 is a natural upgrade for this task.

    Loaded via HuggingFace transformers (weights are license-gated: the HF
    account must have been granted access and HF_TOKEN must be set). The CLS
    token and the patch-token grid are read out of ``last_hidden_state``, whose
    layout is [CLS, <num_register_tokens registers>, <patch tokens>].
    """

    # short name -> gated HF repo id
    _REPO = {
        "dinov3_vits16": "facebook/dinov3-vits16-pretrain-lvd1689m",
        "dinov3_vits16plus": "facebook/dinov3-vits16plus-pretrain-lvd1689m",
        "dinov3_vitb16": "facebook/dinov3-vitb16-pretrain-lvd1689m",
        "dinov3_vitl16": "facebook/dinov3-vitl16-pretrain-lvd1689m",
        "dinov3_vith16plus": "facebook/dinov3-vith16plus-pretrain-lvd1689m",
        "dinov3_vit7b16": "facebook/dinov3-vit7b16-pretrain-lvd1689m",
    }
    _READOUT_MULT = {"cls": 1, "clspatch": 3}
    _IMAGENET_MEAN = [0.485, 0.456, 0.406]
    _IMAGENET_STD = [0.229, 0.224, 0.225]

    def __init__(self, model_name, freeze=True, readout="cls"):
        super().__init__()
        if model_name not in self._REPO:
            raise ValueError(
                f"Unknown DINOv3 variant '{model_name}'. "
                f"Expected one of {list(self._REPO)}.")
        if readout not in self._READOUT_MULT:
            raise ValueError(f"Unknown readout '{readout}'.")
        from transformers import AutoModel
        # Gated download; needs prior access grant + HF_TOKEN. Pinned by repo id
        # (the lvd1689m pretrain checkpoints are the stable released weights).
        self.backbone = AutoModel.from_pretrained(self._REPO[model_name])
        self.num_register_tokens = getattr(
            self.backbone.config, "num_register_tokens", 0)
        self.embed_dim = self.backbone.config.hidden_size
        self.readout = readout
        self.num_features = self.embed_dim * self._READOUT_MULT[readout]

        self.frozen = freeze
        if freeze:
            for p in self.backbone.parameters():
                p.requires_grad = False

        self.register_buffer(
            "mean", torch.tensor(self._IMAGENET_MEAN).view(1, 3, 1, 1))
        self.register_buffer(
            "std", torch.tensor(self._IMAGENET_STD).view(1, 3, 1, 1))

    def train(self, mode=True):
        super().train(mode)
        if self.frozen:
            self.backbone.eval()
        return self

    def forward(self, x):
        x = (x - self.mean) / self.std
        h = self.backbone(pixel_values=x).last_hidden_state  # [B, 1+R+N, D]
        cls = h[:, 0]
        if self.readout == "cls":
            return cls
        patches = h[:, 1 + self.num_register_tokens:]          # [B, N, D]
        return torch.cat(
            [cls, patches.mean(dim=1), patches.amax(dim=1)], dim=1)


class SupervisedCNNEncoder(nn.Module):
    """A stronger ImageNet-supervised CNN backbone (ResNet, ConvNeXt, EfficientNet).

    The baseline's MobileNetV2 was chosen for on-vessel efficiency, not accuracy.
    For a pure accuracy push we allow larger supervised CNNs, whose ImageNet
    features transfer to the masked crops more readily than DINOv2's semantic
    self-supervised features (which overfit here). The classifier head is stripped
    so forward() returns the pooled feature vector, and — unlike the baseline,
    which feeds raw [0,1] — ImageNet normalization is applied here (these weights
    expect it), matching how DINOv2Encoder handles its input.
    """
    _IMAGENET_MEAN = [0.485, 0.456, 0.406]
    _IMAGENET_STD = [0.229, 0.224, 0.225]

    def __init__(self, model_name, freeze=False):
        super().__init__()
        weights = models.get_model_weights(model_name).DEFAULT
        backbone = getattr(models, model_name)(weights=weights)
        # Strip the classification head, recording the feature width.
        if hasattr(backbone, "fc") and isinstance(backbone.fc, nn.Linear):
            self.num_features = backbone.fc.in_features           # ResNet
            backbone.fc = nn.Identity()
        elif hasattr(backbone, "classifier"):
            clf = backbone.classifier
            if isinstance(clf, nn.Sequential):
                for i in range(len(clf) - 1, -1, -1):             # last Linear
                    if isinstance(clf[i], nn.Linear):
                        self.num_features = clf[i].in_features
                        clf[i] = nn.Identity()
                        break
            elif isinstance(clf, nn.Linear):
                self.num_features = clf.in_features
                backbone.classifier = nn.Identity()
        else:
            raise ValueError(f"Cannot strip head of '{model_name}'.")
        self.backbone = backbone
        if freeze:
            for p in self.backbone.parameters():
                p.requires_grad = False
        self.register_buffer(
            "mean", torch.tensor(self._IMAGENET_MEAN).view(1, 3, 1, 1))
        self.register_buffer(
            "std", torch.tensor(self._IMAGENET_STD).view(1, 3, 1, 1))

    def forward(self, x):
        return self.backbone((x - self.mean) / self.std)


# torchvision CNNs routed to SupervisedCNNEncoder (mobilenet_v2 keeps its own
# exact path below for faithful-baseline reproduction).
_SUPERVISED_CNNS = {
    "resnet50", "resnet34", "resnet101", "convnext_tiny", "convnext_small",
    "efficientnet_b0", "efficientnet_b3", "regnet_y_1_6gf",
}


class Model(nn.Module):
    def __init__(self, bbox_input=False, plane_input=False, freeze_backend=True,
                 model_size=None, model_name="mobilenet_v2"):
        super().__init__()

        # Load feature extractor
        if model_name.startswith("dinov2"):
            # ----- Vision Foundation Model encoder (drop-in replacement) -----
            # Produces a single global feature vector, just like the pooled
            # MobileNetV2 feature below; the MLP head and inputs are identical.
            #
            # The readout is encoded as a "-<readout>" suffix on MODEL_BACKEND
            # (e.g. "dinov2_vits14-clspatch"), so the architecture is fully
            # determined by that one config string — eval_length_estimators.py,
            # which only passes MODEL_BACKEND through, rebuilds the exact same
            # model with no changes on its side. No suffix => "cls" (original).
            base_name, _, readout = model_name.partition("-")
            readout = readout or "cls"
            self.features = DINOv2Encoder(
                base_name, freeze=freeze_backend, readout=readout)
            n_inputs = self.features.num_features
        elif model_name.startswith("dinov3"):
            # ----- DINOv3 VFM encoder (same readout suffix convention) -------
            base_name, _, readout = model_name.partition("-")
            readout = readout or "cls"
            self.features = DINOv3Encoder(
                base_name, freeze=freeze_backend, readout=readout)
            n_inputs = self.features.num_features
        elif model_name in _SUPERVISED_CNNS:
            # ----- Larger ImageNet-supervised CNN (accuracy-first) -----------
            self.features = SupervisedCNNEncoder(
                model_name, freeze=freeze_backend)
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
