import torch
from torchvision import models
from torch import optim, cuda
import torch.nn as nn


class Model(nn.Module):
    def __init__(self, bbox_input=False, plane_input=False, freeze_backend=True,
                 model_size=None, model_name="mobilenet_v2"):
        super().__init__()

        # Load feature extractor
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
