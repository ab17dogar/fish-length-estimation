from pycocotools.coco import COCO
import torch
from torchvision import transforms, datasets, models
from torch import optim, cuda
from torch.utils.data import DataLoader, Dataset, sampler
import torch.nn as nn
import os
import pandas as pd
from PIL import Image
from skimage import io
import numpy as np
import matplotlib.pyplot as plt
import time
import json
import copy
import cv2
from stocaching import SharedCache

from crop_augment import crop_augment


class FishLengthDataset(Dataset):
    # Source: https://pytorch.org/tutorials/beginner/data_loading_tutorial.html
    def __init__(self, gt_path, pred_path=None, groups=None,
                 resize=True, mode="train", use_caching=False,
                 crop_to_bbox=False, model_input_bbox=False,
                 normalize_bbox=False, model_input_plane=False,
                 masking_type=None, augment_color=False, augment_crop=False):

        self.resize = resize # enable/disable resizing to 224x224 for the CNN
        self.mode = mode # "train" for training CNN, "eval" outputs extra information
        
        self.dataset_dir = os.path.dirname(gt_path)
        self.plane_equations = self.load_plane_equations()

        # Pre-processing / formatting
        self.crop_to_bbox = crop_to_bbox
        self.normalize_bbox = normalize_bbox
        self.masking_type = masking_type
        
        # Input to the model
        self.model_input_bbox = model_input_bbox        
        self.model_input_plane = model_input_plane

        # Augmentations
        self.augment_color = augment_color
        self.augment_crop = augment_crop

        # Load groundtruth
        self.coco_gt = COCO(gt_path)
        self.annotations = self.coco_gt

        # Identify relevant img ids based on group
        self.img_ids = []
        for img_id in self.coco_gt.getImgIds():
            curr_img = self.coco_gt.loadImgs(img_id)[0]
            if(curr_img["group"] in groups):
                self.img_ids.append(img_id)
                #print(curr_img["file_name"])

        # Load annotations
        self.ann_ids = self.annotations.loadAnns(self.annotations.getAnnIds(imgIds=self.img_ids))
                
        # Load predicted masks (if specified)
        # and find mapping between groundtruth and predictions
        self.coco_pred = None
        self.pred2gt_map = None
        if(pred_path is not None):
            # Force eval mode when using predictions
            # i.e. we should not train when using predictions
            self.mode = "eval"

            # Load mappings from predictions to groundtruth
            # and find the mapping in case it does not already exists
            self.pred2gt_map = self.get_pred_gt_mapping(pred_path)

            # Re-load annotations to only include the ones with a known mapping to gt
            # and annotations included in the specified groups
            self.annotations = self.coco_pred
            ids_with_mapping = self.pred2gt_map.keys()            
            ids_within_groups = self.annotations.getAnnIds(imgIds=self.img_ids)
            ids_overlap = list(set(ids_with_mapping) & set(ids_within_groups))                       
            self.ann_ids = self.annotations.loadAnns(ids_overlap)

            
        print("loaded annotations: ", len(self.ann_ids))

        # initialize the cache        
        self.cache = None
        if(use_caching):
            dataset_len = len(self.ann_ids)   # number of samples in the full dataset
            data_dims = (3, 224, 224)   # data dims (not including batch)

            self.cache = SharedCache(
                size_limit_gib=2,
                dataset_len=dataset_len,
                data_dims=data_dims,
                dtype=torch.float32,
            )
            
    
    def compute_iou(self, mask1, mask2):
        # Intersection
        intersection = np.logical_and(mask1, mask2)
        intersection_sum = np.sum(intersection)
        if(intersection_sum == 0.0):
            return 0.0
        # Union
        union = np.logical_or(mask1, mask2)
        # Compute IoU
        iou = intersection_sum / np.sum(union)
        return iou


    def get_pred_gt_mapping(self, pred_path):
        # Load predictions
        self.coco_pred = self.coco_gt.loadRes(pred_path)

        # Dict with mappings from prediction id to groundtruth
        pred2gt_map = {}
        
        # Load file with mappings if it exists
        path_pred2gt_map = os.path.join(os.path.dirname(pred_path), "map_pred2gt.json")            
        if(os.path.isfile(path_pred2gt_map)):
            #Load file
            with open(path_pred2gt_map) as json_file:
                pred2gt_map = json.load(json_file)

                # convert dict keys from string to int
                pred2gt_map = {int(k): pred2gt_map[k] for k in pred2gt_map}                

                
        # Loop through all predictions and find mapping to groundtruth
        # if it does not already exists
        last_img_id = None
        gt_masks = None        
        predictions = self.coco_pred.loadAnns(self.coco_pred.getAnnIds(imgIds=self.img_ids))
        
        for i,p in enumerate(predictions):
            print(f"checking pred2gt map: {i}/{len(predictions)}")
            
            # Skip very low confidence predictions
            if(p["score"] < 0.1): 
                continue

            # Skip if mapping is already know
            if(p["id"] in pred2gt_map):
                continue
            
            # extract masks for current image
            curr_img_id = p['image_id']
            if(last_img_id != curr_img_id):

                #get gt mask candidates from img_path
                curr_gt_masks = []
                curr_img_gt = self.coco_gt.loadAnns(self.coco_gt.getAnnIds(imgIds=[curr_img_id]))

                # load gt masks - once
                gt_masks = []
                for g in curr_img_gt:
                    gt_masks.append((g,self.coco_gt.annToMask(g)))

                    
            # find best gt mask on downsampled masks
            best_iou = 0.0
            best_gt = None
            pred_mask = self.coco_pred.annToMask(p)
        
            for g in gt_masks:
                curr_iou = self.compute_iou(g[1][::10], pred_mask[::10])
                if(curr_iou > best_iou):
                    best_iou = curr_iou
                    best_gt = g[0]

            if(best_iou == 0):
                print("SKIPPING CURRENT PREDICTION DUE TO IOU=0 WITH GT!")
                continue

            pred2gt_map[p["id"]] = (best_gt, best_iou)
            last_img_id = curr_img_id
            print(f" - added missing mapping for id {p['id']} in image id: {curr_img_id}")

            # Debug only
            #plt.imshow(self.coco_gt.annToMask(best_gt))
            #plt.imshow(pred_mask)
            #plt.title(f"{best_iou} vs {curr_iou} - conf: {p['score']}")
            #plt.savefig(f"debug-id{p['id']}.png")


        # Save updated mappings to json
        with open(path_pred2gt_map, "w") as out:
            json.dump(pred2gt_map, out)
            
        return pred2gt_map
        
    def load_plane_equations(self):
        path = os.path.join(self.dataset_dir, "camera_calibration", "plane_params.json")
        plane_equations = {}
        if(os.path.exists(path)):
            with open(path) as f:
                data = json.load(f)

                for k in data.keys():
                    group_no = k.split("_")[-1]
                    plane_equations[int(group_no)] = np.array([data[k]])[0]
        return plane_equations


    def get_group_from_path(self, path):
        elements = path.split("/")
        for e in elements:
            if("group_" in e):
                group_no = int(e.replace("group_",""))
                return group_no
        return None

    def extract_square_roi(self, bbox, image):
        roi = image[bbox[1]:bbox[1]+bbox[3],bbox[0]:bbox[0]+bbox[2]]

        # pad to size
        width = bbox[2]
        height = bbox[3]

        if height>width:
            w_pad = int((height-width)/2.0)
            padding=((0,0),(w_pad,w_pad),(0,0))
        else:
            h_pad = int((width-height)/2.0)
            padding=((h_pad,h_pad),(0,0),(0,0))
        roi = np.pad(roi,padding)
        return roi        
    
    def __len__(self):
        return len(self.ann_ids)


    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        # current annotation
        pred = self.ann_ids[idx]

        # load and use mapping between predictions
        # and groundtruth, if it exists
        if(self.pred2gt_map is not None):
            best_gt, best_iou = self.pred2gt_map[pred["id"]]
        else:
            best_gt = self.ann_ids[idx]
            best_iou = 1.0

        # fetch current image
        curr_img_id = pred["image_id"]
        curr_img_path = os.path.join(self.dataset_dir, self.coco_gt.imgs[curr_img_id]["file_name"])
        group_no = self.coco_gt.imgs[curr_img_id]["group"]

        # extract gt length
        length = np.array([best_gt["length"]], dtype=np.float32)#.reshape(-1, 2)
        target = torch.from_numpy(length)

        # load mask
        mask = self.annotations.annToMask(pred)

        # extract bbox from mask (just to be safe)       
        non_zero = np.where(mask != 0)
        
        # Use MS COCO for the bbox format
        top_left = np.min(non_zero[1]), np.min(non_zero[0])
        width = np.max(non_zero[1]) - np.min(non_zero[1])
        height = np.max(non_zero[0]) - np.min(non_zero[0])
        bbox = (top_left[0], top_left[1], width, height)

        # # NOT IMPLEMENTED - apply crop augmentation
        # if(self.augment_crop):
        #     image = crop_augment(aaimage, max_augs=2)


        # retrieve data from cache if it's there
        if(self.cache is not None):
            image = self.cache.get_slot(idx)
        else:
            image = None

        # read image if failed to retrieve image from cache
        if image is None:
            image = io.imread(curr_img_path)
        
            # apply masking
            if(self.masking_type == "binary"):
                # use binary mask only
                binary = np.ones_like(image)*255
                binary = binary * mask[:,:,None]
                image = binary
            
            if(self.masking_type == "rgb"):
                # apply binary mask to the image
                rgb = copy.deepcopy(image)
                rgb = rgb * mask[:,:,None]
                image = rgb
                               
            # used cropped version instead of entire image
            if(self.crop_to_bbox):
                cropped = self.extract_square_roi(bbox, image)
                image = cropped

                # Resize image to 224x224 for the network
            if(self.resize):
                trans = transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Resize(size=(224,224), antialias=True),
                ])        
                image = trans(image)        

            # update cache
            if(self.cache is not None):
                self.cache.set_slot(idx, image)
                

        # Apply augmentation
        if(self.augment_color and self.masking_type != "binary"):
            aug_color = transforms.Compose([
                transforms.ToPILImage(),
                transforms.ColorJitter(brightness=0.2,
                                       contrast=0.5,
                                       saturation=0.4,
                                       hue=0.3),
                transforms.ToTensor(),
            ])
            image = aug_color(image)
                
        # prepare data list
        data = []
        data.append(image)

            
        # provide bounding box coords to network
        if(self.model_input_bbox):

            # Normalize bounding box coords
            if(self.normalize_bbox): # TODO: hard-coded image size, find better way to do this!
                img_width = 2464
                img_height = 2056
                bbox = (bbox[0]/img_width,
                        bbox[1]/img_height,
                        bbox[2]/img_width,
                        bbox[3]/img_height)                

            # Add to data list
            bbox = np.array(bbox).astype(np.float32)
            data.append(torch.from_numpy(bbox))

        # Input plane information to network
        if(self.model_input_plane):
            group_no = self.get_group_from_path(curr_img_path)
            plane = self.plane_equations[group_no]
            data.append(torch.from_numpy(plane).float())            

        if(self.mode == "eval"):
            eval_data = {}
            eval_data["pred"] = pred 
            eval_data["gt"] = best_gt
            eval_data["img_path"] = curr_img_path 
            eval_data["group_no"] = group_no
            eval_data["best_iou"] = best_iou
            eval_data["img_id"] = curr_img_id
            return data, eval_data
        return data, target
