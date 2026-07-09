import numpy as np
import cv2
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
import os
import matplotlib.pyplot as plt
import csv
import sys
import time
import os
from skimage import io
import copy
import argparse
import configparser
import glob
import pandas as pd
import torch
import json
import sys
from torchvision import transforms, datasets, models
from torch.utils.data import Dataset

# Imports for CNN based length estimation
sys.path.append("cnn")
from Model import Model
from FishLengthDataset import FishLengthDataset

# Import for skeletonization based length estimation
sys.path.append("skeletonization")
from LengthEstimator import LengthEstimator

# Import for camera utils
sys.path.append("camera_calibration")
from Camera import Camera
from Checkerboard import Checkerboard


class LengthEstimatorSkeletonization():
    def __init__(self, cam_cal_path):
        self.image_scale = 1.0 #downscale to speed-up processing
        self.group_ses = {}
        self.method_skele="zhang"
        self.poly_deg=4 #polynomial degree
        self.sub_skele=1.0 #percentage of points to keep of the skeleton
        self.sub_fit=1.0 #percentage of points to keep of the curve
        self.clip_convex=True #clip to convex hull
        self.calibrated_cam = Camera()
        self.calibrated_cam.load_from_json(os.path.join(cam_cal_path, "intrinsic_params.json"))
        self.plane_equations = {}
        self.plane_path = os.path.join(cam_cal_path, "plane_params.json")
        self.homographies = {}
        self.homographies_path = os.path.join(cam_cal_path, "homographies.json")
        self.undistort_images = True if "undistort-True" in cam_cal_path else False

    def predict_length(self, mask, group_no, img_path, visualize=False):
        # Convert image to fit skeletonization
        mask = mask[0] # convert from list to single image
        mask = mask[:,:,0] # convert from 3 channels binary to single channel
        mask[mask > 1] = 1 # scale from 0 to 1 instead of 0 to 255
        
        skele_start = time.time()

        if(self.undistort_images):
            print("undistorting image")
            mask = cv2.undistort(mask,
                                 self.calibrated_cam.cam_mat,
                                 self.calibrated_cam.dist,
                                 None,
                                 self.calibrated_cam.cam_mat)
            
        
        #run polyfit on predicted mask to get points along the fish
        binary_mask = cv2.resize(
            mask,
            (0, 0),
            fx = self.image_scale,
            fy = self.image_scale
        )

        poly_fit_solution = LengthEstimator.get_poly_fit(
            binary_mask=binary_mask,
            skeleton_method=self.method_skele, #skeletonization method, this one is the fastest
            degree=self.poly_deg, #polynomial degree
            subsample_skeleton=self.sub_skele, #percentage of points to keep of the skeleton
            subsample_fit=self.sub_fit, #percentage of points to keep of the curve
            clip=self.clip_convex #clip to convex hull
        )

        # Estimate length
        skeleton_pts = poly_fit_solution[0].copy()
        skeleton_pts /= self.image_scale

        homo = self.get_homography(group_no)

        # Visualize points
        if(visualize):
            
            img = cv2.imread(img_path)
            if(undistort):
                img = cv2.undistort(img,
                                    self.calibrated_cam.cam_mat,
                                    self.calibrated_cam.dist,
                                    None,
                                    self.calibrated_cam.cam_mat)

            fig = plt.figure()
            plt.imshow(img)
            plt.imshow(mask, alpha=0.5)

        # Get length in cm
        skeleton_pts = skeleton_pts.reshape(-1,1,2)
        obj_pts = cv2.perspectiveTransform(skeleton_pts, homo).reshape(-1,2)
        obj_dist = np.sum(np.linalg.norm((obj_pts[1:] - obj_pts[:-1]), axis=1))
        obj_dist /= 10.0 # convert from mm to cm!

        print("dist homography: ", obj_dist)        
        return obj_dist

    def get_homography(self, group_no):
        if(group_no in self.homographies):
            return self.homographies[int(group_no)]
        else:
            if(self.homographies_path is not None and os.path.exists(self.homographies_path)):
                with open(self.homographies_path) as f:
                    data = json.load(f)

                    # Add all
                    for k in data.keys():
                        curr_group_no = k.split("_")[-1]
                        self.homographies[int(curr_group_no)] = np.array(data[k], dtype=np.float32)
        return self.homographies[int(group_no)]

class LengthEstimatorCNN():
    def __init__(self, model_path):
        self.conf = self.load_config_file(model_path)
        self.model = self.load_model(model_path)
        self.plane_equations = {}
        self.plane_path = "/workspace/autofish_dataset/camera_calibration/plane_params.json"
        self.prev_img_path = None
        self.prev_img = None

    def load_config_file(self, path):
        # Load config file
        conf_path = glob.glob(os.path.join(os.path.dirname(path), "*.cfg"))[0]
        conf = configparser.ConfigParser()
        conf.read(conf_path)
        return conf

    def load_model(self, model_path):
        # Load model
        model = Model(bbox_input=self.conf.getboolean("Model", "MODEL_INPUT_BBOX"),
                      plane_input=self.conf.getboolean("Model", "MODEL_INPUT_PLANE"),
                      model_name=self.conf.get("Model", "MODEL_BACKEND"))

        model.load_state_dict(torch.load(model_path, weights_only=True))
        model.eval()
        model.cuda()
        return model


    def get_plane_eq(self, group_no):
        if(group_no in self.plane_equations):
            return self.plane_equations[int(group_no)]
        else:
            if(self.plane_path is not None and os.path.exists(self.plane_path)):
                with open(self.plane_path) as f:
                    data = json.load(f)

                    # Add all
                    for k in data.keys():
                        curr_group_no = k.split("_")[-1]
                        self.plane_equations[int(curr_group_no)] = np.array(data[k], dtype=np.float32)
        return self.plane_equations[int(group_no)]
            
    def predict_length(self, network_input, group_no, img_path):
        # Move to GPU
        network_input  = [d.cuda().unsqueeze(0) for d in network_input]
        
        # Run it through the network
        output = self.model(network_input)

        pred_length_cm = output[0].cpu().detach().numpy()[0]
        return pred_length_cm
    


def predict_all_lengths(length_estimator, dataset,
                    csv_out="output/length-estimation.csv", conf_thres=0.01):

    # Check if output file already exists
    if(os.path.isfile(csv_out)):
        # Check number of lines in the file
        with open(csv_out, "r") as f:
            num_lines = len(f.readlines())
            print(f"file: {csv_out} already exists with {num_lines} lines")

            if(num_lines == (len(dataset)+1)):
                print(f" - skip predicting, number of lines correspond with dataset length")
                return
            else:
                print(f" - overwriting old prediction")        
    
    
    # Setup csv file for the results
    os.makedirs(os.path.dirname(csv_out), exist_ok=True)        
    csv_f = open(csv_out, 'w')
    csv_writer = csv.writer(csv_f)
    csv_writer.writerow(["img_path",
                         "gt_label",
                         "gt_id",
                         "gt_length_cm",
                         "gt_bbox",
                         "iou",
                         "pred_length_cm",
                         "pred_label",
                         "pred_img_id",
                         "pred_id",
                         "pred_conf"])

    last_img_id = None
    gt_masks = None

    # loop through predictions
    #for k,p in enumerate(predictions):
    for k,d in enumerate(dataset):
        loop_start = time.time()
        print("processing: {0}/{1}".format(k+1, len(dataset)))

        pred = d[1]["pred"]
        gt = d[1]["gt"]
        
        if(pred_path is None):
            pred_conf = 1.0
        else:
            pred_conf = pred['score']
        if(pred_conf < conf_thres):
            continue

        # extract general info
        curr_img_path = d[1]["img_path"]
        group_no = d[1]["group_no"]
        curr_img_id = d[1]["img_id"]
        best_iou = d[1]["best_iou"]

        # extract groundtruth info
        gt_label = dataset.annotations.cats[gt['category_id']]['name'] 
        gt_id = gt['fish_id']
        gt_bbox = gt['bbox']

        # extract predicted info
        pred_label = dataset.annotations.cats[pred['category_id']]['name'] 
        pred_id = pred['id']

        # estimate the length
        data = d[0] # data can either be an image (binary mask for skelenization) but also contain extra information (bbox for CNN length estimation)
        pred_length_cm = length_estimator.predict_length(data, group_no, curr_img_path)

        gt_length_cm = gt['length']
        #print(f" - predicted: {round(float(pred_length_cm),2)} vs gt: {gt_length_cm}")

        # save the results
        csv_writer.writerow([curr_img_path,
                             gt_label,
                             gt_id,
                             gt_length_cm,
                             gt_bbox,
                             best_iou,
                             pred_length_cm,
                             pred_label,
                             curr_img_id,
                             pred_id,
                             pred_conf])

        csv_f.flush()

        loop_end = time.time()
        #print("loop time: ", (loop_end - loop_start))
    csv_f.close()
     

#implement multiprocessing?
#https://stackoverflow.com/questions/26596714/python-writing-to-a-single-file-with-queue-while-using-multiprocessing-pool
    
if __name__=="__main__":
    # example
    # python eval_length_estimators.py --gt_path /workspace/autofish_dataset/annotations.json --pred_path coco_results-vikki/coco_instances_results.json --cnn_model_path cnn/output/test_aug_vanilla/model.pt

    # Parse args
    parser = argparse.ArgumentParser()
    parser.add_argument("--cnn_model_path",
                        default=None,
                        help="Optional - CNN-based approach if specified")
    parser.add_argument("--cam_cal_path",
                        default=None,
                        help="Optional - skeletonization-based approach if specified")
    parser.add_argument("--gt_path")
    parser.add_argument("--pred_path",
                        default=None,
                        help="Optional - if not specified the masks from the gt are used")
    parser.add_argument("--groups",
                        nargs='+',
                        type=int,
                        default=None,
                        help="Optional - only eval specific groups")

    args = parser.parse_args()


    # predictions - from the detector e.g. "coco_instances_results.json"
    pred_path = args.pred_path

    # groundtruth - i.e. the annotations.json in the dataset
    gt_path = args.gt_path

    # test groups
    if(args.groups is None):
        test_groups = [10, 14, 20, 21, 22]
    else:
        test_groups = args.groups
    
    if(args.cam_cal_path is not None): # Skeletonization-based
        length_est = LengthEstimatorSkeletonization(args.cam_cal_path)
        csv_out = os.path.join(args.cam_cal_path, "eval")

        dataset = FishLengthDataset(gt_path=gt_path,
                                    pred_path=pred_path,
                                    groups=test_groups,
                                    mode="eval",
                                    resize=False,
                                    masking_type="binary")
        
    if(args.cnn_model_path is not None): #CNN-based
        length_est = LengthEstimatorCNN(args.cnn_model_path)
        model_dir = os.path.dirname(args.cnn_model_path)
        model_name =  args.cnn_model_path.split("/")[-1].replace(".pt", "")
        csv_out = os.path.join(model_dir, "eval", model_name)
        #csv_out = f"output/cnn-{model_name}"

        config = length_est.conf
        dataset = FishLengthDataset(gt_path=gt_path,
                                    pred_path=pred_path,
                                    groups=test_groups,
                                    mode="eval",
                                    crop_to_bbox = config.getboolean("Preprocessing", "CROP_TO_BBOX", fallback=False),
                                    masking_type = config.get("Preprocessing", "MASKING_TYPE", fallback=None),
                                    model_input_bbox = config.getboolean("Model", "MODEL_INPUT_BBOX", fallback=False),
                                    model_input_plane = config.getboolean("Model", "MODEL_INPUT_PLANE", fallback=False),
                                    normalize_bbox = config.getboolean("Model", "NORMALIZE_BBOX", fallback=False))

    # Set output csv
    if(args.pred_path is None):
        csv_out = os.path.join(csv_out, "from-gt", "eval-output.csv")
    else:
        csv_out = os.path.join(csv_out, "from-pred", "eval-output.csv")
        groups = None

    # Predict lengths
    predict_all_lengths(length_est, dataset, csv_out=csv_out, conf_thres=0.01)
