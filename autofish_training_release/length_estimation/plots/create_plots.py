import numpy as np
import csv
import os
import argparse
import json
import matplotlib.pyplot as plt
from pycocotools.coco import COCO
import seaborn as sns
from tabulate import tabulate
from skimage import io
from PIL import ImageColor
import cv2
import random 

import sys
sys.path.append("../skeletonization")
from LengthEstimator import LengthEstimator

def plot_skeletonization(ann_id, ann_path, legend=True):
    sns.reset_orig()
    # Load mask from annotations
    coco = COCO(ann_path)
    ann = coco.loadAnns([ann_id])[0]
    mask = coco.annToMask(ann)

    # Get skeleton
    skeleton = LengthEstimator.skeletonize(binary_mask=mask, method="zhang")

    # Get convex hull
    convex_hull = LengthEstimator.compute_convex_hull(binary_mask=mask)
    convex_hull = list(convex_hull.reshape(-1,2))
    convex_hull.append(convex_hull[0])
    convex_hull = np.array(convex_hull)



    # Get poly fit
    poly_fit = LengthEstimator.get_poly_fit(
        binary_mask=mask,
        skeleton_method="zhang", #skeletonization method, this one is the fastest
        degree=4, #polynomial degree
        subsample_skeleton=None, #percentage of points to keep of the skeleton
        subsample_fit=None, #percentage of points to keep of the curve
        clip=False #clip to convex hull
    )[0]

    # Sub-sample while keeping the end points
    end_points = poly_fit[[0,-1]]
    poly_fit = poly_fit[0:-1:30]
    poly_fit = list(end_points) + list(poly_fit)
    poly_fit = np.array(poly_fit)
    
    # Plot it!
    linewidth=2.0
    fig = plt.figure()
    ax = fig.add_subplot(111)

    # Load image
    curr_img_id = ann["image_id"]
    curr_img_path = os.path.join("/workspace/autofish_dataset/",
                                 coco.imgs[curr_img_id]["file_name"])

    rgb_img = io.imread(curr_img_path)

    # extract bbox from mask (just to be safe)       
    non_zero = np.where(mask != 0)
        
    # Use MS COCO for the bbox format
    top_left = np.min(non_zero[1]), np.min(non_zero[0])
    width = np.max(non_zero[1]) - np.min(non_zero[1])
    height = np.max(non_zero[0]) - np.min(non_zero[0])
    bbox = (top_left[0], top_left[1], width, height)


    # apply skeleton mask
    kernel = np.ones((3, 3), np.uint8)
    skeleton = cv2.dilate(skeleton, kernel, iterations=3)
    new_color = ImageColor.getcolor("#55a868", "RGB")
    rgb_img[skeleton > 0, :] = new_color# 255.0

    pad = 100
    rgb_img = rgb_img[bbox[1]-pad:bbox[1]+bbox[3]+pad,bbox[0]-pad:bbox[0]+bbox[2]+pad]

    colors = sns.color_palette('deep').as_hex()
    
    #plt.imshow(mask)
    plt.imshow(rgb_img)

    convex_hull = convex_hull - top_left + pad
    poly_fit = poly_fit - top_left + pad
    
    plt.plot(convex_hull[:,0], convex_hull[:,1], linewidth=linewidth, label="convex hull", zorder = 1)
    plt.scatter(poly_fit[:,0], poly_fit[:,1], color = colors[1], marker='x', s=20, label="poly fit", linewidth=2, zorder = 2)


    import matplotlib.lines as mlines


    
    convex_legend = mlines.Line2D([], [], color=colors[0], marker='_', linestyle='None',
                              markersize=10, label='convex hull')
    poly_legend = mlines.Line2D([], [], color=colors[1], marker='x', linestyle='None',
                               markersize=10, label='poly fit')
    skele_legend = mlines.Line2D([], [], color=colors[2], marker='_', linestyle='None',
                                    markersize=10, label='skeleton')

    if(legend):
        plt.legend(handles=[convex_legend, poly_legend, skele_legend])
        
    plt.xticks(ticks=[], labels=[])
    plt.yticks(ticks=[], labels=[])

    plt.tight_layout()
    
    plt.savefig(f"visualize-skeletonization-ann_id{ann_id}.pdf")    
    
    #sns.scatter(x
    


def format_label(label, ignore_sides=False):
    splits = label.split('-')
    curr_label = "{0}-{1}".format(splits[-1], splits[0])

    if(ignore_sides):
        curr_label = curr_label[:-1]

    return curr_label

def img_no_from_path(path):
    elements = path.split("/")
    img_ele = elements[-1]
    img_no = int(img_ele.split(".")[0])
    #print(img_no)
    return img_no

def load_data(csv_path, split="all", conf_min=0.9, iou_min=0.0, iou_max=1.0, ignore_sides=True):
    data = {}
    all_errors = []
    mape_errors = []
    gt_masks = []
    pred_fish_lengths = {}
    gt_fish_lengths = {}
    with open(csv_path, newline='') as csvfile:
        reader = csv.DictReader(csvfile)

        for row in reader:
            if(float(row['pred_conf']) < conf_min):
                continue

            if(float(row['iou']) > iou_max or float(row['iou']) < iou_min):
               continue
            
            gt_label = row['gt_id'] 
            img_no = img_no_from_path(row['img_path'])

            if(split=="low"):
                if(img_no > 40):
                    continue

            if(split=="high"):
                if(img_no <= 40):
                    continue

            fish_type = row['gt_label'] #gt_label.split('-')[0]

            curr_label = gt_label.replace("saithe", "other")
            if(curr_label not in pred_fish_lengths):
                pred_fish_lengths[curr_label] = []
            pred_fish_lengths[curr_label].append(float(row['pred_length_cm']))            

            #if(fish_type != "cod"):
            #    continue

            if(gt_label not in data):
                data[gt_label] = []

            if(curr_label not in gt_fish_lengths):
                gt_fish_lengths[curr_label] = float(row['gt_length_cm'])
            error = float(row['pred_length_cm']) - float(row['gt_length_cm'])
            data[gt_label].append(error)
            all_errors.append(error)
            mape = 100.0/(gt_fish_lengths[curr_label]/np.abs(all_errors[-1]))
            mape_errors.append(mape)

            if('gt_mask' in row):
                gt_masks.append(list(row['gt_mask']))

    return [all_errors, mape_errors], pred_fish_lengths, gt_fish_lengths, gt_masks


def create_density_histograms(data_dict, output_name):
    sns.set_style("darkgrid")
    ##  Create plots!
    fig = plt.figure(figsize=(4,4))
    ax = fig.add_subplot(111)
    
    hist_range = (-5, 5)

    prop_cycle = plt.rcParams['axes.prop_cycle']
    colors = prop_cycle.by_key()['color']
    for i,k in enumerate(data_dict.keys()):

        data1 = data_dict[k][0]
        mean = np.mean(data1)
        std = np.std(data1)
        clipped_data = np.clip(data1, hist_range[0], hist_range[1])
        sns.histplot(data=clipped_data, bins=40, binrange=hist_range,
                     label=f"{k} ({mean:.2f} \u00B1 {std:.2f})", kde=True ) #, palette="deep")

        plt.legend(loc="upper right")


        
    #fig.suptitle(csv_path)

    #plt.xticks(ticks=[], labels=[])
    #plt.yticks(ticks=[], labels=[])
    ax.set(ylabel=None)
    ax.set_yticklabels([])

    ax.set(xlabel="length error (cm)")
    
    #plt.legend(loc="upper right")
    os.makedirs(os.path.dirname(output_name), exist_ok=True)
    plt.tight_layout()
    plt.savefig(output_name) #, dpi=600)


# Re-calculates the length estimation error
# when averaging all the predictions for a single fish/ID
def calc_error_average_prediction(pred_lengths, gt_lengths):
    errors = []
    mapes = []
    for p in pred_lengths:
        err = gt_lengths[p] - np.mean(pred_lengths[p])
        errors.append(err)
        mape = 100.0/(gt_lengths[p] / np.abs(err))
        mapes.append(mape)
    return [errors, mapes]


def create_metric_table(methods, metric, transpose=False, average=False):
    splits = ["low", "high", "all"]
    
    print(f"table for {metric}")
    table = []
    table.append([""] + list(splits))


    for mask_source in ["from-gt", "from-pred"]:

        for p in methods:
            method_name = p
            curr_path = methods[p]

            curr_row = []
            curr_row.append(f"{method_name} ({mask_source})")

            for split in splits:
                data, pred_lengths, gt_lengths, _  = load_data(os.path.join(curr_path, mask_source, "eval-output.csv"), split=split)

                if(average):
                    data = calc_error_average_prediction(pred_lengths, gt_lengths)

                if(metric == "mae"):
                    metric1 = np.mean(np.abs(data[0]))
                    metric2 = np.mean(data[1])
                    curr_row.append(f"{metric1:.2f} ({metric2:.2f}%)")
                else:
                    metric1 = np.mean(data[0])
                    metric2 = np.std(data[0])
                    curr_row.append(f"{metric1:.2f} ({metric2:.2f})")
                    
            table.append(curr_row)

    if(transpose):
        table = list(zip(*table))
    print(tabulate(table, tablefmt="fancy_grid"))
    print(tabulate(table, tablefmt="latex"))


def plot_histograms_methods_compare(methods):   
    for mask_source in ["from-gt", "from-pred"]:
    
        for split in ["all", "low", "high"]:

            data_dict = {}
            for m in methods:
                method_name = m
                curr_path = methods[m]

                data_dict[method_name], _, _, _ = load_data(os.path.join(curr_path, mask_source, "eval-output.csv"), split=split)

            output = os.path.join("results", "compare_methods", mask_source,
                                  f"histogram-methods-compare_{mask_source}_split-{split}.pdf")
            create_density_histograms(data_dict, output_name=output)


    
    
def plot_histograms_gt_vs_pred(methods):
    for p in methods:
        method_name = p
        curr_path = methods[p]
    
        for split in ["all", "low", "high"]:
            data_dict = {}
            data_dict["gt-masks"], _, _, _ = load_data(os.path.join(curr_path, "from-gt", "eval-output.csv"), split=split)
            data_dict["pd-masks"], _, _, _  = load_data(os.path.join(curr_path, "from-pred", "eval-output.csv"), split=split)

            output = os.path.join("results", "compare_gt_vs_pred", method_name,
                                  f"histogram-gt-vs-pred_{method_name}_split-{split}.pdf")
            create_density_histograms(data_dict, output_name=output)


def plot_length_error_versus_confidence(path, mask_source="from-pred", split="all"):
    errors = []
    confidences = []

    csv_path = os.path.join(path, mask_source, "eval-output.csv")
    
    with open(csv_path, newline='') as csvfile:
        reader = csv.DictReader(csvfile)

        for row in reader:
            img_no = img_no_from_path(row['img_path'])
            
            if(split=="low"):
                if(img_no > 40):
                    continue

            if(split=="high"):
                if(img_no <= 40):
                    continue

            errors.append(float(row['pred_length_cm']) - float(row['gt_length_cm']))
            confidences.append(float(row['pred_conf']))

    #errors = np.abs(errors)
    #errors = np.clip(errors, -5.0, 5.0)

    errors = np.array(errors)
    
    err_thresholded = []
    num_after_threshold = []
    thresholds = np.linspace(0.0, 1.0, 99)
    for t in thresholds:
        print(t)

        errors_above_thresh = errors[confidences > t]
        num_after_threshold.append(len(errors_above_thresh))
        err_thresholded.append(np.mean(errors_above_thresh))
    
    #print(len(errors))

    plt.figure()
    sns.lineplot(x=thresholds, y=err_thresholded)
    sns.scatterplot(x=thresholds, y=err_thresholded)
    plt.savefig("length-error_vs_confidence.pdf") #, dpi=600)

    plt.figure()
    sns.lineplot(x=thresholds, y=num_after_threshold)
    sns.scatterplot(x=thresholds, y=num_after_threshold)
    plt.savefig("num-predictions_vs_confidence.pdf") #, dpi=600)


def plot_worst_cases(methods):
    split="high"
    num_cases=8
    
    for m in methods:
        method_name = m
        curr_path = methods[m]


        #errors, pred_fish_lengths, gt_fish_lengths, gt_masks
        
        _, pred_lengths, gt_lengths, _  = load_data(os.path.join(curr_path, "from-pred", "eval-output.csv"), split=split)

        #print(len(pred_lengths))
        #print(len(gt_lengths))

        # Calc average error
        avg_errors = {}
        for ann_id in pred_lengths:
            print(ann_id)
            print(pred_lengths[ann_id])
            print(gt_lengths[ann_id])


            avg_errors[ann_id] = np.mean(np.abs(np.array(pred_lengths[ann_id]) - gt_lengths[ann_id]))
            
        # sort dict by average errprs
        avg_errors = dict(sorted(avg_errors.items(), key=lambda item: item[1]))

        # extract ids for worst performing fish
        worst_fish_ids = list(avg_errors.keys())[-num_cases:]
        #worst_fish_ids = [253, 22, 265, 290, 74, 368, 228]

        # create boxplot with the worst fish ids
        plt.figure()

        worst_fish_pred = [pred_lengths[i] for i in worst_fish_ids]
        print(worst_fish_pred)
        sns.boxplot(data=worst_fish_pred)

        worst_fish_gt = [gt_lengths[i] for i in worst_fish_ids]
        #sns.scatterplot(data=worst_fish_gt)
        plt.scatter(range(len(worst_fish_gt)), worst_fish_gt)
        
        plt.xticks(range(len(worst_fish_ids)), worst_fish_ids)
        #plt.show()

        plt.ylim(0, 70)

        output = os.path.join("results", "worst_cases", f"boxplot_worst-cases_{method_name}_split-{split}.pdf")
        os.makedirs(os.path.dirname(output), exist_ok=True)
        plt.savefig(output)
        #create_density_histograms(data_dict, output_name=output)



def boxplot_errors_old(methods):
    sns.set_style("darkgrid")
    split="all"
    
    for m in methods:
        method_name = m
        curr_path = methods[m]

        
        _, pred_lengths, gt_lengths, _  = load_data(os.path.join(curr_path, "from-pred", "eval-output.csv"), split=split)


        # Calc average error
        avg_errors = {}
        for ann_id in pred_lengths:
            print(ann_id)
            print(pred_lengths[ann_id])
            print(gt_lengths[ann_id])


            avg_errors[ann_id] = np.mean(np.abs(np.array(pred_lengths[ann_id]) - gt_lengths[ann_id]))
            
        # sort dict by average errprs
        avg_errors = dict(sorted(avg_errors.items(), key=lambda item: item[1]))

        # extract ids for worst performing fish
        worst_fish_ids = list(avg_errors.keys())

        
        # sort dict by gt length
        sorted_gt_lengths = gt_lengths #dict(sorted(gt_lengths.items(), key=lambda item: item[1]))
        gt_lengths_ids = list(sorted_gt_lengths.keys())[:18]
        gt_lengths_ids = worst_fish_ids[-18:]


        sorted_pred_lengths = [np.mean(pred_lengths[i]) for i in gt_lengths_ids]
        sorted_pred_lengths = np.array(sorted_pred_lengths)
        sorted_pred_lengths_std = [np.std(pred_lengths[i]) for i in gt_lengths_ids]
        sorted_pred_lengths_std = np.array(sorted_pred_lengths_std)
        sorted_gt_lengths = [gt_lengths[i] for i in gt_lengths_ids]

        print(sorted_pred_lengths)
        print(sorted_pred_lengths_std)
        
        # plot it
        fig, ax = plt.subplots()
        # ax.plot(gt_lengths_ids, sorted_pred_lengths, '-', label="predicted")
        #ax.plot(gt_lengths_ids, sorted_gt_lengths, '-', label="ground truth")
        # ax.fill_between(gt_lengths_ids, sorted_pred_lengths - sorted_pred_lengths_std,
        #                 sorted_pred_lengths + sorted_pred_lengths_std, alpha=0.2)


        for i,k in enumerate(gt_lengths_ids):
            y = pred_lengths[k]
            x = [float(i+1)] * len(y)
            x = np.array(x)
            #x += np.linspace(-0.3, 0.3, len(y))

            label=None
            if(i == 0):
                label="predictions"
                
            #ax.scatter(x, y, facecolors='none', label=label, edgecolors='black', alpha=0.5, s=10)

        data = [pred_lengths[i] for i in gt_lengths_ids]
        ax.boxplot(data, showfliers=False)

        ax.scatter(np.arange(len(sorted_gt_lengths))+1, sorted_gt_lengths,
                   marker='x', color='red', label="ground truth")

        #plt.yticks(np.arange(24, 52, 1.0))
            

        #output = os.path.join("results", "worst_cases", f"boxplot_worst-cases_{method_name}_split-{split}.pdf")
        #os.makedirs(os.path.dirname(output), exist_ok=True)
        plt.legend()
        output = "test.png"        
        plt.savefig(output, dpi=600)
        #create_density_histograms(data_dict, output_name=output)


def plot_mae_vs_samples(methods):
    sns.set_style("darkgrid")
    split="all"



    fig, ax = plt.subplots(figsize=(5,4))
    data = {}
    for m in methods:
        method_name = m
        curr_path = methods[m]

        
        _, pred_lengths, gt_lengths, _  = load_data(os.path.join(curr_path, "from-pred", "eval-output.csv"), split=split)



        
        # different number of samples
        maes = {}
        for n in np.arange(40)+1:
            maes[n] = []
            # repeat sampling
            for s in np.arange(20):
            
                errors = []
                for p in pred_lengths:
                    random.shuffle(pred_lengths[p])
                    subsampled_pred_lengths = pred_lengths[p][:n]
                    err = gt_lengths[p] - np.median(subsampled_pred_lengths)
                    errors.append(err)
                maes[n].append(np.mean(np.abs(errors)))

        # calc means and std
        means = np.array([np.mean(maes[n]) for n in maes])
        stds = np.array([np.std(maes[n]) for n in maes])
        
        ax.fill_between(range(1,41), means - stds, means + stds, alpha=0.2)
            
        plt.plot(range(1,41), means, label=m)

    x_tiks = np.arange(0,41,5)
    x_tiks[0] = 1
    plt.xticks(x_tiks)

    #y_tiks = np.arange(0.3,1.9,0.1)
    #plt.yticks(y_tiks)
    
    plt.xlabel("samples per fish")
    plt.ylabel("MAE (cm)")
    plt.legend()
    ax.set_xlim(1,40)
    plt.tight_layout()
    plt.savefig("mae_vs_samples.pdf") #, dpi=600)

                



def boxplot_errors(methods):
    sns.set_style("darkgrid")
    split="all"


    data = {}
    
    for m in methods:
        method_name = m
        curr_path = methods[m]

        
        _, pred_lengths, gt_lengths, _  = load_data(os.path.join(curr_path, "from-pred", "eval-output.csv"), split=split)


        # Calc average error
        avg_errors = {}
        for ann_id in pred_lengths:


            #avg_errors[ann_id] = np.mean(np.abs(np.array(pred_lengths[ann_id]) - gt_lengths[ann_id]))
            avg_errors[ann_id] = np.std(np.array(pred_lengths[ann_id]))
            
        # sort dict by average errprs
        avg_errors = dict(sorted(avg_errors.items(), key=lambda item: item[1]))

        # extract ids for worst performing fish
        worst_fish_ids = list(avg_errors.keys())

        
        # sort dict by gt length
        sorted_gt_lengths = gt_lengths #dict(sorted(gt_lengths.items(), key=lambda item: item[1]))
        gt_lengths_ids = list(sorted_gt_lengths.keys())[:18]
        #gt_lengths_ids = worst_fish_ids[-4:]
        gt_lengths_ids = ["57", "104", "265", "330", "325", "339"]
        print(gt_lengths_ids)


        sorted_pred_lengths = [np.mean(pred_lengths[i]) for i in gt_lengths_ids]
        sorted_pred_lengths = np.array(sorted_pred_lengths)
        sorted_pred_lengths_std = [np.std(pred_lengths[i]) for i in gt_lengths_ids]
        sorted_pred_lengths_std = np.array(sorted_pred_lengths_std)
        sorted_gt_lengths = [gt_lengths[i] for i in gt_lengths_ids]

        #print(sorted_pred_lengths)
        #print(sorted_pred_lengths_std)
        
        # plot it

        # ax.plot(gt_lengths_ids, sorted_pred_lengths, '-', label="predicted")
        #ax.plot(gt_lengths_ids, sorted_gt_lengths, '-', label="ground truth")
        # ax.fill_between(gt_lengths_ids, sorted_pred_lengths - sorted_pred_lengths_std,
        #                 sorted_pred_lengths + sorted_pred_lengths_std, alpha=0.2)


        for i,k in enumerate(gt_lengths_ids):
            y = pred_lengths[k]
            x = [float(i+1)] * len(y)
            x = np.array(x)
            #x += np.linspace(-0.3, 0.3, len(y))

            label=None
            if(i == 0):
                label="predictions"
                
            #ax.scatter(x, y, facecolors='none', label=label, edgecolors='black', alpha=0.5, s=10)

        data[m] = [pred_lengths[i] for i in gt_lengths_ids]
        # ax.boxplot(data, showfliers=False)

        # ax.scatter(np.arange(len(sorted_gt_lengths))+1, sorted_gt_lengths,
        #            marker='x', color='red', label="ground truth")



    fig, ax = plt.subplots(figsize=(5,4))
    pos = np.arange(0,len(data["SKL"]))
    print(pos)
    print(len(data["SKL"]))

    offset = 0.2
    colors = sns.color_palette('deep').as_hex()
    
    for i,_ in enumerate(gt_lengths_ids):

        for k,m in enumerate(["SKL", "REG"]):
            y = data[m][i]
            x = [float(i)] * len(y)
            x = np.array(x)

            if(m == "SKL"):
                x += offset
            else:
                x -= offset

            label = None
            if(i == 0):
                label = m
            
            #x += np.linspace(-0.3, 0.3, len(y))
            ax.scatter(x, y, facecolors='none', label=label, edgecolors=colors[k], s=15)


        # y = data["cnn"][i]
        # x = [float(i)] * len(y)
        # x = np.array(x)-offset
        # #x += np.linspace(-0.3, 0.3, len(y))
        # ax.scatter(x, y, facecolors='none', label=label, edgecolors=colors[1], alpha=0.5, s=20)

    box_width = 0.25
    bp = ax.boxplot(data["SKL"], showfliers=False, positions=pos+offset, widths=box_width)
    for median in bp['medians']:
        median.set_color('black')
        
    bp = ax.boxplot(data["REG"], showfliers=False, positions=pos-offset, widths=box_width)
    for median in bp['medians']:
        median.set_color('black')

    gt_tiks = np.arange(len(sorted_gt_lengths)) #+0.5
    ax.scatter(gt_tiks, sorted_gt_lengths, zorder=100, s=60,
               marker='X', color=colors[2], label="GT", edgecolors="black")

    #print(gt_tiks)
    plt.xticks(gt_tiks, gt_lengths_ids)
    plt.xlabel("fish ID")
    plt.ylabel("estimated length (cm)")
    plt.legend()
    output = "length_est_boxplot_highest_std.pdf"        
    plt.savefig(output) #, dpi=600)


        
    
if __name__ == '__main__':
    methods = {}
    #methods["skl"] = "../output/skeletonization/"
    methods["SKL"] = "../camera_calibration/output/undistort-True_cb-thickness_0.0/eval/"
    #methods["cnn"] = "../cnn/output/from-askes-pc/test_v4/complete/test_v4_crop_True_mask_rgb_aug-color_norm-bb/eval/model/"
    methods["REG"] = "../cnn/output/from-askes-pc/test_v4/1000epochs_test_v4_crop_True_mask_rgb_aug-color_norm-bb_train-backend/eval/model-epoch200/"


    ann_path = "/workspace/autofish_dataset/annotations.json"

    #plot_worst_cases(methods)
    boxplot_errors(methods)
    plot_mae_vs_samples(methods)
    #a
    

    # Visualize skeletons
    # #plot_skeletonization(8, ann_path)
    # plot_skeletonization(502, ann_path)
    # plot_skeletonization(508, ann_path, legend=False)
    #plot_skeletonization(597, ann_path) # good clean fish
    #plot_skeletonization(538, ann_path, legend=False) # occlusion
    #plot_skeletonization(587, ann_path, legend=False) # missing mask + occlusion
    
    
    # for i in range(100):
    #     try:
    #         plot_skeletonization(i+500, ann_path)
    #     except:
    #         print("")
    

    #plot_length_error_versus_confidence(methods["skele"])
    
    #plot_histograms_gt_vs_pred(methods)
    plot_histograms_methods_compare(methods)
    create_metric_table(methods, metric="mae", transpose=False)
    create_metric_table(methods, metric="mae", transpose=False, average=True)                   
            
    # High / low plots
    #output = os.path.join(os.path.dirname(csv_path), "high_low_splits")
    #create_density_histograms(data_dict, output_name=output)

    # # And with averaging across each fish/ID
    # data_dict = {}
    # data_dict["all"], pred_lengths, gt_lengths, _ = load_data(csv_path, conf_min=0.9, iou_min=0, split="all") 
    # data_dict["all-avg"] = calc_error_average_prediction(pred_lengths, gt_lengths)
    # output = os.path.join(os.path.dirname(csv_path), "with_averaging")
    # create_density_histograms(data_dict, output_name=output)
    
            
