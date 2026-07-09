import numpy as np
import csv
import os
import argparse
import json
import matplotlib.pyplot as plt
from pycocotools.coco import COCO

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

def load_data(csv_path, split="all", conf_min=0.8, iou_min=0.0, iou_max=1.0, ignore_sides=True):
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

            
            gt_label = row['gt_id'] #format_label(, ignore_sides=ignore_sides)
            #print(gt_label)

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
    # Save stats to file
    output_name_stats = os.path.join(output_name, "stats.txt")
    os.makedirs(os.path.dirname(output_name_stats), exist_ok=True)        
    f = open(output_name_stats, "w+")
    for k in data_dict.keys():
        data = data_dict[k]
        f.write("{0}\n".format(k))
        f.write(" num: {0}\n".format(len(data[0])))
        f.write(" mean: {0}\n".format(np.mean(data[0])))
        f.write(" mean absolute error: {0}\n".format(np.mean(np.abs(data[0]))))
        f.write(" mean absolute percentage: {0}\n".format(np.mean(data[1])))
        f.write(" median: {0}\n".format(np.median(data[0])))
        f.write(" std: {0}\n".format(np.std(data[0])))

    # Read back content of file before closing it
    print("Stats: ")
    f.seek(0)
    [print(l, end=" ") for l in f.readlines()]
    f.close()

    
    # Also try to save to json file
    json_dict = {}
    for k in data_dict.keys():        
        data = data_dict[k]
        json_dict[k] = {}
        json_dict[k]["num"] = len(data[0])
        json_dict[k]["mean"] = np.mean(data[0])
        json_dict[k]["mean_abs_err"] = np.mean(np.abs(data[0]))
        json_dict[k]["mean_abs_per_err"] = np.mean(data[1])
        json_dict[k]["median"] = np.median(data[0])
        json_dict[k]["std"] = np.std(data[0])
        
    with open(output_name_stats.replace(".txt", ".json"), "w") as f:               
        json.dump(json_dict, f)      
        

    ##  Create plots!
    # Plot distributions
    fig, ax = plt.subplots(nrows=1, ncols=len(data_dict.keys())+1, figsize=(12,6))
    
    #hist_range = (mean-2*std, mean+2*std)
    #hist_range = (-10, 10)
    hist_range = (-5, 5)


    prop_cycle = plt.rcParams['axes.prop_cycle']
    colors = prop_cycle.by_key()['color']
    for i,k in enumerate(data_dict.keys()):
        data1 = data_dict[k][0]
        ax[i].hist(np.clip(data1, hist_range[0], hist_range[1]), bins=40, range=hist_range, alpha=0.5, color=colors[i], density=True) #, label="pre")
        #ax[i].hist(data1, bins=40, range=hist_range, alpha=0.5, color=colors[i], density=True) #, label="pre")
        ax[i].grid(which = "major", linewidth = 1)
        ax[i].grid(which = "minor", linewidth = 0.2)
        ax[i].set_xlim([hist_range[0]*1.2, hist_range[1]*1.2])
        #ax[i].set_ylim([0, 500])
        ax[i].set_ylim([0, 1])
        ax[i].set_xlabel("error (cm)")
        ax[i].set_ylabel("instances")


        std = np.std(data1)
        mean = np.mean(data1)
        #median = np.median(data1)
        mae = np.mean(np.abs(data1))
        mape = np.mean(data_dict[k][1])

        ax[i].set_title(f"density: {k}\nmae: {round(mae,3)}\nmape: {round(mape,3)}\nmean: {round(mean,3)} std: {round(std,3)}")
        ax[i].minorticks_on()


        ax[-1].hist(data1, bins=40, range=hist_range, alpha=0.5, label=k, density=True) #, label="pre")
        ax[-1].grid(which = "major", linewidth = 1)
        ax[-1].grid(which = "minor", linewidth = 0.2)
        ax[i].set_xlim([hist_range[0]*1.2, hist_range[1]*1.2])
        #ax[-1].set_ylim([0, 500])
        ax[-1].set_ylim([0, 1])
        ax[-1].set_xlabel("error (cm)")
        ax[-1].set_ylabel("instances")
        ax[-1].set_title("combined")
        ax[-1].minorticks_on()
        ax[-1].legend(loc="upper right")
    fig.suptitle(csv_path)
    plt.tight_layout()
    #plt.legend(loc="upper right")
    plt.savefig(os.path.join(output_name, "hist.png"), dpi=600)


def create_error_heatmap(errors, gt_masks, output_name):
    print("errors: ", len(errors))
    print("masks: ", len(gt_masks))

    xs = []
    ys = []
    weights = []

    coco = COCO()
    
    for i in range(len(errors)):
        curr_mask = gt_masks[i]
        curr_error = errors[i]

        mask = coco.annToMask(curr_mask)
        print(mask)

        

        
        #weights = np.append(weights, [curr_error]*



    heatmap, xedges, yedges = np.histogram2d(x, y, bins=(1024,1024))
    extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]]

    plt.clf()
    plt.imshow(heatmap.T, extent=extent, origin='lower')
    plt.show()
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_name, "heatmap.png"), dpi=600)


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
    

if __name__ == '__main__':
    # Parse args
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv")
    parser.add_argument("--csv_compare",
                        default=None,
                        help="Optional csv to compare against")
    args = parser.parse_args()
    csv_path = args.csv
    
    # # Create heat map
    # data_dict = {}
    # errors, _, _, gt_masks = load_data(csv_path, conf_min=0.9, iou_min=0, split="all")    
    # output = os.path.join(os.path.dirname(csv_path), "heatmap")
    # create_error_heatmap(errors, gt_masks, output_name=output)
    
    # High, low, all histograms
    data_dict = {}
    data_dict["all"], _, _, _ = load_data(csv_path, conf_min=0.9, iou_min=0, split="all")
    data_dict["low"], _, _, _  = load_data(csv_path, conf_min=0.9, iou_min=0, split="low")
    data_dict["high"], _, _, _ = load_data(csv_path, conf_min=0.9, iou_min=0, split="high")
    output = os.path.join(os.path.dirname(csv_path), "high_low_splits")
    create_density_histograms(data_dict, output_name=output)

    # And with averaging across each fish/ID
    data_dict = {}
    data_dict["all"], pred_lengths, gt_lengths, _ = load_data(csv_path, conf_min=0.9, iou_min=0, split="all") 
    data_dict["all-avg"] = calc_error_average_prediction(pred_lengths, gt_lengths)
    output = os.path.join(os.path.dirname(csv_path), "with_averaging")
    create_density_histograms(data_dict, output_name=output)
    

    # Create comparisons if second CSV specified
    if(args.csv_compare is not None):
        csv_compare_path = args.csv_compare

        csv1_name = f"{csv_path.split('/')[-2]}-{csv_path.split('/')[-1]}".replace(".csv", "")
        csv2_name = f"{csv_compare_path.split('/')[-2]}-{csv_compare_path.split('/')[-1]}".replace(".csv", "")
        output = os.path.join(os.path.dirname(os.path.dirname(csv_path)),
                              f"compare_{csv1_name}_vs_{csv2_name}")

        data_dict = {}
        data_dict[csv1_name], _, _, _ = load_data(csv_path, conf_min=0.9, iou_min=0, split="all")
        data_dict[csv2_name], _, _, _  = load_data(csv_compare_path, conf_min=0.9, iou_min=0, split="all")
        create_density_histograms(data_dict, output_name=output)
        
    
    


# ax.boxplot(data_dict.values(), showfliers=False)
# ax.set_xticklabels(data_dict.keys())
# ax.grid(which = "major", linewidth = 1)
# ax.grid(which = "minor", linewidth = 0.2)
# ax.minorticks_on()
# ax.set_ylabel("error (cm)")
# #ax.set_ylim([-5, 5])
# ax.set_title("predictions - different densities")
# plt.xticks(rotation=90)
# plt.tight_layout()
# #plt.show()
# plt.savefig(output_name, dpi=1200)

# # Plot for havforsker meeting
# fig, ax = plt.subplots(figsize=(3,3.25))
# hist_range = (-5.5,2.5)
# data1 = load_data("output/newest_predictions_wavg.csv", conf_min=0.9, iou_min=0, split="all")
# #data1 = np.clip(data1, hist_range[0], hist_range[1])
# ax.hist(data1, bins=40, range=hist_range, alpha=0.5, label="all") #, density=True) #, label="pre")
# ax.grid(which = "major", linewidth = 1)
# ax.grid(which = "minor", linewidth = 0.2)
# ax.set_xlim([-7, 3])
# ax.set_ylim([0, 400])
# ax.set_xlabel("error (cm)")
# ax.set_ylabel("instances")
# ax.set_title("all densities")
# ax.minorticks_on()
# #plt.legend(loc="upper right")
# #plt.show()
# plt.tight_layout()
# plt.savefig('predictions_distribution_man_vs_pre_all.png', dpi=600)

# # # Plot for havforsker meeting
# #fig, ax = plt.subplots(figsize=(3,3.25))
# # #hist_range = (-5.5,2.5)
# # data1 = load_data("output/newest_predictions.csv", conf_min=0.9, iou_min=0.5, split="all")
# # #data1 = np.clip(data1, hist_range[0], hist_range[1])
# # ax.hist(data1, bins=40, range=hist_range, alpha=0.5, label="all")
# # #ax.grid(which = "major", linewidth = 1)
# # #ax.grid(which = "minor", linewidth = 0.2)
# # #ax.set_xlabel("error (cm)")
# # #ax.set_ylabel("probability")
# # #ax.set_title("length errors - manual vs predicted masks")
# # #ax.minorticks_on()
# # #plt.show()
# # #plt.tight_layout()
# # #plt.savefig('predictions_distribution_man_vs_pre_low.png', dpi=600)

# # Plot for havforsker meeting
# fig, ax = plt.subplots(figsize=(3,3.25))
# #hist_range = (-5.5,2.5)
# data1 = load_data("output/newest_predictions_shifted_1_5mm_wavg.csv", conf_min=0.9, iou_min=0, split="high")
# #data1 = np.clip(data1, hist_range[0], hist_range[1])
# ax.hist([0])
# ax.hist(data1, bins=40, range=hist_range, alpha=0.5, label="high")
# ax.grid(which = "major", linewidth = 1)
# ax.grid(which = "minor", linewidth = 0.2)
# ax.set_xlim([-7, 3])
# ax.set_ylim([0, 400])
# ax.set_xlabel("error (cm)")
# ax.set_ylabel("instances")
# ax.set_title("high density")
# ax.minorticks_on()
# #plt.show()
# #plt.legend(loc="upper right")
# plt.tight_layout()
# plt.savefig('predictions_distribution_man_vs_pre_high.png', dpi=600)

# # Plot for havforsker meeting
# fig, ax = plt.subplots(figsize=(3,3.25))
# #hist_range = (-5.5,2.5)
# data1 = load_data("output/newest_predictions_shifted_1_5mm_wavg.csv", conf_min=0.9, iou_min=0, split="low")
# #data1 = np.clip(data1, hist_range[0], hist_range[1])
# ax.hist([0])
# ax.hist([0])
# ax.hist(data1, bins=40, range=hist_range, alpha=0.5, label="low")
# ax.grid(which = "major", linewidth = 1)
# ax.grid(which = "minor", linewidth = 0.2)
# ax.set_xlim([-7, 3])
# ax.set_ylim([0, 400])
# ax.set_xlabel("error (cm)")
# ax.set_ylabel("instances")
# ax.set_title("low density")
# ax.minorticks_on()
# #plt.show()
# #plt.legend(loc="upper right")
# plt.tight_layout()
# plt.savefig('predictions_distribution_man_vs_pre_low.png', dpi=600)



# fig, ax = plt.subplots(figsize=(3,3.25))
# #hist_range = (-5.5,2.5)
# data1 = load_data("output/test_homography/new_approach_5mm.csv", conf_min=0.9, iou_min=0, split="low")
# #data1 = np.clip(data1, hist_range[0], hist_range[1])
# ax.hist([0])
# ax.hist([0])
# ax.hist(data1, bins=40, range=hist_range, alpha=0.5, label="low")
# ax.grid(which = "major", linewidth = 1)
# ax.grid(which = "minor", linewidth = 0.2)
# ax.set_xlim([-7, 3])
# ax.set_ylim([0, 400])
# ax.set_xlabel("error (cm)")
# ax.set_ylabel("instances")
# ax.set_title("low density")
# ax.minorticks_on()
# #plt.show()
# #plt.legend(loc="upper right")
# plt.tight_layout()
# plt.savefig('predictions_distribution_man_vs_pre_low_new.png', dpi=600)

# # # Plot for niels
# # fig, ax = plt.subplots(figsize=(3,3.25))
# # hist_range = (-17.5,2.5)
# # data1 = load_data("output.csv", conf_min=0.9, iou_min=0.9)
# # #data1 = np.clip(data1, hist_range[0], hist_range[1])
# # ax.hist(data1, bins=40, range=hist_range, alpha=0.5, density=True) #, label="pre")
# # ax.grid(which = "major", linewidth = 1)
# # ax.grid(which = "minor", linewidth = 0.2)
# # ax.set_xlabel("error (cm)")
# # ax.set_ylabel("probability")
# # #ax.set_title("length errors - manual vs predicted masks")
# # ax.minorticks_on()
# # #plt.show()
# # plt.tight_layout()
# # plt.savefig('niels_length_errors_man_vs_pre.png', dpi=600)


# # # Create boxplot
# # fig, ax = plt.subplots(figsize=(4,5.5))
# # data_dict = {}
# # data_dict["manual\n masks"] = load_data("near_gt_4dof_sample_9.csv", conf_min=0.8, iou_min=0.5)
# # data_dict["predicted\n masks"] = load_data("near_pred_4dof_sample_9.csv", conf_min=0.8, iou_min=0.5)

# # for k in data_dict.keys():
# #     data = data_dict[k]
# #     print(k)
# #     print(" mean: ", np.mean(data))
# #     print(" median: ", np.median(data))
# #     print(" std: ", np.std(data))

# # ax.boxplot(data_dict.values(), showfliers=False)
# # ax.set_xticklabels(data_dict.keys())
# # ax.grid(which = "major", linewidth = 1)
# # ax.grid(which = "minor", linewidth = 0.2)
# # ax.minorticks_on()
# # ax.set_ylabel("error (cm)")
# # ax.set_title("length errors - manual vs predicted masks")
# # #plt.xticks(rotation=90)
# # #plt.show()
# # plt.savefig('length_errors_man_vs_pre_box.png', dpi=1200)

# # # Compare estimateds length from gt and predicted masks
# # fig, ax = plt.subplots(figsize=(8,4))
# # hist_range = (-10,2.5)
# # data2 = load_data("gt_4dof_sample_9.csv")
# # data2 = np.clip(data2, hist_range[0], hist_range[1])
# # ax.hist(data2, bins=40, alpha=0.5, label="man")
# # data1 = load_data("pred_4dof_sample_9.csv")
# # data1 = np.clip(data1, hist_range[0], hist_range[1])
# # ax.hist(data1, bins=40, alpha=0.5, label="pre")
# # ax.grid(which = "major", linewidth = 1)
# # ax.grid(which = "minor", linewidth = 0.2)
# # ax.set_xlabel("error (cm) - clipped")
# # ax.set_ylabel("number of samples")
# # ax.set_title("length errors - manual vs predicted masks")
# # ax.legend()
# # ax.minorticks_on()
# # #plt.show()
# # plt.savefig('length_errors_man_vs_pre.png', dpi=600)

# # # Compare estimated lengths at Iou>0.5 and Iou>0.9
# # fig, ax = plt.subplots(figsize=(8,4))
# # #all_errors = np.clip(all_errors, -5, 5)
# # hist_range = (-10,2.5)
# # data2 = load_data("pred_4dof_sample_9.csv", iou_min=0.5)
# # data2 = np.clip(data2, hist_range[0], hist_range[1])
# # ax.hist(data2, bins=40, alpha=0.5, label="iou>0.5")
# # data1 = load_data("pred_4dof_sample_9.csv", iou_min=0.9)
# # data1 = np.clip(data1, hist_range[0], hist_range[1])
# # ax.hist(data1, bins=40, alpha=0.5, label="iou>0.9")
# # ax.grid(which = "major", linewidth = 1)
# # ax.grid(which = "minor", linewidth = 0.2)
# # ax.set_xlabel("error (cm) - clipped")
# # ax.set_ylabel("number of samples")
# # ax.set_title("length errors - predicted masks (IoU 0.5 vs 0.9)")
# # ax.legend()
# # ax.minorticks_on()
# # #plt.show()
# # plt.savefig('length_errors_pre_ious_05_09.png', dpi=600)


# # # Compare estimated lengths at Iou>0.5 and Iou>0.95
# # fig, ax = plt.subplots(figsize=(8,4))
# # #all_errors = np.clip(all_errors, -5, 5)
# # hist_range = (-10,2.5)
# # data2 = load_data("pred_4dof_sample_9.csv", iou_min=0.5)
# # data2 = np.clip(data2, hist_range[0], hist_range[1])
# # ax.hist(data2, bins=40, alpha=0.5, label="iou>0.5")
# # data1 = load_data("pred_4dof_sample_9.csv", iou_min=0.95)
# # data1 = np.clip(data1, hist_range[0], hist_range[1])
# # ax.hist(data1, bins=40, alpha=0.5, label="iou>0.95")
# # ax.grid(which = "major", linewidth = 1)
# # ax.grid(which = "minor", linewidth = 0.2)
# # ax.set_xlabel("error (cm) - clipped")
# # ax.set_ylabel("number of samples")
# # ax.set_title("length errors - predicted masks (IoU 0.5 vs 0.95)")
# # ax.legend()
# # ax.minorticks_on()
# # #plt.show()
# # plt.savefig('length_errors_pre_ious_05_095.png', dpi=600)

# # # Compare estimated lengths at Iou>0.5 and Iou>0.8
# # fig, ax = plt.subplots(figsize=(8,4))
# # #all_errors = np.clip(all_errors, -5, 5)
# # hist_range = (-10,2.5)
# # data2 = load_data("pred_4dof_sample_9.csv", iou_min=0.5)
# # data2 = np.clip(data2, hist_range[0], hist_range[1])
# # ax.hist(data2, bins=40, alpha=0.5, label="iou>0.5")
# # data1 = load_data("pred_4dof_sample_9.csv", iou_min=0.8)
# # data1 = np.clip(data1, hist_range[0], hist_range[1])
# # ax.hist(data1, bins=40, alpha=0.5, label="iou>0.8")
# # ax.grid(which = "major", linewidth = 1)
# # ax.grid(which = "minor", linewidth = 0.2)
# # ax.set_xlabel("error (cm) - clipped")
# # ax.set_ylabel("number of samples")
# # ax.set_title("length errors - predicted masks (IoU 0.5 vs 0.8)")
# # ax.legend()
# # ax.minorticks_on()
# # #plt.show()
# # plt.savefig('length_errors_pre_ious_05_08.png', dpi=600)

# # # Compare estimated lengths at different confs
# # fig, ax = plt.subplots(figsize=(8,4))
# # #all_errors = np.clip(all_errors, -5, 5)
# # hist_range = (-10,2.5)
# # data2 = load_data("pred_4dof_sample_9.csv", conf_min=0.8)
# # data2 = np.clip(data2, hist_range[0], hist_range[1])
# # ax.hist(data2, bins=40, alpha=0.5, label="conf>0.8")
# # data1 = load_data("pred_4dof_sample_9.csv", conf_min=0.95)
# # data1 = np.clip(data1, hist_range[0], hist_range[1])
# # ax.hist(data1, bins=40, alpha=0.5, label="conf>0.95")
# # ax.grid(which = "major", linewidth = 1)
# # ax.grid(which = "minor", linewidth = 0.2)
# # ax.set_xlabel("error (cm) - clipped")
# # ax.set_ylabel("number of samples")
# # ax.set_title("length errors - predicted masks (conf 0.95 vs 0.8)")
# # ax.legend()
# # ax.minorticks_on()
# # #plt.show()
# # plt.savefig('length_errors_pre_confs_08_095.png', dpi=600)
