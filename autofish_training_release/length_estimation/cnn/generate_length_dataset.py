import csv
import os
import numpy as np
from pycocotools.coco import COCO
from skimage import io
import matplotlib.pyplot as plt
import copy
import argparse

def process_group(group_no):
    gt_lengths = load_gt("gt_lengths_cm.csv")
    gt_path = "/media/shbe/data/datasets/mine/autofish/autofish_groups/group_{0}/jai/annotations/coco/manual/dataset.json".format(group_no)
    img_dir = "/media/shbe/data/datasets/mine/autofish/autofish_groups/group_{0}/jai/rgb".format(group_no)

    output_dir = "group{0}".format(group_no)
    os.mkdir(output_dir)

    coco = COCO(gt_path)

    images = coco.loadImgs(coco.getImgIds())

    counter = 0
    with open(os.path.join(output_dir, "lengths.csv"),"w") as f:
        for idx, image_dict in enumerate(images):
            print(f"Working on image {image_dict['file_name']}")

            # Load image
            img_name = image_dict["file_name"]
            img_path = os.path.join(img_dir, img_name)
            image = io.imread(img_path)
            image = image/np.iinfo(image.dtype).max
            image = (image * 255).astype(np.uint8)

            # Look through all categories
            cat_ids = coco.getCatIds()
            for c in cat_ids:
                ann_ids = coco.getAnnIds(imgIds=image_dict["id"], catIds=c)

                if(len(ann_ids) == 0):
                    #print("SKIPPING")
                    continue

                # Merge masks for the same id
                if(len(ann_ids) > 1):
                    print("keep: ", ann_ids[0])
                    print(" - merge: ", ann_ids[1:])

                    for merge_id in ann_ids[1:]:
                        #print("merge_id: ", merge_id)
                        coco.anns[ann_ids[0]]["segmentation"].append(coco.anns[merge_id]["segmentation"][0])

                curr_ann = coco.anns[ann_ids[0]]

                # Load mask
                mask = coco.annToMask(curr_ann)
                image_masked = copy.deepcopy(image)
                image_masked[mask == 0] = 0

                # Load label and find length
                label = coco.loadCats(curr_ann["category_id"])[0]["name"]
                label = label.replace("L-", "-").replace("R-", "-")
                label = label.replace("saithe", "other")
                length = gt_lengths[label]
                print(f"{label} with length: {length} cm")

                # Save image
                img_name = "{:02d}-{:05d}.png".format(group_no, counter)
                io.imsave(os.path.join(output_dir, img_name), image_masked)

                #print(curr_ann)
                #print(img_name)

                # Write length to csv
                f.write(f"{img_name},{length}")
                f.write("\n")
                counter = counter + 1


    #catIds = coco.getCatIds() #imgIds=image["id"])
    #annIds = coco.getAnnIds(catIds=catIds, iscrowd=None)
    #anns = coco.loadAnns(annIds)

    #with open('lengths.csv','w') as f:

        # for i,ann in enumerate(anns):
        #     # Load image
        #     img_name = coco.loadImgs(ann["image_id"])[0]["file_name"]
        #     img_path = os.path.join(img_dir, img_name)
        #     image = io.imread(img_path)
        #     image = image/np.iinfo(image.dtype).max
        #     image = (image * 255).astype(np.uint8)

        #     # Load mask
        #     mask = coco.annToMask(ann)
        #     image_masked = image
        #     image_masked[mask == 0] = 0

        #     # Load label and find length
        #     label = coco.loadCats(ann["category_id"])[0]["name"]
        #     label = label.replace("L-", "-").replace("R-", "-")
        #     length = gt_lengths[label]
        #     print(f"{label} with length: {length} cm")

        #     # Save image
        #     img_name = "{:02d}-{:05d}.png".format(group_no, i)
        #     print(image_masked.shape)
        #     print(image_masked.dtype)
        #     io.imsave(img_name, image_masked)

        #     # Write length to csv
        #     f.write(f"{img_name},{length}")
        #     f.write("\n")

        #plt.imshow(image_masked)
        #plt.show()


def load_gt(gt_path):
    gt = {}
    with open(gt_path, newline='') as csvfile:
        reader = csv.DictReader(csvfile)

        for row in reader:
           curr_label =  "{0}-{1}".format(row['id'],row['fish'])
           gt[curr_label] = float(row['length'])
    return gt



if __name__=="__main__":
    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--group_no", help="Group number")
    arguments = parser.parse_args()
    p = parser.parse_args()

    process_group(int(p.group_no))
