import torch
#from torch.cuda.amp import autocast
import os
import numpy as np
import cv2
import argparse
import yaml
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
# detectron2
from detectron2.config import get_cfg
from detectron2.projects.deeplab import add_deeplab_config
from detectron2.data import MetadataCatalog
from detectron2.modeling import build_model
from detectron2.checkpoint import DetectionCheckpointer
# Mask2Former
import sys
sys.path.append("/workspace/Mask2Former")
from mask2former import add_maskformer2_config
from register_autofish_dataset import Autofish
# custom
from length_estimator import LengthEstimator

#RGB
VISUALIZER_COLORS = [(255, 0, 0),
          (0, 255, 0),
          (0, 0, 255),
          (0, 255, 255),
          (255, 0, 255),
          (255, 255, 0)
          ]

class Predictor:
    def __init__(self, cfg):
        self.cfg = cfg.clone()
        self.model = build_model(self.cfg)
        self.model.eval()
        checkpointer = DetectionCheckpointer(self.model)
        checkpointer.load(cfg.MODEL.WEIGHTS)


    def __call__(self, original_image, augmentations=None, convert_BGR_to_RGB=False, convert_RGB_to_BGR=False, conversion_function=None):
        assert_msg = f"Pick only one conversion. convert_BGR_to_RGB {convert_BGR_to_RGB} convert_RGB_to_BGR {convert_RGB_to_BGR}"
        assert not all([convert_BGR_to_RGB, convert_RGB_to_BGR])==True, assert_msg
        if any([convert_BGR_to_RGB, convert_RGB_to_BGR])==True and conversion_function is not None:
            print("The provided conversion function takes precedences over other selected conversion methods.")

        with torch.no_grad():  # https://github.com/sphinx-doc/sphinx/issues/4258
            if conversion_function is not None:
                original_image = conversion_function(original_image)
            elif convert_BGR_to_RGB or convert_RGB_to_BGR:
                original_image = original_image[:, :, ::-1]

            height, width = original_image.shape[:2]

            if augmentations != None:
                image = augmentations.get_transform(original_image).apply_image(original_image)
            else:
                image = original_image

            image = torch.as_tensor(image.astype("float32").transpose(2, 0, 1))

            inputs = {"image": image, "height": height, "width": width}
            predictions = self.model([inputs])[0]
            return predictions


class Visualizer:
    @classmethod
    def draw(cls, image, masks, scores, labels, draw_text, color_map=None, text_scale=1):
        # We'll start by creating a completely transparent overlay
        overlay = image.copy()

        for mask, score, cls in zip(masks, scores, labels):
            if not np.all(mask==0):
                # Convert mask to contour
                contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                # Generate a random color for visualization
                if color_map is None:
                    color = tuple(np.random.randint(0, 256, 3).tolist())
                else:
                    color = color_map[cls]
                # Fill contour (mask) on the overlay with semi-transparency
                cv2.drawContours(overlay, contours, -1, color, -1)  # -1 fill the contour
                # Draw contour border on the image
                cv2.drawContours(image, contours, -1, color, 2)
                # Get center of the mask
                y_coords, x_coords = np.where(mask == 1)
                # Compute the mean of the x and y coordinates
                cX = int(np.mean(x_coords))
                cY = int(np.mean(y_coords))

                if draw_text:
                    # Overlay the class label and the score at the center of the mask
                    label = f"{cls} {score*100:.0f}%"
                    # Calculate text width & height to create the background rectangle
                    (text_width, text_height), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, fontScale=text_scale, thickness=1)
                    padding = int(np.round(2*text_scale))
                    rect_start_point = (cX - text_width // 2 - padding, cY - text_height // 2 - padding)
                    rect_end_point = (cX + text_width // 2 + padding, cY + text_height // 2 + padding)
                    cv2.rectangle(image, rect_start_point, rect_end_point, (0, 0, 0), -1)
                    cv2.rectangle(overlay, rect_start_point, rect_end_point, (0, 0, 0), -1)
                    # Now, place the text on top of the rectangle
                    cv2.putText(image, label, (cX - text_width // 2, cY + text_height // 2), cv2.FONT_HERSHEY_SIMPLEX, fontScale=text_scale, color=color, thickness=1)
                    cv2.putText(overlay, label, (cX - text_width // 2, cY + text_height // 2), cv2.FONT_HERSHEY_SIMPLEX, fontScale=text_scale, color=color, thickness=1)

        # Alpha blend the overlay with the original image
        alpha = 0.5
        cv2.addWeighted(overlay, alpha, image, 1 - alpha, 0, image)
        return image


# get the minimal required config for the predictor
def get_config(model_config):
    cfg = get_cfg()
    add_deeplab_config(cfg)
    add_maskformer2_config(cfg)
    model = model_config["model"]
    cfg.merge_from_file(model)
    cfg.MODEL.WEIGHTS = os.path.join(model_config["output_dir"], model_config["test_weights"])
    cfg.TEST.DETECTIONS_PER_IMAGE=model_config["no_of_predictions"]
    dataset_classes = MetadataCatalog.get(model_config['test_dataset'][0]).thing_classes
    cfg.MODEL.SEM_SEG_HEAD.NUM_CLASSES = len(dataset_classes)
    cfg.freeze()
    return cfg, dataset_classes


def generate_legend(color_map, output):
    # Create a separate figure for the legend
    fig, ax = plt.subplots(figsize=(len(color_map) * 2.5, 1))
    #fig.set_dpi(200)

    # Create legend patches
    legend_patches = [Patch(color=[r / 255, g / 255, b / 255], label=species) for species, (r, g, b) in color_map.items()]

    # Create legend in the separate figure
    #ax.legend(handles=legend_patches, loc='center')
    legend = ax.legend(handles=legend_patches, loc='center', ncol=len(color_map), frameon=False, fontsize=20)

    # Set font size for legend text

    # Set legend patch size
    for patch in legend.legend_handles:
        patch.set_height(10)  # Adjust the height of legend patches
        patch.set_width(30)   # Adjust the width of legend patches


    # Remove the axis to make it a clean legend figure
    ax.axis('off')
    plt.savefig(os.path.join(output, "legend.png"))


def resize_image(image, scale):
        height, width, _ = image.shape
        target_dim = (int(width*scale), int(height*scale))
        return cv2.resize(image, target_dim, interpolation=cv2.INTER_AREA)


def concatenate_images(img1, img2, bar_height=10):
    black_bar = np.zeros((bar_height, img1.shape[1], img1.shape[2]), dtype=img1.dtype)
    return np.vstack((img1, black_bar, img2))


def parse_arguments():
    parser = argparse.ArgumentParser()

    parser.add_argument('-i', '--input', required=True, type=str, help='Path to a folder with images')
    parser.add_argument('-o', '--output', required=True, type=str, help='Output folder')
    parser.add_argument('-y', '--yaml', required=True, type=str, help='Path to a YAML file')

    #parser.add_argument('--labels', type=bool, default=True, help='Draw predicted labels along with the masks')
    parser.add_argument('--no_labels', action='store_true', help='Draw predicted labels along with the masks')
    parser.add_argument('--confidence', type=float, default=-1, help='Filter predictions lower than the given confidence level')
    parser.add_argument('--length', action='store_true', help='Visualize the length estimation output')

    args = parser.parse_args()
    return args


def main(args):
    print(args)
    model_config =  yaml.safe_load(open(args.yaml))

    dataset = Autofish.instance_from_yaml(model_config)
    dataset.register_all_autofish_splits()
    config, classes = get_config(model_config=model_config)

    predictor = Predictor(cfg=config)

    os.makedirs(os.path.join(".", args.output), exist_ok=True)

    color_map = dict(zip(classes, VISUALIZER_COLORS))
    generate_legend(color_map, args.output)
    # Convert color_map to BGR as we are using opencv as our image library
    for k,v in color_map.items():
        v_reversed = tuple(list(v)[::-1])
        color_map[k] = v_reversed

    images = os.listdir(args.input)
    img_formats = ["jpeg", "jpg", "tiff", "png"]
    images = [i for i in images if i.split(".")[-1] in img_formats]

    # Set confidence level.
    if args.confidence != -1:
        confidence = args.confidence
    elif model_config.get('confidence'):
        confidence = model_config['confidence']
    else:
        print("No confidence level was provided")
        exit()

    # Draw labels
    if args.no_labels:
        draw_text = False
    else:
        draw_text= True

    for idx, image in enumerate(images):
        image_filename = os.path.join(args.input, image)
        print(f"working on {image_filename}")
        image = cv2.imread(image_filename)

        #resize image to the training size
        #can also be done with the predictor using resize augmentations as an input
        image = resize_image(image, model_config['test_augmentations']['scale'])

        predictions = predictor(image, conversion_function= lambda img: img[:, :, ::-1])
        pred_instances = predictions['instances'].to('cpu')
        pred_instances = pred_instances[pred_instances.scores > confidence]
        pred_masks = pred_instances.pred_masks.numpy()
        pred_labels = pred_instances.pred_classes.numpy()
        pred_labels = [classes[id] for id in pred_labels]
        pred_scores = pred_instances.scores.numpy()

        img_with_predictions = Visualizer.draw(image=image.copy(),
                                               masks=pred_masks,
                                               scores=pred_scores,
                                               labels=pred_labels,
                                               draw_text=draw_text,
                                               color_map=color_map,
                                               text_scale=model_config['test_augmentations']['scale']
                                               )

        img_to_save = concatenate_images(image, img_with_predictions)

        #output_image = image_filename.split("/")[-1]
        #output_image = output_image.replace(".tiff", ".png")
        #output_filename = os.path.join(".", args.output, output_image)

        output_filename = os.path.join(".", args.output, image_filename.split("/")[-1])
        cv2.imwrite(output_filename, img_to_save)

        #length_estimation
        if args.length:
            estimated_lengths = []
            for mask in pred_masks:
                mask = mask.astype(np.uint8)
                fit = LengthEstimator.get_poly_fit(
                    binary_mask=mask,
                    skeleton_method="zhang", #skeletonization method, this one is the fastest
                    degree=4, #polynomial degree
                    #subsample_skeleton=1, #percentage of points to keep of the skeleton
                    #subsample_fit=1, #percentage of points to keep of the curve
                    clip=True #clip to convex hull
                    )
                estimated_lengths.append(fit[0])
            img_with_lengths = LengthEstimator.draw_predicted_lengths(image, estimated_lengths)
            img_to_save = concatenate_images(img_with_predictions, img_with_lengths)
            output_filename = os.path.join(".", args.output, f"img_{idx}_lengths.png")
            cv2.imwrite(output_filename, img_to_save)


if __name__=="__main__":
    main(parse_arguments())
