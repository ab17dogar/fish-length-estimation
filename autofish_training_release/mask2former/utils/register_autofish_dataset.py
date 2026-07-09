import os
from pycocotools.coco import COCO
import numpy as np
import copy
import random
from detectron2.data import MetadataCatalog, DatasetCatalog, datasets
from detectron2.structures import BoxMode
import shutil
from tabulate import tabulate

LABELS_CONFIGURATIONS = {
    'C1': {
        'classes': ["whiting", "cod", "haddock", "hake", "horse_mackerel", "other"],
        'id_species_map': {
            "whiting": 0,
            "cod": 1,
            "haddock": 2,
            "hake": 3,
            "horse_mackerel": 4,
            "saithe": 5,
            "other": 5,
        },
    },

    'C2': {
        'classes': ["whiting", "cod", "haddock", "other"],
        'id_species_map': {
            "whiting": 0,
            "cod": 1,
            "haddock": 2,
            "hake": 3,
            "horse_mackerel": 3,
            "saithe": 3,
            "other": 3,
        },
    },

    'C3': {
        'classes': ["cod-like", "other"],
        'id_species_map': {
            "whiting": 0,
            "cod": 0,
            "haddock": 0,
            "hake": 1,
            "horse_mackerel": 1,
            "saithe": 1,
            "other": 1,
        },
    },

    'C4': {
        'classes': ["fish"],
        'id_species_map': {
            "whiting": 0,
            "cod": 0,
            "haddock": 0,
            "hake": 0,
            "horse_mackerel": 0,
            "saithe": 0,
            "other": 0,
        },
    }
}


class Autofish:
    def __init__(self, root_dir, train_split, val_split, test_split, labels_configuration=None, classes=None, id_species_map=None):
        self.root_dir = root_dir
        self.train_split = train_split
        self.val_split = val_split
        self.test_split = test_split
        self.exclusions = {}

         # if classes and id_species_map arguments are used explicitly, use these instead of the label_configuration argument
        if classes != None and id_species_map != None:
            self.classes = classes
            self.id_species_map = id_species_map
        elif labels_configuration != None:
            self.classes = LABELS_CONFIGURATIONS[labels_configuration]['classes']
            self.id_species_map = LABELS_CONFIGURATIONS[labels_configuration]['id_species_map']
        else:
            print("Provide label_configuration or classes and id_species_map argument as the input.")
            exit()

        # create splits, remove val split it no validation groups are specified
        self.splits = ["autofish_train", "autofish_val", "autofish_test"]
        if len(self.val_split) == 0:
            self.splits.remove("autofish_val")

        # create dictionary with information about each create split, e.g., camera type, image type, groups
        self.split_configurations = self.build_split_configurations(self.splits)


    @classmethod
    def instance_from_yaml(cls, yaml_config):
        #_yaml_config = copy.deepcopy(yaml_config)
        _yaml_config = yaml_config

        if _yaml_config.get('subsample_train_split') and _yaml_config['subsample_train_split']['subsample']:
            _yaml_config['train_split'] = random.sample(_yaml_config['train_split'], _yaml_config['subsample_train_split']['no_of_groups'])
            print(_yaml_config['train_split'])

        if "classes" and "id_species_map" in _yaml_config:
            instance = cls(
                root_dir=_yaml_config["root_dir"],
                train_split=_yaml_config["train_split"],
                val_split=_yaml_config["val_split"],
                test_split=_yaml_config["test_split"],
                classes=_yaml_config["classes"],
                id_species_map=_yaml_config["id_species_map"]
            )
        elif "labels_configuration" in _yaml_config:
            instance = cls(
                root_dir=_yaml_config["root_dir"],
                train_split=_yaml_config["train_split"],
                val_split=_yaml_config["val_split"],
                test_split=_yaml_config["test_split"],
                labels_configuration=_yaml_config["labels_configuration"]
            )

        if _yaml_config.get('exclusions'):
            instance.exclusions = _yaml_config['exclusions']

        return instance


    def build_split_configurations(self, splits):
        configurations = {}
        for split in splits:
            configurations[split] = {
                'groups': 0
            }
            #groups
            if split.endswith("train"):
                configurations[split]['groups'] = self.train_split
            elif split.endswith("val"):
                configurations[split]['groups'] = self.val_split
            elif split.endswith("test"):
                configurations[split]['groups'] = self.test_split
        return configurations


    def get_category_id(self, cat_name):
        return self.id_species_map[cat_name]
    

    def register_split(self, groups:list):
        """
        Registers a dataset split for Detectron2.

        This method processes the COCO annotations, updates the category IDs based on the provided mapping,
        and prepares the dataset dictionary for the specified groups.

        Args:
            groups (list): List of group names to include in the dataset split.

        Returns:
            list: A list of dictionaries, each representing an image and its annotations in the dataset split.
        """
        dataset_dicts = []
        json_file = os.path.join(self.root_dir, 'annotations.json')
        coco_annotation = COCO(annotation_file=json_file)

        # Update category_id in annotations based on mapping
        for ann in coco_annotation.dataset['annotations']:
            ann['category_id'] = self.get_category_id(coco_annotation.loadCats(ann['category_id'])[0]['name'])
            ann['bbox_mode'] = BoxMode.XYWH_ABS
            
        for group in groups:
            images = coco_annotation.loadImgs(coco_annotation.getImgIds())
            images = [img for img in images if img['group'] == group]
            
            if self.exclusions.get(group):
                exclusions = [f"{str(img).zfill(5)}" for img in self.exclusions.get(group)]
            else:
                exclusions = []

            for image in images:
                if image["file_name"].split(".")[0] in list(set(exclusions)):
                    pass
                else:
                    record = {}
                    record["file_name"] = os.path.join(self.root_dir, image["file_name"]) 
                    record["image_id"] = image["id"]
                    record["height"] = image["height"]
                    record["width"] = image["width"]

                    # Get all annotations for the current image
                    ann_ids = coco_annotation.getAnnIds(imgIds=image["id"], iscrowd=None)
                    anns = coco_annotation.loadAnns(ann_ids)
                    anns = [ann for ann in anns if ann["category_id"] in self.id_species_map.values()]
                    record["annotations"] = anns
                    dataset_dicts.append(record)
        return dataset_dicts


    def register_autofish_split(self, split:str):
        assert split in self.splits, f"{split} is not available. Available splits are {self.splits}"
        groups = self.split_configurations[split]['groups']

        DatasetCatalog.register(split, lambda groups=groups: self.register_split(groups))
        MetadataCatalog.get(split).set(thing_classes=self.classes)
        MetadataCatalog.get(split).set(evaluator_type="coco")


    def register_all_autofish_splits(self):
        for split in self.splits:
            self.register_autofish_split(split)


def parse_arguments():
    parser = argparse.ArgumentParser()
    # Define the argument for the YAML file path
    parser.add_argument('-y', '--yaml', required=True, type=str, help='Path to the YAML file.')
    parser.add_argument('-s', '--save', action='store_true', help='Indicate if you are saving the dataset as a coco json file')
    parser.add_argument('-o', '--output', type=str, help='Where to save a coco json file')
    parser.add_argument('-v', '--visualize', action='store_true', help='Visualize samples from the registered dataset. These are saved in mask2former/output')
    args = parser.parse_args()
    return args


def print_dataset_statistics(dataset_instance):
    histograms = []
    table_labels = copy.deepcopy(dataset_instance.classes)
    table_labels.append("total")
    for split in dataset_instance.splits:
        hist = generate_hist_of_instances_per_class(split, dataset_instance.classes)
        hist.append(sum(hist))
        histograms.append(hist)
        histograms[-1].insert(0, split)

    #compute autofish_total (train+val+test)
    total_hist = [0 for i in range(len(table_labels))]
    total_hist.insert(0, "autofish_total")
    for hist in histograms:
        if hist[0] in ["autofish_train", "autofish_val", "autofish_test"]:
            for i in range(1, len(hist)):
                total_hist[i] += hist[i]
    histograms.append(total_hist)

    table_labels.insert(0, "dataset_split")
    print(tabulate(histograms, headers=table_labels))


def generate_hist_of_instances_per_class(dataset, class_names):
        dataset_dict = DatasetCatalog.get(dataset)
        num_classes = len(class_names)
        hist_bins = np.arange(num_classes + 1)
        histogram = np.zeros((num_classes,), dtype=int)
        for entry in dataset_dict:
            annos = entry["annotations"]
            classes = np.asarray(
                [x["category_id"] for x in annos if not x.get("iscrowd", 0)], dtype=int
            )
            if len(classes):
                assert classes.min() >= 0, f"Got an invalid category_id={classes.min()}"
                assert (
                    classes.max() < num_classes
                ), f"Got an invalid category_id={classes.max()} for a dataset of {num_classes} classes"
            histogram += np.histogram(classes, bins=hist_bins)[0]

        histogram = [v for _, v in enumerate(histogram)]
        return histogram


def draw_samples(dataset, metadata, window_name, number_of_samples = 3):
    for i, d in enumerate(random.sample(dataset, number_of_samples)):
        img = cv2.imread(d["file_name"])
        visualizer = Visualizer(img[:, :, ::-1], metadata=metadata, scale=0.5)
        out = visualizer.draw_dataset_dict(d)
        os.makedirs('../output/', exist_ok=True)
        cv2.imwrite(f'../output/visualization_{i}.png', out.get_image()[:, :, ::-1])


if __name__=="__main__":
    from detectron2.utils.visualizer import Visualizer
    import cv2
    import yaml
    import argparse

    args = parse_arguments()
    config = yaml.safe_load(open((args.yaml)))

    dataset = Autofish.instance_from_yaml(config)
    dataset.register_all_autofish_splits()

    print_dataset_statistics(dataset)
    
    if args.save:
        os.makedirs(args.output)
        for split in dataset.splits:
            datasets.convert_to_coco_json(split,output_file=os.path.join(args.output, f"{split}.json"), allow_cached=False)
        shutil.copy2(args.yaml, args.output)

    if args.visualize:
        dataset = DatasetCatalog.get('autofish_train')
        draw_samples(dataset, None, "jai_test_mini")
