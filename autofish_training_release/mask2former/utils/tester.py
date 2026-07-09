#common modules
from torch import argsort
from torch.cuda.amp import autocast
import os
import csv
import copy
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import cv2
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
import argparse
import yaml
#detectron2 modules
from detectron2.evaluation import COCOEvaluator, inference_on_dataset
from detectron2.utils.logger import setup_logger
import detectron2.utils.comm as comm
from detectron2.data import build_detection_test_loader, DatasetMapper, MetadataCatalog
#Mask2Former
import sys
sys.path.append("/workspace/Mask2Former")
from mask2former import COCOInstanceNewBaselineDatasetMapper
#custom modules
from inference import Predictor, Visualizer
import augmentations
from register_autofish_dataset import Autofish, generate_hist_of_instances_per_class

# TODO: Add the functionality of computing confusion matrix from coco results file

class Tester:
    def __init__(self, yaml, train_cfg, data_augmentations, coco_results:bool=False):
        self.test_cfg = copy.deepcopy(train_cfg)
        self.test_cfg.DATASETS.ROOT_DIR = yaml['root_dir']
        self.test_cfg.DATASETS.TEST = yaml["test_dataset"]
        assert len(self.test_cfg.DATASETS.TEST) == 1, "Evaluation across multiple datasets is not supported yet"
        self.test_cfg.MODEL.WEIGHTS = os.path.join(yaml["output_dir"], yaml["test_weights"])
        self.test_cfg.TEST.DETECTIONS_PER_IMAGE=yaml["no_of_predictions"]
        self.test_cfg.freeze()
        self.data_augmentations = data_augmentations
        if coco_results:
            assert os.path.isfile(os.path.join(self.test_cfg.OUTPUT_DIR, "coco_instances_results.json")), "No coco results found"
            self.predictor = None
        else:
            self.predictor = Predictor(self.test_cfg)
        self.classes = MetadataCatalog.get(self.test_cfg.DATASETS.TEST[0]).thing_classes
        self.number_of_instances = generate_hist_of_instances_per_class(self.test_cfg.DATASETS.TEST[0], self.classes)


    def compute_ap(self, data_loader_batch_size = 1, data_loader_num_worker = 1, task=None, fast=False, output_suffix=None):
        #########################################################################
        #########################################################################
        def compute_AP_for_every_IoU(results):
            print("Computing AP for every IoU threshold")
            gt = os.path.join(self.test_cfg.OUTPUT_DIR, f"{self.test_cfg.DATASETS.TEST[0]}_coco_format.json")
            pred = os.path.join(self.test_cfg.OUTPUT_DIR, "coco_instances_results.json")
            
            coco_gt = COCO(gt)
            predictions = coco_gt.loadRes(pred)
            eval_res = COCOeval(coco_gt, predictions)
            if output_suffix == 'low':
                density_imgs = [ids for ids in coco_gt.getImgIds() if len(coco_gt.getAnnIds(imgIds=ids)) < 14]
                eval_res.params.imgIds = density_imgs
            elif output_suffix == 'high':
                density_imgs = [ids for ids in coco_gt.getImgIds() if len(coco_gt.getAnnIds(imgIds=ids)) >= 14]
                eval_res.params.imgIds = density_imgs
            eval_res.evaluate()
            eval_res.accumulate()
            #eval_res.summarize()
            p = eval_res.eval["params"]
            areaRng = 'all'
            maxDets = 100
            aind = [i for i, aRng in enumerate(p.areaRngLbl) if aRng == areaRng]
            mind = [i for i, mDet in enumerate(p.maxDets) if mDet == maxDets]

            IoUs, APs = [], []
            for i in range(eval_res.eval['precision'].shape[2]):
                class_precision = eval_res.eval['precision'][:, : ,i, aind, mind]
                ap = class_precision.mean()  # Average over all IoU thresholds and recall levels
                results['segm'][f"AP-{coco_gt.loadCats(ids=[i])[0]['name']}"] = ap*100
            for iouThr in p.iouThrs:
                if iouThr is not None:
                    s = copy.deepcopy(eval_res.eval['precision'])
                    t = np.where(iouThr == p.iouThrs)[0]
                    s = s[t]
                    s = s[:,:,:,aind,mind]
                    IoUs.append(np.round(iouThr, 2))
                    APs.append(s.mean())
                    #print(f"AP@{np.round(iouThr, 2)} {np.round(s.mean(), 3)}")

            for IoU, ap in zip(IoUs, APs):
                if int(IoU*100) != 50 or int(IoU*100) != 75:
                    results['segm'][f'AP{int(100*IoU)}'] = ap*100
            print(results)
            return results
        #########################################################################
        #########################################################################
        def write_csv(output, data):
            # Modify the structure
            modified_data = {}
            for category, metrics in data.items():
                for key, value in metrics.items():
                    modified_data[f"{category}/{key}"] = value

            # Write to CSV
            with open(output, 'w', newline='') as csvfile:
                writer = csv.writer(csvfile)
                # Write the headers
                writer.writerow(modified_data.keys())
                # Write the data
                writer.writerow(modified_data.values())
            print(f"Saved results to {output}")
        #########################################################################
        #########################################################################
        
        evaluator = COCOEvaluator(self.test_cfg.DATASETS.TEST[0],
                    tasks=task,
                    output_dir=self.test_cfg.OUTPUT_DIR,
                    use_fast_impl=fast,
                    allow_cached_coco=False)

        data_loader = build_detection_test_loader(
            self.test_cfg,
            self.test_cfg.DATASETS.TEST[0],
            mapper = DatasetMapper(
                self.test_cfg,
                is_train=False,
                augmentations=self.data_augmentations
            ),
            batch_size = data_loader_batch_size,
            num_workers = data_loader_num_worker
        )
        
        if self.predictor:
            with autocast():
                results = inference_on_dataset(self.predictor.model, data_loader, evaluator)
        else:
            results = {}
            results["segm"] = {}

        results = compute_AP_for_every_IoU(results)
        output_filename = "ap_{}.csv".format(self.test_cfg.DATASETS.TEST[0])

        if output_suffix is not None:
            output_filename = "ap_{}_{}.csv".format(self.test_cfg.DATASETS.TEST[0], output_suffix)


        write_csv(output=os.path.join(self.test_cfg.OUTPUT_DIR, output_filename), data=results)


    def compute_confusion_matrix(self, level="instance_level", compute_FP_FN=True, IoU=0.5, confidence=0.5, data_loader_num_worker=1, normalize=True):

        def normalize_confusion_matrix(confusion_matrix):
            #check if FPs and FNs were computed
            if confusion_matrix.shape[0] == len(self.classes):
                col_sums = np.sum(confusion_matrix, axis=0)
                return confusion_matrix/col_sums
            else:
                #normalize predictions including FPs
                cm_predictions = confusion_matrix[:, :-1]
                col_sums = np.sum(cm_predictions, axis=0)
                normalized_predictions = cm_predictions/col_sums
                #normalize FNs
                cm_missed_predictions = confusion_matrix[:, -1:]
                number_of_instances = np.append(np.array(self.number_of_instances), 1)
                normalized_missed_predictions = cm_missed_predictions/number_of_instances[:, np.newaxis]
                return np.hstack((normalized_predictions, normalized_missed_predictions))

        def plot_confusion_matrix(confusion_matrix, output):
            # Assuming you have 6 classes + background
            #labels = copy.deepcopy(self.classes)
            labels_predictions = copy.deepcopy(self.classes)
            labels_ground_truth = copy.deepcopy(self.classes)
            if confusion_matrix.shape[0] != len(labels_predictions):
                labels_predictions.append("bg (FN)")
                labels_ground_truth.append("bg (FP)")

            plt.figure(figsize=(10, 8))
            # Using seaborn to plot the heatmap
            sns.heatmap(confusion_matrix, annot=True, fmt='.3f', cmap='viridis', xticklabels=labels_predictions, yticklabels=labels_ground_truth)
            plt.ylabel('True Class')
            plt.xlabel('Predicted Class')
            plt.title('Confusion Matrix')
            plt.savefig(output)
            print(f"Saving a confusion matrix to {output}")
        
        def compute_from_model():
            data_loader = build_detection_test_loader(
            self.test_cfg,
            self.test_cfg.DATASETS.TEST[0],
            mapper = COCOInstanceNewBaselineDatasetMapper(
                is_train=True,
                image_format=self.test_cfg.INPUT.FORMAT,
                tfm_gens = self.data_augmentations
            ),
            batch_size = 1,
            num_workers = data_loader_num_worker
            )

            if level=="instance_level" and compute_FP_FN is False:
                num_classes = len(self.classes)
            else:
                num_classes = len(self.classes) + 1 #number of classes + 1 for background

            confusion_matrix = np.zeros((num_classes, num_classes), dtype=np.float128)
            itr_counter = 0
            data_loader_itr = iter(data_loader)

            for input in data_loader_itr:
                print(f"Confusion Matrix: Processing image {input[0]['file_name']} {itr_counter}/{len(data_loader_itr)}")
                img = input[0]["image"].permute(1, 2, 0).cpu().detach().numpy()

                with autocast():
                    predictions = self.predictor(img)

                gt_instances = input[0]["instances"]
                pred_instances = predictions['instances'].to('cpu')
                pred_instances = self._sort_predictions(pred_instances)

                if level == "instance_level":
                    confusion_matrix += self._compute_instance_level_confusion_matrix(
                        gt=gt_instances,
                        predictions=pred_instances,
                        IoU_threshold=IoU,
                        confidence_threshold=confidence,
                        FP_FN=compute_FP_FN,
                    )

                if level == "pixel_level":
                    if compute_FP_FN:
                        print("FP and FN are not computed for a pixel-level confusion matrix by default.")
                    confusion_matrix += self._compute_pixel_level_confusion_matrix(
                        gt=gt_instances,
                        predictions=pred_instances,
                        confidence_threshold=confidence
                    )
                itr_counter += 1
            return confusion_matrix

        def compute_from_coco():
            raise Exception('Confusion matrix from coco results file has not been implemented. Please use model checkpoint')

        
        if self.predictor:
            confusion_matrix = compute_from_model()
        else:
            confusion_matrix = compute_from_coco()

        if normalize:
            confusion_matrix = normalize_confusion_matrix(confusion_matrix)

        output_filename = None
        if level == "instance_level":
            output_filename = f"cm_{level}@IoU{IoU}conf{confidence}FPFN{compute_FP_FN}_{self.test_cfg.DATASETS.TEST[0]}"
        if level == "pixel_level":
            output_filename = f"cm_{level}@conf{confidence}_{self.test_cfg.DATASETS.TEST[0]}"
        np.save(os.path.join(self.test_cfg.OUTPUT_DIR, output_filename + ".npy"), confusion_matrix)
        plot_confusion_matrix(confusion_matrix, output=os.path.join(self.test_cfg.OUTPUT_DIR, output_filename + ".png"))


    def _sort_predictions(self, predictions_tensor):
        sorted_indices = argsort(predictions_tensor.scores, descending=True)
        # Use the sorted indices to rearrange the other tensors
        predictions_tensor.pred_masks = predictions_tensor.pred_masks[sorted_indices]
        predictions_tensor.pred_boxes = predictions_tensor.pred_boxes[sorted_indices]
        predictions_tensor.scores = predictions_tensor.scores[sorted_indices]
        predictions_tensor.pred_classes = predictions_tensor.pred_classes[sorted_indices]
        return predictions_tensor


    def _compute_instance_level_confusion_matrix(self, gt, predictions, IoU_threshold, confidence_threshold, FP_FN):
        # initialize empty confusion matrix
        if FP_FN:
            num_classes = len(self.classes) + 1
        else:
            num_classes = len(self.classes)

        confusion_matrix = np.zeros((num_classes, num_classes), dtype=np.uint64)

        # get image ground truth
        gt_masks = gt.gt_masks.numpy()
        gt_labels = gt.gt_classes.numpy()
        gt_matches = np.zeros(len(gt_masks))

        # Filter the instances using the input confidence threshold
        predictions = predictions[predictions.scores > confidence_threshold]
        # get predictions
        pred_masks = predictions.pred_masks.numpy()
        pred_labels = predictions.pred_classes.numpy()

        for mask, label in zip(pred_masks, pred_labels):
            IoU_with_gt = []
            for gt_mask in gt_masks:
                IoU_with_gt.append(self._compute_iou(mask, gt_mask))
            max_IoU = max(IoU_with_gt)

            if max_IoU > IoU_threshold:
                idx = IoU_with_gt.index(max_IoU)
                if label == gt_labels[idx] and gt_matches[idx] != 1:
                    #print("TP")
                    confusion_matrix[gt_labels[idx], label] += 1
                    gt_matches[idx] = 1
                elif label == gt_labels[idx] and gt_matches[idx] == 1 and FP_FN:
                    #print("another match for gt with an already existing match, FP (double detection)")
                    confusion_matrix[-1, label] += 1
                elif label != gt_labels[idx]:
                    #print("misclassification, classes do not match no need to check for a match with the ground truth")
                    confusion_matrix[gt_labels[idx], label] += 1
            else:
                #print(f"BACKGROUND MISCLASSIFICATION pred_label {label} max_IoU {max_IoU} IoU{IoU_with_gt}, FN")
                #cv2.imwrite("misclassification_gt.png", gt_masks[IoU_with_gt.index(max_IoU)]*255)
                #cv2.imwrite("misclassification_pred.png", mask*255)
                #print("a predicted mask doesn't match any of the annotations, FP")
                if FP_FN:
                    confusion_matrix[-1, label] += 1

        if FP_FN:
            unmatched_gt = np.where(gt_matches == 0)[0]
            for idx in unmatched_gt:
                confusion_matrix[gt_labels[idx], -1] += 1

        return confusion_matrix


    def _compute_iou(self, mask1, mask2):
        # Intersection
        intersection = np.logical_and(mask1, mask2)
        # Union
        union = np.logical_or(mask1, mask2)
        # Compute IoU
        iou = np.sum(intersection) / np.sum(union)
        return iou


    def _compute_pixel_level_confusion_matrix(self, gt, predictions, confidence_threshold):
        #########################################################################
        #########################################################################
        def compute_gt_map(gt_masks, gt_labels, bg_label):
            gt_map = np.full(gt_masks[0].shape, bg_label)
            for mask, label in zip(gt_masks, gt_labels):
                gt_map[mask == 1] = label
            return gt_map
        #########################################################################
        #########################################################################

        # get image ground truth
        gt_masks = gt.gt_masks.numpy()
        gt_labels = gt.gt_classes.numpy()
        gt_map = compute_gt_map(gt_masks=gt_masks, gt_labels=gt_labels, bg_label=len(self.classes))

        # Filter the instances using the input confidence threshold
        predictions = predictions[predictions.scores > confidence_threshold]
        # get predictions
        pred_masks = predictions.pred_masks.numpy()
        pred_labels = predictions.pred_classes.numpy()

        # Turn instance segmentation to semantic
        ## Create empty semantic map for each class
        num_classes = len(self.classes) + 1  # no. of classes + 1 background
        semantic_maps = [np.zeros(gt_map.shape, dtype=np.uint64) for i in range(num_classes)]
        ## Fill in semantic class maps with predictions
        for mask, label in zip(pred_masks, pred_labels):
            semantic_maps[label][mask == 1] = 1
            semantic_maps[-1] = np.bitwise_or(semantic_maps[-1], semantic_maps[label]) # Semantic background map is create from combined predictions
        semantic_maps[-1] = np.logical_not(semantic_maps[-1]) # Invert the background semantic map

        # Create image-level confusion matrix
        confusion_matrix = np.zeros((num_classes, num_classes), dtype=np.uint64)
        for gt_class in range(num_classes):
            for label, predicted_map in enumerate(semantic_maps):
                    confusion_matrix[gt_class, label] += np.sum((gt_map == gt_class) & (predicted_map == 1))
        return confusion_matrix


    def draw_predictions(self, draw_text=True, text_scale = 1, confidence=0.5, data_loader_num_worker=1):
        if not os.path.exists(os.path.join(self.test_cfg.OUTPUT_DIR, f"predicted_images_{self.test_cfg.DATASETS.TEST[0]}")):
            os.makedirs(os.path.join(self.test_cfg.OUTPUT_DIR, f"predicted_images_{self.test_cfg.DATASETS.TEST[0]}"))
        if self.predictor:
            data_loader = build_detection_test_loader(
                self.test_cfg,
                self.test_cfg.DATASETS.TEST[0],
                mapper = COCOInstanceNewBaselineDatasetMapper(
                    is_train=True,
                    image_format=self.test_cfg.INPUT.FORMAT,
                    tfm_gens = self.data_augmentations
                ),
                batch_size = 1,
                num_workers = data_loader_num_worker
            )

            data_loader_itr = iter(data_loader)
            itr_counter = 0

            for input in data_loader_itr:
                print(f"Drawing predictions: Processing image {input[0]['file_name']} {itr_counter}/{len(data_loader_itr)}")
                image = input[0]["image"].permute(1, 2, 0).cpu().detach().numpy()
                
                
                with autocast():
                    predictions = self.predictor(image)
                pred_instances = predictions['instances'].to('cpu')
                pred_instances = pred_instances[pred_instances.scores > confidence]
                pred_masks = pred_instances.pred_masks.numpy()
                pred_labels = pred_instances.pred_classes.numpy()
                pred_labels = [self.classes[id] for id in pred_labels]
                pred_scores = pred_instances.scores.numpy()
                

                # image.copy() because using image as an input directly doesn't work, dont ask why...
                img_with_predictions = Visualizer.draw(image=image.copy(),
                                                masks=pred_masks,
                                                scores=pred_scores,
                                                labels=pred_labels,
                                                draw_text=draw_text,
                                                text_scale=text_scale
                                                )

                file_path = os.path.join(self.test_cfg.OUTPUT_DIR, f"predicted_images_{self.test_cfg.DATASETS.TEST[0]}", f"{itr_counter}.png")
                # opencv expects the BGR format as an input, so in order to save it as an RGB, convert RGB to BGR before saving
                cv2.imwrite(file_path, img_with_predictions[:, :, ::-1])
                itr_counter += 1
        else:
            coco_gt = COCO(os.path.join(self.test_cfg.OUTPUT_DIR, f"{self.test_cfg.DATASETS.TEST[0]}_coco_format.json"))
            coco_dt = coco_gt.loadRes(os.path.join(self.test_cfg.OUTPUT_DIR, "coco_instances_results.json"))

            itr_counter = 0
            for image_id in coco_dt.getImgIds():
                # Load image metadata and the image itself
                img_info = coco_gt.loadImgs(image_id)[0]
                img_path = os.path.join(self.test_cfg.DATASETS.ROOT_DIR, img_info['file_name'])  # Update to your image folder
                img = cv2.imread(img_path)[..., ::-1]  # Convert BGR to RGB for Matplotlib
    
                # Display the image
                plt.figure(figsize=(12, 8))
                plt.imshow(img)
                ax = plt.gca()

                # Get predictions for the image
                predictions = coco_dt.loadAnns(coco_dt.getAnnIds(imgIds=[image_id]))
                coco_dt.showAnns(predictions)

                plt.axis("off")
                file_path = os.path.join(self.test_cfg.OUTPUT_DIR, f"predicted_images_{self.test_cfg.DATASETS.TEST[0]}", f"{itr_counter}.png")
                plt.savefig(file_path)
                itr_counter += 1
      

    # compute_error_cases IS AN EXPERIMENTAL evaluation; it seems to work, but this should be tested more properly
    def compute_error_cases(self, IoU=0.5, confidence=0.5, data_loader_num_worker=1):
        #########################################################################
        #########################################################################
        def get_group_id_from_file_name(image_filename):
            paths = image_filename.split("/")
            group_name = [path for path in paths if "group_" in path]
            assert len(group_name) == 1, f"Something went wrong when extracting group name {group_name}"
            group_id = str(group_name[0])
            group_id = int(group_id.split("_")[1])
            return group_id
        #########################################################################
        #########################################################################

        data_loader = build_detection_test_loader(
            self.test_cfg,
            self.test_cfg.DATASETS.TEST[0],
            mapper = COCOInstanceNewBaselineDatasetMapper(
                is_train=True,
                image_format=self.test_cfg.INPUT.FORMAT,
                tfm_gens = self.data_augmentations
            ),
            batch_size = 1,
            num_workers = data_loader_num_worker
        )

        itr_counter = 0
        data_loader_itr = iter(data_loader)

        out = os.path.join(self.test_cfg.OUTPUT_DIR, f"errors_{self.test_cfg.DATASETS.TEST[0]}.yaml")
        #file clean up from the last time, otherwise we just append to an already existing file
        if os.path.exists(out):
            os.remove(out)

        for input in data_loader_itr:
            print(f"Computing error cases: Processing image {input[0]['file_name']} {itr_counter}/{len(data_loader_itr)}")
            img = input[0]["image"].permute(1, 2, 0).cpu().detach().numpy()
            #prepare predictions
            with autocast():
                predictions = self.predictor(img)
            pred_instances = predictions['instances'].to('cpu')
            pred_instances = self._sort_predictions(pred_instances)
            # Filter the instances using the input confidence threshold
            pred_instances = pred_instances[pred_instances.scores > confidence]
            # get predictions
            pred_masks = pred_instances.pred_masks.numpy()
            pred_labels = pred_instances.pred_classes.numpy()

            # get image ground truth
            gt = input[0]["instances"]
            gt_masks = gt.gt_masks.numpy()
            gt_labels = gt.gt_classes.numpy()
            gt_matches = np.zeros(len(gt_masks))

            # create output dictionary
            error_cases = {
                'group': get_group_id_from_file_name(input[0]["file_name"]),
                #'image': input[0]["file_name"],
                'image_id': input[0]["image_id"],
                'confidence_threshold': confidence,
                'IoU_threshold': IoU,
                'TP': 0,
                'FPdd': 0, #+1 for any double detection
                'FPbg': [], #saved as the list of IoUs that did not pass the IoU check
                'FP': 0,
                'misclassifications': 0, #+1 for misclassification
                'FN': 0,
                'recall': 0,
                'precision': 0,
                'f1_score': 0
            }

            image_error_summary = {
                input[0]["file_name"]: error_cases,
            }

            for mask, label in zip(pred_masks, pred_labels):
                IoU_with_gt = []
                for gt_mask in gt_masks:
                    IoU_with_gt.append(self._compute_iou(mask, gt_mask))
                max_IoU = max(IoU_with_gt)

                if max_IoU > IoU:
                    idx = IoU_with_gt.index(max_IoU)
                    if label == gt_labels[idx] and gt_matches[idx] != 1:
                        error_cases['TP'] += 1
                        gt_matches[idx] = 1
                    elif label == gt_labels[idx] and gt_matches[idx] == 1:
                        error_cases['FPdd'] += 1
                    elif label != gt_labels[idx]:
                        error_cases['misclassifications'] += 1
                else:
                    error_cases['FPbg'].append(float(np.round(max_IoU, 3)))

            unmatched_gt = np.where(gt_matches == 0)[0]
            error_cases['FN'] = len(unmatched_gt)
            error_cases['recall'] = error_cases['TP']/len(gt_masks)
            error_cases['FP'] = error_cases['FPdd'] + len(error_cases['FPbg'])
            error_cases['precision'] = error_cases['TP']/(error_cases['TP']+error_cases['FP'])
            error_cases['f1_score'] = (2*error_cases['precision']*error_cases['recall'])/(error_cases['precision']+error_cases['recall'])

            #image_error_summary["error_summary"] = error_cases

            #if len(error_cases['FPbg']) != 0:
            #    print(error_cases['FPbg'])
            #    print(type(error_cases['FPbg']))
            #    print(type(error_cases['FPbg'][0]))

            with open(out, 'a') as file:
                yaml.dump(image_error_summary, file)

            itr_counter += 1


    def compute_pr_curves(self, IoU_thresholds=None, confidence_thresholds=None, plot_IoU_PR = True, plot_conf_PR = True, data_loader_num_worker=1):
        #########################################################################
        #########################################################################
        def compute_TP_FP_FN(predictions, gt, confidence, IoU):
            _predictions = copy.deepcopy(predictions)
            _predictions = _predictions[_predictions.scores > confidence]
            # get predictions
            pred_masks = _predictions.pred_masks.numpy()
            pred_labels = _predictions.pred_classes.numpy()

            # get image ground truth
            _gt = copy.deepcopy(gt)
            gt_masks = _gt.gt_masks.numpy()
            gt_labels = _gt.gt_classes.numpy()
            gt_matches = np.zeros(len(gt_masks))

            #tp_fp_fn = np.asarray([0, 0, 0])
            tp_fp_fn = {
                'TP': 0,
                'FP': 0,
                'FN': 0,
            }
            for mask, label in zip(pred_masks, pred_labels):
                IoU_with_gt = []
                for gt_mask in gt_masks:
                    IoU_with_gt.append(self._compute_iou(mask, gt_mask))
                max_IoU = max(IoU_with_gt)

                if max_IoU > IoU:
                    idx = IoU_with_gt.index(max_IoU)
                    if label == gt_labels[idx] and gt_matches[idx] != 1: #TP
                        #tp_fp_fn[0] += 1
                        tp_fp_fn['TP'] += 1
                        gt_matches[idx] = 1
                    else:
                        #tp_fp_fn[1] += 1  #FP -> either a double detection or a misclassification
                        tp_fp_fn['FP'] += 1
                else:
                    #tp_fp_fn[1] += 1 #FP -> prediction was made, but a prediction doesn't match any annotations
                    tp_fp_fn['FP'] += 1

            unmatched_gt = np.where(gt_matches == 0)[0]
            for idx in unmatched_gt:
                #tp_fp_fn[2] += 1 #FN -> fish was not detected given the confidence and IoU level
                tp_fp_fn['FN'] += 1

            if tp_fp_fn['TP'] == 0:
                print(tp_fp_fn)
                #exit()
            return tp_fp_fn
        #########################################################################
        #########################################################################
        def plot_PR_curve(interval, thresholds, results, title, output):
            plt.figure(dpi=200)
            for threshold in thresholds:
                precision, recall = [], []
                for result in results:
                    if result[interval] == threshold:
                        precision.append(result['TP']/(result['TP']+result['FP']))
                        recall.append(result['TP']/(result['TP']+result['FN']))
                print(f"interval {interval} threshold {threshold} P {len(precision)}, R {len(recall)} P:{precision} R:{recall}")
                _color = [np.random.uniform() for _ in range(3)]
                plt.scatter(recall, precision, color=(0, 0, 0), marker='o', facecolors='none')
                plt.plot(recall, precision, color=_color, label=f"{interval}_{threshold}")
            plt.legend()
            plt.title(title)
            #plt.xticks([.1, .2, .3, .4, .5, .6, .7, .8, .9, 1])
            #plt.yticks([.1, .2, .3, .4, .5, .6, .7, .8, .9, 1])
            plt.ylabel("Precision")
            plt.xlabel("Recall")
            plt.tight_layout()
            plt.savefig(output)
            print(f"Saved PR curve to {output}")
        #########################################################################
        #########################################################################

        data_loader = build_detection_test_loader(
            self.test_cfg,
            self.test_cfg.DATASETS.TEST[0],
            mapper = COCOInstanceNewBaselineDatasetMapper(
                is_train=True,
                image_format=self.test_cfg.INPUT.FORMAT,
                tfm_gens = self.data_augmentations
            ),
            batch_size = 1,
            num_workers = data_loader_num_worker
        )
        data_loader_itr = iter(data_loader)

        results = []
        indexing_map = []
        #confidence_interval = np.round(np.arange(0, 1, 0.1), 2)[::-1]
        confidence_interval = ([0, 0.2, 0.4, 0.6, 0.8, 0.9])[::-1]
        IoU_interval = ([0.5, 0.6, 0.7, 0.8, 0.9, 0.95])[::-1]
        #IoU_interval = np.round(np.arange(0, 1, 0.1), 2)[::-1]
        for conf in confidence_interval:
            for IoU in IoU_interval:
                results.append({
                    'confidence': conf,
                    'IoU': IoU,
                    'TP': 0,
                    'FP': 0,
                    'FN': 0,
                })
                indexing_map.append((conf, IoU))

        itr_counter = 0
        for input in data_loader_itr:
            print(f"Computing PR curve: Processing image {input[0]['file_name']} {itr_counter}/{len(data_loader_itr)}")
            img = input[0]["image"].permute(1, 2, 0).cpu().detach().numpy()
            #prepare predictions
            with autocast():
                predictions = self.predictor(img)
            gt_instances = input[0]["instances"]
            pred_instances = predictions['instances'].to('cpu')
            pred_instances = self._sort_predictions(pred_instances)

            for conf in confidence_interval:
                for IoU in IoU_interval:
                    print(f"    Computing TP, FP, FN for confidence {conf} and IoU of {IoU}")
                    tp_fp_fn = compute_TP_FP_FN(predictions=pred_instances, gt=gt_instances, confidence=conf, IoU=IoU)
                    idx = indexing_map.index((conf, IoU))
                    results[idx]['TP'] += tp_fp_fn['TP']
                    results[idx]['FP'] += tp_fp_fn['FP']
                    results[idx]['FN'] += tp_fp_fn['FN']

            itr_counter += 1


        if plot_conf_PR:
            if confidence_thresholds is None:
                confidence_thresholds = confidence_interval
            plot_PR_curve(interval="confidence",
                      thresholds=confidence_thresholds,
                      results=results,
                      title=f"IoUs {IoU_interval}",
                      output=os.path.join(self.test_cfg.OUTPUT_DIR, f"pr_confidence_{confidence_thresholds}")
                      )

        if plot_IoU_PR:
            if IoU_thresholds is None:
                IoU_thresholds = IoU_interval

            plot_PR_curve(interval="IoU",
                        thresholds=IoU_thresholds,
                        results=results,
                        title=f"confidences {confidence_interval}",
                        output=os.path.join(self.test_cfg.OUTPUT_DIR, f"pr_IoU_{IoU_thresholds}.png")
                        )


def parse_arguments():
    parser = argparse.ArgumentParser()
    # Define the argument for the YAML file path
    parser.add_argument('-y', '--yaml', type=str, help='Path to the YAML file.')
    parser.add_argument('--coco_results', action='store_true', help='Use results saved in coco format for evaluation.')
    parser.add_argument('--density', choices=['low', 'high', 'All'], type=str, help='Run the test on images of low density, high density or all.')
    parser.add_argument('--draw', action='store_true', help='Draw predictions.')

    cm_group = parser.add_argument_group('Confusion Matrix')
    cm_group.add_argument('--cm', action='store_true', help='Compute confusion matrix.')
    cm_group.add_argument('--level', choices=['instance_level', 'pixel_level'], default='instance_level', help='Level of the confusion matrix.')
    cm_group.add_argument('--compute_error_cases', action='store_true', help='Compute error cases.')
    cm_group.add_argument('--IoU', default=0.5, type=float, help='IoU threshold for the confusion matrix.')
    cm_group.add_argument('--confidence', default=0.5, type=float, help='Confidence threshold for the confusion matrix.')
    cm_group.add_argument('--normalize', default=True, type=bool, help='Normalize the confusion matrix.')
    cm_group.add_argument('--data_loader_num_worker', default=1, type=int, help='Number of workers for the data loader.')
    cm_group.add_argument('--compute_FP_FN', default=True, type=bool, help='Compute False Positives and False Negatives.')

    pr_group = parser.add_argument_group('PR Curves')
    pr_group.add_argument('--compute_pr_curves', action='store_true', help='Compute PR curves.')
    pr_group.add_argument('--IoU_thresholds', nargs='+', type=float, help='IoU thresholds for the PR curves.')
    pr_group.add_argument('--confidence_thresholds', nargs='+', type=float, help='Confidence thresholds for the PR curves.')
    pr_group.add_argument('--plot_IoU_PR', action='store_true', help='Plot PR curves for IoUs.')
    pr_group.add_argument('--plot_conf_PR', action='store_true', help='Plot PR curves for confidences.')
  
    args = parser.parse_args()
    return args


def main():
    '''
    Main issues with this code currently:
    This expects a yaml file that has been produced during training. Meaning it is not readily compatible with someone who downloaded the model
    weights from e.g. github and wants to test them.
    '''
    setup_logger()
    args = parse_arguments()
    training_yaml = yaml.safe_load(open(args.yaml))

    # DATASET SETUP
    dataset = Autofish.instance_from_yaml(training_yaml)
    dataset.register_all_autofish_splits()

    # COMPILE AUGMENTATIONS
    train_augmentations, test_augmentations =augmentations.get_augmentations(training_yaml)

    # DETECTRON2 CONFIG SETUP
    import train
    cfg = train.get_train_config(yaml_cfg=training_yaml)
    model_tester = Tester(yaml=training_yaml, train_cfg=cfg, data_augmentations=test_augmentations, coco_results=args.coco_results)
    setup_logger(output=cfg.OUTPUT_DIR, distributed_rank=comm.get_rank(), name="mask2former")

    model_tester.compute_ap(data_loader_num_worker=training_yaml['dataloader_workers'], output_suffix=args.density)

    if args.cm:
        model_tester.compute_confusion_matrix(
            level=args.level,
            compute_FP_FN=args.compute_FP_FN,
            IoU=args.IoU,
            confidence=args.confidence,
            data_loader_num_worker=args.data_loader_num_worker,
            normalize=args.normalize
        )

    if args.compute_error_cases:
        model_tester.compute_error_cases(
            IoU=args.IoU,
            confidence=args.confidence,
            data_loader_num_worker=args.data_loader_num_worker
        )

    if args.draw:
        model_tester.draw_predictions(
            draw_text=True,
            text_scale = 0.3,
            confidence=0.75,
            data_loader_num_worker=10
        )


if __name__=="__main__":
    main()
