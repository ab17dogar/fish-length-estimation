# Length estimation #

---

## Skeletonization-based approach

### 1) Calibration (once)
The camera calibration calculate both the intrinsic parameters of the camera along with homographies mapping between the camera and  each of the checkerboards. This information is used in the skeletonization-based approach in order to correct for e.g. lens distortion but also to relate pixel lengths to centimeters. This calibration only needs to be done once.

- Run the camera calibration (may take upto 15 mins):
```
cd autofish_training/length_estimation/
python camera_calibration/run_camera_calibration.py \
       --autofish_dir /path/to/autofish/autofish_dataset_release/autofish/
```
The output should be saved in ``camera_calibration/output/undistort-True/`` as .json-files.


### 2) Inference and evaluation
The skeletonization-based approach can be used by executing the ``eval_length_estimators.py``-script while specifying the ``--cam_cal_path`` argument (should point to the output from the previous calibration). The approach can be evaluated on the groundtruth masks provided with the dataset by just specifying the ``--gt_path`` argument. Additionaly, the approach can also be evaluated on actual predicted masks from the Mask2Former network by specifying the ``--pred_path`` argument (note: the ``--gt_path`` argument must still be specified for evaluation purposes).

- Run the evaluation script:
```
python eval_length_estimators.py \
       --gt_path /workspace/autofish_dataset/annotations.json \
       --pred_path /path/to/mask2former_results/coco_instances_results.json \
       --cam_cal_path camera_calibration/output/undistort-True/
```
The output can be found in ``camera_calibration/output/undistort-True/eval/from-[gt|pred]/eval-output.csv`` (depending on if predicted masks were provided or not)

---

## CNN-based approach

### 1) Training (once)
In order to train the CNN for length regression a configuration file is required. A default configuration file is located at: ``cnn/configs/default.cfg`` and includes all the parameters and options used to train the model reported in the paper (with the exception of reducing the number of epochs to 10 and batch size to 16). Make sure to update the paths specified in the configuration file (should not be necessary if running in Docker). 

- Start training the CNN-based length estimation approach:
```
python cnn/train.py --config cnn/configs/default.cfg
```
The trained model should be saved as ``cnn/output/cnn-default/model.pt`` - this directory will also include other information such as the training and validation loss.


### 2) Inference and evaluation
The CNN-based approach can be used by executing the ``eval_length_estimators.py``-script while specifying the ``--cnn_model_path`` argument (should point to the previously trained model). The approach can be evaluated on the groundtruth masks provided with the dataset by just specifying the ``--gt_path`` argument. Additionaly, the approach can also be evaluated on actual predicted masks from the Mask2Former network by specifying the ``--pred_path`` argument (note: the ``--gt_path`` argument must still be specified for evaluation purposes).

- Run the evaluation script:
```
python eval_length_estimators.py \
       --gt_path /workspace/autofish_dataset/annotations.json \
       --pred_path /path/to/mask2former_results/coco_instances_results.json \
       --cnn_model_path cnn/output/cnn-default/model.pt
```
The output can be found in ``cnn/output/cnn-default/eval/model/from-[gt|pred]/eval-output.csv`` (depending on if predicted masks were provided or not)

