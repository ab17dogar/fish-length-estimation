# Mask2Former (Instance segmentation and classification) #

---

## Training

- Download the pre-trained weights for the different backbones for Mask2Former (R50 and SwinB). You can download them directly from the [Mask2Former ModelZoo](https://github.com/facebookresearch/Mask2Former/blob/main/MODEL_ZOO.md) or use the supplied script (recommended):
```
cd autofish_training/mask2former/weights
bash download_weights.sh
```
It might take a couple of minutes but it only has to be done once!

- Setup a configuration file. For an example see `autofish_training/configs/training_example_r50.yaml`

- Train a model using the configuration file:
   ```
   cd /workspace/autofish_training/mask2former/utils
   python train.py --yaml ../configs/training_example_r50.yaml
   ```
   the output is saved to: `autofish_training/mask2former/output/training_test`

## Inference
Run inference on group 10 in the dataset:
   ```bash
   cd /workspace/autofish_training/mask2former/utils
   python inference.py --yaml ../configs/training_example_r50.yaml -i /workspace/autofish_dataset/group\_10/jai/rgb/ -o ../output/training_test/inference_grp10 --confidence 0.0
   ```
   the output is saved to: `autofish_training/mask2former/output/training_test/inference_grp10`

## Test
Evaluate a trained model:
   ```
   cd /workspace/autofish_training/mask2former/utils
   python tester.py --yaml ../configs/training_example_r50.yaml
   ```
