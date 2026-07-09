# Fish Length Estimation

Reproduction and extension of the **AutoFish** CNN baseline for estimating fish
length (in cm) from RGB images with bounding boxes. The baseline uses a
MobileNetV2 encoder feeding an MLP regression head; the goal of this project is
to reproduce it, then compare against a Vision Transformer (DINOv2) encoder,
especially under limited labeled-data scenarios.

## Repository structure

```
autofish_training_release/       # AutoFish training code (pristine authors' code; runs in their CUDA Docker)
  length_estimation/
    cnn/
      Model.py                   # MobileNetV2 encoder + MLP head
      FishLengthDataset.py       # dataset / dataloader
      train.py                   # training entry point
      configs/                   # default / baseline / smoke / paper configs
      output/                    # trained checkpoints (.pt) + loss curves
    eval_length_estimators.py    # evaluation (CNN + skeletonization approaches)
    compute_metrics.py           # MAE / MAPE reporting
    camera_calibration/          # camera calibration utilities
  mask2former/                   # instance segmentation + classification
ODE_fish_length_estimation.md    # ODE protocol documentation
AutoFish_MACVI_WACV25_vbn.pdf    # reference paper
```

## Data

The AutoFish image dataset (~10 GB) is **not** tracked in this repository. Download
it from [vap.aau.dk/autofish](https://vap.aau.dk/autofish/) and place it under
`data/autofish/` (the path referenced by the training/eval configs).

## Training & evaluation

The baseline is reproduced with the **authors' unmodified training code** inside
their CUDA Docker container (see the AutoFish README for the Docker workflow).
The dataset mounts at `/workspace/autofish_dataset/`, so the configs need no path
edits. Configs live in `autofish_training_release/length_estimation/cnn/configs/`:

- `smoke.cfg` — 1-group / 1-epoch pre-flight sanity check
- `paper.cfg` — **full paper settings**: MobileNetV2 + MLP head, batch 32, 200
  epochs, Adam lr 1e-3, L1 loss, RGB-masked bbox crop @224 + normalized bbox,
  color augmentation only (paper Sec. 4.2.2 / Table 2)
- `baseline.cfg` — reduced 10-epoch / batch-16 quick reproduction
- `default.cfg` — the authors' original released config

Run the paper reproduction from inside the container. **Train** from
`length_estimation/cnn/`:

```
python train.py --config configs/paper.cfg           # -> output/cnn-paper/model.pt (+ paper.cfg copied alongside)
```

Then **evaluate** from `length_estimation/` (the eval script does
`sys.path.append("cnn")`, so it must run from this dir):

```
python eval_length_estimators.py \
    --gt_path /workspace/autofish_dataset/annotations.json \
    --cnn_model_path cnn/output/cnn-paper/model.pt     # test groups default to the paper's
# -> cnn/output/cnn-paper/eval/model/from-gt/eval-output.csv
python compute_metrics.py --csv cnn/output/cnn-paper/eval/model/from-gt/eval-output.csv
```

The paper reports the REG model at **MAE ≈ 0.99 cm** on the combined test set
(Table 4); this reproduction targets that figure. `compute_metrics.py` prints
MAE/MAPE per subset (separated / touching / combined) and per species.

## Attribution

The training code and dataset originate from the AutoFish project
([vap.aau.dk/autofish](https://vap.aau.dk/autofish/)); see the accompanying paper
`AutoFish_MACVI_WACV25_vbn.pdf`. The training/eval scripts under
`autofish_training_release/` are the authors' unmodified code; this repository
adds reproduction configs (`paper.cfg`, `smoke.cfg`, `baseline.cfg`) and
evaluation/metric tooling (`compute_metrics.py`).
