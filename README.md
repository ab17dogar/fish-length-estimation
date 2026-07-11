# Fish Length Estimation

Reproduction and extension of the **AutoFish** CNN baseline for estimating fish
length (in cm) from RGB images with bounding boxes. The baseline uses a
MobileNetV2 encoder feeding an MLP regression head; the goal of this project is
to reproduce it, then compare against a Vision Transformer (DINOv2) encoder,
especially under limited labeled-data scenarios.

## Repository structure

```
autofish_training_release/       # AutoFish training code (authors' code; only cnn/Model.py extended)
  length_estimation/
    cnn/
      Model.py                   # MobileNetV2 OR DINOv2 (VFM) encoder + MLP head
      FishLengthDataset.py       # dataset / dataloader (unchanged)
      train.py                   # training entry point (unchanged)
      configs/                   # default/baseline/smoke/paper + vfm/vfm_smoke
        label_efficiency/        # generated reduced-label configs (baseline & VFM)
      output/                    # trained checkpoints (.pt) + loss curves (gitignored)
    eval_length_estimators.py    # evaluation (CNN + skeletonization approaches, unchanged)
    compute_metrics.py           # MAE/MAPE/RMSE/R2/bias per subset/species/fish-id
    analyze_predictions.py       # failure-case & prediction-bias analysis + plots
    plot_label_efficiency.py     # MAE-vs-training-data curve (baseline vs VFM)
    VFM_EXPERIMENTS.md           # VFM variant + full experiment/analysis guide
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

Since this evaluates on **ground-truth masks** (`--gt_path` only), the target is
the paper's `REG^gt` result — **MAE 0.82 cm** combined (0.67 separated / 0.96
touching; Table 4). The widely-quoted 0.62/1.38 cm (and 0.99 cm combined) are
`REG^pd`, on *predicted* Mask2Former masks — a different condition, reproducible
by additionally passing `--pred_path`. `compute_metrics.py` reports
MAE/MAPE/RMSE/R²/median/bias per subset (separated / touching / combined), per
species, and per fish id.

## VFM variant (encoder swap)

The extension replaces **only the encoder** — MobileNetV2 → a Vision Foundation
Model (DINOv2 ViT-S/14) — keeping the same MLP head, inputs, data, optimizer,
loss, and schedule. The single code change is in `cnn/Model.py` (selected by the
`MODEL_BACKEND` config key; the `mobilenet_v2` path is byte-for-byte the
original). To keep the comparison honest, the encoder and the training regime
are separated into a `{MobileNetV2, DINOv2} × {frozen, fine-tuned}` matrix
(`baseline_frozen.cfg`, `paper.cfg`, `vfm.cfg`, `vfm_finetune.cfg`); the clean
encoder-only comparison is frozen-vs-frozen (`baseline_frozen` vs `vfm`).

```
python train.py --config configs/vfm.cfg                # -> output/cnn-vfm/model.pt
```

The full experiment matrix, the reduced-label study (3 arms × data ladder ×
random subsets), and the failure/bias analysis are documented in
[`length_estimation/VFM_EXPERIMENTS.md`](autofish_training_release/length_estimation/VFM_EXPERIMENTS.md).

## Attribution

The training code and dataset originate from the AutoFish project
([vap.aau.dk/autofish](https://vap.aau.dk/autofish/)); see the accompanying paper
`AutoFish_MACVI_WACV25_vbn.pdf`. The scripts under `autofish_training_release/`
are the authors' code, used unmodified **except** for `cnn/Model.py`, which is
extended additively to allow a DINOv2 (VFM) encoder alongside the original
MobileNetV2. This repository also adds reproduction/VFM/label-efficiency configs
and evaluation/metric tooling (`compute_metrics.py`).
