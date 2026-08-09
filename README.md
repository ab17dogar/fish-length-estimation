# Fish Length Estimation — AutoFish REG baseline, improved, and a VFM study

Reproduction and extension of the **AutoFish** CNN baseline for estimating fish
length (in cm) from an RGB fish crop plus its bounding box. The published
baseline (REG) uses a MobileNetV2 encoder feeding an MLP regression head.

This repository (1) reproduces that baseline, (2) **improves it** with a better
training recipe, and (3) investigates whether a **Vision Foundation Model (VFM)**
encoder (DINOv2 / DINOv3) can do better.

**Status / headline finding.** So far, **VFMs do not beat the CNN baseline on this
task** — frozen VFM features underperform even a bounding-box-only regressor, and
fine-tuning a large ViT overfits the dataset's ~454 unique fish. Meanwhile an
improved MobileNetV2 recipe **beats the published number (0.82 → 0.68 cm)**. Work
is **ongoing**: we are testing further approaches (higher input resolution,
partial fine-tuning, DINOv3, unmasked context crops, ensembling) both to give the
VFM every fair chance and to build a strong, well-supported claim about *why* a
VFM does not improve CNN regression for fine-grained length estimation here.

## Current results (test groups, ground-truth masks — the `REG^gt` condition)

MAE in cm, lower is better. Combined = separated + touching test sets.

| Model / control | separated | touching | **combined** | notes |
|---|---|---|---|---|
| **MobileNetV2, improved recipe (ours)** | **0.544** | **0.818** | **0.681** | **beats the baseline** |
| REG^gt — AutoFish paper (target) | 0.67 | 0.96 | 0.82 | published baseline |
| MobileNetV2, faithful reproduction (bs16) | 0.82 | 1.06 | 0.94 | validates the pipeline |
| Mask-geometry regressor (GBM, no image) | 0.75 | 1.13 | 0.94 | hand-crafted geometry ceiling |
| Bbox-only regressor (GBM, no image) | 0.73 | 1.19 | 0.96 | the bounding-box floor |
| DINOv2 ViT-S/14, frozen (CLS) | 1.41 | 1.52 | 1.46 | VFM frozen < bbox floor |
| DINOv2 ViT-S/14, frozen (CLS+patch) | 1.81 | 1.91 | 1.86 | richer readout, still worse |
| DINOv2 ViT-S/14, fine-tuned | 1.96 | 2.04 | 2.00 | overfits |
| ResNet50 / ConvNeXt-T, fine-tuned | — | — | (overfit) | bigger CNNs overfit too |

With **multi-sample aggregation** (median predicted length over multiple images of
the same fish — the paper's own "with IDs" scenario), the improved model reaches
**≈0.46 cm at 5 images/fish and ≈0.40 cm at 20**, near the ground-truth precision
limit (lengths were measured to the nearest 5 mm).

**Two takeaways that shape the story.** (a) A regressor on the 4 bounding-box
coordinates *alone* already reaches 0.96 cm, and mask geometry caps at 0.94 —
so the paper's 0.82 sits *below* any hand-crafted geometry, i.e. the learned image
model contributes real, sub-geometry signal. (b) Frozen VFM features land *above*
the bbox floor and fine-tuning overfits, so on this small, geometry-dominated task
a compact supervised CNN remains the better encoder. Establishing this rigorously
is the current focus.

## What we changed vs. the authors' code

The `main` branch holds the **faithful baseline** (authors' code essentially
unmodified). This `vfm-extension` branch extends it additively so a VFM/CNN
encoder can be compared under a stronger, still-honest recipe:

- `cnn/Model.py` — DINOv2, DINOv3 and larger supervised-CNN (ResNet/ConvNeXt)
  encoders alongside MobileNetV2, with `cls` / `cls+patch` readouts selected via
  the `MODEL_BACKEND` config string (so evaluation reconstructs the model with no
  eval-side changes).
- `cnn/FishLengthDataset.py`, `cnn/train.py`, `eval_length_estimators.py` —
  new **config keys** (`CACHE_SIZE_GIB`, `RESIZE_TO`, `NUM_WORKERS`, `DROP_LAST`);
  defaults preserve the authors' behaviour.
- `cnn/train_vfm.py` — a stronger training loop: AdamW + layer-wise LR decay,
  cosine schedule + warmup, bf16 mixed precision (fits fine-tuning in ~2 GB of
  shared GPU), gradient clipping, optional length-preserving flip augmentation,
  **BatchNorm recalibration before evaluation**, and partial fine-tuning
  (`FREEZE_BLOCKS_BELOW`). It writes the same on-disk layout as `train.py`, so
  `eval_length_estimators.py` evaluates its checkpoints unchanged.

The improvement over the published 0.82 comes from the training recipe — most
notably **batch size 32 and BatchNorm-statistics recalibration** before eval;
isolating each factor's contribution is part of the ongoing work.

## Repository layout (extension additions)

```
autofish_training_release/length_estimation/
  cnn/
    Model.py                     # MobileNetV2 / DINOv2 / DINOv3 / ResNet / ConvNeXt + MLP head
    train.py                     # authors' training loop (+ config keys)
    train_vfm.py                 # stronger training loop (AMP, LLRD, BN recalib, partial FT)
    FishLengthDataset.py         # dataset (+ cache-size / resolution config keys)
    configs/
      paper.cfg, paper16.cfg     # baseline (batch 32) and faithful batch-16 reproduction
      extension/                 # {frozen, fine-tuned} x {MobileNetV2, DINOv2, ResNet, ConvNeXt}
      le_ext/                    # frozen label-efficiency sweep configs
      generate_extension_configs.py, generate_le_extension.py
    output/                      # checkpoints, eval CSVs, metrics (gitignored)
  eval_length_estimators.py      # evaluation (+ resolution config key)
  compute_metrics.py             # MAE/MAPE/RMSE/R2/bias per subset/species/fish-id
  bbox_only_control.py           # length from bbox coords only (the 0.96 floor)
  mask_geometry_control.py       # length from mask geometry only (the 0.94 ceiling)
  aggregation_curve.py           # MAE vs. #images aggregated per fish
  collate_results.py             # cross-run comparison table
  retarget_data_path.sh          # point configs at a local dataset path
  run_extension.sh, run_gpu_queue.sh   # resumable train -> eval -> metrics runners
ENVIRONMENT_NOTES.md             # running without Docker (venv), every deviation documented
AutoFish_MACVI_WACV25_vbn.pdf    # reference paper
```

## Data

The AutoFish images (~11 GB) are **not** tracked here. Download from
[huggingface.co/datasets/vapaau/autofish](https://huggingface.co/datasets/vapaau/autofish)
(or [vap.aau.dk/autofish](https://vap.aau.dk/autofish/)). The COCO-format
`annotations.json` carries per-instance `length`, `group`, `fish_id`, `side_up`
and species. Splits are by group (test = groups 10/14/20/21/22), so no fish
appears in both train and test.

## Environment

The reference environment is the authors' CUDA Docker image. Where Docker is
unavailable this project runs in a plain Python venv; **`ENVIRONMENT_NOTES.md`
documents every deviation** (package pins, the `numpy < 2` requirement, a
`stocaching` cache-race fix, etc.). After downloading the data, point the configs
at it once:

```
bash retarget_data_path.sh /abs/path/to/autofish/annotations.json
```

## Training & evaluation

**Baseline (authors' code)**, run from `length_estimation/cnn/`:

```
python train.py --config configs/paper.cfg          # -> output/cnn-paper/model.pt
```

**Extension / improved recipe**, via the runners (train -> eval -> metrics,
resumable), run from `length_estimation/`:

```
GT=/abs/path/annotations.json PYTHON=python \
  bash run_extension.sh x-mnet-b32           # the improved MobileNetV2 (0.68 cm)
```

Evaluation runs from `length_estimation/` (the eval script does
`sys.path.append("cnn")`). It uses **ground-truth masks** (`--gt_path`), so the
comparison target is the paper's `REG^gt` (0.82 combined; the widely quoted
0.62/1.38 cm are the *predicted-mask* `REG^pd` condition — a different setup).
`compute_metrics.py`, `collate_results.py` and `aggregation_curve.py` produce the
per-subset / per-species metrics, the cross-run table, and the aggregation curve.

## Attribution

Training code and dataset originate from the AutoFish project
([vap.aau.dk/autofish](https://vap.aau.dk/autofish/); paper
`AutoFish_MACVI_WACV25_vbn.pdf`). The code under `autofish_training_release/` is
the authors' work, extended additively as described above. All added tooling
(`train_vfm.py`, the controls, aggregation, collation, runners, configs) is part
of this study.
