# VFM Encoder Variant — Experiments Guide

Vision Foundation Model (VFM) variant of the AutoFish length regressor and the
experiments that answer the seminar's research question:

> How well does the established deep-learning baseline perform for fish length
> estimation, and does replacing its encoder with a Vision Foundation Model
> improve performance, in particular when labeled data are limited?

Everything runs in the authors' Docker container (see the top-level
`autofish_training_release/README.md`), dataset mounted at
`/workspace/autofish_dataset/`. Train from `length_estimation/cnn/`; evaluate
and analyze from `length_estimation/`.

---

## 1. What changed (and what did not)

The single source edit is in [`cnn/Model.py`](cnn/Model.py): a `DINOv2Encoder`
module was added and the backbone is selected by the `MODEL_BACKEND` config key.
When `MODEL_BACKEND = mobilenet_v2` the code path is byte-for-byte the original
baseline; when it starts with `dinov2` the DINOv2 ViT is used instead.
`train.py`, `FishLengthDataset.py`, and `eval_length_estimators.py` are the
authors' **unmodified** code.

**Prediction target.** One continuous value per fish instance — its length in
centimeters (the per-instance `length` annotation) — regressed from the
RGB-masked square bounding-box crop (224×224) plus the 4 normalized bbox
coordinates. Identical target and inputs for both encoders.

**VFM encoder.** DINOv2 ViT-S/14 via `torch.hub`, pinned to a fixed commit for
reproducible architecture/checkpoint keys (`trust_repo=True` so a `docker run`
session does not hang on a trust prompt). It returns the CLS-token embedding
(384-d), mirroring MobileNetV2's pooled feature. 224 is a multiple of the patch
size 14, so no resize change is needed. Available backbones (`MODEL_BACKEND`):
`dinov2_vits14` (default), `dinov2_vitb14`, `dinov2_vitl14`, `dinov2_vitg14`.

**Two asymmetries that swapping the encoder necessarily introduces** (stated
plainly rather than hidden behind an "everything else is identical" claim):

1. **Input scaling.** `FishLengthDataset` feeds raw `[0,1]` tensors (ToTensor +
   Resize, no normalization). DINOv2 was pretrained with ImageNet normalization,
   which is applied *inside* the encoder; the MobileNetV2 baseline consumes the
   raw `[0,1]` input as in the original code. So the two encoders see
   differently-scaled inputs — the correct, constraint-respecting choice, but a
   real difference.
2. **Head input width.** The head's first `Linear` is `384+4` (DINOv2) vs
   `1280+4` (MobileNetV2); its design (widths, BN, depth) is otherwise identical.

Everything else — data, splits, RGB-masked bbox crop, normalized-bbox input,
Adam lr 1e-3, L1 loss, batch 32, 200 epochs, color-only augmentation — matches.

---

## 2. A controlled comparison (2×2 matrix)

"Does a VFM encoder help?" must not confound the *encoder* with the *training
regime*. So four configs form a `{MobileNetV2, DINOv2} × {frozen, fine-tuned}`
matrix (all runnable with **no** code edit):

| Config | Encoder | Regime | Role |
|---|---|---|---|
| `configs/baseline_frozen.cfg` | MobileNetV2 | frozen | encoder-only **control** |
| `configs/paper.cfg` | MobileNetV2 | fine-tuned | baseline as published |
| `configs/vfm.cfg` | DINOv2 ViT-S/14 | frozen | **VFM variant (primary)** |
| `configs/vfm_finetune.cfg` | DINOv2 ViT-S/14 | fine-tuned | VFM ceiling |

- The clean, encoder-isolating answer is **`baseline_frozen` vs `vfm`** (same
  frozen-feature-extractor regime, only the encoder differs).
- **`paper` vs `vfm_finetune`** compares the two encoders when both are
  fine-tuned. Caveat: `train.py` fixes Adam lr to 1e-3 for all params, which is
  high for ViT fine-tuning — treat `vfm_finetune` as an as-is data point.
- `vfm.cfg` (frozen DINOv2) is the **primary** VFM result: frozen VFM features
  are the standard strong protocol and the setting most relevant to the
  limited-data question.

---

## 3. Run the experiments

### 3.1 Main comparison (full data)

```bash
cd length_estimation/cnn
python train.py --config configs/vfm_smoke.cfg          # optional pre-flight
for c in paper baseline_frozen vfm vfm_finetune; do
    python train.py --config configs/$c.cfg             # -> output/cnn-<name>/model.pt
done
```

Evaluate on the held-out test groups and print metrics (from `length_estimation/`):

```bash
cd length_estimation
for run in cnn-paper cnn-baseline-frozen cnn-vfm cnn-vfm-finetune; do
  python eval_length_estimators.py \
      --gt_path /workspace/autofish_dataset/annotations.json \
      --cnn_model_path cnn/output/$run/model.pt
  python compute_metrics.py --csv cnn/output/$run/eval/model/from-gt/eval-output.csv --tag $run
done
```

`compute_metrics.py` reports **MAE, MAPE, RMSE, R², median AE, and signed bias**,
overall / per subset (separated / touching / combined) / per species, plus the
**per-fish-id averaged** error (each fish is imaged ~40×; averaging views is the
lower-variance, deployment-relevant estimate the AutoFish authors also compute).
It writes a `metrics.json` next to each eval CSV.

**Reproduction target (evaluate on GT masks, i.e. `--gt_path` only).** The
AutoFish REG model on ground-truth masks (Table 4, `REG^gt`) is **MAE 0.82 cm**
combined (0.67 separated / 0.96 touching). The often-quoted 0.62 cm / 1.38 cm and
0.99 cm combined are `REG^pd`, i.e. on *predicted* Mask2Former masks — a
different, harder condition. Since the commands above use GT masks, compare
against `REG^gt = 0.82 cm`. To reproduce the end-to-end `REG^pd` numbers, also
pass `--pred_path <mask2former coco_instances_results.json>`; results then land
under `.../eval/model/from-pred/`.

### 3.2 Label-efficiency study

`configs/generate_label_efficiency.py` emits 39 configs: three arms
(`baseline_ft`, `baseline_frozen`, `vfm`) × the data ladder {1, 2, 4, 8, 15}
training groups. For each size < 15 it draws **3 different random group subsets**
(shared across arms, so comparisons are paired) — because varying `RANDOM_SEED`
alone would *not* change which groups are used (`train.py` reads `TRAIN_GROUPS`
verbatim), so the low-data variance must come from the subset draw, as in the
paper (which averaged 10 random subsets per size; 3 here for tractability, raise
`K_SUBSETS` for tighter bands). Subsets whose instance count would leave a
size-1 final batch (which crashes the head's BatchNorm1d) are rejected at
generation time.

```bash
cd length_estimation/cnn
for cfg in configs/label_efficiency/le-*.cfg; do python train.py --config "$cfg"; done
cd ..
for d in cnn/output/le-*; do
  python eval_length_estimators.py \
      --gt_path /workspace/autofish_dataset/annotations.json \
      --cnn_model_path "$d/model.pt"
  python compute_metrics.py --csv "$d/eval/model/from-gt/eval-output.csv" --tag "$(basename $d)"
done
python plot_label_efficiency.py --runs-dir cnn/output      # -> label_efficiency_curve.png
```

`plot_label_efficiency.py` reads every run's `metrics.json` and plots mean
combined-test MAE vs. #training-groups per arm, with a min–max band across
subsets. Hypothesis: with the regime held fixed (`baseline_frozen` vs `vfm`), the
DINOv2 features give a smaller gap to full-data performance at 1–4 groups —
i.e. better label efficiency. Comparing `baseline_frozen` vs `vfm` isolates the
encoder; `baseline_ft` is the paper-regime reference.

> Compute note: the frozen arms train only the ~0.89M head and are cheap; the
> fine-tuned `baseline_ft` arm is the expensive one. If compute is tight,
> prioritise the frozen-vs-frozen pair.

### 3.3 Failure-case & prediction-bias analysis

```bash
cd length_estimation
python analyze_predictions.py --csv cnn/output/cnn-vfm/eval/model/from-gt/eval-output.csv
# and likewise for cnn-paper / cnn-baseline-frozen to compare
```

`analyze_predictions.py` writes to `.../analysis/`: signed-bias tables per
species / subset / GT-length bin (reveals systematic over/under-estimation and
regression-to-the-mean), a predicted-vs-GT scatter and a signed-error-vs-length
scatter (coloured by species), and a `worst_N.csv` of the largest-error
instances (with `img_path`, species, subset, gt, pred, error, iou) for
qualitative inspection.

---

## 4. Task-description coverage

- **Prediction target documented** — §1 here and the project ODE report.
- **Data splits + preprocessing justified** — group-level splits prevent
  fish-level leakage; RGB-masked bbox crop preserves the pixel→cm relationship;
  no geometric augmentation for the same reason. Shared by both encoders.
- **Baseline reproduced** — `paper.cfg`, authors' unmodified code (target
  `REG^gt` MAE 0.82 cm on GT masks).
- **VFM variant, comparable pipeline** — `vfm.cfg`; only the encoder differs,
  with the two asymmetries in §1 stated explicitly.
- **Suitable regression metrics** — MAE/MAPE/RMSE/R²/median/bias + per-fish-id,
  per subset and species (`compute_metrics.py`).
- **Reduced-label study** — §3.2, three arms × data ladder × random subsets.
- **Failure cases & biases** — §3.3 (`analyze_predictions.py`).
- **Optional 2nd-dataset (North Sea) transfer** — zero-shot: convert that
  dataset to the same COCO length-annotation format, run
  `eval_length_estimators.py --gt_path <its annotations.json> --cnn_model_path
  cnn/output/cnn-vfm/model.pt`, then `compute_metrics.py`. Note
  `compute_metrics`'s subset split expects file names `.../<group>/NNNNN.png`
  with index ≤40 = separated; adjust or ignore the subset split if the second
  dataset does not follow that convention.
