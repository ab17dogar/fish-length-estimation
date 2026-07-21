# Project Handoff — Fish Length Estimation (AutoFish baseline + VFM encoder)

Self-contained context transfer. A new session should be able to pick up from
here without re-deriving anything. **Read this fully before touching the repo.**

---

## 1. The task

University of Rostock seminar, *Deep Learning for Maritime Vision Applications*.
Supervisor: Bohan Zhuang. Group: Abu Bakar, M. Shahman Butt, Laksh Jiwani.
The user's goal is explicitly **the highest possible grade**.

**Research question:** How well does an established deep-learning baseline
perform for fish length estimation, and does replacing its encoder with a Vision
Foundation Model (VFM) improve performance, *particularly when labeled data are
limited*? Optional extension: transfer to a second (North Sea) dataset — **not
released, so out of scope**.

**Required deliverables** (from `task_description.pdf`):
1. define + document the prediction target
2. prepare data splits and justify preprocessing
3. reproduce/approximate the baseline as closely as feasible
4. implement a VFM-based variant with a comparable pipeline
5. evaluate both with suitable regression metrics
6. study performance as labeled training data is reduced
7. analyze failure cases and prediction biases across samples/species
8. (optional) transfer to a second dataset

## 2. Sources

- **Paper:** Bengtson et al., "AutoFish: Dataset and Benchmark for Fine-Grained
  Analysis of Fish", WACVW 2025. PDF in repo root.
- **Official code:** `https://bitbucket.org/autofish/autofish_training_release`
  (cloned into `autofish_training_release/`, tracked directly in our git — it has
  **no** nested `.git`).
- **Dataset:** `https://huggingface.co/datasets/vapaau/autofish` — 1500 images,
  18,158 instance annotations, 25 groups, 454 unique fish. COCO-format
  `annotations.json` with per-instance `length` (cm), `group`, `fish_id`,
  `side_up`, `category_id` (species).

## 3. THE HARD CONSTRAINT (do not violate)

> User: *"do not change the codebase. just replace the cnn encoder with the vfm encoder."*

- The **only** source file edited is `length_estimation/cnn/Model.py` (additively).
- `train.py`, `FishLengthDataset.py`, `eval_length_estimators.py` are **verified
  byte-identical** to a fresh bitbucket clone. Keep them that way.
- New config files, analysis scripts, and docs are additive and are fine.
- Consequence: `train.py` hard-codes `optim.Adam(model.parameters())` (lr 1e-3,
  no per-group lr) and `.cuda()`. Do **not** "fix" these.

Also: the user requires **no Claude/AI mention** in commits — no co-author
trailer, no AI attribution. Commits are authored by `Abu Bakar
<abu.bakar@uni-rostock.de>`.

## 4. The baseline (what we reproduce)

MobileNetV2 (ImageNet-pretrained, final classifier layer stripped → 1280-d)
⊕ 4 normalized bbox coords → `Linear(1000)+BN+ReLU → Linear(500)+BN+ReLU →
Linear(1)`. L1 loss, Adam lr 1e-3, batch 32, 200 epochs.
Input: ground-truth-mask RGB-masked square bbox crop, resized to 224×224.
Color jitter only — **no geometric augmentation** (it would break the pixel→cm
relationship). Splits are by group (each fish lives in exactly one group, so no
fish-level leakage):

- train `[2,3,4,5,7,8,9,12,13,15,16,18,19,23,24]`
- val `[1,6,11,17,25]`
- test `[10,14,20,21,22]`

Image index within a group: `00001–00040` = **separated**, `00041–00060` =
**touching**; both = **combined**.

**Paper Table 4 (memorize — we got this wrong once):**

| | Separated | Touching | Combined |
|---|---|---|---|
| REG on **GT masks** (`REG^gt`) | 0.67 | 0.96 | **0.82 cm** |
| REG on **predicted masks** (`REG^pd`) | 0.62 | 1.38 | 0.99 cm |

Our eval uses `--gt_path` only (GT masks), so **the target is 0.82 cm**, not
0.99. The famous 0.62/1.38 abstract numbers are the predicted-mask condition.

## 5. What was implemented (the VFM variant)

`Model.py` gained a `DINOv2Encoder` and the backbone is chosen by the **existing**
`MODEL_BACKEND` config key — so `train.py`/dataset/eval need zero changes.
`MODEL_BACKEND=mobilenet_v2` → byte-for-byte original path.

`DINOv2Encoder` specifics (all deliberate):
- Loads DINOv2 via `torch.hub`, **pinned** to commit
  `facebookresearch/dinov2:7764ea0f912e53c92e82eb78a2a1631e92725fc8`, with
  `trust_repo=True` (prevents an interactive trust prompt hanging `docker run`,
  and freezes the architecture so checkpoint keys match across containers).
- Returns the **CLS-token embedding** (384-d for ViT-S/14) — the analogue of
  MobileNetV2's pooled feature. Dims mapped for vits14/vitb14/vitl14/vitg14.
- Applies **ImageNet normalization inside the encoder**, because
  `FishLengthDataset` feeds raw `[0,1]` tensors and must stay untouched.
- Overrides `.train()` so a **frozen** encoder stays in `eval()` mode → no
  drop-path stochasticity in the frozen features.
- 224 is a multiple of patch size 14, so no resize change was needed.

## 6. Key decisions and WHY (preserve these)

1. **Frozen DINOv2 is the primary VFM setting.** It's the standard strong VFM
   feature-extractor protocol, it's the setting most relevant to the
   limited-label question, and it works with the fixed lr 1e-3 (which trains the
   head fine but is too high to properly fine-tune a ViT — and we may not edit
   `train.py`). Framed as a *choice*, not a hard constraint.
2. **A 2×2 matrix avoids a confound.** Comparing `vfm.cfg` (frozen DINOv2)
   against `paper.cfg` (fine-tuned MobileNetV2) changes **two** variables
   (encoder *and* regime). So `baseline_frozen.cfg` (frozen MobileNetV2) and
   `vfm_finetune.cfg` were added. **The clean encoder-only answer is
   frozen-vs-frozen: `baseline_frozen` vs `vfm`.** Never claim "difference is
   attributable to the encoder alone" for `paper` vs `vfm`.
3. **Two asymmetries are unavoidable and must be stated honestly:** DINOv2 sees
   ImageNet-normalized input while the baseline sees raw `[0,1]`; and the head's
   first Linear is 384+4 vs 1280+4.
4. **Label-efficiency variance must come from the group SUBSET, not the seed.**
   `train.py` only calls `torch.manual_seed`, which changes init/shuffle/aug but
   **not** which groups are used (`TRAIN_GROUPS` is read verbatim). So the
   generator emits multiple random *subsets* per size, shared across arms
   (paired comparison).

## 7. Verified facts — do NOT re-derive

- **Runtime test passed** (local, torch 2.12/MPS): both encoder paths output
  `(B,1)`; frozen VFM → only 0.89M trainable (head), encoder 0 trainable and
  `encoder.training=False`; fine-tune → 22.9M trainable. Pinned hub ref loads
  and returns `(2,384)`.
- **Faithfulness verified** by diffing against a fresh bitbucket clone: only
  `Model.py` differs; its "removed" lines are just the original MobileNetV2 lines
  re-indented into the `else` branch.
- **Per-group instance counts** (embedded in the generator):
  `{1:640, 2:760, 3:600, 4:880, 5:880, 6:640, 7:560, 8:800, 9:960, 10:920,
  11:640, 12:640, 13:880, 14:840, 15:800, 16:720, 17:799, 18:600, 19:560,
  20:640, 21:679, 22:680, 23:560, 24:560, 25:920}`
- **BatchNorm size-1 risk checked:** `train.py` uses `drop_last=False`, and the
  head starts with `BatchNorm1d`, so a final batch of size 1 would crash
  training. All 39+4 configs were checked — **none** produce it; the generator
  rejects unsafe subsets.
- All analysis scripts were **tested end-to-end on synthetic eval CSVs**.

## 8. Repo state

GitHub: `https://github.com/ab17dogar/fish-length-estimation` (public, owner
`ab17dogar`). `data/` (11 GB) and `cnn/output/` are gitignored.

- `main` — `15588df` baseline reproduction (pristine authors' code + configs)
- `vfm-extension` — `70fde5b` ← **current working branch**, all VFM work
  - `5e6d3e9` VFM encoder swap + controlled experiment suite
  - `480f433` `run_all.sh` one-shot runner
  - `70fde5b` `run_all.sh` MPLBACKEND=Agg headless fix

**`vfm-extension` has NOT been merged into `main`.**

### File inventory (under `autofish_training_release/length_estimation/`)

| Path | Status |
|---|---|
| `cnn/Model.py` | **modified** (additive DINOv2 encoder) |
| `cnn/train.py`, `cnn/FishLengthDataset.py`, `eval_length_estimators.py` | pristine — keep |
| `cnn/configs/{paper,baseline_frozen,vfm,vfm_finetune,vfm_smoke}.cfg` | 2×2 matrix + smoke |
| `cnn/configs/{default,baseline,smoke}.cfg` | authors'/earlier configs |
| `cnn/configs/generate_label_efficiency.py` | emits 39 configs (3 arms × {1,2,4,8,15} × subsets) |
| `cnn/configs/label_efficiency/le-*.cfg` | 39 generated configs |
| `compute_metrics.py` | MAE/MAPE/RMSE/R²/median/**signed bias**/per-fish-id; writes `metrics.json` + runs CSV |
| `analyze_predictions.py` | bias by species/subset/length-bin, scatters, worst-N CSV |
| `plot_label_efficiency.py` | MAE-vs-#groups curve, mean + min–max band per arm |
| `run_all.sh` | one-shot runner, phases `smoke\|main\|le\|analysis\|all` |
| `VFM_EXPERIMENTS.md` | full experiment + analysis guide |

## 9. CURRENT STATE — the critical part

**All code is complete, verified, and pushed. ZERO experiments have been run.**

- The only outputs on disk are stale: `cnn/output/cnn-baseline` from an old
  10-epoch/batch-16 local run → **MAE 3.03 cm**, i.e. 3.7× worse than the paper's
  0.82. **The baseline is NOT yet reproduced.** Also `cnn-smoke` (throwaway).
- No VFM results, no label-efficiency curve, no failure/bias results.
- ⚠️ **`ODE_fish_length_estimation.md` still presents that 3.03 cm in its results
  table against the paper's 0.67/0.96.** As-is it reads as a failed
  reproduction. It must be replaced with real 200-epoch numbers (or clearly
  marked preliminary) before submission.

**Blocker:** the GPU server (`po2498@atlantic`) returns
`permission denied ... /var/run/docker.sock` — the user is not in the `docker`
group. Unresolved options offered: (a) `sudo usermod -aG docker $USER`,
(b) Apptainer/Singularity if available, (c) skip Docker and use a venv.
**The user has not yet said which applies.**

## 10. Next steps (in order)

1. **Unblock the server** (pick one of a/b/c above). If no sudo, the venv route
   is fastest: torch + torchvision, pandas, scikit-image, opencv-python,
   pycocotools, stocaching, matplotlib, seaborn, numpy — then run `run_all.sh`
   with `GT=/path/to/annotations.json`.
2. **`bash run_all.sh smoke`** — 1-group/1-epoch check; also triggers the
   one-time DINOv2 download (needs internet on first run).
3. **`bash run_all.sh main`** — the 4 headline runs.
4. **Verify the baseline lands near 0.82 cm (GT masks)** before claiming
   reproduction. This single number gates the credibility of everything else.
5. **`bash run_all.sh le`** — the 39-run sweep (resumable; skips existing
   `model.pt` unless `FORCE=1`).
6. **`bash run_all.sh analysis`** — failure/bias + label-efficiency curve.
7. **Rewrite the ODE report's results** with real numbers; write up the
   comparison (frozen-vs-frozen as the headline), the label-efficiency finding,
   and the bias/failure analysis.

### How to run (inside the authors' Docker)

```bash
docker pull shbe/mask2former_container
docker tag  shbe/mask2former_container mask2former_container  # run_docker.sh hard-codes this name
cd autofish_training_release/docker
bash run_docker.sh /abs/path/to/autofish_data mask2former_container
# inside the container:
cd /workspace/autofish_training/length_estimation
bash run_all.sh smoke && bash run_all.sh main    # use tmux for the long sweep
```

Docker bind-mounts repo → `/workspace/autofish_training` and dataset →
`/workspace/autofish_dataset`; nothing needs to be physically copied into
`/workspace`. Results appear back on the host under `cnn/output/`.

## 11. Gotchas

- **`eval_length_estimators.py` must run from `length_estimation/`**, not
  `cnn/` — it does `sys.path.append("cnn")`. `train.py` runs from `cnn/`.
- `train.py` copies the config next to `model.pt`; eval globs `*.cfg` there to
  rebuild the model — this is how eval knows to construct the DINOv2 variant.
- Eval writes to `cnn/output/<run>/eval/model/from-gt/eval-output.csv`.
- Docker image already has every dependency (torch 2.0.1+cu117, numpy 1.x where
  `np.Inf` still works, pandas via seaborn, scikit-image, pycocotools,
  stocaching). No MSDeformAttn/CUDA-kernel compile needed for length estimation.
- The dataset's `SharedCache` is hard-coded to 2 GiB in pristine code, so only
  ~1/3 of train crops cache — slower but faithful. Do not "fix".
- A prior session's local `.venv` (Python 3.12, torch 2.12 MPS) exists for
  analysis/scratch work; the real runs happen on the server.
