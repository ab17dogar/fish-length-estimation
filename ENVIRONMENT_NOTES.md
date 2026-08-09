# Environment notes — running the study without the authors' Docker image

The AutoFish release ships a Docker image (`shbe/mask2former_container`) that
pins every dependency. On the GPU server used for these runs Docker is not
available to the account (no `docker` group, no sudo), so the study runs in a
plain virtualenv instead.

This file records **every** deviation from the shipped setup, so the results can
be reproduced and so the write-up can state honestly what was and was not
identical to the published environment.

**No file in `autofish_training_release/` was modified** other than
`length_estimation/cnn/Model.py` (the sanctioned VFM encoder swap). In
particular `train.py`, `FishLengthDataset.py` and `eval_length_estimators.py`
remain byte-identical to the bitbucket release.

---

## 1. Interpreter and packages

Created with [`uv`](https://astral.sh/uv):

```
uv venv --python 3.11 fishenv          # resolved to the system 3.12.3
uv pip install "numpy<2" torch torchvision --torch-backend=cu124
uv pip install pandas scikit-image opencv-python-headless pycocotools \
               stocaching matplotlib seaborn huggingface_hub pyransac3d \
               scikit-learn
```

Key versions: torch 2.6.0+cu124, numpy 1.26.4, opencv 4.11, stocaching 0.2.1.

**numpy must stay below 2.0.** `train.py:94` uses `np.Inf`, which numpy 2
removed. Pinning numpy is what keeps the authors' file untouched.

`opencv-python-headless` replaces `opencv-python` because the server has no X
libraries; the `cv2` API used by the repo is identical.

`pyransac3d` and `scikit-learn` are not in the repo's requirements but are
needed: the former is imported transitively by `eval_length_estimators.py`
(via `camera_calibration/Camera.py`), the latter only by our own
`bbox_only_control.py`.

## 2. Dataset location

The configs ship with the Docker bind-mount path
`/workspace/autofish_dataset/annotations.json`. `/workspace` cannot be created
without root, so the dataset lives elsewhere and `retarget_data_path.sh`
rewrites `MASKS_GT` in every config:

```
bash retarget_data_path.sh /abs/path/to/autofish_data/annotations.json
```

It is idempotent and reversible (pass the Docker path to restore). Re-run it
after `generate_label_efficiency.py`, which re-emits the Docker default.

Only `annotations.json` and `group_01..group_25` are downloaded (11 GB).
`camera_calibration/` and `unlabeled_images/` are skipped: all configs set
`MODEL_INPUT_PLANE: False`, and the one code path that reads
`camera_calibration/plane_params.json` is guarded by `os.path.exists` and is
never reached.

## 3. `stocaching` patched to allow cache-slot overwrite

**What.** In the venv only, the default of `allow_overwrite` in
`SharedCache.set_slot` was flipped from `False` to `True`
(`fishenv/lib/python3.12/site-packages/stocaching/__init__.py`; the pristine
file is kept beside it as `__init__.py.orig`).

**Why.** `FishLengthDataset.__getitem__` caches a decoded crop with
`self.cache.set_slot(idx, image)` (line 316) after a `get_slot` miss. With
`num_workers=7` this raises

```
RuntimeError: Tried to overwrite non-empty slot idx=... in SharedCache.
```

reproducibly on the label-efficiency configs. The trigger is the *small
training set*: a 1-group run finishes a training epoch in ~2.5 s, and the crash
lands in the first validation pass. Configs with the full 15 training groups
(`paper`, `vfm`, `baseline_frozen`) are slow enough that it does not fire. The
behaviour is identical in every published `stocaching` release (0.1.0 through
0.2.1), so it is not version drift — it is a latent race in the interaction
between the authors' caching pattern and the dataloader workers.

**Why this is safe.** The cached value is a pure function of `idx`: read image →
apply mask → crop to bbox → resize to 224 (`FishLengthDataset.py:285-316`).
Colour augmentation is applied *after* the cache write (line 319 onward) and is
never cached. Re-writing a slot therefore stores byte-identical data, so
tolerating an overwrite cannot change any result — it only stops the crash.

The alternative, `CACHE_IMAGES: False`, is also correct but ~25x slower
(every epoch would re-decode ~4,400 full 2464x2056 PNGs), which would have made
the label-efficiency sweep infeasible.

## 4. `vfm_smoke.cfg` raised from 1 epoch to 2

`train.py:266` computes `total_time / (epoch)` in its final summary, which
raises `ZeroDivisionError` whenever `NUM_EPOCHS: 1`. This is a pre-existing bug
in the authors' code and affects only 1-epoch runs. Rather than edit `train.py`,
the smoke config now uses 2 epochs. No experimental config was ever 1 epoch, so
no reported result is affected.

## 5. Hardware constraint (not a code deviation)

The GPU (RTX 5000 Ada, 32 GB) is shared with an unrelated long-running vLLM
server holding ~30 GB, leaving ~2.2 GB. Measured peak usage at batch 32:

| config | encoder | regime | peak | fits |
|---|---|---|---|---|
| `vfm` | DINOv2 ViT-S/14 | frozen | 0.30 GB | yes |
| `baseline_frozen` | MobileNetV2 | frozen | 0.40 GB | yes |
| `paper` | MobileNetV2 | fine-tuned | ~2.5 GB | **no** |
| `vfm_finetune` | DINOv2 ViT-S/14 | fine-tuned | ~2.5 GB | **no** |

The frozen arms — which are the clean encoder-only comparison — run fine. The
fine-tuned arms wait for GPU memory; `run_all.sh` skips any run whose `model.pt`
already exists, so re-running a phase resumes exactly where it left off.

Training is CPU-bound, not GPU-bound: the 2 GiB `SharedCache` hard-coded in
`FishLengthDataset.py` holds only ~31% of the 15-group training set, so most
samples re-decode a full-resolution PNG every epoch (GPU utilisation sits near
0%). This is why two training streams run concurrently.
