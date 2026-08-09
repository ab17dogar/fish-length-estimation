"""Stronger training loop for the VFM (and baseline) length regressors.

This is the AutoFish-extension counterpart to the authors' ``train.py``. The
byte-faithful reproduction on the ``main`` branch keeps ``train.py`` exactly as
published (plain ``Adam(model.parameters())`` at a fixed lr 1e-3, colour-only
augmentation, no schedule); that run is the honest baseline anchor. This script
is what lets the VFM actually compete, by giving it the training recipe a ViT
needs — none of which is expressible through the original ``train.py``:

  * **AdamW** with decoupled weight decay.
  * **Layer-wise LR decay** across the DINOv2 blocks + a separate (higher) head
    LR, so a fixed global lr 1e-3 no longer wrecks the pretrained features.
  * **Cosine schedule with linear warmup.**
  * **Length-preserving flip augmentation** (horizontal/vertical flips of the
    square crop; both leave the fish's pixel length — and therefore its cm
    length — unchanged, unlike the scaling/geometric augments the paper rightly
    forbids). Applied to the image tensor only; the normalized bbox side-channel
    is unchanged.
  * **Gradient clipping.**

It reuses the authors' ``Model``, ``FishLengthDataset`` and ``setup_dataloaders``
unchanged, and writes the same on-disk contract as ``train.py``
(``<output>/model.pt`` = best-val weights, plus a copy of the config), so
``eval_length_estimators.py`` evaluates a model trained here with no changes.

Extra config keys (all optional; defaults reproduce a sane ViT fine-tune):

    [Training]
    OPTIMIZER      : adamw            adamw | adam
    HEAD_LR        : 1e-3             LR for the regression head (and bbox path)
    BACKBONE_LR    : 1e-4             base LR for the *last* backbone layer
    LAYER_DECAY    : 0.75             per-layer LR decay toward the input
    WEIGHT_DECAY   : 0.05             decoupled weight decay (AdamW)
    WARMUP_EPOCHS  : 5
    SCHEDULER      : cosine           cosine | none
    FLIP_AUG       : True             length-preserving h/v flips
    GRAD_CLIP      : 1.0              max grad norm (0 disables)

Usage (from length_estimation/cnn/):
    python train_vfm.py --config configs/vfm_ft.cfg
"""

import argparse
import configparser
import json
import math
import os
import shutil
from timeit import default_timer as timer

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn

from Model import Model, DINOv2Encoder
from train import setup_dataloaders


# --------------------------------------------------------------------------- #
# Parameter groups with layer-wise LR decay
# --------------------------------------------------------------------------- #
def _dinov2_layer_id(param_name, num_blocks):
    """Depth index (0 = input/embeddings, num_blocks+1 = final norm/head side).

    Names come from DINOv2Encoder.backbone: 'patch_embed.*', 'cls_token',
    'pos_embed', 'register_tokens', 'blocks.<i>.*', 'norm.*'.
    """
    if param_name.startswith(("cls_token", "pos_embed", "register_tokens",
                              "mask_token", "patch_embed")):
        return 0
    if param_name.startswith("blocks."):
        return int(param_name.split(".")[1]) + 1
    # final norm and anything else: treat as the deepest layer
    return num_blocks + 1


def build_param_groups(model, head_lr, backbone_lr, layer_decay, weight_decay):
    """Layer-wise-decayed groups for the backbone + a flat group for the head.

    Returns a list of param-group dicts for the optimizer. Frozen parameters
    (requires_grad=False) are skipped, so a frozen encoder yields head-only
    groups automatically. Biases and 1-D params (norms) get no weight decay.
    """
    enc = model.features
    is_dino = isinstance(enc, DINOv2Encoder)
    groups = {}

    def add(name, param, base_lr, layer_scale):
        if not param.requires_grad:
            return
        decay = weight_decay if (param.ndim > 1) else 0.0
        key = (round(base_lr * layer_scale, 12), decay)
        groups.setdefault(key, {"params": [], "lr": key[0],
                                "weight_decay": decay})["params"].append(param)

    if is_dino:
        num_blocks = len(enc.backbone.blocks)
        for name, p in enc.backbone.named_parameters():
            lid = _dinov2_layer_id(name, num_blocks)
            # deepest layer -> scale 1; each step toward input -> *layer_decay
            scale = layer_decay ** (num_blocks + 1 - lid)
            add(name, p, backbone_lr, scale)
    else:
        for name, p in enc.named_parameters():
            add(name, p, backbone_lr, 1.0)

    # Head + everything that is not the encoder (e.g. the MLP regressor).
    for name, p in model.named_parameters():
        if name.startswith("features."):
            continue
        add(name, p, head_lr, 1.0)

    return list(groups.values())


# --------------------------------------------------------------------------- #
# Length-preserving augmentation
# --------------------------------------------------------------------------- #
def recalibrate_bn(model, loader, amp, amp_dtype, max_batches=50):
    """Refresh the head's BatchNorm running stats to match current features.

    When the encoder is fine-tuned, its output distribution shifts every step,
    so the head BatchNorm's running mean/var (used at eval) lag the live batch
    stats (used at train) — train loss falls smoothly while val explodes. This
    recomputes those running stats over a slice of TRAINING data with the
    encoder in eval mode (deterministic features, no drop-path), so eval — and
    the saved checkpoint that eval_length_estimators.py later loads — sees
    well-calibrated stats. No-op if the model has no BatchNorm.
    """
    bn = [m for m in model.modules()
          if isinstance(m, nn.modules.batchnorm._BatchNorm)]
    if not bn:
        return
    model.eval()                      # deterministic features everywhere...
    saved = {}
    for m in bn:
        saved[m] = m.momentum
        m.reset_running_stats()
        m.momentum = None             # cumulative average over the slice
        m.train()                     # ...but let BN layers update their stats
    with torch.no_grad():
        for i, (data, _) in enumerate(loader):
            if i >= max_batches:
                break
            data = [d.cuda() for d in data]
            with torch.autocast("cuda", dtype=amp_dtype, enabled=amp):
                model(data)
    for m in bn:
        m.momentum = saved[m]
    model.eval()


def flip_augment(img):
    """Random horizontal/vertical flips of a [B,3,H,W] crop batch.

    Both flips preserve the fish's pixel extent, hence its length label, so
    unlike scaling/rotation they are safe for the px->cm regression. Applied
    per-batch (cheap, on-GPU); the bbox side-channel is left untouched.
    """
    if torch.rand(()) < 0.5:
        img = torch.flip(img, dims=[3])  # horizontal
    if torch.rand(()) < 0.5:
        img = torch.flip(img, dims=[2])  # vertical
    return img


# --------------------------------------------------------------------------- #
# Schedule
# --------------------------------------------------------------------------- #
def make_lr_lambda(warmup_steps, total_steps, scheduler):
    def fn(step):
        if step < warmup_steps:
            return (step + 1) / max(1, warmup_steps)
        if scheduler == "none":
            return 1.0
        prog = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.5 * (1.0 + math.cos(math.pi * min(1.0, prog)))
    return fn


# --------------------------------------------------------------------------- #
# Train
# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    args = ap.parse_args()

    cfg = configparser.ConfigParser()
    cfg.read(args.config)

    seed = cfg.getint("Training", "RANDOM_SEED", fallback=42)
    torch.manual_seed(seed)
    np.random.seed(seed)

    gf = cfg.getfloat
    optimizer_name = cfg.get("Training", "OPTIMIZER", fallback="adamw").lower()
    head_lr = gf("Training", "HEAD_LR", fallback=1e-3)
    backbone_lr = gf("Training", "BACKBONE_LR", fallback=1e-4)
    layer_decay = gf("Training", "LAYER_DECAY", fallback=0.75)
    weight_decay = gf("Training", "WEIGHT_DECAY", fallback=0.05)
    warmup_epochs = gf("Training", "WARMUP_EPOCHS", fallback=5)
    scheduler = cfg.get("Training", "SCHEDULER", fallback="cosine").lower()
    flip_aug = cfg.getboolean("Training", "FLIP_AUG", fallback=True)
    grad_clip = gf("Training", "GRAD_CLIP", fallback=1.0)
    amp = cfg.getboolean("Training", "AMP", fallback=False)
    bn_recalib = cfg.getboolean("Training", "BN_RECALIB", fallback=True)
    n_epochs = cfg.getint("Training", "NUM_EPOCHS")
    batch_size = cfg.getint("Training", "BATCH_SIZE")
    # bf16 autocast lets a ViT be fine-tuned inside the ~2 GB of shared GPU we
    # have (peak ~0.8 GB at batch 16) — no GradScaler needed for bf16's range.
    amp_dtype = torch.bfloat16

    model = Model(
        bbox_input=cfg.getboolean("Model", "MODEL_INPUT_BBOX"),
        plane_input=cfg.getboolean("Model", "MODEL_INPUT_PLANE"),
        model_size=cfg.get("Model", "MODEL_SIZE", fallback=None),
        freeze_backend=cfg.getboolean("Model", "FREEZE_BACKEND", fallback=True),
        model_name=cfg.get("Model", "MODEL_BACKEND")).cuda()

    loss_name = cfg.get("Training", "LOSS", fallback="l1")
    criterion = {"l1": nn.L1Loss(), "l2": nn.MSELoss(),
                 "smooth-l1": nn.SmoothL1Loss()}[loss_name]

    # Partial fine-tuning: freeze the first N transformer blocks (and the input
    # embeddings), adapting only the top blocks + head. A 22M-param ViT overfits
    # AutoFish's ~454 unique fish under full fine-tuning; freezing the lower
    # layers is the standard remedy on small data. 0 = full fine-tune.
    freeze_below = cfg.getint("Training", "FREEZE_BLOCKS_BELOW", fallback=0)
    if freeze_below > 0 and hasattr(model.features, "backbone") \
            and hasattr(model.features.backbone, "blocks"):
        bb = model.features.backbone
        for name in ("cls_token", "pos_embed", "register_tokens", "mask_token"):
            p = getattr(bb, name, None)
            if isinstance(p, torch.nn.Parameter):
                p.requires_grad = False
        if hasattr(bb, "patch_embed"):
            for p in bb.patch_embed.parameters():
                p.requires_grad = False
        for i, blk in enumerate(bb.blocks):
            if i < freeze_below:
                for p in blk.parameters():
                    p.requires_grad = False
        n_frozen = sum(1 for i in range(len(bb.blocks)) if i < freeze_below)
        print(f"partial fine-tune: froze embeddings + blocks 0..{freeze_below-1} "
              f"({n_frozen}/{len(bb.blocks)} blocks); training the rest + head")

    groups = build_param_groups(model, head_lr, backbone_lr, layer_decay,
                                weight_decay)
    n_train_params = sum(p.numel() for g in groups for p in g["params"]) / 1e6
    if optimizer_name == "adam":
        optimizer = torch.optim.Adam(groups)
    else:
        optimizer = torch.optim.AdamW(groups)

    output_dir = cfg.get("Config", "OUTPUT_DIR")
    os.makedirs(output_dir, exist_ok=True)
    shutil.copy(args.config, output_dir)

    loaders = setup_dataloaders(cfg)
    train_loader, val_loader = loaders["train"], loaders["val"]
    steps_per_epoch = len(train_loader)
    total_steps = steps_per_epoch * n_epochs
    warmup_steps = int(round(warmup_epochs * steps_per_epoch))
    lr_lambda = make_lr_lambda(warmup_steps, total_steps, scheduler)
    lr_sched = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    print(f"train_vfm: backend={cfg.get('Model','MODEL_BACKEND')} "
          f"frozen={cfg.getboolean('Model','FREEZE_BACKEND', fallback=True)} "
          f"opt={optimizer_name} head_lr={head_lr} backbone_lr={backbone_lr} "
          f"layer_decay={layer_decay} wd={weight_decay} warmup={warmup_epochs} "
          f"sched={scheduler} flip={flip_aug} clip={grad_clip}")
    print(f"trainable params: {n_train_params:.2f}M in {len(groups)} groups | "
          f"{steps_per_epoch} steps/epoch x {n_epochs} epochs")

    history = []
    best_val = math.inf
    best_epoch = -1
    t0 = timer()

    for epoch in range(n_epochs):
        model.train()
        train_loss = 0.0
        for data, target in train_loader:
            data = [d.cuda() for d in data]
            target = target.cuda()
            if flip_aug:
                data[0] = flip_augment(data[0])
            optimizer.zero_grad()
            with torch.autocast("cuda", dtype=amp_dtype, enabled=amp):
                loss = criterion(model(data), target)
            loss.backward()
            if grad_clip > 0:
                nn.utils.clip_grad_norm_(
                    (p for g in optimizer.param_groups for p in g["params"]),
                    grad_clip)
            optimizer.step()
            lr_sched.step()
            train_loss += loss.item() * target.size(0)

        # Recalibrate head BatchNorm stats to the just-updated encoder features
        # before validating and saving (fixes eval-time BN lag under fine-tuning).
        if bn_recalib:
            recalibrate_bn(model, train_loader, amp, amp_dtype)
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for data, target in val_loader:
                data = [d.cuda() for d in data]
                target = target.cuda()
                with torch.autocast("cuda", dtype=amp_dtype, enabled=amp):
                    val_loss += criterion(model(data), target).item() * target.size(0)

        train_loss /= len(train_loader.dataset)
        val_loss /= len(val_loader.dataset)
        history.append([train_loss, val_loss])
        cur_lrs = [pg["lr"] for pg in optimizer.param_groups]
        print(f"Epoch {epoch:3d}  train {train_loss:.4f}  val {val_loss:.4f}  "
              f"lr[{min(cur_lrs):.2e}..{max(cur_lrs):.2e}]"
              f"{'  *best' if val_loss < best_val else ''}")

        # Save best-val weights as model.pt (the contract eval.py expects).
        if val_loss < best_val:
            best_val = val_loss
            best_epoch = epoch
            torch.save(model.state_dict(), os.path.join(output_dir, "model.pt"))

    # Never-empty safety: if val never improved (shouldn't happen), still save.
    if not os.path.exists(os.path.join(output_dir, "model.pt")):
        torch.save(model.state_dict(), os.path.join(output_dir, "model.pt"))

    hist = np.array(history)
    np.savetxt(os.path.join(output_dir, "train_loss.csv"), hist[:, 0])
    np.savetxt(os.path.join(output_dir, "val_loss.csv"), hist[:, 1])
    plt.figure()
    plt.plot(hist[:, 0], label="train")
    plt.plot(hist[:, 1], label="val")
    plt.xlabel("epoch"); plt.ylabel(f"{loss_name} loss"); plt.legend()
    plt.title(os.path.basename(output_dir.rstrip("/")))
    plt.savefig(os.path.join(output_dir, "loss.png"))
    plt.close()

    with open(os.path.join(output_dir, "train_summary.json"), "w") as f:
        json.dump({"best_epoch": best_epoch, "best_val_loss": best_val,
                   "epochs": n_epochs, "minutes": (timer() - t0) / 60.0,
                   "trainable_params_M": n_train_params}, f, indent=2)
    print(f"\nBest val {best_val:.4f} at epoch {best_epoch}. "
          f"{(timer()-t0)/60:.1f} min. -> {output_dir}/model.pt")


if __name__ == "__main__":
    main()
