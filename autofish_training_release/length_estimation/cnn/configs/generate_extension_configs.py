"""Generate the extension-study configs (trained with ../train_vfm.py).

Matrix = {encoder} x {regime} at full training data (15 groups), all sharing the
stronger recipe (AdamW, layer-wise LR decay, cosine+warmup, length-preserving
flips, whole-split RAM cache). The published baselines (paper.cfg = fine-tuned
MobileNetV2 via the authors' train.py; baseline_frozen.cfg) remain the faithful
anchors and are NOT regenerated here.

Encoders:
    mnet   MobileNetV2                (same-recipe CNN reference)
    vits   DINOv2 ViT-S/14, CLS readout
    vitsp  DINOv2 ViT-S/14, CLS+patch readout   (our main lever)
    vitbp  DINOv2 ViT-B/14, CLS+patch readout   (bigger backbone)

Regimes:
    frozen    encoder frozen, head only          (fits in ~0.3-0.4 GB now)
    ft        encoder fine-tuned                  (needs the freed GPU)

Run from this directory:  python generate_extension_configs.py
Then retarget data paths:  bash ../retarget_data_path.sh /path/annotations.json
"""
import os

DATA = "/workspace/autofish_dataset/annotations.json"  # retargeted later
TRAIN = "[2, 3, 4, 5, 7, 8, 9, 12, 13, 15, 16, 18, 19, 23, 24]"
VAL = "[1, 6, 11, 17, 25]"
TEST = "[10, 14, 20, 21, 22]"

BACKENDS = {
    "mnet": "mobilenet_v2",
    "vits": "dinov2_vits14",
    "vitsp": "dinov2_vits14-clspatch",
    "vitbp": "dinov2_vitb14-clspatch",
}

TEMPLATE = """\
# Extension study config (train with ../train_vfm.py, NOT train.py).
# {tag}: encoder={backend}, regime={regime}.
[Config]
OUTPUT_DIR: ./output/{tag}/
MASKS_GT: {data}
TRAIN_GROUPS: {train}
VAL_GROUPS: {val}
TEST_GROUPS: {test}

[Training]
CACHE_IMAGES: True
CACHE_SIZE_GIB: 8
NUM_EPOCHS: {epochs}
BATCH_SIZE: {batch}
NUM_WORKERS: {workers}
DROP_LAST: {drop_last}
AMP: {amp}
RANDOM_SEED: 42
LOSS: l1
OPTIMIZER: adamw
HEAD_LR: {head_lr}
BACKBONE_LR: {backbone_lr}
LAYER_DECAY: 0.75
WEIGHT_DECAY: 0.05
WARMUP_EPOCHS: {warmup}
SCHEDULER: cosine
FLIP_AUG: False
GRAD_CLIP: 1.0

[Preprocessing]
CROP_TO_BBOX: True
MASKING_TYPE: rgb

[Model]
MODEL_BACKEND: {backend}
FREEZE_BACKEND: {frozen}
MODEL_SIZE: small
MODEL_INPUT_PLANE: False
MODEL_INPUT_BBOX: True
NORMALIZE_BBOX: True

[Augmentation]
AUGMENT_COLOR_TRAIN_IMAGES: True
AUGMENT_CROP_TRAIN_IMAGES: False
"""

def main():
    here = os.path.dirname(os.path.abspath(__file__))
    outdir = os.path.join(here, "extension")
    os.makedirs(outdir, exist_ok=True)
    n = 0
    for enc, backend in BACKENDS.items():
        for regime in ("frozen", "ft"):
            frozen = (regime == "frozen")
            tag = f"x-{enc}-{regime}"
            cfg = TEMPLATE.format(
                tag=tag, backend=backend, regime=regime, data=DATA,
                train=TRAIN, val=VAL, test=TEST,
                # Fine-tuning: bf16 + batch 16 fits the shared ~2 GB GPU and
                # converges faster, so fewer epochs. Frozen: fp32, batch 32.
                epochs=150 if frozen else 150,
                warmup=5 if frozen else 10,
                # Fine-tuning: gentle LRs so the encoder drifts slowly and the
                # head's BatchNorm running stats stay calibrated (high LR made
                # val diverge while train fell — a BN-lag instability).
                head_lr="1e-3" if frozen else "2e-4",
                backbone_lr="0" if frozen else "2e-5",
                frozen="True" if frozen else "False",
                batch=32 if frozen else 16,
                drop_last="False" if frozen else "True",
                amp="False" if frozen else "True",
                workers=7 if frozen else 14,
            )
            with open(os.path.join(outdir, tag + ".cfg"), "w") as f:
                f.write(cfg)
            n += 1
    print(f"wrote {n} configs to {outdir}")

if __name__ == "__main__":
    main()
