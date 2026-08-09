"""Bbox-only control: how much of the length signal is in the image at all?

The REG network is fed TWO things: the 224x224 RGB-masked crop AND the four
normalized bbox coordinates. Because the camera is fixed and the fish lie flat
on a table, the bbox alone already encodes most of the scale information --- so
a length regressor could score well while barely using the image.

That matters directly for this study's research question. Comparing encoders
(MobileNetV2 vs DINOv2) is only meaningful to the extent that the ENCODER, not
the bbox side-channel, drives the prediction. This script quantifies the
side-channel by fitting length regressors on the bbox coordinates ALONE:

    mean      predict the training-set mean length (no input at all)
    linear    ordinary least squares on the 4 normalized bbox coords
    gbm       gradient boosting on the same 4 coords (nonlinear ceiling)

Whatever MAE the 'gbm' row reaches is roughly the score a network could obtain
while ignoring its encoder entirely. The encoder's true contribution is the gap
between that number and the full model's MAE.

Inputs are replicated EXACTLY as FishLengthDataset builds them:
  * the bbox is recomputed from the segmentation mask (not the COCO bbox field),
    using width = max - min, matching FishLengthDataset.__getitem__;
  * coords are normalized by the same hard-coded 2464 x 2056 image size.

Group splits match the configs. Results are reported on the TEST groups, broken
down by the separated / touching / combined subsets (image index <= 40 is
separated, > 40 is touching), so the numbers sit directly alongside the paper's
Table 4 and our metrics.json outputs.

Usage:
    python bbox_only_control.py --gt_path /path/to/annotations.json
"""

import argparse
import json
import os
import re

import numpy as np
from pycocotools.coco import COCO
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import LinearRegression

# Same splits as the configs.
TRAIN_GROUPS = [2, 3, 4, 5, 7, 8, 9, 12, 13, 15, 16, 18, 19, 23, 24]
VAL_GROUPS = [1, 6, 11, 17, 25]
TEST_GROUPS = [10, 14, 20, 21, 22]

# Hard-coded in FishLengthDataset (normalize_bbox branch).
IMG_W, IMG_H = 2464, 2056

# Images 00001-00040 hold separated fish, 00041-00060 touching fish.
SEPARATED_MAX_INDEX = 40


def image_index(file_name):
    """'group_07/00042.png' -> 42."""
    return int(re.search(r"(\d+)\.png$", file_name).group(1))


def build_table(coco):
    """Return per-instance features/labels, replicating FishLengthDataset."""
    rows = []
    for ann_id in sorted(coco.anns):
        ann = coco.anns[ann_id]
        if ann.get("length") is None:
            continue
        img = coco.imgs[ann["image_id"]]

        # Recompute the bbox from the mask exactly as the dataset does.
        mask = coco.annToMask(ann)
        nz = np.where(mask != 0)
        if nz[0].size == 0:
            continue
        x, y = np.min(nz[1]), np.min(nz[0])
        w = np.max(nz[1]) - np.min(nz[1])
        h = np.max(nz[0]) - np.min(nz[0])

        rows.append({
            "group": img["group"],
            "separated": image_index(img["file_name"]) <= SEPARATED_MAX_INDEX,
            "species": coco.cats[ann["category_id"]]["name"],
            "length": ann["length"],
            "feat": (x / IMG_W, y / IMG_H, w / IMG_W, h / IMG_H),
        })
    return rows


def subset_of(rows, groups):
    idx = [i for i, r in enumerate(rows) if r["group"] in groups]
    X = np.array([rows[i]["feat"] for i in idx], dtype=np.float64)
    y = np.array([rows[i]["length"] for i in idx], dtype=np.float64)
    return idx, X, y


def report(name, y_true, y_pred, mask_sep):
    """Print MAE for separated / touching / combined, like the paper's table."""
    err = np.abs(y_pred - y_true)
    out = {
        "separated": float(err[mask_sep].mean()),
        "touching": float(err[~mask_sep].mean()),
        "combined": float(err.mean()),
    }
    print(f"  {name:8s}  separated {out['separated']:5.2f}   "
          f"touching {out['touching']:5.2f}   combined {out['combined']:5.2f}")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gt_path", required=True)
    ap.add_argument("--out", default="cnn/output/bbox_only_control.json")
    args = ap.parse_args()

    coco = COCO(args.gt_path)
    rows = build_table(coco)
    print(f"instances with a length label: {len(rows)}")

    _, Xtr, ytr = subset_of(rows, TRAIN_GROUPS)
    _, Xva, yva = subset_of(rows, VAL_GROUPS)
    te_idx, Xte, yte = subset_of(rows, TEST_GROUPS)
    sep_te = np.array([rows[i]["separated"] for i in te_idx])
    print(f"train {len(ytr)}   val {len(yva)}   test {len(yte)} "
          f"({int(sep_te.sum())} separated / {int((~sep_te).sum())} touching)\n")

    print("MAE (cm) on the TEST groups, bbox coordinates only:")
    results = {}

    results["mean"] = report(
        "mean", yte, np.full_like(yte, ytr.mean()), sep_te)

    linear = LinearRegression().fit(Xtr, ytr)
    results["linear"] = report("linear", yte, linear.predict(Xte), sep_te)

    gbm = HistGradientBoostingRegressor(
        loss="absolute_error", random_state=42).fit(Xtr, ytr)
    results["gbm"] = report("gbm", yte, gbm.predict(Xte), sep_te)

    print("\nThe 'gbm' combined MAE is roughly what a network could reach while "
          "ignoring\nits encoder entirely; the encoder's real contribution is "
          "the margin below it.")

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as f:
        json.dump({"n_train": len(ytr), "n_test": len(yte),
                   "mae_by_model": results}, f, indent=2)
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
