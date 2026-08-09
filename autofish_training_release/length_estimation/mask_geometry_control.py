"""Mask-geometry control: how far can length be predicted from the GT mask's
shape alone (no learned image features, no camera calibration)?

The REG baseline feeds the network the *axis-aligned* bounding box, which is a
loose length proxy for a fish lying diagonally. But the ground-truth mask (which
the baseline already consumes as an RGB crop) carries far stronger geometry: the
rotated minimum-area rectangle and fitted ellipse give an *orientation-invariant*
major-axis length that hugs the fish. The paper's other baseline (SKL) exploits
exactly this (skeleton on the mask) and reaches 0.59 cm on separated fish.

This script quantifies that signal by regressing length on cheap mask-derived
features, to see whether it beats REG^gt = 0.82 cm combined and where the ceiling
is. If it does, the path to beating the baseline is to feed these features to the
model alongside the image (a "geometry-augmented" head), not a bigger encoder.

Features per instance (all in pixels; a GBM learns the ~constant px->cm scale, as
the bbox control does), computed from coco.annToMask:
    bbox w,h ; rotated-rect major,minor ; ellipse major,minor ;
    area ; perimeter ; centroid x,y (position, for mild perspective variation)

Splits/subsets match the configs and the paper. Reports MAE by separated /
touching / combined, alongside the bbox-only floor (0.96) and REG^gt (0.82).

    python mask_geometry_control.py --gt_path /path/to/annotations.json
"""
import argparse
import json
import os
import re

import cv2
import numpy as np
from pycocotools.coco import COCO
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import LinearRegression

TRAIN_GROUPS = [2, 3, 4, 5, 7, 8, 9, 12, 13, 15, 16, 18, 19, 23, 24]
TEST_GROUPS = [10, 14, 20, 21, 22]
SEP_MAX_IDX = 40


def image_index(file_name):
    return int(re.search(r"(\d+)\.png$", file_name).group(1))


def mask_features(mask):
    """Cheap shape descriptors from a binary mask."""
    ys, xs = np.where(mask > 0)
    if xs.size < 6:
        return None
    x0, x1, y0, y1 = xs.min(), xs.max(), ys.min(), ys.max()
    bw, bh = float(x1 - x0), float(y1 - y0)
    area = float(mask.sum())
    cx, cy = float(xs.mean()), float(ys.mean())

    cnts, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL,
                               cv2.CHAIN_APPROX_SIMPLE)
    cnt = max(cnts, key=cv2.contourArea)
    perim = float(cv2.arcLength(cnt, True))
    (_, _), (rw, rh), _ = cv2.minAreaRect(cnt)   # rotated rectangle
    rr_major, rr_minor = float(max(rw, rh)), float(min(rw, rh))
    if len(cnt) >= 5:
        (_, _), (ew, eh), _ = cv2.fitEllipse(cnt)
        el_major, el_minor = float(max(ew, eh)), float(min(ew, eh))
    else:
        el_major, el_minor = rr_major, rr_minor
    return [bw, bh, rr_major, rr_minor, el_major, el_minor,
            area, perim, cx, cy]


def build(coco):
    rows = []
    for aid in sorted(coco.anns):
        ann = coco.anns[aid]
        if ann.get("length") is None:
            continue
        feat = mask_features(coco.annToMask(ann))
        if feat is None:
            continue
        img = coco.imgs[ann["image_id"]]
        rows.append({
            "group": img["group"],
            "separated": image_index(img["file_name"]) <= SEP_MAX_IDX,
            "length": ann["length"],
            "feat": feat,
        })
    return rows


def subset(rows, groups):
    idx = [i for i, r in enumerate(rows) if r["group"] in groups]
    X = np.array([rows[i]["feat"] for i in idx], dtype=np.float64)
    y = np.array([rows[i]["length"] for i in idx], dtype=np.float64)
    return idx, X, y


def report(name, y, pred, sep):
    e = np.abs(pred - y)
    print(f"  {name:26s} sep {e[sep].mean():.3f}  touch {e[~sep].mean():.3f}  "
          f"combined {e.mean():.3f}")
    return e.mean()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gt_path", required=True)
    ap.add_argument("--out", default="cnn/output/mask_geometry_control.json")
    args = ap.parse_args()

    coco = COCO(args.gt_path)
    rows = build(coco)
    _, Xtr, ytr = subset(rows, TRAIN_GROUPS)
    te, Xte, yte = subset(rows, TEST_GROUPS)
    sep = np.array([rows[i]["separated"] for i in te])
    print(f"instances: {len(rows)} | train {len(ytr)} | test {len(yte)}\n")
    print("MAE (cm) on TEST, mask-geometry features:")

    feat_names = ["bbox_w", "bbox_h", "rr_major", "rr_minor", "el_major",
                  "el_minor", "area", "perim", "cx", "cy"]
    res = {}

    # single strongest feature: rotated-rect major axis, linear
    j = feat_names.index("rr_major")
    lin1 = LinearRegression().fit(Xtr[:, [j]], ytr)
    res["rr_major_linear"] = report("rr_major only (linear)", yte,
                                    lin1.predict(Xte[:, [j]]), sep)

    lin = LinearRegression().fit(Xtr, ytr)
    res["all_linear"] = report("all feats (linear)", yte, lin.predict(Xte), sep)

    gbm = HistGradientBoostingRegressor(
        loss="absolute_error", max_iter=400, random_state=42).fit(Xtr, ytr)
    res["all_gbm"] = report("all feats (GBM)", yte, gbm.predict(Xte), sep)

    print("\n  reference:  bbox-only floor 0.96 | REG^gt (paper) 0.82 | "
          "SKL^gt separated 0.59")
    best = min(res.values())
    print(f"\n  best mask-geometry combined MAE: {best:.3f}  "
          f"({'BEATS' if best < 0.82 else 'does NOT beat'} REG^gt 0.82)")

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    json.dump({"mae": res, "features": feat_names}, open(args.out, "w"), indent=2)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
