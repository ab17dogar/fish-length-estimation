"""Compute length-estimation regression metrics (MAE, MAPE) from an eval-output.csv.

Subsets follow AutoFish Fig. 3: within each group, image index 1-40 are the
'separated' sets (Set1, Set2), and 41-60 are the 'touching' set (All).
'combined' = all test fish together.
"""
import argparse
import os
import re
import numpy as np
import pandas as pd


def subset_for_path(img_path):
    # file like .../group_10/00001.png
    m = re.search(r"(\d+)\.png$", os.path.basename(img_path))
    idx = int(m.group(1))
    return "separated" if idx <= 40 else "touching"


def metrics(df):
    gt = df["gt_length_cm"].to_numpy(dtype=float)
    pred = df["pred_length_cm"].to_numpy(dtype=float)
    abs_err = np.abs(pred - gt)
    mae = abs_err.mean()
    mape = (abs_err / gt).mean() * 100.0
    return mae, mape, len(df)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True)
    args = ap.parse_args()

    df = pd.read_csv(args.csv)
    df["subset"] = df["img_path"].map(subset_for_path)

    print(f"\nFile: {args.csv}")
    print(f"{'Subset':<12}{'MAE (cm)':>12}{'MAPE (%)':>12}{'N':>8}")
    print("-" * 44)
    for name in ["separated", "touching"]:
        sub = df[df["subset"] == name]
        if len(sub):
            mae, mape, n = metrics(sub)
            print(f"{name:<12}{mae:>12.3f}{mape:>11.2f}%{n:>8}")
    mae, mape, n = metrics(df)
    print(f"{'combined':<12}{mae:>12.3f}{mape:>11.2f}%{n:>8}")

    # Per-species breakdown (combined)
    print(f"\n{'Species':<14}{'MAE (cm)':>12}{'MAPE (%)':>12}{'N':>8}")
    print("-" * 46)
    for sp in sorted(df["gt_label"].unique()):
        sub = df[df["gt_label"] == sp]
        mae, mape, n = metrics(sub)
        print(f"{sp:<14}{mae:>12.3f}{mape:>11.2f}%{n:>8}")


if __name__ == "__main__":
    main()
