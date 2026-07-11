"""Failure-case and prediction-bias analysis from an eval-output.csv.

Task-description item: "analyzing failure cases and prediction biases across
different samples or species." compute_metrics.py summarises error magnitudes;
this script characterises WHERE and HOW the model is wrong:

  1. Signed-bias tables (mean error, + = over-estimate) per species, per subset,
     and per ground-truth length bin -> reveals systematic over/under-estimation
     and whether the model regresses toward the mean (over-estimates small fish,
     under-estimates large ones).
  2. Plots (PNG): predicted-vs-GT length scatter (with y=x), and signed-error
     vs GT length, both coloured by species.
  3. A worst-N table (largest |error|) as CSV, with img_path, species, subset,
     gt, pred, error, iou -> the concrete cases to inspect qualitatively.

All outputs go to <csv_dir>/analysis/ (or --out-dir). Additive tooling; the
authors' code is untouched.

    python analyze_predictions.py --csv .../eval-output.csv [--top-n 40]
"""
import argparse
import os
import re

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def subset_for_path(img_path):
    idx = int(re.search(r"(\d+)\.png$", os.path.basename(img_path)).group(1))
    return "separated" if idx <= 40 else "touching"


def _bias_table(df, by, label):
    print(f"\nSigned bias by {label} (mean error, + = over-estimate):")
    print(f"{label:14}{'bias':>9}{'MAE':>9}{'N':>7}")
    print("-" * 39)
    rows = []
    for key, g in df.groupby(by):
        err = (g["pred_length_cm"] - g["gt_length_cm"])
        rows.append((str(key), err.mean(), err.abs().mean(), len(g)))
    for name, bias, mae, n in sorted(rows, key=lambda r: r[1]):
        print(f"{name:14}{bias:>9.3f}{mae:>9.3f}{n:>7}")
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True)
    ap.add_argument("--out-dir", default=None)
    ap.add_argument("--top-n", type=int, default=40)
    ap.add_argument("--n-bins", type=int, default=6, help="GT-length bins for bias")
    args = ap.parse_args()

    df = pd.read_csv(args.csv)
    df["subset"] = df["img_path"].map(subset_for_path)
    df["error"] = df["pred_length_cm"] - df["gt_length_cm"]
    df["abs_error"] = df["error"].abs()
    df["len_bin"] = pd.cut(df["gt_length_cm"], bins=args.n_bins)

    out_dir = args.out_dir or os.path.join(os.path.dirname(args.csv), "analysis")
    os.makedirs(out_dir, exist_ok=True)

    print(f"File: {args.csv}   (N = {len(df)} instances)")
    print(f"Overall bias (mean signed error): {df['error'].mean():+.3f} cm")
    _bias_table(df, "gt_label", "species")
    _bias_table(df, "subset", "subset")
    _bias_table(df, "len_bin", "GT-length bin")

    species = sorted(df["gt_label"].unique())
    cmap = plt.get_cmap("tab10")
    colors = {s: cmap(i % 10) for i, s in enumerate(species)}

    # Plot 1: predicted vs GT length
    fig, ax = plt.subplots(figsize=(6, 6))
    for s in species:
        d = df[df["gt_label"] == s]
        ax.scatter(d["gt_length_cm"], d["pred_length_cm"], s=6, alpha=0.4,
                   color=colors[s], label=s)
    lo = float(min(df["gt_length_cm"].min(), df["pred_length_cm"].min()))
    hi = float(max(df["gt_length_cm"].max(), df["pred_length_cm"].max()))
    ax.plot([lo, hi], [lo, hi], "k--", lw=1, label="y = x")
    ax.set_xlabel("Ground-truth length (cm)")
    ax.set_ylabel("Predicted length (cm)")
    ax.set_title("Predicted vs. ground-truth length")
    ax.legend(fontsize=7, markerscale=2)
    fig.tight_layout()
    p1 = os.path.join(out_dir, "pred_vs_gt.png")
    fig.savefig(p1, dpi=130)
    plt.close(fig)

    # Plot 2: signed error vs GT length
    fig, ax = plt.subplots(figsize=(7, 5))
    for s in species:
        d = df[df["gt_label"] == s]
        ax.scatter(d["gt_length_cm"], d["error"], s=6, alpha=0.4,
                   color=colors[s], label=s)
    ax.axhline(0, color="k", lw=1)
    ax.set_xlabel("Ground-truth length (cm)")
    ax.set_ylabel("Signed error: pred - gt (cm)")
    ax.set_title("Prediction bias vs. fish length")
    ax.legend(fontsize=7, markerscale=2)
    fig.tight_layout()
    p2 = os.path.join(out_dir, "error_vs_length.png")
    fig.savefig(p2, dpi=130)
    plt.close(fig)

    # Worst-N failure cases
    cols = [c for c in ["img_path", "gt_label", "subset", "gt_id",
                        "gt_length_cm", "pred_length_cm", "error", "iou"]
            if c in df.columns]
    worst = df.sort_values("abs_error", ascending=False).head(args.top_n)[cols]
    p3 = os.path.join(out_dir, f"worst_{args.top_n}.csv")
    worst.to_csv(p3, index=False)

    print(f"\nWorst {min(args.top_n, len(df))} cases -> {p3}")
    print(worst.head(10).to_string(index=False))
    print(f"\nwrote plots: {p1}\n             {p2}")


if __name__ == "__main__":
    main()
