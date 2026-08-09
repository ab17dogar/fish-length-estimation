"""Multi-sample length accuracy: MAE vs. number of samples aggregated per fish.

The AutoFish dataset images every fish ~40 times. The paper's Fig. 10 shows that
taking the MEDIAN predicted length over several images of the same fish drops the
MAE dramatically (combined REG: 0.99 cm at 1 sample -> ~0.4 cm at 40), and flags
maintaining fish IDs via tracking / re-identification as future work. This script
reproduces that analysis for our models and, crucially, lets several models be
overlaid so we can show which encoder aggregates best.

For each model's eval-output.csv and each sample budget k, we repeatedly draw k
random predictions per fish (only fish with >= k samples), take the median,
and average |median - gt| over fish; the mean +/- std over the random draws gives
the curve and its band. The k=1 point is the ordinary single-shot MAE.

This is a deployment-relevant, honest way to beat the single-shot baseline: it is
applied identically to every model, and it is exactly the "with IDs" scenario the
authors describe. Report it ALONGSIDE the single-shot numbers, never as a
replacement for them.

    python aggregation_curve.py \
        --run cnn/output/x-vitsp-ft/eval/model/from-gt/eval-output.csv:Ours \
        --run cnn/output/cnn-paper/eval/model/from-gt/eval-output.csv:REG(paper) \
        --subset combined --out cnn/output/aggregation_curve.png
"""
import argparse
import os
import re

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def subset_mask(df, subset):
    idx = df["img_path"].map(
        lambda p: int(re.search(r"(\d+)\.png$", os.path.basename(p)).group(1)))
    if subset == "separated":
        return idx <= 40
    if subset == "touching":
        return idx > 40
    return pd.Series(True, index=df.index)  # combined


def aggregation_curve(df, ks, n_draws=200, seed=0):
    """MAE(mean,std) of per-fish median length over k random samples, per k."""
    rng = np.random.default_rng(seed)
    by_fish = [g[["gt_length_cm", "pred_length_cm"]].to_numpy()
               for _, g in df.groupby("gt_id")]
    means, stds = [], []
    for k in ks:
        pools = [a for a in by_fish if len(a) >= k]
        if not pools:
            means.append(np.nan); stds.append(np.nan); continue
        draw_maes = []
        for _ in range(n_draws):
            errs = []
            for a in pools:
                sel = rng.choice(len(a), size=k, replace=False)
                med = np.median(a[sel, 1])
                errs.append(abs(med - a[sel, 0][0]))  # gt is constant per fish
            draw_maes.append(np.mean(errs))
        means.append(float(np.mean(draw_maes)))
        stds.append(float(np.std(draw_maes)))
    return np.array(means), np.array(stds)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", action="append", required=True,
                    help="path/to/eval-output.csv:Label (repeatable)")
    ap.add_argument("--subset", default="combined",
                    choices=["separated", "touching", "combined"])
    ap.add_argument("--max-k", type=int, default=20)
    ap.add_argument("--draws", type=int, default=200)
    ap.add_argument("--out", default="cnn/output/aggregation_curve.png")
    args = ap.parse_args()

    ks = list(range(1, args.max_k + 1))
    plt.figure(figsize=(6, 4))
    print(f"subset={args.subset}   MAE (cm) by #samples/fish")
    for spec in args.run:
        path, _, label = spec.partition(":")
        label = label or os.path.basename(os.path.dirname(os.path.dirname(path)))
        df = pd.read_csv(path)
        df = df[subset_mask(df, args.subset)]
        mean, std = aggregation_curve(df, ks, n_draws=args.draws)
        plt.plot(ks, mean, marker="o", ms=3, label=label)
        plt.fill_between(ks, mean - std, mean + std, alpha=0.15)
        shown = {1: mean[0], 5: mean[4] if len(mean) > 4 else np.nan,
                 args.max_k: mean[-1]}
        print(f"  {label:22s} k=1 {shown[1]:.3f} | k=5 {shown[5]:.3f} | "
              f"k={args.max_k} {shown[args.max_k]:.3f}")
    plt.axhline(0.82, ls="--", c="gray", lw=1, label="REG$^{gt}$ single-shot (0.82)")
    plt.xlabel("samples aggregated per fish (median)")
    plt.ylabel(f"MAE (cm) — {args.subset}")
    plt.title("Length MAE vs. multi-sample aggregation")
    plt.legend(fontsize=8); plt.grid(alpha=0.3); plt.tight_layout()
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    plt.savefig(args.out, dpi=140)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
