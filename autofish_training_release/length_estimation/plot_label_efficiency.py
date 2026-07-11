"""Aggregate label-efficiency runs into a MAE-vs-training-data curve.

Consumes the per-run metrics written by compute_metrics.py. Point it at the
directory holding the le-* run folders (each with eval/model/from-gt/metrics.json)
OR at a runs CSV produced with compute_metrics.py --runs-csv.

Run tags follow le-<arm>-<NN>grp-s<sid> (see configs/generate_label_efficiency.py),
so the arm and #training-groups are parsed from the tag. For each (arm, #groups)
it plots the mean combined-test MAE across the random subsets, with a shaded
min-max band, so the baseline (fine-tuned / frozen) and the VFM can be compared
as labeled data shrink.

    # after running compute_metrics.py on every le-* run:
    python plot_label_efficiency.py --runs-dir cnn/output
    # or, from an aggregated CSV:
    python plot_label_efficiency.py --runs-csv cnn/output/label_efficiency_runs.csv
"""
import argparse
import glob
import json
import os
import re
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

TAG_RE = re.compile(r"le-(?P<arm>.+)-(?P<n>\d+)grp-s(?P<sid>\d+)")
ARM_STYLE = {
    "baseline_ft": ("MobileNetV2 (fine-tuned)", "tab:orange", "o"),
    "baseline_frozen": ("MobileNetV2 (frozen)", "tab:green", "s"),
    "vfm": ("DINOv2 ViT-S/14 (frozen)", "tab:blue", "^"),
}


def rows_from_runs_dir(runs_dir):
    rows = []
    for mj in glob.glob(os.path.join(runs_dir, "le-*", "eval", "model",
                                     "from-gt", "metrics.json")):
        with open(mj) as f:
            m = json.load(f)
        rows.append({"tag": m["tag"],
                     "combined_mae": m["by_subset"]["combined"]["mae"]})
    return rows


def rows_from_csv(path):
    df = pd.read_csv(path)
    return df[["tag", "combined_mae"]].to_dict("records")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs-dir", default=None,
                    help="dir containing le-* run folders with metrics.json")
    ap.add_argument("--runs-csv", default=None,
                    help="aggregated CSV from compute_metrics.py --runs-csv")
    ap.add_argument("--out", default="label_efficiency_curve.png")
    args = ap.parse_args()

    if args.runs_csv:
        rows = rows_from_csv(args.runs_csv)
    elif args.runs_dir:
        rows = rows_from_runs_dir(args.runs_dir)
    else:
        ap.error("give --runs-dir or --runs-csv")

    # (arm, n_groups) -> [combined MAE across subsets]
    series = defaultdict(lambda: defaultdict(list))
    for r in rows:
        m = TAG_RE.search(r["tag"])
        if not m:
            print(f"skip unparseable tag: {r['tag']}")
            continue
        series[m["arm"]][int(m["n"])].append(float(r["combined_mae"]))

    fig, ax = plt.subplots(figsize=(7, 5))
    summary = []
    for arm in sorted(series, key=lambda a: list(ARM_STYLE).index(a) if a in ARM_STYLE else 99):
        label, color, marker = ARM_STYLE.get(arm, (arm, None, "o"))
        ns = sorted(series[arm])
        means = [float(np.mean(series[arm][n])) for n in ns]
        los = [float(np.min(series[arm][n])) for n in ns]
        his = [float(np.max(series[arm][n])) for n in ns]
        ax.plot(ns, means, marker=marker, color=color, label=label)
        ax.fill_between(ns, los, his, color=color, alpha=0.15)
        for n, mu in zip(ns, means):
            summary.append((arm, n, mu, len(series[arm][n])))

    ax.set_xlabel("Number of training groups (labeled data)")
    ax.set_ylabel("Combined-test MAE (cm)")
    ax.set_title("Label efficiency: length-estimation error vs. training data")
    ax.set_xticks([1, 2, 4, 8, 15])
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(args.out, dpi=140)
    print(f"wrote {args.out}\n")
    print(f"{'arm':18}{'#groups':>9}{'mean MAE':>10}{'#subsets':>10}")
    for arm, n, mu, k in summary:
        print(f"{arm:18}{n:>9}{mu:>10.3f}{k:>10}")


if __name__ == "__main__":
    main()
