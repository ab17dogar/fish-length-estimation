"""Length-estimation regression metrics from an eval-output.csv.

Reports, overall and split by subset (separated / touching / combined) and by
species:
  - MAE, MAPE            (absolute error; the paper's Table 4 metrics)
  - RMSE                 (penalises large misses, e.g. occlusion outliers)
  - R2                   (fraction of length variance explained)
  - median AE            (robust central error)
  - bias (signed ME)     (systematic over/under-estimation; + = over-estimate)

It also reports the PER-FISH-ID averaged-prediction error: every fish is imaged
~40 times, so averaging the predicted length per fish id (as the AutoFish
authors do) gives a lower-variance, deployment-relevant estimate.

Subsets follow AutoFish Fig. 3: within each group image index 1-40 are the
'separated' sets, 41-60 are the 'touching' set; 'combined' = all test fish.

Machine-readable output (for the label-efficiency aggregation): writes a full
metrics JSON next to the CSV (or to --out-json), and optionally appends one flat
summary row to --runs-csv, tagged with --tag.

    python compute_metrics.py --csv .../eval-output.csv [--tag vfm] \
        [--runs-csv output/label_efficiency_runs.csv]
"""
import argparse
import csv as _csv
import json
import os
import re

import numpy as np
import pandas as pd


def subset_for_path(img_path):
    m = re.search(r"(\d+)\.png$", os.path.basename(img_path))
    idx = int(m.group(1))
    return "separated" if idx <= 40 else "touching"


def _reg_metrics(gt, pred):
    gt = np.asarray(gt, dtype=float)
    pred = np.asarray(pred, dtype=float)
    err = pred - gt
    abs_err = np.abs(err)
    ss_res = float(np.sum(err ** 2))
    ss_tot = float(np.sum((gt - gt.mean()) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    return {
        "n": int(len(gt)),
        "mae": float(abs_err.mean()),
        "mape": float((abs_err / gt).mean() * 100.0),
        "rmse": float(np.sqrt((err ** 2).mean())),
        "r2": r2,
        "median_ae": float(np.median(abs_err)),
        "bias": float(err.mean()),
    }


def _per_fish_metrics(df):
    """Average the prediction over all views of each fish id, then score."""
    g = df.groupby("gt_id").agg(gt=("gt_length_cm", "first"),
                                pred=("pred_length_cm", "mean"))
    return _reg_metrics(g["gt"].to_numpy(), g["pred"].to_numpy())


def _print_table(title, rows):
    print(f"\n{title}")
    print(f"{'':16}{'MAE':>8}{'RMSE':>8}{'MAPE%':>8}{'R2':>8}"
          f"{'medAE':>8}{'bias':>8}{'N':>7}")
    print("-" * 71)
    for name, m in rows:
        print(f"{name:16}{m['mae']:>8.3f}{m['rmse']:>8.3f}{m['mape']:>8.2f}"
              f"{m['r2']:>8.3f}{m['median_ae']:>8.3f}{m['bias']:>8.3f}{m['n']:>7}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True)
    ap.add_argument("--tag", default=None,
                    help="run label for the machine-readable summary row")
    ap.add_argument("--conf-min", type=float, default=0.0,
                    help="drop rows with pred_conf below this (default 0 = keep all)")
    ap.add_argument("--out-json", default=None,
                    help="where to write the full metrics JSON (default: next to --csv)")
    ap.add_argument("--runs-csv", default=None,
                    help="append one flat summary row here (for label-efficiency aggregation)")
    args = ap.parse_args()

    df = pd.read_csv(args.csv)
    if args.conf_min > 0:
        df = df[df["pred_conf"].astype(float) >= args.conf_min]
    df["subset"] = df["img_path"].map(subset_for_path)

    tag = args.tag or os.path.basename(os.path.dirname(os.path.dirname(args.csv)))
    result = {"tag": tag, "csv": os.path.abspath(args.csv),
              "n_train_groups": None, "by_subset": {}, "by_species": {},
              "per_fish_combined": _per_fish_metrics(df)}

    subset_rows = []
    for name in ["separated", "touching"]:
        sub = df[df["subset"] == name]
        if len(sub):
            m = _reg_metrics(sub["gt_length_cm"], sub["pred_length_cm"])
            result["by_subset"][name] = m
            subset_rows.append((name, m))
    combined = _reg_metrics(df["gt_length_cm"], df["pred_length_cm"])
    result["by_subset"]["combined"] = combined
    subset_rows.append(("combined", combined))

    species_rows = []
    for sp in sorted(df["gt_label"].unique()):
        m = _reg_metrics(*[df[df["gt_label"] == sp][c]
                           for c in ("gt_length_cm", "pred_length_cm")])
        result["by_species"][sp] = m
        species_rows.append((sp, m))

    print(f"\nFile: {args.csv}   tag: {tag}")
    _print_table("By image subset:", subset_rows)
    _print_table("By species (combined subsets):", species_rows)
    pf = result["per_fish_combined"]
    print(f"\nPer-fish-id averaged prediction (combined):  "
          f"MAE {pf['mae']:.3f} cm  MAPE {pf['mape']:.2f}%  "
          f"bias {pf['bias']:.3f} cm  (N fish = {pf['n']})")

    out_json = args.out_json or os.path.join(os.path.dirname(args.csv), "metrics.json")
    with open(out_json, "w") as f:
        json.dump(result, f, indent=2)
    print(f"\nwrote {out_json}")

    if args.runs_csv:
        c = result["by_subset"]["combined"]
        s = result["by_subset"].get("separated", {})
        t = result["by_subset"].get("touching", {})
        row = {
            "tag": tag,
            "combined_mae": c["mae"], "combined_mape": c["mape"],
            "combined_rmse": c["rmse"], "combined_r2": c["r2"], "combined_bias": c["bias"],
            "separated_mae": s.get("mae"), "touching_mae": t.get("mae"),
            "perfish_mae": pf["mae"], "n": c["n"],
        }
        exists = os.path.isfile(args.runs_csv)
        os.makedirs(os.path.dirname(os.path.abspath(args.runs_csv)), exist_ok=True)
        with open(args.runs_csv, "a", newline="") as f:
            w = _csv.DictWriter(f, fieldnames=list(row.keys()))
            if not exists:
                w.writeheader()
            w.writerow(row)
        print(f"appended summary row (tag={tag}) to {args.runs_csv}")


if __name__ == "__main__":
    main()
