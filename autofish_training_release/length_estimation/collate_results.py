"""Collate every run's metrics.json into one comparison table (our Table 4).

Scans cnn/output/*/eval/model/from-gt/metrics.json (written by compute_metrics.py)
and prints a table of single-shot test MAE on the separated / touching / combined
subsets, plus the per-fish-averaged combined MAE, one row per run. Two fixed
reference rows are always included for context:

    REG^gt (paper)   0.67 / 0.96 / 0.82   the baseline we aim to beat (GT masks)
    bbox-only floor  0.73 / 1.19 / 0.96   our control: 4 bbox coords, no image

Rows are sorted by combined MAE (best first) so it is obvious at a glance which
configurations beat the 0.82 cm baseline. Emits Markdown to stdout and, with
--csv, a machine-readable copy.

    python collate_results.py [--output-dir cnn/output] [--csv cnn/output/results_table.csv]
"""
import argparse
import glob
import json
import os


REFERENCE_ROWS = [
    ("REG^gt (paper baseline)", 0.67, 0.96, 0.82, None),
    ("bbox-only floor (ours)", 0.73, 1.19, 0.96, None),
]


def load_runs(output_dir):
    rows = []
    for mj in sorted(glob.glob(
            os.path.join(output_dir, "*", "eval", "model", "from-gt",
                         "metrics.json"))):
        run = mj.split(os.sep)[-5]
        try:
            m = json.load(open(mj))
        except Exception:
            continue
        bs = m.get("by_subset", {})

        def mae(k):
            return bs.get(k, {}).get("mae", float("nan"))
        per_fish = m.get("per_fish_combined", {}).get("mae", None)
        rows.append((run, mae("separated"), mae("touching"),
                     mae("combined"), per_fish))
    return rows


def fmt(x):
    return "  -  " if x is None else f"{x:.3f}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--output-dir", default="cnn/output")
    ap.add_argument("--csv", default=None)
    args = ap.parse_args()

    runs = load_runs(args.output_dir)
    runs.sort(key=lambda r: (r[3] if r[3] == r[3] else 9e9))  # nan-safe
    all_rows = REFERENCE_ROWS + [("", None, None, None, None)] + runs

    header = ["run/config", "separated", "touching", "combined", "per-fish(comb)"]
    print("| " + " | ".join(header) + " |")
    print("|" + "|".join(["---"] * len(header)) + "|")
    for name, sep, tou, comb, pf in all_rows:
        beat = ""
        if comb is not None and comb == comb and name not in dict(
                (r[0], r) for r in REFERENCE_ROWS):
            beat = "  ✅" if comb < 0.82 else ""
        print(f"| {name}{beat} | {fmt(sep)} | {fmt(tou)} | {fmt(comb)} | {fmt(pf)} |")

    if args.csv:
        import csv as _csv
        with open(args.csv, "w", newline="") as f:
            w = _csv.writer(f)
            w.writerow(header)
            for name, sep, tou, comb, pf in all_rows:
                if not name:
                    continue
                w.writerow([name, sep, tou, comb, pf])
        print(f"\nwrote {args.csv}")


if __name__ == "__main__":
    main()
