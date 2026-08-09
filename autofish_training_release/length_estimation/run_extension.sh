#!/usr/bin/env bash
# Train (via train_vfm.py) -> eval on GT masks -> compute metrics, for the
# extension configs passed as arguments (paths or bare tags under
# cnn/configs/extension/). Resumable: skips a run whose model.pt already exists
# unless FORCE=1. Per-run logs in cnn/output/logs/.
#
#   GT=/path/annotations.json PYTHON=.../python bash run_extension.sh x-vitsp-ft x-vits-frozen
#   bash run_extension.sh cnn/configs/extension/*.cfg
set -uo pipefail
export MPLBACKEND=Agg
export PYTHONUNBUFFERED=1

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CNN="$HERE/cnn"
GT="${GT:-/workspace/autofish_dataset/annotations.json}"
PYTHON="${PYTHON:-python}"
FORCE="${FORCE:-0}"
LOGDIR="$CNN/output/logs"; mkdir -p "$LOGDIR"
RUNS_CSV="$CNN/output/runs_summary.csv"

cfg_path() {  # tag or path -> absolute cfg path
    case "$1" in
        *.cfg) [[ "$1" = /* ]] && echo "$1" || echo "$HERE/$1" ;;
        *) echo "$CNN/configs/extension/$1.cfg" ;;
    esac
}
cfg_name() { grep -E '^OUTPUT_DIR' "$1" | sed -E 's#.*/output/##; s#/##g; s#[[:space:]]##g'; }

FAILS=()
for arg in "$@"; do
    cfg="$(cfg_path "$arg")"
    [[ -f "$cfg" ]] || { echo "!! no config: $arg"; FAILS+=("cfg:$arg"); continue; }
    name="$(cfg_name "$cfg")"
    model="$CNN/output/$name/model.pt"

    if [[ "$FORCE" != "1" && -f "$model" ]]; then
        echo "[skip-train] $name"
    else
        echo "[train] $name"
        if ! ( cd "$CNN" && "$PYTHON" train_vfm.py --config "$cfg" ) \
                > "$LOGDIR/train-$name.log" 2>&1; then
            echo "  !! train FAILED: $name (see $LOGDIR/train-$name.log)"; FAILS+=("train:$name"); continue
        fi
    fi

    echo "[eval]  $name"
    if ! ( cd "$HERE" && "$PYTHON" eval_length_estimators.py \
                --gt_path "$GT" --cnn_model_path "cnn/output/$name/model.pt" ) \
            > "$LOGDIR/eval-$name.log" 2>&1; then
        echo "  !! eval FAILED: $name"; FAILS+=("eval:$name"); continue
    fi
    csv="$CNN/output/$name/eval/model/from-gt/eval-output.csv"
    [[ -f "$csv" ]] || { echo "  !! no eval CSV: $name"; FAILS+=("evalcsv:$name"); continue; }

    echo "[metrics] $name"
    ( cd "$HERE" && "$PYTHON" compute_metrics.py --csv "$csv" --tag "$name" --runs-csv "$RUNS_CSV" ) \
        > "$LOGDIR/metrics-$name.log" 2>&1 || FAILS+=("metrics:$name")
done

echo "==== done ===="
if [[ ${#FAILS[@]} -eq 0 ]]; then echo "failures: none"; else printf 'FAIL %s\n' "${FAILS[@]}"; fi
