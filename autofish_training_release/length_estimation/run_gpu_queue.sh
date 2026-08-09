#!/usr/bin/env bash
# Fire the GPU-hungry (fine-tuning) runs, in priority order, once the shared GPU
# has enough free memory. Resumable: each run is skipped if its model.pt exists.
#
# 1. paper            fine-tuned MobileNetV2 via the FAITHFUL train.py -> the
#                     0.82 cm reproduction that anchors every comparison.
# 2. x-vitsp-ft       fine-tuned DINOv2 CLS+patch (train_vfm.py) -> main
#                     contender to beat 0.82.
# 3. x-mnet-ft        fine-tuned MobileNetV2 under the SAME strong recipe ->
#                     fair, recipe-controlled encoder comparison.
# 4. x-vits-ft        fine-tuned DINOv2 CLS-only -> readout ablation.
# 5. x-vitbp-ft       fine-tuned DINOv2-B CLS+patch -> bigger-backbone push.
#
# Usage (once GPU is free):
#   GT=/path/annotations.json bash run_gpu_queue.sh
# Optionally gate on free memory (MiB): MIN_FREE_MIB=8000 bash run_gpu_queue.sh
set -uo pipefail
export MPLBACKEND=Agg
export PYTHONUNBUFFERED=1
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CNN="$HERE/cnn"
GT="${GT:-/workspace/autofish_dataset/annotations.json}"
PYTHON="${PYTHON:-python}"
MIN_FREE_MIB="${MIN_FREE_MIB:-0}"
LOGDIR="$CNN/output/logs"; mkdir -p "$LOGDIR"

if [[ "$MIN_FREE_MIB" != "0" ]]; then
    free=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1)
    if (( free < MIN_FREE_MIB )); then
        echo "GPU free ${free} MiB < required ${MIN_FREE_MIB} MiB — not starting."; exit 3
    fi
    echo "GPU free ${free} MiB >= ${MIN_FREE_MIB} MiB — proceeding."
fi

# --- 1. faithful paper reproduction (train.py) ---------------------------------
if [[ -f "$CNN/output/cnn-paper/model.pt" ]]; then
    echo "[skip] cnn-paper (model.pt exists)"
else
    echo "[train] cnn-paper (faithful train.py)"
    ( cd "$CNN" && "$PYTHON" train.py --config configs/paper.cfg ) \
        > "$LOGDIR/train-cnn-paper.log" 2>&1 \
        && echo "  paper trained" || echo "  !! paper FAILED (see log)"
fi
if [[ -f "$CNN/output/cnn-paper/model.pt" ]]; then
    ( cd "$HERE" && "$PYTHON" eval_length_estimators.py --gt_path "$GT" \
        --cnn_model_path cnn/output/cnn-paper/model.pt ) \
        > "$LOGDIR/eval-cnn-paper.log" 2>&1
    csv="$CNN/output/cnn-paper/eval/model/from-gt/eval-output.csv"
    [[ -f "$csv" ]] && ( cd "$HERE" && "$PYTHON" compute_metrics.py --csv "$csv" \
        --tag cnn-paper --runs-csv "$CNN/output/runs_summary.csv" ) \
        > "$LOGDIR/metrics-cnn-paper.log" 2>&1
fi

# --- 2..5 fine-tuned extension arms (train_vfm.py) -----------------------------
bash "$HERE/run_extension.sh" x-vitsp-ft x-mnet-ft x-vits-ft x-vitbp-ft

echo "==== GPU queue done ===="
( cd "$HERE" && "$PYTHON" collate_results.py --output-dir cnn/output \
    --csv cnn/output/results_table.csv ) || true
