#!/usr/bin/env bash
# =============================================================================
# One-shot runner for the AutoFish length-estimation study:
#   train -> evaluate (on GT masks) -> compute metrics -> analysis/plots,
# for the baseline, the VFM variant, the 2x2 encoder/regime matrix, and the
# label-efficiency sweep.
#
# WHERE TO RUN: inside the authors' Docker container (see the top-level
# README's Docker workflow), where the repo is mounted at
# /workspace/autofish_training and the dataset at /workspace/autofish_dataset.
# Run it from this directory (length_estimation/):
#
#     cd /workspace/autofish_training/length_estimation
#     bash run_all.sh            # everything: main matrix -> label efficiency -> analysis
#
# PHASES (first argument):
#     smoke      quick 1-group/1-epoch VFM sanity check (~1 min) then stop
#     main       train+eval+metrics for the 4 matrix configs
#                (paper, baseline_frozen, vfm, vfm_finetune)
#     le         train+eval+metrics for the 39 label-efficiency configs
#     analysis   failure/bias analysis on the main models + label-efficiency curve
#     all        main -> le -> analysis   (default)
#
# USEFUL ENV VARS:
#     GT=/path/annotations.json   dataset annotations (default: Docker mount)
#     PYTHON=python               interpreter (default: python)
#     FORCE=1                     retrain even if a model.pt already exists
#                                 (default: skip already-trained runs -> resumable)
#
# The script CONTINUES past a failed run and prints a summary of failures at the
# end; per-run stdout/stderr go to cnn/output/logs/. Re-running resumes: any run
# whose model.pt already exists is skipped unless FORCE=1.
# =============================================================================
set -uo pipefail

# Force matplotlib's non-interactive backend so plotting works on a headless
# server (the Docker run passes -e DISPLAY, which is invalid with no X server).
export MPLBACKEND=Agg

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"   # .../length_estimation
CNN="$HERE/cnn"
GT="${GT:-/workspace/autofish_dataset/annotations.json}"
PYTHON="${PYTHON:-python}"
FORCE="${FORCE:-0}"
PHASE="${1:-all}"

LOGDIR="$CNN/output/logs"
RUNS_CSV="$CNN/output/runs_summary.csv"
mkdir -p "$LOGDIR"

MAIN_CFGS=(paper baseline_frozen vfm vfm_finetune)

FAILURES=()
N_TRAINED=0
N_EVALED=0

# OUTPUT_DIR basename (= run name) from a config file.
cfg_name() { grep -E '^OUTPUT_DIR' "$1" | sed -E 's#.*/output/##; s#/##g; s#[[:space:]]##g'; }

train_one() {  # $1 = absolute cfg path
    local cfg="$1" name model
    name="$(cfg_name "$cfg")"
    model="$CNN/output/$name/model.pt"
    if [[ "$FORCE" != "1" && -f "$model" ]]; then
        echo "  [skip-train] $name (model.pt exists; FORCE=1 to retrain)"
        return 0
    fi
    echo "  [train] $name"
    if ( cd "$CNN" && "$PYTHON" train.py --config "$cfg" ) \
            > "$LOGDIR/train-$name.log" 2>&1; then
        N_TRAINED=$((N_TRAINED+1))
    else
        echo "    !! train FAILED: $name  (see $LOGDIR/train-$name.log)"
        FAILURES+=("train:$name")
    fi
}

eval_one() {  # $1 = run name
    local name="$1" model csv
    model="$CNN/output/$name/model.pt"
    [[ -f "$model" ]] || { echo "  [skip-eval] $name (no model.pt)"; return 0; }
    echo "  [eval]  $name"
    if ! ( cd "$HERE" && "$PYTHON" eval_length_estimators.py \
                --gt_path "$GT" --cnn_model_path "cnn/output/$name/model.pt" ) \
            > "$LOGDIR/eval-$name.log" 2>&1; then
        echo "    !! eval FAILED: $name  (see $LOGDIR/eval-$name.log)"
        FAILURES+=("eval:$name"); return 0
    fi
    csv="$CNN/output/$name/eval/model/from-gt/eval-output.csv"
    [[ -f "$csv" ]] || { echo "    !! no eval CSV: $name"; FAILURES+=("evalcsv:$name"); return 0; }
    if ( cd "$HERE" && "$PYTHON" compute_metrics.py \
                --csv "$csv" --tag "$name" --runs-csv "$RUNS_CSV" ) \
            > "$LOGDIR/metrics-$name.log" 2>&1; then
        N_EVALED=$((N_EVALED+1))
    else
        echo "    !! metrics FAILED: $name  (see $LOGDIR/metrics-$name.log)"
        FAILURES+=("metrics:$name")
    fi
}

do_main() {
    echo "== PHASE main: 4-config encoder/regime matrix =="
    for c in "${MAIN_CFGS[@]}"; do train_one "$CNN/configs/$c.cfg"; done
    for c in "${MAIN_CFGS[@]}"; do eval_one "$(cfg_name "$CNN/configs/$c.cfg")"; done
}

do_le() {
    echo "== PHASE le: label-efficiency sweep (39 runs) =="
    [[ -d "$CNN/configs/label_efficiency" ]] || \
        ( cd "$CNN/configs" && "$PYTHON" generate_label_efficiency.py )
    # cheapest/most-important arms first (vfm, frozen baseline), then fine-tuned.
    local cfgs=()
    for arm in vfm baseline_frozen baseline_ft; do
        for f in "$CNN"/configs/label_efficiency/le-"$arm"-*.cfg; do
            [[ -e "$f" ]] && cfgs+=("$f")
        done
    done
    for cfg in "${cfgs[@]}"; do train_one "$cfg"; done
    for cfg in "${cfgs[@]}"; do eval_one "$(cfg_name "$cfg")"; done
}

do_analysis() {
    echo "== PHASE analysis: failure/bias + label-efficiency curve =="
    for name in cnn-paper cnn-baseline-frozen cnn-vfm cnn-vfm-finetune; do
        csv="$CNN/output/$name/eval/model/from-gt/eval-output.csv"
        [[ -f "$csv" ]] || continue
        echo "  [analyze] $name"
        ( cd "$HERE" && "$PYTHON" analyze_predictions.py --csv "$csv" ) \
            > "$LOGDIR/analysis-$name.log" 2>&1 || FAILURES+=("analyze:$name")
    done
    echo "  [curve] label_efficiency_curve.png"
    ( cd "$HERE" && "$PYTHON" plot_label_efficiency.py \
            --runs-dir cnn/output --out cnn/output/label_efficiency_curve.png ) \
        > "$LOGDIR/le-curve.log" 2>&1 || FAILURES+=("le-curve")
}

do_smoke() {
    echo "== PHASE smoke: VFM sanity check =="
    train_one "$CNN/configs/vfm_smoke.cfg"
    eval_one "$(cfg_name "$CNN/configs/vfm_smoke.cfg")"
}

echo "AutoFish length-estimation runner"
echo "  phase=$PHASE  GT=$GT  PYTHON=$PYTHON  FORCE=$FORCE"
echo "  logs -> $LOGDIR ; summary CSV -> $RUNS_CSV"
echo

case "$PHASE" in
    smoke)    do_smoke ;;
    main)     do_main ;;
    le)       do_le ;;
    analysis) do_analysis ;;
    all)      do_main; do_le; do_analysis ;;
    *) echo "unknown phase '$PHASE' (use: smoke|main|le|analysis|all)"; exit 2 ;;
esac

echo
echo "==================== SUMMARY ===================="
echo "trained: $N_TRAINED   evaluated+metrics: $N_EVALED"
if [[ ${#FAILURES[@]} -eq 0 ]]; then
    echo "failures: none"
else
    echo "failures (${#FAILURES[@]}):"
    printf '  - %s\n' "${FAILURES[@]}"
fi
echo "per-run metrics.json live under cnn/output/<run>/eval/model/from-gt/"
echo "summary rows: $RUNS_CSV"
[[ "$PHASE" == "all" || "$PHASE" == "analysis" ]] && \
    echo "label-efficiency curve: cnn/output/label_efficiency_curve.png"
