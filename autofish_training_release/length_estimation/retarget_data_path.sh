#!/usr/bin/env bash
# =============================================================================
# Point every config's MASKS_GT at a local dataset copy.
#
# The configs ship with the authors' Docker path
# (/workspace/autofish_dataset/annotations.json), which is correct when the
# dataset is bind-mounted by docker/run_docker.sh. On a machine where Docker is
# unavailable and the study runs in a plain virtualenv, the dataset lives at an
# arbitrary path instead, and train.py reads MASKS_GT straight from the config
# (only eval takes --gt_path, which run_all.sh already supplies via GT=).
#
# This rewrites MASKS_GT in place. It is idempotent (safe to re-run) and
# reversible (pass the Docker path to restore it). Re-run it after
# generate_label_efficiency.py, which re-emits configs with the Docker default.
#
#   bash retarget_data_path.sh /abs/path/to/autofish_data/annotations.json
#   bash retarget_data_path.sh /workspace/autofish_dataset/annotations.json  # restore
# =============================================================================
set -euo pipefail

GT="${1:?usage: retarget_data_path.sh /abs/path/to/annotations.json}"
[[ -f "$GT" ]] || { echo "no such annotations file: $GT" >&2; exit 1; }

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
n=0
while IFS= read -r -d '' cfg; do
    # MASKS_GT is written as "MASKS_GT: <path>"; replace only the value.
    sed -i -E "s#^(MASKS_GT:[[:space:]]*).*#\1$GT#" "$cfg"
    n=$((n+1))
done < <(find "$HERE/cnn/configs" -name '*.cfg' -print0)

echo "retargeted MASKS_GT -> $GT  ($n config files)"
