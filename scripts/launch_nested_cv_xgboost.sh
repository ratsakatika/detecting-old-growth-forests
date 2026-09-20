#!/usr/bin/env bash
# Launch one XGBoost run_nested_cv.py process per feature set concurrently.
#
# The feature set is the parallel axis: concurrent processes share the OS page
# cache for the memmaps, so each cached matrix is read once and reused. Per-fit
# CPU threads default inside run_nested_cv.py to (cores // concurrency); the GPU
# semaphore caps concurrent GPU fits. The CNN launcher runs its matrix
# concurrently too, with a tunable cap; see scripts/launch_nested_cv_cnn.sh.
#
# Usage:
#   scripts/launch_nested_cv_xgboost.sh [N_JOBS] [GPU_SLOTS] [-- extra args ...]
#
# Examples:
#   scripts/launch_nested_cv_xgboost.sh                 # 8 threads each, 4 GPU slots
#   scripts/launch_nested_cv_xgboost.sh 8 4 --sampling-mode inverse_prevalence
#   scripts/launch_nested_cv_xgboost.sh 8 2 --smoke
set -euo pipefail

cd "$(dirname "$0")/.."

FEATURE_SETS=(baseline baseline_conventional_eo baseline_tessera baseline_alphaearth)
N_JOBS="${1:-8}"
GPU_SLOTS="${2:-4}"
shift "$(( $# < 2 ? $# : 2 ))" || true
[[ "${1:-}" == "--" ]] && shift || true

PYTHON=".venv/bin/python"
pids=()
for fs in "${FEATURE_SETS[@]}"; do
  echo "Launching nested CV for ${fs} (n_jobs=${N_JOBS}, gpu_slots=${GPU_SLOTS})."
  "${PYTHON}" -m scripts.run_nested_cv \
    --architecture xgboost \
    --feature-set "${fs}" \
    --concurrency "${#FEATURE_SETS[@]}" \
    --n-jobs "${N_JOBS}" \
    --gpu-slots "${GPU_SLOTS}" \
    "$@" &
  pids+=($!)
done

status=0
for pid in "${pids[@]}"; do
  wait "${pid}" || status=1
done
exit "${status}"
