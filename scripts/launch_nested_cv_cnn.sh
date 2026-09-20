#!/usr/bin/env bash
# Launch the receptive-field CNN run_nested_cv jobs for a chosen set of patch
# sizes at a chosen concurrency, as a bounded-concurrency pool over the
# (feature_set, patch_size) configurations. Splitting the matrix by patch size
# lets the heavy sizes run at lower concurrency: the P3/P5 patch caches are
# modest, but the P7 caches are very large on disk (about 111 GB for TESSERA,
# roughly 197 GB across the four feature sets). They are memory-mapped, so
# running fewer P7 configs at once keeps the OS page cache from thrashing.
#
# Per-run resources are sized from MAX_CONCURRENT: --n-jobs = cores /
# MAX_CONCURRENT (floored at 2), --torch-threads 2, and --gpu-slots =
# MAX_CONCURRENT so the file-lock GPU semaphore admits every concurrent fit
# rather than serialising them.
#
# Usage:
#   scripts/launch_nested_cv_cnn.sh MAX_CONCURRENT PATCH_SIZE [PATCH_SIZE ...] [-- extra args]
#
# Examples (run the phases one after another, not at the same time; ";" lets the
# second phase start even if a job in the first failed):
#   scripts/launch_nested_cv_cnn.sh 4 3 5 ; scripts/launch_nested_cv_cnn.sh 2 7
#   scripts/launch_nested_cv_cnn.sh 4 3 5 -- --smoke
#
# Resume note: --resume takes a per-config run id, so it does not compose with a
# multi-config launch; resume a single crashed config by calling run_nested_cv
# directly with --resume <run_id>.
set -euo pipefail
cd "$(dirname "$0")/.."

FEATURE_SETS=(baseline baseline_conventional_eo baseline_tessera baseline_alphaearth)

if (( $# < 1 )); then
  echo "usage: scripts/launch_nested_cv_cnn.sh MAX_CONCURRENT PATCH_SIZE [PATCH_SIZE ...]" \
       "[-- extra args]" >&2
  exit 2
fi
MAX_CONCURRENT="$1"
shift
# Positional patch sizes up to an optional "--"; everything after "--" is
# forwarded to each run_nested_cv invocation.
PATCH_SIZES=()
while (( $# )) && [[ "$1" != "--" ]]; do PATCH_SIZES+=("$1"); shift; done
[[ "${1:-}" == "--" ]] && shift || true
(( ${#PATCH_SIZES[@]} )) || { echo "error: give at least one patch size (3, 5 or 7)." >&2; exit 2; }
# Mirror utils.terminology.CNN_PATCH_SIZES; reject typos before launching jobs.
for p in "${PATCH_SIZES[@]}"; do
  case "${p}" in
    3 | 5 | 7) ;;
    *)
      echo "error: invalid patch size '${p}' (expected 3, 5 or 7)." >&2
      exit 2
      ;;
  esac
done

CORES="$(nproc)"
N_JOBS=$(( CORES / MAX_CONCURRENT < 2 ? 2 : CORES / MAX_CONCURRENT ))
TORCH_THREADS=2
PYTHON=".venv/bin/python"

echo "CNN phase: patch sizes [${PATCH_SIZES[*]}] x ${#FEATURE_SETS[@]} feature sets," \
     "max_concurrent=${MAX_CONCURRENT}, n_jobs=${N_JOBS}, torch_threads=${TORCH_THREADS}."

status=0
running=0
for p in "${PATCH_SIZES[@]}"; do
  for fs in "${FEATURE_SETS[@]}"; do
    echo "Launching CNN nested CV for ${fs} (patch ${p}x${p})."
    "${PYTHON}" -m scripts.run_nested_cv \
      --architecture "cnn_${p}x${p}" \
      --patch-size "${p}" \
      --feature-set "${fs}" \
      --n-jobs "${N_JOBS}" \
      --torch-threads "${TORCH_THREADS}" \
      --concurrency "${MAX_CONCURRENT}" \
      --gpu-slots "${MAX_CONCURRENT}" \
      "$@" &
    running=$(( running + 1 ))
    if (( running >= MAX_CONCURRENT )); then
      wait -n || status=1
      running=$(( running - 1 ))
    fi
  done
done
while (( running > 0 )); do
  wait -n || status=1
  running=$(( running - 1 ))
done

if (( status == 0 )); then
  echo "CNN phase complete (patch sizes [${PATCH_SIZES[*]}])."
else
  echo "CNN phase finished with at least one failed job." >&2
fi
exit "${status}"
