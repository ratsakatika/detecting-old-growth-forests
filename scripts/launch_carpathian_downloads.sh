#!/usr/bin/env bash
# Launch the full Carpathian data acquisition for the stage-020 AOA analysis.
#
# Runs scripts/build_carpathian_massif.py first (every other script depends on
# the massif polygon), then the four acquisition scripts concurrently. Each
# script is idempotent and resumable, so re-running this launcher only fetches
# whatever is missing. Per-script logs live under each output directory
# (data/processed/rasters/carpathians/*/logs/ and
# data/processed/vectors/corine_land_cover/logs/).
#
# Usage:
#   scripts/launch_carpathian_downloads.sh [GRID_RES] [-- extra args ...]
#
# Examples:
#   scripts/launch_carpathian_downloads.sh            # 100 m analysis grid
#   scripts/launch_carpathian_downloads.sh 100 -- --dry-run
set -euo pipefail

cd "$(dirname "$0")/.."

GRID_RES="${1:-100}"
shift "$(( $# < 1 ? $# : 1 ))" || true
[[ "${1:-}" == "--" ]] && shift || true

PYTHON=".venv/bin/python"

echo "Extracting the Carpathian massif polygon (prerequisite)."
"${PYTHON}" -m scripts.build_carpathian_massif

echo "Launching acquisition scripts concurrently (grid ${GRID_RES} m)."
pids=()
names=()
for script in download_carpathians_corine download_carpathians_worldcover \
              download_carpathians_baseline download_carpathians_tessera; do
  args=()
  [[ "${script}" != "download_carpathians_corine" ]] && args+=(--grid-res "${GRID_RES}")
  echo "  ${script} ${args[*]} $*"
  "${PYTHON}" -m "scripts.${script}" "${args[@]}" "$@" &
  pids+=($!)
  names+=("${script}")
done

status=0
for i in "${!pids[@]}"; do
  if wait "${pids[$i]}"; then
    echo "DONE  ${names[$i]}"
  else
    echo "FAILED ${names[$i]} (idempotent: rerun this launcher to resume)"
    status=1
  fi
done
exit "${status}"
