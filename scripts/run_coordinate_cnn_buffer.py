"""Score the coordinate-only control's CNN cells at the within-AOI and buffered arms.

The nested-CV run for ``(cnn_*, xy_coords)`` already gives the within-AOI (0 km) arm: its
held-out parcel predictions are the 0 km cell. This script adds the spatially-gapped arm by
refitting each outer fold with that fold's tuned hyperparameters on training data with the
buffer applied, exactly as ``scripts/run_performance_matrix.py`` does for the modelling
stacks -- it reuses that module's own helpers, so the control's cells are produced the same
way as every other cell in the figure.

One row per (architecture, buffer_km) is written to
``results/coordinate_cnn_buffer/<architecture>.csv`` with the pooled PR-AUC and its
block-bootstrap interval, in the column names ``performance_matrix.csv`` uses, so notebooks
013 and 022 can merge them straight into the matrix. Each invocation covers one buffer
distance and merges its arms into whatever the file already holds, so the arms accumulate
across runs instead of the later run clobbering the earlier one.

Run it once per architecture and buffer::

    python -m scripts.run_coordinate_cnn_buffer --architecture cnn_3x3
    python -m scripts.run_coordinate_cnn_buffer --architecture cnn_3x3 --buffer-km 20
"""

import argparse
import json
import logging
import os
from collections.abc import Sequence
from pathlib import Path
from typing import Final

import numpy as np
import pandas as pd

from scripts.run_gap_analysis import (
    _buffered_excluded_pids,
    _cnn_buffered_fold_results,
    _inner_thresholds,
    _latest_cnn_run,
    _load_cnn_fold_hps,
    _load_labelled_geoms,
)
from scripts.run_nested_cv import _patch_size_from_architecture
from utils.bootstrap import load_block_assignment
from utils.cv import NestedCVConfig
from utils.logging import make_run_logger
from utils.paths import ProjectPaths, ensure_writable_dirs, get_project_paths
from utils.reporting import METRIC_NAMES, metric_matrix_row
from utils.terminology import BOOTSTRAP_REPS_CROSS_CONFIG, FOLD_IDS, N_PER_PARCEL, SEED

logger = logging.getLogger(__name__)

_FEATURE_SET: Final[str] = "xy_coords"
_SOURCE_ROLE: Final[str] = "main_nested_cv"
_RESULTS_ROLE: Final[str] = "coordinate_cnn_buffer"
_LABELS_FILENAME: Final[str] = "ogf_reference_labels_partitioned.gpkg"
_DEFAULT_BUFFER_KM: Final[int] = 10
_DEFAULT_N_JOBS: Final[int] = min(8, os.cpu_count() or 1)


def _within_aoi_frames(run_dir: Path) -> dict[int, pd.DataFrame]:
    """Per-fold held-out parcel predictions from the nested-CV run (the 0 km arm)."""
    frames: dict[int, pd.DataFrame] = {}
    for fold in FOLD_IDS:
        path = run_dir / f"parcel_predictions_fold{fold}.parquet"
        if path.is_file():
            frames[fold] = pd.read_parquet(path).reset_index(drop=True)
    return frames


def _row(
    frames: dict[int, pd.DataFrame],
    inner: dict[int, float],
    block: pd.DataFrame,
    *,
    architecture: str,
    buffer_km: int,
    reps: int,
    seed: int,
    n_jobs: int,
) -> dict[str, object]:
    """One matrix row: pooled metrics and their intervals for a single arm."""
    stats = metric_matrix_row(frames, inner, block, reps=reps, seed=seed, n_jobs=n_jobs)
    row: dict[str, object] = {
        "feature_set": _FEATURE_SET,
        "architecture": architecture,
        "buffer_km": int(buffer_km),
        "folds_used": len(frames),
    }
    for metric in METRIC_NAMES:
        for stat in ("pooled", "ci_lo", "ci_hi"):
            row[f"{metric}__{stat}"] = stats[f"{metric}__{stat}"]
    return row


def _merge_rows(out_path: Path, rows: pd.DataFrame) -> pd.DataFrame:
    """Fold this run's arms into any arms already scored for the architecture.

    One invocation covers a single buffer distance, so the arms accumulate across runs
    (0/10 km from one, 20 km from the next). Rows are keyed on ``buffer_km``: rerunning an
    arm replaces it rather than duplicating it, and arms scored earlier are kept.
    """
    if out_path.is_file():
        existing = pd.read_csv(out_path)
        rows = pd.concat([existing, rows], ignore_index=True)
        rows = rows.drop_duplicates(subset=["buffer_km"], keep="last")
    return rows.sort_values("buffer_km").reset_index(drop=True)


def _run(args: argparse.Namespace, paths: ProjectPaths, run_dir: Path) -> Path:
    """Score both arms for one architecture and write the CSV."""
    run_logger = make_run_logger(__name__, run_dir)
    source = _latest_cnn_run(paths.results / _SOURCE_ROLE, args.architecture, _FEATURE_SET)
    run_logger.info("Source nested-CV run: %s", source.relative_to(paths.repo_root))

    best_params, cnn_epochs, _inputs = _load_cnn_fold_hps(source)
    if not best_params:
        raise FileNotFoundError(
            f"No completed folds with hyperparameters in {source}; the search must finish first."
        )
    run_logger.info("Folds with saved hyperparameters: %s", sorted(best_params))

    labels_path = paths.labels / _LABELS_FILENAME
    labelled = _load_labelled_geoms(labels_path)
    block = load_block_assignment(labels_path)
    inner = _inner_thresholds(source)

    # The patch cache and pixel index the CNN trains on (built by the search; reused here).
    from scripts.build_patch_cache import ensure_patch_cache
    from scripts.run_nested_cv import _load_pixel_index

    patch_size = _patch_size_from_architecture(args.architecture)
    cache_path = ensure_patch_cache(_FEATURE_SET, patch_size, paths=paths, logger=run_logger)
    patch_cache = np.load(cache_path, mmap_mode="r")
    manifest = json.loads(cache_path.with_suffix(".json").read_text(encoding="utf-8"))
    grid_shape = (int(manifest["grid_height"]), int(manifest["grid_width"]))
    pixel_index, _index_inputs = _load_pixel_index(paths, _FEATURE_SET, run_logger)

    config = NestedCVConfig(
        feature_set=_FEATURE_SET,
        architecture=args.architecture,
        sampling_mode="none",
        n_per_parcel=args.n_per_parcel,
        seed=args.seed,
        device=args.device,
        n_jobs=args.n_jobs,
        fold_ids=FOLD_IDS,
    )

    excluded = _buffered_excluded_pids(labelled, buffer_km=args.buffer_km, seed=args.seed)
    buffered, folds_used = _cnn_buffered_fold_results(
        config,
        pixel_index,
        patch_cache,
        best_params,
        cnn_epochs,
        excluded,
        patch_size=patch_size,
        grid_shape=grid_shape,
        run_logger=run_logger,
        config_key=f"{args.architecture}__{_FEATURE_SET}",
    )

    arms = {
        0: _within_aoi_frames(source),
        args.buffer_km: {
            int(fold): result.parcel_predictions
            for fold, result in zip(folds_used, buffered, strict=True)
        },
    }
    rows = []
    for buffer_km, frames in arms.items():
        if not frames:
            run_logger.warning("Arm %d km has no folds; skipped.", buffer_km)
            continue
        row = _row(
            frames,
            inner,
            block,
            architecture=args.architecture,
            buffer_km=buffer_km,
            reps=args.reps,
            seed=args.seed,
            n_jobs=args.n_jobs,
        )
        rows.append(row)
        run_logger.info(
            "%s at %d km (%d folds): pooled parcel PR-AUC %.4f [%.4f, %.4f].",
            args.architecture,
            buffer_km,
            row["folds_used"],
            row["pr_auc__pooled"],
            row["pr_auc__ci_lo"],
            row["pr_auc__ci_hi"],
        )

    out_path = run_dir / f"{args.architecture}.csv"
    table = _merge_rows(out_path, pd.DataFrame(rows))
    table.to_csv(out_path, index=False)
    run_logger.info("Wrote %s (arms: %s km).", out_path, list(table["buffer_km"]))
    return out_path


def _parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--architecture", required=True, choices=["cnn_3x3", "cnn_5x5", "cnn_7x7"])
    parser.add_argument("--buffer-km", type=int, default=_DEFAULT_BUFFER_KM)
    parser.add_argument("--device", default="cuda", choices=["cpu", "cuda"])
    parser.add_argument("--n-jobs", type=int, default=_DEFAULT_N_JOBS)
    parser.add_argument("--n-per-parcel", type=int, default=N_PER_PARCEL)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--reps", type=int, default=BOOTSTRAP_REPS_CROSS_CONFIG)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    """Score the coordinate-only CNN cells for one architecture.

    Args:
        argv: Argument list, or ``None`` to read from ``sys.argv``.

    Returns:
        Process exit code (0 on success).
    """
    args = _parse_args(argv)
    paths = get_project_paths()
    ensure_writable_dirs(paths)
    run_dir = paths.results / _RESULTS_ROLE
    run_dir.mkdir(parents=True, exist_ok=True)
    _run(args, paths, run_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
