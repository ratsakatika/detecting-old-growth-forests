"""Parcel-level performance matrix and pre-specified contrasts (0 km and buffered arms).

For every feature set x architecture the script reports the eight metric families of
:mod:`utils.reporting` at the headline (0 km) and at each training-exclusion buffer in
``_BUFFER_KMS`` (10 and 20 km). The 0 km arm reads the main nested-CV out-of-fold parcel
predictions; each buffered arm is a fixed-HP buffered refit reusing the main-CV best
params (no Optuna). A configuration whose main-CV run has not been trained yet is written
as a pending (NaN) row and fills in on a re-run, so the matrix tracks the still-running
CNN matrix. Every refit and inference runs on the configured device (GPU by default). The
buffered refits are cached per source run and buffer under
``data/cache/performance_matrix_buffered/``, so a re-run only recomputes cells whose
main-CV run changed (or gained folds); pass ``--refresh-buffered`` to recompute every cell.

The tables under ``results/performance_matrix/<run_id>/`` carry one row per buffer in
``_BUFFERS`` (0 km plus ``_BUFFER_KMS``):

  - ``performance_matrix.csv``   - 4 feature sets x 4 architectures x |_BUFFERS| rows.
  - ``feature_set_contrasts.csv``- (conventional_eo, alphaearth, tessera) minus baseline
    (XGBoost) at each buffer, paired block-bootstrap differences.
  - ``embedding_contrasts.csv``  - the three EO-representation pairwise contrasts at each buffer.
  - ``architecture_contrasts.csv``- (cnn 3x3/5x5/7x7) minus xgboost, four feature sets,
    at each buffer, paired block-bootstrap differences.

Each row carries the 88 ``{metric}__{stat}`` columns; contrasts add a raw two-sided
p per metric and a Holm-adjusted p plus rejection flag for the PR-AUC family. The
notebook ``013`` displays the tables; computation lives here (mirroring 010/013).
"""

from __future__ import annotations

import argparse
import json
import logging
import os
from collections.abc import Sequence
from datetime import UTC, datetime
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Final

import numpy as np
import pandas as pd

from scripts.run_gap_analysis import (
    _BEST_PARAMS_TEMPLATE,
    _PARCEL_PRED_TEMPLATE,
    _buffered_excluded_pids,
    _cnn_buffered_fold_results,
    _latest_cnn_run,
    _load_cnn_fold_hps,
    _load_labelled_geoms,
    _xgb_buffered_fold_results,
)
from scripts.run_sampling_sensitivity import _latest_run, _load_best_params, _load_pixel_data
from utils.bootstrap import holm_adjusted_p_values, load_block_assignment
from utils.cv import NestedCVConfig
from utils.env_capture import write_environment_snapshot
from utils.io import ColumnSpec, Schema, write_csv, write_json
from utils.logging import make_run_logger
from utils.manifest import write_input_manifest
from utils.paths import ProjectPaths, ensure_writable_dirs, get_project_paths
from utils.reporting import (
    METRIC_NAMES,
    matrix_columns,
    metric_matrix_row,
    paired_row,
)
from utils.terminology import (
    ARCHITECTURES,
    BOOTSTRAP_REPS_CROSS_CONFIG,
    FEATURE_SETS,
    FOLD_IDS,
    N_PER_PARCEL,
    SEED,
)

logger = logging.getLogger(__name__)

_LOGGER_NAME: Final[str] = "run_performance_matrix"
_SOURCE_ROLE: Final[str] = "main_nested_cv"
_BUFFERED_CACHE: Final[str] = "performance_matrix_buffered"
_RESULTS_ROLE: Final[str] = "performance_matrix"
_LABELS_FILENAME: Final[str] = "ogf_reference_labels_partitioned.gpkg"
_XGB: Final[str] = "xgboost"
_BASELINE: Final[str] = "baseline"
# Non-zero training-exclusion buffers (km): each is a fixed-HP buffered refit reported
# alongside the 0 km headline arm. Add a distance here to report a further buffer.
_BUFFER_KMS: Final[tuple[int, ...]] = (10, 20)
_BUFFERS: Final[tuple[int, ...]] = (0, *_BUFFER_KMS)
_HP_TRIALS_TEMPLATE: Final[str] = "hp_trials_fold{fold}.csv"
# Feature sets contrasted against the baseline (the EO-signal family).
_FS_CONTRASTS: Final[tuple[str, ...]] = (
    "baseline_conventional_eo",
    "baseline_alphaearth",
    "baseline_tessera",
)
# Pairwise contrasts among the EO representations (are they indistinguishable?).
_EMBEDDING_CONTRASTS: Final[tuple[tuple[str, str], ...]] = (
    ("baseline_alphaearth", "baseline_conventional_eo"),
    ("baseline_tessera", "baseline_conventional_eo"),
    ("baseline_tessera", "baseline_alphaearth"),
)
_HOLM_ALPHA: Final[float] = 0.05

_FLOAT = ColumnSpec("float64")
_STR = ColumnSpec("object")
_INT = ColumnSpec("int64")
_BOOL = ColumnSpec("bool")
_METRIC_COLUMNS: Final[dict[str, ColumnSpec]] = {column: _FLOAT for column in matrix_columns()}
_P_COLUMNS: Final[dict[str, ColumnSpec]] = {f"{name}__p": _FLOAT for name in METRIC_NAMES}

MATRIX_SCHEMA: Final[Schema] = {
    "feature_set": _STR,
    "architecture": _STR,
    "buffer_km": _INT,
    "folds_used": _INT,
    **_METRIC_COLUMNS,
}

CONTRAST_SCHEMA: Final[Schema] = {
    "contrast": _STR,
    "set_a": _STR,
    "set_b": _STR,
    "buffer_km": _INT,
    "folds_used": _INT,
    **_METRIC_COLUMNS,
    **_P_COLUMNS,
    "pr_auc__p_holm": _FLOAT,
    "pr_auc__reject_holm": _BOOL,
}

# One cached cell: the per-fold parcel predictions and the per-fold inner thresholds.
_Cell = tuple[dict[int, pd.DataFrame], dict[int, float]]
# Keyed by (feature_set, architecture, buffer_km).
_Cells = dict[tuple[str, str, int], _Cell]


# --------------------------------------------------------------------------- #
# Locating headline runs and their per-fold inner thresholds.
# --------------------------------------------------------------------------- #


def _headline_run(paths: ProjectPaths, architecture: str, feature_set: str) -> Path | None:
    """Newest main-CV run for one architecture and feature set, or ``None`` if absent."""
    source_root = paths.results / _SOURCE_ROLE
    try:
        if architecture == _XGB:
            return _latest_run(source_root, feature_set)
        return _latest_cnn_run(source_root, architecture, feature_set)
    except FileNotFoundError:
        return None


def _completed_folds(run_dir: Path) -> list[int]:
    """Folds with both saved parcel predictions and hyperparameter trials."""
    return [
        fold
        for fold in FOLD_IDS
        if (run_dir / _PARCEL_PRED_TEMPLATE.format(fold=fold)).is_file()
        and (run_dir / _HP_TRIALS_TEMPLATE.format(fold=fold)).is_file()
    ]


def _inner_thresholds_from_run(run_dir: Path, folds: Sequence[int]) -> dict[int, float]:
    """Per-fold parcel inner threshold: the best trial's ``inner_parcel_threshold``.

    The best trial is the one maximising the inner mean parcel PR-AUC (the selection
    metric), reproducing the ``inner_threshold`` column of ``per_fold_metrics.csv`` and
    working before that run-level file is written (a still-running CNN matrix).
    """
    thresholds: dict[int, float] = {}
    for fold in folds:
        trials = pd.read_csv(run_dir / _HP_TRIALS_TEMPLATE.format(fold=fold))
        completed = trials[trials["state"] == "COMPLETE"]
        best = completed.loc[completed["pr_auc"].idxmax()]
        thresholds[int(fold)] = float(best["inner_parcel_threshold"])
    return thresholds


def _headline_frames(run_dir: Path, folds: Sequence[int]) -> dict[int, pd.DataFrame]:
    """Read the saved out-of-fold parcel predictions for the given folds."""
    return {
        int(fold): pd.read_parquet(run_dir / _PARCEL_PRED_TEMPLATE.format(fold=fold))
        for fold in folds
    }


# --------------------------------------------------------------------------- #
# Buffered (10 km) refits, reusing the run_gap_analysis fixed-HP machinery.
# --------------------------------------------------------------------------- #


def _config(feature_set: str, architecture: str, args: argparse.Namespace) -> NestedCVConfig:
    return NestedCVConfig(
        feature_set=feature_set,
        architecture=architecture,
        sampling_mode="none",
        n_per_parcel=args.n_per_parcel,
        seed=args.seed,
        device=args.device,
        n_jobs=args.n_jobs,
        fold_ids=FOLD_IDS,
    )


def _buffered_frames(
    paths: ProjectPaths,
    architecture: str,
    feature_set: str,
    headline_run: Path,
    excluded: dict[int, np.ndarray],
    args: argparse.Namespace,
    run_logger: logging.Logger,
) -> tuple[dict[int, pd.DataFrame], list[Path]]:
    """Refit one config at the 10 km buffer; return ``{fold: parcel predictions}``."""
    pixel_index, memmap, inputs = _load_pixel_data(paths, feature_set)
    config = _config(feature_set, architecture, args)
    if architecture == _XGB:
        best_params = _load_best_params(headline_run)
        fold_results = _xgb_buffered_fold_results(
            config, pixel_index, memmap, best_params, excluded
        )
        inputs += [headline_run / _BEST_PARAMS_TEMPLATE.format(fold=f) for f in FOLD_IDS]
        return {int(fr.outer_fold): fr.parcel_predictions for fr in fold_results}, inputs

    from scripts.build_patch_cache import ensure_patch_cache

    patch_size = int(architecture.removeprefix("cnn_").split("x")[0])
    cache_path = ensure_patch_cache(feature_set, patch_size, paths=paths, logger=run_logger)
    patch_cache = np.load(cache_path, mmap_mode="r")
    manifest = json.loads(cache_path.with_suffix(".json").read_text(encoding="utf-8"))
    grid_shape = (int(manifest["grid_height"]), int(manifest["grid_width"]))
    best_params, cnn_epochs, hp_inputs = _load_cnn_fold_hps(headline_run)
    fold_results, _ = _cnn_buffered_fold_results(
        config,
        pixel_index,
        patch_cache,
        best_params,
        cnn_epochs,
        excluded,
        patch_size=patch_size,
        grid_shape=grid_shape,
        run_logger=run_logger,
        config_key=f"{architecture}__{feature_set}",
    )
    frames = {int(fr.outer_fold): fr.parcel_predictions for fr in fold_results}
    return frames, [*inputs, cache_path, *hp_inputs]


def _cached_buffered_frames(
    paths: ProjectPaths,
    architecture: str,
    feature_set: str,
    headline_run: Path,
    folds: Sequence[int],
    excluded: dict[int, np.ndarray],
    buffer_km: int,
    args: argparse.Namespace,
    run_logger: logging.Logger,
) -> tuple[dict[int, pd.DataFrame], list[Path]]:
    """Buffered refit for one cell, cached per source run and buffer so unchanged cells are reused.

    The expensive part of the matrix is the buffered refit; the cache key is the headline run
    name and the buffer distance, so a new ``main_nested_cv`` run for a cell (or more completed
    folds, or a change to ``--seed`` / ``--n-per-parcel``) misses and recomputes, while every
    unchanged cell is loaded from ``paths.cache``. ``--refresh-buffered`` forces a recompute.
    Stale entries from superseded runs are harmless and can be cleared by deleting
    ``paths.cache / performance_matrix_buffered``.
    """
    cache_dir = (
        paths.cache
        / _BUFFERED_CACHE
        / f"{architecture}__{feature_set}__{headline_run.name}__{buffer_km}km"
    )
    fold_paths = {int(fold): cache_dir / f"buffered_fold{fold}.parquet" for fold in folds}
    guard = {"seed": args.seed, "n_per_parcel": args.n_per_parcel, "folds": sorted(fold_paths)}
    meta_path = cache_dir / "cache_meta.json"
    if (
        not args.refresh_buffered
        and meta_path.is_file()
        and all(p.is_file() for p in fold_paths.values())
    ):
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        if all(meta.get(key) == value for key, value in guard.items()):
            run_logger.info(
                "%s / %s: reusing cached buffered refit (%s).",
                feature_set,
                architecture,
                headline_run.name,
            )
            frames = {fold: pd.read_parquet(path) for fold, path in fold_paths.items()}
            return frames, [Path(item) for item in meta.get("inputs", [])]

    frames, cell_inputs = _buffered_frames(
        paths, architecture, feature_set, headline_run, excluded, args, run_logger
    )
    cache_dir.mkdir(parents=True, exist_ok=True)
    for fold, frame in frames.items():
        frame.to_parquet(cache_dir / f"buffered_fold{fold}.parquet")
    meta_path.write_text(
        json.dumps({**guard, "inputs": [str(item) for item in cell_inputs]}), encoding="utf-8"
    )
    return frames, cell_inputs


# --------------------------------------------------------------------------- #
# Row assembly.
# --------------------------------------------------------------------------- #


def _matrix_row(
    feature_set: str,
    architecture: str,
    buffer_km: int,
    frames: dict[int, pd.DataFrame],
    inner_thresholds: dict[int, float],
    block_assignment: pd.DataFrame,
    args: argparse.Namespace,
) -> dict[str, object]:
    """One performance-matrix row: identity columns plus the 88 metric statistics."""
    stats = metric_matrix_row(
        frames,
        inner_thresholds,
        block_assignment,
        reps=args.reps,
        seed=args.seed,
        n_jobs=args.n_jobs,
    )
    row: dict[str, object] = {
        "feature_set": feature_set,
        "architecture": architecture,
        "buffer_km": int(buffer_km),
        "folds_used": int(stats.pop("folds_used")),
    }
    row.update(stats)
    return row


def _contrast_row(
    set_a: str,
    set_b: str,
    buffer_km: int,
    config_a: _Cell,
    config_b: _Cell,
    block_assignment: pd.DataFrame,
    args: argparse.Namespace,
) -> dict[str, object]:
    """One paired-contrast row (A minus B) with per-metric p-values."""
    frames_a, inner_a = config_a
    frames_b, inner_b = config_b
    stats = paired_row(
        frames_a,
        frames_b,
        inner_a,
        inner_b,
        block_assignment,
        reps=args.reps,
        seed=args.seed,
        n_jobs=args.n_jobs,
    )
    row: dict[str, object] = {
        "contrast": f"{set_a} - {set_b}",
        "set_a": set_a,
        "set_b": set_b,
        "buffer_km": int(buffer_km),
        "folds_used": int(stats.pop("folds_used")),
        "pr_auc__p_holm": float("nan"),
        "pr_auc__reject_holm": False,
    }
    row.update(stats)
    return row


def _apply_holm(table: pd.DataFrame, group_columns: list[str]) -> pd.DataFrame:
    """Holm-adjust the PR-AUC p within each family (rows sharing ``group_columns``)."""
    table = table.copy()
    for _, index in table.groupby(group_columns).groups.items():
        finite = [i for i in index if np.isfinite(table.loc[i, "pr_auc__p"])]
        if not finite:
            continue
        adjusted = holm_adjusted_p_values([float(table.loc[i, "pr_auc__p"]) for i in finite])
        for position, i in enumerate(finite):
            table.loc[i, "pr_auc__p_holm"] = adjusted[position]
            table.loc[i, "pr_auc__reject_holm"] = bool(adjusted[position] < _HOLM_ALPHA)
    return table


# --------------------------------------------------------------------------- #
# Orchestration.
# --------------------------------------------------------------------------- #


def _build_matrix(
    paths: ProjectPaths,
    block_assignment: pd.DataFrame,
    excluded_by_buffer: dict[int, dict[int, np.ndarray]],
    args: argparse.Namespace,
    run_logger: logging.Logger,
) -> tuple[pd.DataFrame, _Cells, list[Path]]:
    """Build the performance matrix and the per-cell ``(frames, inner thresholds)`` cache.

    One row per (feature set, architecture, buffer) over ``_BUFFERS`` (0 km plus every buffer
    in ``_BUFFER_KMS``); each non-zero buffer is an independently cached fixed-HP refit.
    """
    rows: list[dict[str, object]] = []
    cells: _Cells = {}
    inputs: list[Path] = []
    for feature_set in args.feature_sets:
        for architecture in args.architectures:
            run = _headline_run(paths, architecture, feature_set)
            inner: dict[int, float] = {}
            headline: dict[int, pd.DataFrame] = {}
            buffered_by_km: dict[int, dict[int, pd.DataFrame]] = {km: {} for km in _BUFFER_KMS}
            if run is not None and (folds := _completed_folds(run)):
                inner = _inner_thresholds_from_run(run, folds)
                headline = _headline_frames(run, folds)
                for buffer_km in _BUFFER_KMS:
                    buffered_by_km[buffer_km], cell_inputs = _cached_buffered_frames(
                        paths,
                        architecture,
                        feature_set,
                        run,
                        folds,
                        excluded_by_buffer[buffer_km],
                        buffer_km,
                        args,
                        run_logger,
                    )
                    inputs += cell_inputs
            cells[(feature_set, architecture, 0)] = (headline, inner)
            for buffer_km in _BUFFER_KMS:
                cells[(feature_set, architecture, buffer_km)] = (buffered_by_km[buffer_km], inner)
            for buffer_km, frames in (
                (0, headline),
                *((km, buffered_by_km[km]) for km in _BUFFER_KMS),
            ):
                rows.append(
                    _matrix_row(
                        feature_set, architecture, buffer_km, frames, inner, block_assignment, args
                    )
                )
            run_logger.info(
                "%s / %s: %d headline folds, %s buffered folds.",
                feature_set,
                architecture,
                len(headline),
                ", ".join(f"{km} km: {len(buffered_by_km[km])}" for km in _BUFFER_KMS),
            )
    matrix = pd.DataFrame(rows, columns=list(MATRIX_SCHEMA))
    return matrix, cells, inputs


def _contrast_table(rows: list[dict[str, object]], group_columns: list[str]) -> pd.DataFrame:
    """Assemble, Holm-adjust and dtype a contrast table (typed even when empty)."""
    table = pd.DataFrame(rows, columns=list(CONTRAST_SCHEMA))
    if table.empty:
        return table.astype({name: _allowed_dtype(spec) for name, spec in CONTRAST_SCHEMA.items()})
    return _apply_holm(table, group_columns)


def _allowed_dtype(spec: ColumnSpec) -> str:
    return spec.dtype if isinstance(spec.dtype, str) else spec.dtype[0]


def _feature_set_contrasts(
    cells: _Cells,
    block_assignment: pd.DataFrame,
    args: argparse.Namespace,
) -> pd.DataFrame:
    """(EO feature sets minus baseline) for XGBoost, at each buffer."""
    rows: list[dict[str, object]] = []
    if _XGB in args.architectures and _BASELINE in args.feature_sets:
        for buffer_km in _BUFFERS:
            for feature_set in (f for f in _FS_CONTRASTS if f in args.feature_sets):
                rows.append(
                    _contrast_row(
                        feature_set,
                        _BASELINE,
                        buffer_km,
                        cells[(feature_set, _XGB, buffer_km)],
                        cells[(_BASELINE, _XGB, buffer_km)],
                        block_assignment,
                        args,
                    )
                )
    return _contrast_table(rows, ["buffer_km"])


def _embedding_contrasts(
    cells: _Cells,
    block_assignment: pd.DataFrame,
    args: argparse.Namespace,
) -> pd.DataFrame:
    """Pairwise contrasts among the EO representations (XGBoost), at each buffer.

    Tests whether AlphaEarth, TESSERA and Conventional-EO are distinguishable - the
    headline ordering versus whether it survives the buffer (Point 2).
    """
    rows: list[dict[str, object]] = []
    if _XGB in args.architectures:
        for buffer_km in _BUFFERS:
            for set_a, set_b in _EMBEDDING_CONTRASTS:
                if set_a in args.feature_sets and set_b in args.feature_sets:
                    rows.append(
                        _contrast_row(
                            set_a,
                            set_b,
                            buffer_km,
                            cells[(set_a, _XGB, buffer_km)],
                            cells[(set_b, _XGB, buffer_km)],
                            block_assignment,
                            args,
                        )
                    )
    return _contrast_table(rows, ["buffer_km"])


def _architecture_contrasts(
    cells: _Cells,
    block_assignment: pd.DataFrame,
    args: argparse.Namespace,
) -> pd.DataFrame:
    """(CNN architectures minus XGBoost) for every feature set, at each buffer."""
    cnn_architectures = [a for a in args.architectures if a != _XGB]
    rows: list[dict[str, object]] = []
    if _XGB in args.architectures:
        for buffer_km in _BUFFERS:
            for feature_set in args.feature_sets:
                for architecture in cnn_architectures:
                    rows.append(
                        _contrast_row(
                            f"{architecture}__{feature_set}",
                            f"{_XGB}__{feature_set}",
                            buffer_km,
                            cells[(feature_set, architecture, buffer_km)],
                            cells[(feature_set, _XGB, buffer_km)],
                            block_assignment,
                            args,
                        )
                    )
    return _contrast_table(rows, ["set_b", "buffer_km"])


def _library_versions() -> dict[str, str]:
    versions: dict[str, str] = {}
    for library in ("numpy", "pandas", "xgboost", "scikit-learn", "torch"):
        try:
            versions[library] = version(library)
        except PackageNotFoundError:
            versions[library] = "unknown"
    return versions


def _run(args: argparse.Namespace, paths: ProjectPaths, run_dir: Path) -> dict[str, object]:
    """Build and write the matrix and the two contrast tables."""
    run_logger = make_run_logger(_LOGGER_NAME, run_dir)
    labels_path = paths.labels / _LABELS_FILENAME
    block_assignment = load_block_assignment(labels_path)
    labelled = _load_labelled_geoms(labels_path)
    excluded_by_buffer = {
        buffer_km: _buffered_excluded_pids(labelled, buffer_km=buffer_km, seed=args.seed)
        for buffer_km in _BUFFER_KMS
    }

    matrix, cells, inputs = _build_matrix(
        paths, block_assignment, excluded_by_buffer, args, run_logger
    )
    feature_set_contrasts = _feature_set_contrasts(cells, block_assignment, args)
    embedding_contrasts = _embedding_contrasts(cells, block_assignment, args)
    architecture_contrasts = _architecture_contrasts(cells, block_assignment, args)

    write_csv(matrix, run_dir / "performance_matrix.csv", MATRIX_SCHEMA)
    write_csv(feature_set_contrasts, run_dir / "feature_set_contrasts.csv", CONTRAST_SCHEMA)
    write_csv(embedding_contrasts, run_dir / "embedding_contrasts.csv", CONTRAST_SCHEMA)
    write_csv(architecture_contrasts, run_dir / "architecture_contrasts.csv", CONTRAST_SCHEMA)
    write_input_manifest(
        [labels_path, *inputs], run_dir / "input_manifest.json", repo_root=paths.repo_root
    )
    write_environment_snapshot(run_dir / "env_snapshot.txt")

    metadata = {
        "run_id": run_dir.name,
        "created_at": datetime.now(UTC).isoformat(),
        "script": "scripts/run_performance_matrix.py",
        "buffer_km": list(_BUFFER_KMS),
        "feature_sets": list(args.feature_sets),
        "architectures": list(args.architectures),
        "n_reps": args.reps,
        "n_blocks": int(block_assignment["bootstrap_id"].nunique()),
        "seed": args.seed,
        "device": args.device,
        "n_jobs": args.n_jobs,
        "hp_reuse": "per-fold best_params from the main nested CV (no Optuna search)",
        "library_versions": _library_versions(),
    }
    write_json(metadata, run_dir / "run_metadata.json")
    return metadata


def _parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--architectures",
        nargs="+",
        default=list(ARCHITECTURES),
        choices=list(ARCHITECTURES),
        help="Architectures to include (default: all four).",
    )
    parser.add_argument(
        "--feature-sets",
        nargs="+",
        default=list(FEATURE_SETS),
        choices=list(FEATURE_SETS),
        help="Feature sets to include (default: all four).",
    )
    parser.add_argument("--reps", type=int, default=BOOTSTRAP_REPS_CROSS_CONFIG)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--n-per-parcel", type=int, default=N_PER_PARCEL)
    parser.add_argument("--n-jobs", type=int, default=min(8, os.cpu_count() or 1))
    parser.add_argument("--device", default="cuda", choices=["cpu", "cuda"])
    parser.add_argument(
        "--refresh-buffered",
        action="store_true",
        help="Recompute the buffered (10 km) refits even when a cached result exists.",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    """Build the performance matrix and contrast tables, returning a process code."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
    args = _parse_args(argv)
    paths = get_project_paths()
    ensure_writable_dirs(paths)
    run_dir = paths.results / _RESULTS_ROLE / datetime.now(UTC).strftime("%Y%m%d_%H%M%S")
    run_dir.mkdir(parents=True, exist_ok=True)
    _run(args, paths, run_dir)
    logger.info("Wrote performance matrix to %s.", run_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
