"""Seed sweep: retraining variability of the pooled parcel PR-AUC (XGBoost).

The block bootstrap in ``run_performance_matrix`` conditions on the fitted models:
it resamples the evaluation blocks but never retrains, so model-fitting variability
(stochastic row/column subsampling inside XGBoost) is not propagated. This script
quantifies that missing component directly. Each headline XGBoost feature set is
refit with the main nested-CV per-fold hyperparameters (fixed HPs, no Optuna) under
several seeds, at the 0 km headline arm and at the 10 km training-exclusion buffer,
and the between-seed spread of the pooled out-of-fold parcel PR-AUC / ROC-AUC is
reported next to the evaluation bootstrap interval.

The seed enters exactly where the headline pipeline is stochastic: the XGBoost
``seed`` (row subsampling ``subsample`` and column subsampling ``colsample_bytree``
are tuned well below 1). The buffered exclusions are geometric and therefore
identical across seeds; the default seed list starts with the paper seed (42), whose
0 km arm must reproduce the saved main nested-CV predictions (logged as an
integrity check, mirroring the gap-0-vs-source check of ``run_gap_analysis``).

Outputs under ``results/seed_sweep/<run_id>/``:

  - ``predictions/{fs}__{km}km__seed{seed}__fold{fold}.parquet`` - per-fold
    out-of-fold parcel predictions (the resumable unit; re-running with
    ``--resume <run_id>`` skips completed fits).
  - ``seed_sweep_metrics.csv``   - one row per (feature set, buffer, seed): pooled
    and per-fold PR-AUC / ROC-AUC.
  - ``seed_sweep_contrasts.csv`` - one row per pre-specified contrast (the three
    EO feature sets minus baseline, plus the three pairwise EO-representation
    contrasts), buffer and seed: the paired difference of the pooled metrics.
  - ``run_metadata.json`` / ``input_manifest.json`` / ``env_snapshot.txt`` / logs.

The notebook ``014_robustness_checks.ipynb`` displays the results; computation lives
here.
"""

from __future__ import annotations

import argparse
import logging
import os
from collections.abc import Sequence
from datetime import UTC, datetime
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Final

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

from scripts.run_gap_analysis import (
    _buffered_excluded_pids,
    _fold_result_for_exclusion,
    _load_labelled_geoms,
)
from scripts.run_performance_matrix import _EMBEDDING_CONTRASTS
from scripts.run_sampling_sensitivity import _latest_run, _load_best_params, _load_pixel_data
from utils.cv import NestedCVConfig
from utils.env_capture import write_environment_snapshot
from utils.io import (
    PARCEL_PREDICTIONS_SCHEMA,
    ColumnSpec,
    Schema,
    write_csv,
    write_json,
    write_parquet,
)
from utils.logging import make_run_logger
from utils.manifest import write_input_manifest
from utils.paths import ProjectPaths, ensure_writable_dirs, get_project_paths
from utils.terminology import FEATURE_SETS, FOLD_IDS, N_PER_PARCEL, SEED

logger = logging.getLogger(__name__)

_LOGGER_NAME: Final[str] = "run_seed_sweep"
_SOURCE_ROLE: Final[str] = "main_nested_cv"
_RESULTS_ROLE: Final[str] = "seed_sweep"
_ARCHITECTURE: Final[str] = "xgboost"
_BASELINE: Final[str] = "baseline"
_PREDICTIONS_DIR: Final[str] = "predictions"
# Paper seed first (its 0 km arm reproduces the headline run), then four fresh seeds.
_DEFAULT_SEEDS: Final[tuple[int, ...]] = (SEED, 1, 2, 3, 4)
_DEFAULT_BUFFER_KMS: Final[tuple[int, ...]] = (0, 10)
_PARCEL_PRED_TEMPLATE: Final[str] = "parcel_predictions_fold{fold}.parquet"

_FLOAT = ColumnSpec("float64")
_STR = ColumnSpec("object")
_INT = ColumnSpec("int64")

_METRIC_COLUMNS: Final[dict[str, ColumnSpec]] = {
    f"{metric}__{stat}": _FLOAT
    for metric in ("pr_auc", "roc_auc")
    for stat in ("pooled", "fold1", "fold2", "fold3", "fold4", "fold5", "fold6")
}

METRICS_SCHEMA: Final[Schema] = {
    "feature_set": _STR,
    "buffer_km": _INT,
    "seed": _INT,
    **_METRIC_COLUMNS,
}

CONTRASTS_SCHEMA: Final[Schema] = {
    "contrast": _STR,
    "set_a": _STR,
    "set_b": _STR,
    "buffer_km": _INT,
    "seed": _INT,
    "pr_auc__diff": _FLOAT,
    "roc_auc__diff": _FLOAT,
}


def _prediction_path(run_dir: Path, feature_set: str, buffer_km: int, seed: int, fold: int) -> Path:
    return (
        run_dir / _PREDICTIONS_DIR / f"{feature_set}__{buffer_km}km__seed{seed}__fold{fold}.parquet"
    )


def _pooled(frames: dict[int, pd.DataFrame]) -> pd.DataFrame:
    parts = [frame.assign(outer_fold=int(fold)) for fold, frame in sorted(frames.items())]
    return pd.concat(parts, ignore_index=True)


def _metric_row(
    feature_set: str, buffer_km: int, seed: int, frames: dict[int, pd.DataFrame]
) -> dict:
    """Pooled and per-fold ranking metrics for one (feature set, buffer, seed) cell."""
    row: dict[str, object] = {"feature_set": feature_set, "buffer_km": buffer_km, "seed": seed}
    pooled = _pooled(frames)
    y_pooled = pooled["y_true"].to_numpy().astype(np.int64)
    p_pooled = pooled["p_mean"].to_numpy().astype(np.float64)
    row["pr_auc__pooled"] = float(average_precision_score(y_pooled, p_pooled))
    row["roc_auc__pooled"] = float(roc_auc_score(y_pooled, p_pooled))
    for fold, frame in sorted(frames.items()):
        y = frame["y_true"].to_numpy().astype(np.int64)
        p = frame["p_mean"].to_numpy().astype(np.float64)
        row[f"pr_auc__fold{fold}"] = float(average_precision_score(y, p))
        row[f"roc_auc__fold{fold}"] = float(roc_auc_score(y, p))
    return row


def _contrast_row(
    set_a: str,
    set_b: str,
    buffer_km: int,
    seed: int,
    frames_a: dict[int, pd.DataFrame],
    frames_b: dict[int, pd.DataFrame],
) -> dict:
    """Paired difference of the pooled ranking metrics on the shared parcel set."""
    merged = _pooled(frames_a).merge(
        _pooled(frames_b), on="parcel_id", how="inner", suffixes=("_a", "_b")
    )
    y = merged["y_true_a"].to_numpy().astype(np.int64)
    p_a = merged["p_mean_a"].to_numpy().astype(np.float64)
    p_b = merged["p_mean_b"].to_numpy().astype(np.float64)
    return {
        "contrast": f"{set_a} - {set_b}",
        "set_a": set_a,
        "set_b": set_b,
        "buffer_km": buffer_km,
        "seed": seed,
        "pr_auc__diff": float(average_precision_score(y, p_a) - average_precision_score(y, p_b)),
        "roc_auc__diff": float(roc_auc_score(y, p_a) - roc_auc_score(y, p_b)),
    }


def _integrity_check(
    run_dir: Path,
    feature_set: str,
    headline_run: Path,
    frames: dict[int, pd.DataFrame],
    run_logger: logging.Logger,
) -> dict[str, str | float]:
    """Compare the paper-seed 0 km refit against the saved main nested-CV predictions."""
    worst = 0.0
    for fold, frame in frames.items():
        saved = pd.read_parquet(headline_run / _PARCEL_PRED_TEMPLATE.format(fold=fold))
        merged = saved.merge(frame, on="parcel_id", suffixes=("_saved", "_refit"))
        if len(merged) != len(saved):
            run_logger.warning(
                "%s fold %d: parcel sets differ from the headline run.", feature_set, fold
            )
        gap = float(np.abs(merged["p_mean_saved"] - merged["p_mean_refit"]).max())
        worst = max(worst, gap)
    run_logger.info(
        "%s: paper-seed 0 km integrity check, max |p_mean(refit) - p_mean(saved)| = %.3g.",
        feature_set,
        worst,
    )
    return {"feature_set": feature_set, "max_abs_p_mean_gap": worst}


def _typed(table: pd.DataFrame, schema: Schema) -> pd.DataFrame:
    """Coerce a (possibly empty) table to the schema dtypes so validation passes."""
    dtypes = {
        name: spec.dtype if isinstance(spec.dtype, str) else spec.dtype[0]
        for name, spec in schema.items()
    }
    return table.astype(dtypes)


def _run(args: argparse.Namespace, paths: ProjectPaths, run_dir: Path) -> dict[str, object]:
    run_logger = make_run_logger(_LOGGER_NAME, run_dir)
    labels_path = paths.labels / "ogf_reference_labels_partitioned.gpkg"
    labelled = _load_labelled_geoms(labels_path)
    empty = {fold: np.array([], dtype=np.int64) for fold in FOLD_IDS}
    excluded_by_buffer = {
        buffer_km: (
            empty
            if buffer_km == 0
            else _buffered_excluded_pids(labelled, buffer_km=buffer_km, seed=SEED)
        )
        for buffer_km in args.buffer_kms
    }

    metric_rows: list[dict] = []
    contrast_rows: list[dict] = []
    integrity: list[dict[str, str | float]] = []
    inputs: list[Path] = [labels_path]
    # frames[(feature_set, buffer_km, seed)] -> {fold: parcel predictions}
    cells: dict[tuple[str, int, int], dict[int, pd.DataFrame]] = {}

    total = len(args.feature_sets) * len(args.buffer_kms) * len(args.seeds) * len(FOLD_IDS)
    done = 0
    for feature_set in args.feature_sets:
        headline_run = _latest_run(paths.results / _SOURCE_ROLE, feature_set)
        best_params = _load_best_params(headline_run)
        pixel_index, memmap, fs_inputs = _load_pixel_data(paths, feature_set)
        inputs += fs_inputs
        inputs += [headline_run / f"best_params_fold{fold}.json" for fold in FOLD_IDS]
        base_valid = pixel_index["valid"].to_numpy()
        parcel_ids = pixel_index["parcel_id"].to_numpy()
        for buffer_km in args.buffer_kms:
            for seed in args.seeds:
                config = NestedCVConfig(
                    feature_set=feature_set,
                    architecture=_ARCHITECTURE,
                    sampling_mode="none",
                    n_per_parcel=args.n_per_parcel,
                    seed=seed,
                    device=args.device,
                    n_jobs=args.n_jobs,
                    fold_ids=FOLD_IDS,
                )
                frames: dict[int, pd.DataFrame] = {}
                for fold in FOLD_IDS:
                    path = _prediction_path(run_dir, feature_set, buffer_km, seed, fold)
                    if path.is_file():
                        frames[fold] = pd.read_parquet(path)
                        done += 1
                        continue
                    result = _fold_result_for_exclusion(
                        config,
                        pixel_index,
                        memmap,
                        best_params[fold],
                        outer_fold=fold,
                        base_valid=base_valid,
                        parcel_ids=parcel_ids,
                        excluded_pids=excluded_by_buffer[buffer_km][fold],
                    )
                    write_parquet(result.parcel_predictions, path, PARCEL_PREDICTIONS_SCHEMA)
                    frames[fold] = result.parcel_predictions
                    done += 1
                    run_logger.info(
                        "[%d/%d] %s / %d km / seed %d / fold %d done.",
                        done,
                        total,
                        feature_set,
                        buffer_km,
                        seed,
                        fold,
                    )
                cells[(feature_set, buffer_km, seed)] = frames
                metric_rows.append(_metric_row(feature_set, buffer_km, seed, frames))
                if seed == SEED and buffer_km == 0:
                    integrity.append(
                        _integrity_check(run_dir, feature_set, headline_run, frames, run_logger)
                    )

    pairs = [
        (feature_set, _BASELINE)
        for feature_set in args.feature_sets
        if feature_set != _BASELINE and _BASELINE in args.feature_sets
    ] + [
        (set_a, set_b)
        for set_a, set_b in _EMBEDDING_CONTRASTS
        if set_a in args.feature_sets and set_b in args.feature_sets
    ]
    for buffer_km in args.buffer_kms:
        for seed in args.seeds:
            for set_a, set_b in pairs:
                contrast_rows.append(
                    _contrast_row(
                        set_a,
                        set_b,
                        buffer_km,
                        seed,
                        cells[(set_a, buffer_km, seed)],
                        cells[(set_b, buffer_km, seed)],
                    )
                )

    metrics = _typed(pd.DataFrame(metric_rows, columns=list(METRICS_SCHEMA)), METRICS_SCHEMA)
    contrasts = _typed(
        pd.DataFrame(contrast_rows, columns=list(CONTRASTS_SCHEMA)), CONTRASTS_SCHEMA
    )
    write_csv(metrics, run_dir / "seed_sweep_metrics.csv", METRICS_SCHEMA)
    write_csv(contrasts, run_dir / "seed_sweep_contrasts.csv", CONTRASTS_SCHEMA)
    write_input_manifest(
        sorted(set(inputs)), run_dir / "input_manifest.json", repo_root=paths.repo_root
    )
    write_environment_snapshot(run_dir / "env_snapshot.txt")

    versions: dict[str, str] = {}
    for library in ("numpy", "pandas", "xgboost", "scikit-learn"):
        try:
            versions[library] = version(library)
        except PackageNotFoundError:
            versions[library] = "unknown"
    metadata = {
        "run_id": run_dir.name,
        "created_at": datetime.now(UTC).isoformat(),
        "script": "scripts/run_seed_sweep.py",
        "architecture": _ARCHITECTURE,
        "feature_sets": list(args.feature_sets),
        "buffer_kms": list(args.buffer_kms),
        "seeds": list(args.seeds),
        "n_per_parcel": args.n_per_parcel,
        "device": args.device,
        "n_jobs": args.n_jobs,
        "hp_reuse": "per-fold best_params from the main nested CV (no Optuna search)",
        "buffer_exclusion_seed": SEED,
        "integrity_check_paper_seed_0km": integrity,
        "library_versions": versions,
    }
    write_json(metadata, run_dir / "run_metadata.json")
    return metadata


def _parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--feature-sets",
        nargs="+",
        default=list(FEATURE_SETS),
        choices=list(FEATURE_SETS),
        help="XGBoost feature sets to refit (default: all four).",
    )
    parser.add_argument(
        "--seeds",
        nargs="+",
        type=int,
        default=list(_DEFAULT_SEEDS),
        help="Training seeds (default: the paper seed 42 plus 1-4).",
    )
    parser.add_argument(
        "--buffer-kms",
        nargs="+",
        type=int,
        default=list(_DEFAULT_BUFFER_KMS),
        help="Training-exclusion buffers in km (default: 0 and 10).",
    )
    parser.add_argument("--n-per-parcel", type=int, default=N_PER_PARCEL)
    parser.add_argument("--n-jobs", type=int, default=min(8, os.cpu_count() or 1))
    parser.add_argument("--device", default="cuda", choices=["cpu", "cuda"])
    parser.add_argument(
        "--resume",
        default=None,
        help="Existing run id under results/seed_sweep to continue (skips saved fits).",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    """Run the seed sweep, returning a process exit code."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
    args = _parse_args(argv)
    paths = get_project_paths()
    ensure_writable_dirs(paths)
    run_id = args.resume or datetime.now(UTC).strftime("%Y%m%d_%H%M%S")
    run_dir = paths.results / _RESULTS_ROLE / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    _run(args, paths, run_dir)
    logger.info("Wrote seed sweep to %s.", run_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
