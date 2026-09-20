"""Spatial baseline: XGBoost trained on pixel coordinates only.

This runs the selected XGBoost pipeline unchanged, but replaces the predictor
features with just the two pixel-centre coordinates (easting and northing in the
project CRS, expressed in metres from the grid origin). Evaluated under the same six
spatial folds, the same per-parcel sampling, the same nested Optuna search and the
same parcel aggregation as the headline run, it measures how much apparent old-growth
skill is obtainable from spatial position alone.

It is a *spatial null model* (equivalently a coordinates-only or trend-surface model):
a benchmark for spatial structure, not a predictor of old-growth from forest
properties. Because the folds are contiguous spatial blocks, a tree model on (x, y)
must extrapolate to a held-out region it never saw in training; any out-of-fold skill
it nonetheless shows is the residual spatial-autocorrelation signal that leaks across
fold boundaries (the binding range is roughly 6 to 7 km). Reading the feature-set
results against this baseline says how much of their performance is genuine signal
over and above location, and motivates the spatial-buffer (gap) analysis of notebook
009.

The runner mirrors :func:`utils.cv.run_nested_cv` and its inner objective exactly
(identical seeding, sampling, inner-threshold capture, parcel aggregation and metric
computation), so the result is directly comparable to a ``main_nested_cv`` feature-set
run; the only difference is the two-column in-memory feature matrix, built from the
pixel index and the reference grid, so no feature stack is needed and no new feature
set is registered. Outputs are written in the ``main_nested_cv`` layout to
``results/spatial_baseline/<run_id>`` so ``notebooks/009_performance_matrix_and_buffers.ipynb`` (and
the reporting helpers in :mod:`utils.reporting`) can consume them like any other run.
The model runs on GPU (CUDA) by default; ``--device cpu`` is ample for two features.
"""

import argparse
import logging
import os
import time
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Final

import numpy as np
import numpy.typing as npt
import optuna
import pandas as pd
from optuna.samplers import TPESampler

from scripts.run_sampling_sensitivity import _load_pixel_data
from utils.cv import FoldArrays, FoldResult, aggregate_pooled_metrics, build_fold_arrays
from utils.env_capture import write_environment_snapshot
from utils.hp_search import hp_boundary_diagnostics, suggest_xgb_params
from utils.io import (
    FEATURE_IMPORTANCE_SCHEMA,
    HP_BOUNDARY_SCHEMA,
    PARCEL_PREDICTIONS_SCHEMA,
    PER_FOLD_METRICS_SCHEMA,
    PIXEL_PREDICTIONS_SCHEMA,
    POOLED_METRICS_SCHEMA,
    SUMMARY_METRICS_SCHEMA,
    write_csv,
    write_json,
    write_parquet,
)
from utils.logging import make_run_logger
from utils.manifest import write_input_manifest
from utils.metrics import best_f1_threshold, parcel_aggregations, parcel_metrics
from utils.models.xgboost import (
    feature_importance,
    make_xgb_classifier,
    predict_ogf_proba,
    train_xgb_fold,
)
from utils.paths import ProjectPaths, ensure_writable_dirs, get_project_paths
from utils.raster_io import ReferenceGrid, open_reference_grid
from utils.terminology import FOLD_IDS, N_HP_TRIALS, N_PER_PARCEL, SEED

logger = logging.getLogger(__name__)

_LOGGER_NAME: Final[str] = "run_coordinate_xgboost"
_ARCHITECTURE: Final[str] = "xgboost"
_RESULTS_ROLE: Final[str] = "spatial_baseline"
# Not a registered feature set: a label for the run directory and feature-importance rows.
_FEATURE_LABEL: Final[str] = "xy_coords"
_FEATURE_NAMES: Final[tuple[str, str]] = ("easting_m", "northing_m")
# The pixel/parcel set is taken from this feature set's validity mask, so the spatial
# baseline is scored on exactly the pixels and parcels the headline model used.
_DEFAULT_VALID_FROM: Final[str] = "baseline_tessera"
_SAMPLING_MODE: Final[str] = "none"
_FALLBACK_THRESHOLD: Final[float] = 0.5
_INNER_METRICS: Final[tuple[str, ...]] = ("pr_auc", "roc_auc", "f1", "precision", "recall")
_DEFAULT_N_JOBS: Final[int] = min(8, os.cpu_count() or 1)


def coordinate_features(pixel_index: pd.DataFrame, ref: ReferenceGrid) -> npt.NDArray[np.float32]:
    """Build the ``(n_pixels, 2)`` easting/northing matrix from pixel ids and the grid.

    Each pixel's flat id decodes to ``(row, col) = divmod(pixel_id, width)``, and the
    reference grid's affine transform maps the cell centre ``(col + 0.5, row + 0.5)`` to
    projected coordinates. The two columns are easting and northing in metres, shifted to
    the minimum corner so the values stay small and exactly representable in ``float32``;
    the shift is a per-axis monotonic transform, so it does not change any tree split.

    Args:
        pixel_index: Pixel index aligned row-for-row with the feature memmap; needs a
            ``pixel_id`` column (flat id ``row * width + col``).
        ref: The project reference grid (its transform and width decode the ids).

    Returns:
        A C-contiguous ``(n_pixels, 2)`` ``float32`` array of ``(easting_m, northing_m)``,
        in the input row order.
    """
    pixel_id = pixel_index["pixel_id"].to_numpy(dtype=np.int64)
    width = ref.width
    row = pixel_id // width
    col = pixel_id % width
    affine = ref.transform
    centre_col = col + 0.5
    centre_row = row + 0.5
    easting = affine.a * centre_col + affine.b * centre_row + affine.c
    northing = affine.d * centre_col + affine.e * centre_row + affine.f
    easting = easting - easting.min()
    northing = northing - northing.min()
    return np.ascontiguousarray(np.column_stack([easting, northing]).astype(np.float32))


def _parcel_label_map(pixel_index: pd.DataFrame) -> dict[int, int]:
    """Map each parcel id to its single old-growth label (mirrors ``utils.cv._parcel_labels``)."""
    unique = pixel_index.drop_duplicates("parcel_id")
    return {int(p): int(o) for p, o in zip(unique["parcel_id"], unique["ogf"], strict=True)}


def _parcel_truth(parcel_ids: npt.NDArray[np.int64], parcel_label: Mapping[int, int]) -> np.ndarray:
    """Look up the per-parcel label for an array of parcel ids."""
    return np.array([parcel_label[int(p)] for p in parcel_ids], dtype=np.int8)


def _pooled_f1_threshold(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    """F1-maximising threshold on a pooled set, falling back to 0.5 if single-class."""
    if np.unique(y_true).size < 2:
        return _FALLBACK_THRESHOLD
    return float(best_f1_threshold(y_true, y_prob).threshold)


def _fold_probabilities(
    arrays: FoldArrays,
    params: Mapping[str, float | int],
    *,
    device: str,
    n_jobs: int,
    seed: int,
) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    """Train one split and return parcel aggregations plus the held-out pixel labels/probs."""
    probabilities = train_xgb_fold(
        params,
        arrays.X_train,
        arrays.y_train,
        arrays.X_val,
        sample_weight=arrays.train_sample_weight,
        device=device,
        n_jobs=n_jobs,
        seed=seed,
    )
    aggregations = parcel_aggregations(arrays.val_parcel_ids, probabilities)
    return aggregations, arrays.y_val, probabilities


def _inner_objective(
    trial: optuna.trial.Trial,
    *,
    features: np.ndarray,
    pixel_index: pd.DataFrame,
    outer_fold: int,
    inner_folds: Sequence[int],
    parcel_label: Mapping[int, int],
    n_per_parcel: int,
    seed: int,
    device: str,
    n_jobs: int,
) -> float:
    """Mean inner-fold parcel PR-AUC for one trial (mirrors ``utils.cv._inner_objective``)."""
    params = suggest_xgb_params(trial)
    per_fold: list[dict[str, float]] = []
    pixel_truth: list[np.ndarray] = []
    pixel_prob: list[np.ndarray] = []
    parcel_truth: list[np.ndarray] = []
    parcel_prob: list[np.ndarray] = []
    for inner_val in inner_folds:
        train_inner = tuple(f for f in inner_folds if f != inner_val)
        rng = np.random.default_rng([seed, outer_fold, trial.number, inner_val])
        arrays = build_fold_arrays(
            features,
            pixel_index,
            train_folds=train_inner,
            val_fold=inner_val,
            sampling_mode=_SAMPLING_MODE,
            n_per_parcel=n_per_parcel,
            rng=rng,
        )
        aggregations, y_val, probabilities = _fold_probabilities(
            arrays, params, device=device, n_jobs=n_jobs, seed=seed
        )
        truth = _parcel_truth(aggregations["parcel_id"].to_numpy(), parcel_label)
        per_fold.append(parcel_metrics(truth, aggregations["p_mean"].to_numpy()))
        pixel_truth.append(y_val)
        pixel_prob.append(probabilities)
        parcel_truth.append(truth)
        parcel_prob.append(aggregations["p_mean"].to_numpy())

    means = {metric: float(np.mean([m[metric] for m in per_fold])) for metric in _INNER_METRICS}
    for metric, value in means.items():
        trial.set_user_attr(metric, value)
    trial.set_user_attr(
        "inner_pixel_threshold",
        _pooled_f1_threshold(np.concatenate(pixel_truth), np.concatenate(pixel_prob)),
    )
    trial.set_user_attr(
        "inner_parcel_threshold",
        _pooled_f1_threshold(np.concatenate(parcel_truth), np.concatenate(parcel_prob)),
    )
    return means["pr_auc"]


def run_outer_fold(
    outer_fold: int,
    *,
    features: np.ndarray,
    pixel_index: pd.DataFrame,
    parcel_label: Mapping[int, int],
    n_hp_trials: int,
    n_per_parcel: int,
    seed: int,
    device: str,
    n_jobs: int,
    fold_ids: Sequence[int],
    run_logger: logging.Logger,
) -> FoldResult:
    """Run the inner search and outer refit for one outer fold (mirrors ``run_nested_cv``)."""
    start = time.perf_counter()
    inner_folds = tuple(f for f in fold_ids if f != outer_fold)
    sampler_seed = int(np.random.SeedSequence([seed, outer_fold]).generate_state(1)[0])
    study = optuna.create_study(direction="maximize", sampler=TPESampler(seed=sampler_seed))

    def objective(trial: optuna.trial.Trial) -> float:
        return _inner_objective(
            trial,
            features=features,
            pixel_index=pixel_index,
            outer_fold=outer_fold,
            inner_folds=inner_folds,
            parcel_label=parcel_label,
            n_per_parcel=n_per_parcel,
            seed=seed,
            device=device,
            n_jobs=n_jobs,
        )

    study.optimize(objective, n_trials=n_hp_trials)
    best_params = dict(study.best_params)
    best_attrs = study.best_trial.user_attrs
    inner_pixel_threshold = float(best_attrs.get("inner_pixel_threshold", _FALLBACK_THRESHOLD))
    inner_parcel_threshold = float(best_attrs.get("inner_parcel_threshold", _FALLBACK_THRESHOLD))
    run_logger.info(
        "Outer fold %d: best inner parcel PR-AUC %.4f (inner thresholds pixel=%.3f parcel=%.3f).",
        outer_fold,
        study.best_value,
        inner_pixel_threshold,
        inner_parcel_threshold,
    )

    rng = np.random.default_rng([seed, outer_fold])
    arrays = build_fold_arrays(
        features,
        pixel_index,
        train_folds=inner_folds,
        val_fold=outer_fold,
        sampling_mode=_SAMPLING_MODE,
        n_per_parcel=n_per_parcel,
        rng=rng,
    )
    model = make_xgb_classifier(best_params, device=device, n_jobs=n_jobs, seed=seed)
    model.fit(arrays.X_train, arrays.y_train, sample_weight=arrays.train_sample_weight)
    probabilities = predict_ogf_proba(model, arrays.X_val)

    pixel_predictions = pd.DataFrame(
        {
            "pixel_id": arrays.val_pixel_ids.astype(np.int64),
            "outer_fold": np.full(probabilities.shape[0], outer_fold, dtype=np.int8),
            "y_true": arrays.y_val.astype(np.int8),
            "p": probabilities.astype(np.float64),
        }
    )
    aggregations = parcel_aggregations(arrays.val_parcel_ids, probabilities)
    truth = _parcel_truth(aggregations["parcel_id"].to_numpy(), parcel_label)
    parcel_predictions = pd.DataFrame(
        {
            "parcel_id": aggregations["parcel_id"].to_numpy().astype(np.int64),
            "outer_fold": np.full(len(aggregations), outer_fold, dtype=np.int8),
            "y_true": truth.astype(np.int8),
            "p_mean": aggregations["p_mean"].to_numpy().astype(np.float64),
            "n_pixels": aggregations["n_pixels"].to_numpy().astype(np.int64),
        }
    )
    result = FoldResult.from_predictions(
        outer_fold,
        pixel_predictions,
        parcel_predictions,
        best_params=best_params,
        boundary=hp_boundary_diagnostics(best_params),
        importance=feature_importance(model, _FEATURE_NAMES),
        inner_pixel_threshold=inner_pixel_threshold,
        inner_parcel_threshold=inner_parcel_threshold,
        best_inner_pr_auc=float(study.best_value),
        n_train=int(arrays.y_train.shape[0]),
        seconds=time.perf_counter() - start,
    )
    run_logger.info(
        "Outer fold %d done in %.1fs: parcel PR-AUC %.4f, ROC-AUC %.4f.",
        outer_fold,
        result.seconds,
        result.parcel_metrics["pr_auc"],
        result.parcel_metrics["roc_auc"],
    )
    return result


def _persist_fold(run_dir: Path, result: FoldResult) -> None:
    """Write one fold's hyperparameters, diagnostics and out-of-fold predictions."""
    fold = result.outer_fold
    write_json(dict(result.best_params), run_dir / f"best_params_fold{fold}.json")
    write_csv(result.boundary, run_dir / f"boundary_fold{fold}.csv", HP_BOUNDARY_SCHEMA)
    write_csv(result.importance, run_dir / f"importance_fold{fold}.csv", FEATURE_IMPORTANCE_SCHEMA)
    write_parquet(
        result.pixel_predictions,
        run_dir / f"predictions_fold{fold}.parquet",
        PIXEL_PREDICTIONS_SCHEMA,
    )
    write_parquet(
        result.parcel_predictions,
        run_dir / f"parcel_predictions_fold{fold}.parquet",
        PARCEL_PREDICTIONS_SCHEMA,
    )


def _write_pooled_tables(run_dir: Path, results: Sequence[FoldResult]) -> dict[str, float]:
    """Aggregate the folds and write the per-fold, summary and pooled tables."""
    aggregated = aggregate_pooled_metrics(results)
    write_csv(aggregated.per_fold, run_dir / "per_fold_metrics.csv", PER_FOLD_METRICS_SCHEMA)
    write_csv(aggregated.summary, run_dir / "summary_metrics.csv", SUMMARY_METRICS_SCHEMA)
    write_csv(aggregated.pooled_pixel, run_dir / "pooled_pixel_metrics.csv", POOLED_METRICS_SCHEMA)
    write_csv(
        aggregated.pooled_parcel, run_dir / "pooled_parcel_metrics.csv", POOLED_METRICS_SCHEMA
    )

    def pooled(frame: pd.DataFrame, metric: str) -> float:
        return float(frame.loc[frame["metric"] == metric, "value"].iloc[0])

    return {
        "pooled_parcel_pr_auc": pooled(aggregated.pooled_parcel, "pr_auc"),
        "pooled_parcel_roc_auc": pooled(aggregated.pooled_parcel, "roc_auc"),
        "pooled_pixel_pr_auc": pooled(aggregated.pooled_pixel, "pr_auc"),
    }


def _library_versions() -> dict[str, str]:
    """Return versions of the key libraries for the run metadata."""
    versions: dict[str, str] = {}
    for library in ("numpy", "pandas", "xgboost", "scikit-learn", "optuna"):
        try:
            versions[library] = version(library)
        except PackageNotFoundError:
            versions[library] = "unknown"
    return versions


def _parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    """Parse the command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--valid-feature-set",
        default=_DEFAULT_VALID_FROM,
        help="Feature set whose validity mask defines the scored pixels (default headline set).",
    )
    parser.add_argument("--n-trials", type=int, default=N_HP_TRIALS)
    parser.add_argument("--n-per-parcel", type=int, default=N_PER_PARCEL)
    parser.add_argument("--device", default="cuda", choices=["cpu", "cuda"])
    parser.add_argument("--n-jobs", type=int, default=_DEFAULT_N_JOBS)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument(
        "--outer-folds",
        type=int,
        nargs="+",
        default=list(FOLD_IDS),
        help="Outer folds to run (default every fold).",
    )
    return parser.parse_args(argv)


def _run(args: argparse.Namespace, paths: ProjectPaths, run_dir: Path) -> dict:
    """Build coordinate features, run the nested CV and write the artefacts."""
    run_logger = make_run_logger(_LOGGER_NAME, run_dir)
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    pixel_index, _memmap, cache_inputs = _load_pixel_data(paths, args.valid_feature_set)
    ref = open_reference_grid()
    features = coordinate_features(pixel_index, ref)
    parcel_label = _parcel_label_map(pixel_index)
    run_logger.info(
        "Spatial baseline: %d pixels, %d labelled parcels; coordinates from the %s grid.",
        len(pixel_index),
        len(parcel_label),
        ref.crs.to_string() if ref.crs is not None else "reference",
    )

    results: list[FoldResult] = []
    for outer_fold in args.outer_folds:
        result = run_outer_fold(
            outer_fold,
            features=features,
            pixel_index=pixel_index,
            parcel_label=parcel_label,
            n_hp_trials=args.n_trials,
            n_per_parcel=args.n_per_parcel,
            seed=args.seed,
            device=args.device,
            n_jobs=args.n_jobs,
            fold_ids=FOLD_IDS,
            run_logger=run_logger,
        )
        _persist_fold(run_dir, result)
        results.append(result)

    pooled = _write_pooled_tables(run_dir, results)
    run_logger.info(
        "Spatial baseline complete: pooled parcel PR-AUC %.4f, ROC-AUC %.4f.",
        pooled["pooled_parcel_pr_auc"],
        pooled["pooled_parcel_roc_auc"],
    )

    write_input_manifest(cache_inputs, run_dir / "input_manifest.json", repo_root=paths.repo_root)
    write_environment_snapshot(run_dir / "env_snapshot.txt")
    return {
        "created_at": datetime.now(UTC).isoformat(),
        "script": "scripts/run_coordinate_xgboost.py",
        "feature_label": _FEATURE_LABEL,
        "feature_names": list(_FEATURE_NAMES),
        "architecture": _ARCHITECTURE,
        "valid_feature_set": args.valid_feature_set,
        "sampling_mode": _SAMPLING_MODE,
        "n_pixels": int(len(pixel_index)),
        "n_labelled_parcels": int(len(parcel_label)),
        "n_hp_trials": args.n_trials,
        "n_per_parcel": args.n_per_parcel,
        "outer_folds": list(args.outer_folds),
        "device": args.device,
        "n_jobs": args.n_jobs,
        "seed": args.seed,
        "library_versions": _library_versions(),
        **pooled,
    }


def main(argv: Sequence[str] | None = None) -> int:
    """Run the spatial-baseline nested cross-validation and write its artefacts.

    Args:
        argv: Argument list, or ``None`` to read from ``sys.argv``.

    Returns:
        Process exit code (0 on success).
    """
    args = _parse_args(argv)
    paths = get_project_paths()
    ensure_writable_dirs(paths)
    run_id = f"{datetime.now(UTC):%Y%m%d_%H%M%S}__{_ARCHITECTURE}__{_FEATURE_LABEL}"
    run_dir = paths.results / _RESULTS_ROLE / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    summary = _run(args, paths, run_dir)
    write_json(summary, run_dir / "run_metadata.json")
    logger.info("Spatial baseline complete: %s.", run_id)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
