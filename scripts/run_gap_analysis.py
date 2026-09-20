"""Spatial-gap (buffered cross-validation) analysis for XGBoost + baseline_tessera.

Adapting the buffered approach of Ploton et al. (2020) to the spatial block
cross-validation design, this script asks whether residual cross-fold parcel
proximity inflates the performance estimate. For each held-out outer fold,
training parcels whose geometry lies within a buffer ``d`` of the test fold are
excluded before the model is refit; ``d`` is swept over {0, 5, ..., 30} km.

To separate the effect of training-data loss from spatial leakage, a
parcel-matched control is run at every distance: the same number of training
parcels is removed, but chosen at random rather than by spatial buffering. A
buffered curve that falls faster than its matched control is evidence that
proximity, not data volume, was inflating the estimate.

The per-fold hyperparameters from the headline run are reused
(:func:`utils.cv.fixed_hp_fold_result`); no new Optuna search runs. For each
outer fold ``k`` the model trains (with ``best_params_fold{k}``) on the surviving
training parcels and predicts fold ``k``; the folds are then pooled. ``d = 0``
removes nothing, so both arms reproduce the headline fold (a sanity check). The
model runs on GPU (CUDA) by default, much faster for these fits; pass
``--device cpu`` to stay off a GPU that is busy with other jobs. The output feeds
``notebooks/009_performance_matrix_and_buffers.ipynb``.

This adapts the methodology of the archived ``archive/scripts/016_fix_test_gap_matching.py``
(referenced for the parcel-matched control only; no code is imported from the
archive).
"""

import argparse
import json
import logging
import os
from collections.abc import Sequence
from datetime import UTC, datetime
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Final

import geopandas as gpd
import numpy as np
import pandas as pd
from shapely.ops import unary_union

from scripts.run_sampling_sensitivity import (
    _BEST_PARAMS_TEMPLATE,
    _latest_run,
    _load_best_params,
    _load_pixel_data,
    tidy_summary,
)

# The loaders and the tidy-summary join are shared with the sampling-sensitivity
# script, and the parcel block assignment with the block-bootstrap script (same
# selected model and reproducibility-floor pattern).
from utils.bootstrap import load_block_assignment
from utils.cv import (
    FoldResult,
    NestedCVConfig,
    aggregate_pooled_metrics,
    fixed_hp_fold_result,
    fixed_hp_fold_result_cnn,
)
from utils.env_capture import write_environment_snapshot
from utils.io import ColumnSpec, Schema, write_csv, write_json
from utils.logging import make_run_logger
from utils.manifest import write_input_manifest
from utils.paths import ProjectPaths, ensure_writable_dirs, get_project_paths
from utils.reporting import (
    METRIC_NAMES,
    STAT_NAMES,
    inner_thresholds_from_per_fold,
    metric_matrix_row,
)
from utils.terminology import (
    BOOTSTRAP_REPS_CROSS_CONFIG,
    FEATURE_SETS,
    FOLD_IDS,
    N_PER_PARCEL,
    SEED,
)

logger = logging.getLogger(__name__)

_LOGGER_NAME: Final[str] = "run_gap_analysis"

_SOURCE_ROLE: Final[str] = "main_nested_cv"
_RESULTS_ROLE: Final[str] = "spatial_buffer"
_ARCHITECTURE: Final[str] = "xgboost"
_DEFAULT_FEATURE_SET: Final[str] = "baseline_tessera"
# The coordinate-only control (scripts/run_coordinate_xgboost.py): an XGBoost on pixel-centre
# (easting, northing) only, sourced from its results/spatial_baseline run. Not a registered
# feature set, so it is handled specially rather than loaded from a feature cache.
_COORD_FEATURE_SET: Final[str] = "xy_coords"
_COORD_SOURCE_ROLE: Final[str] = "spatial_baseline"
_LABELS_FILENAME: Final[str] = "ogf_reference_labels_partitioned.gpkg"

_DEFAULT_GAP_KM: Final[tuple[int, ...]] = tuple(range(0, 31))
_M_PER_KM: Final[float] = 1000.0
# Which arms to run. Both arms are feature-set specific (each trains on that set's features),
# so every feature set needs its own control to isolate the spatial-buffer effect. A sweep can
# run both arms at once ("both") or one arm at a time ("buffered" / "control") as separate runs
# that the 013 notebook merges per feature set (see performance_change_vs_buffer).
_ARM_SETS: Final[dict[str, tuple[str, ...]]] = {
    "both": ("buffered", "control"),
    "buffered": ("buffered",),
    "control": ("control",),
}
_DEFAULT_N_JOBS: Final[int] = min(8, os.cpu_count() or 1)

_FLOAT = ColumnSpec("float64")
_NULLABLE_FLOAT = ColumnSpec("float64", nullable=True)
_STR = ColumnSpec("object")
_INT = ColumnSpec("int64")

GAP_ANALYSIS_SCHEMA: Final[Schema] = {
    "gap_km": _INT,
    "arm": _STR,
    "level": _STR,
    "metric": _STR,
    "mean": _FLOAT,
    "sd": _NULLABLE_FLOAT,
    "min": _FLOAT,
    "max": _FLOAT,
    "pooled": _NULLABLE_FLOAT,
}

GAP_COUNTS_SCHEMA: Final[Schema] = {
    "gap_km": _INT,
    "arm": _STR,
    "outer_fold": _INT,
    "n_train_parcels": _INT,
    "n_excluded": _INT,
    "n_retained": _INT,
    "n_train_parcels_prevalence": _NULLABLE_FLOAT,
}

# The detailed per-gap table (XGBoost + TESSERA): every metric family of utils.reporting
# in long form, with the pooled bootstrap interval, the per-fold values and their mean/sd,
# one row per (gap, arm, metric, stat). The notebook pivots it to the gaps-as-columns view.
GAP_DETAIL_SCHEMA: Final[Schema] = {
    "gap_km": _INT,
    "arm": _STR,
    "metric": _STR,
    "stat": _STR,
    "value": _NULLABLE_FLOAT,
}


def _load_labelled_geoms(labels_path: Path) -> gpd.GeoDataFrame:
    """Load labelled parcels with ``parcel_id``, ``fold_id``, ``ogf`` and geometry."""
    frame = gpd.read_file(labels_path)
    labelled = frame[frame["ogf"].notna()].copy()
    labelled["parcel_id"] = labelled["parcel_id"].astype(np.int64)
    labelled["fold_id"] = labelled["fold_id"].astype(np.int64)
    labelled["ogf"] = labelled["ogf"].astype(np.int64)
    return labelled[["parcel_id", "fold_id", "ogf", "geometry"]].reset_index(drop=True)


def _train_distances(labelled: gpd.GeoDataFrame, fold: int) -> tuple[gpd.GeoDataFrame, np.ndarray]:
    """Training parcels for one fold and their distance to the held-out fold.

    The distance is from each training parcel (any other fold) to the union of the
    held-out fold's parcels, in the project CRS (metres).
    """
    test_geom = unary_union(labelled.loc[labelled["fold_id"] == fold, "geometry"].to_list())
    train = labelled[labelled["fold_id"] != fold]
    distances = train["geometry"].distance(test_geom).to_numpy()
    return train, distances


def _arm_keep_masks(
    distances: np.ndarray, gap_km: int, *, seed: int, fold: int
) -> dict[str, np.ndarray]:
    """Boolean keep masks over the training parcels for the buffered and control arms.

    ``buffered`` keeps parcels farther than ``gap_km`` from the test fold; ``control``
    keeps the same number, chosen at random with a fixed seed. At ``gap_km == 0``
    nothing is excluded, so the two arms coincide.
    """
    gap_m = gap_km * _M_PER_KM
    n_train = int(distances.size)
    keep_buffered = np.ones(n_train, dtype=bool) if gap_m <= 0 else distances > gap_m
    if gap_m <= 0:
        return {"buffered": keep_buffered, "control": keep_buffered}
    n_keep = int(keep_buffered.sum())
    rng = np.random.default_rng([seed, fold, gap_km])
    keep_control = np.zeros(n_train, dtype=bool)
    keep_idx = (
        rng.choice(n_train, size=n_keep, replace=False) if n_keep < n_train else np.arange(n_train)
    )
    keep_control[keep_idx] = True
    return {"buffered": keep_buffered, "control": keep_control}


def gap_parcel_counts(
    labelled: gpd.GeoDataFrame, *, gap_km_values: Sequence[int], seed: int
) -> pd.DataFrame:
    """Per (gap, arm, fold) training-parcel counts and the retained OGF prevalence.

    Pure geometry and seeded random selection with no model fitting, so it reproduces
    the exact training sets a gap run used and can also be applied retrospectively to
    an existing run. ``n_train_parcels_prevalence`` is the fraction of the retained
    training parcels that are old-growth (NaN when none survive), the quantity that
    distinguishes a spatially buffered exclusion from a prevalence-preserving random
    control.
    """
    rows: list[dict[str, object]] = []
    for fold in FOLD_IDS:
        train, distances = _train_distances(labelled, fold)
        train_ogf = train["ogf"].to_numpy().astype(np.int64)
        n_train = int(train_ogf.size)
        for gap_km in gap_km_values:
            masks = _arm_keep_masks(distances, gap_km, seed=seed, fold=fold)
            for arm, keep_mask in masks.items():
                n_keep = int(keep_mask.sum())
                prevalence = float(train_ogf[keep_mask].mean()) if n_keep else float("nan")
                rows.append(
                    {
                        "gap_km": int(gap_km),
                        "arm": arm,
                        "outer_fold": int(fold),
                        "n_train_parcels": n_train,
                        "n_excluded": n_train - n_keep,
                        "n_retained": n_keep,
                        "n_train_parcels_prevalence": prevalence,
                    }
                )
    return pd.DataFrame(rows, columns=list(GAP_COUNTS_SCHEMA))


def _fold_result_for_exclusion(
    config: NestedCVConfig,
    pixel_index: pd.DataFrame,
    memmap: np.ndarray,
    best_params: dict[str, float | int],
    *,
    outer_fold: int,
    base_valid: np.ndarray,
    parcel_ids: np.ndarray,
    excluded_pids: np.ndarray,
) -> FoldResult:
    """Refit one fold with the excluded training parcels masked out of ``valid``."""
    gap_valid = base_valid & ~np.isin(parcel_ids, excluded_pids)
    fold_index = pixel_index.assign(valid=gap_valid)
    return fixed_hp_fold_result(config, fold_index, memmap, best_params, outer_fold=outer_fold)


_CHECKPOINT_DIR: Final[str] = "fold_checkpoints"
# Written at run start (before any training) so --auto-resume can tell a buffered run from a
# control run and never resume across arm modes.
_RUN_ARMS_FILE: Final[str] = "run_arms.json"


def _fold_checkpoint_paths(run_dir: Path, fold: int) -> tuple[Path, Path]:
    """Parcel- and pixel-prediction checkpoint paths for one outer fold."""
    base = run_dir / _CHECKPOINT_DIR
    return base / f"fold{fold}_parcel.parquet", base / f"fold{fold}_pixel.parquet"


def _latest_unfinished_run(
    results_role_dir: Path, feature_set: str, arms: Sequence[str]
) -> str | None:
    """Newest resumable run id for ``feature_set`` with the same ``arms``, or ``None``.

    A run is resumable when it never wrote ``gap_analysis.csv`` (that file marks a complete
    run), already holds at least one per-fold checkpoint, and recorded the same arms in its
    ``run_arms.json`` marker. Matching on arms stops ``--auto-resume`` from continuing a
    different-arm run: a control sweep must never resume an unfinished buffered sweep, whose
    checkpoints hold buffered predictions. Runs with no marker (older runs) are skipped rather
    than risk a cross-arm resume; continue those with an explicit ``--resume`` instead.
    """
    suffix = f"__{_ARCHITECTURE}__{feature_set}"
    if not results_role_dir.is_dir():
        return None
    want = list(arms)
    resumable: list[str] = []
    for p in results_role_dir.iterdir():
        if not (p.is_dir() and p.name.endswith(suffix)):
            continue
        if (p / "gap_analysis.csv").is_file():
            continue
        if not any((p / _CHECKPOINT_DIR).glob("fold*_parcel.parquet")):
            continue
        marker = p / _RUN_ARMS_FILE
        if not marker.is_file():
            continue
        try:
            run_arms = json.loads(marker.read_text(encoding="utf-8")).get("arms")
        except (json.JSONDecodeError, OSError):
            continue
        if run_arms == want:
            resumable.append(p.name)
    return sorted(resumable)[-1] if resumable else None


def _write_fold_checkpoint(
    run_dir: Path, fold: int, entries: Sequence[tuple[str, int, FoldResult]]
) -> None:
    """Persist a completed fold's per-(arm, gap) predictions so a crash can resume."""
    parcel_path, pixel_path = _fold_checkpoint_paths(run_dir, fold)
    parcel_path.parent.mkdir(parents=True, exist_ok=True)
    parcel = pd.concat(
        [
            fr.parcel_predictions.assign(arm=arm, gap_km=gap, n_train=fr.n_train)
            for arm, gap, fr in entries
        ],
        ignore_index=True,
    )
    pixel = pd.concat(
        [fr.pixel_predictions.assign(arm=arm, gap_km=gap) for arm, gap, fr in entries],
        ignore_index=True,
    )
    parcel.to_parquet(parcel_path)
    pixel.to_parquet(pixel_path)


def _read_fold_checkpoint(
    run_dir: Path, fold: int, best_params_fold: dict[str, float | int]
) -> dict[tuple[str, int], FoldResult]:
    """Rebuild a fold's FoldResults from its checkpoint (the resume path)."""
    parcel_path, pixel_path = _fold_checkpoint_paths(run_dir, fold)
    parcel = pd.read_parquet(parcel_path)
    pixel = pd.read_parquet(pixel_path)
    rebuilt: dict[tuple[str, int], FoldResult] = {}
    for (arm, gap), parcel_group in parcel.groupby(["arm", "gap_km"], sort=False):
        pixel_group = pixel[(pixel["arm"] == arm) & (pixel["gap_km"] == gap)]
        rebuilt[(str(arm), int(gap))] = FoldResult.from_predictions(
            fold,
            pixel_group.drop(columns=["arm", "gap_km"]).reset_index(drop=True),
            parcel_group.drop(columns=["arm", "gap_km", "n_train"]).reset_index(drop=True),
            best_params=dict(best_params_fold),
            n_train=int(parcel_group["n_train"].iloc[0]),
        )
    return rebuilt


def run_gap_analysis(
    pixel_index: pd.DataFrame,
    memmap: np.ndarray,
    best_params: dict[int, dict[str, float | int]],
    labelled: gpd.GeoDataFrame,
    *,
    feature_set: str,
    gap_km_values: Sequence[int],
    arms: Sequence[str],
    device: str,
    n_jobs: int,
    seed: int,
    n_per_parcel: int,
    run_logger: logging.Logger,
    run_dir: Path,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[tuple[str, int], list[FoldResult]]]:
    """Sweep the spatial buffer and the matched control, returning (summary, counts, results).

    Each outer fold's per-(arm, gap) predictions are checkpointed to
    ``<run_dir>/fold_checkpoints/`` as soon as the fold finishes, so a crash mid-sweep
    loses at most the fold in progress; re-running with the same ``run_dir`` (``--resume``)
    reloads the completed folds and continues.
    """
    config = NestedCVConfig(
        feature_set=feature_set,
        architecture=_ARCHITECTURE,
        sampling_mode="none",
        n_per_parcel=n_per_parcel,
        seed=seed,
        device=device,
        n_jobs=n_jobs,
        fold_ids=FOLD_IDS,
    )
    base_valid = pixel_index["valid"].to_numpy()
    parcel_ids = pixel_index["parcel_id"].to_numpy()
    counts_table = gap_parcel_counts(labelled, gap_km_values=gap_km_values, seed=seed)

    # results[(arm, gap_km)] -> list of FoldResult (folds that retained training data).
    results: dict[tuple[str, int], list[FoldResult]] = {}
    for fold in FOLD_IDS:
        parcel_path, pixel_path = _fold_checkpoint_paths(run_dir, fold)
        if parcel_path.is_file() and pixel_path.is_file():
            for (arm, gap_km), fold_result in _read_fold_checkpoint(
                run_dir, fold, best_params[fold]
            ).items():
                results.setdefault((arm, gap_km), []).append(fold_result)
            run_logger.info("Fold %d: loaded saved checkpoint.", fold)
            continue
        train, distances = _train_distances(labelled, fold)
        train_pids = train["parcel_id"].to_numpy().astype(np.int64)
        entries: list[tuple[str, int, FoldResult]] = []
        for gap_km in gap_km_values:
            masks = {
                arm: mask
                for arm, mask in _arm_keep_masks(distances, gap_km, seed=seed, fold=fold).items()
                if arm in arms
            }
            shared: FoldResult | None = None
            for arm, keep_mask in masks.items():
                if not keep_mask.any():
                    run_logger.warning(
                        "Fold %d gap %d km arm %s: no training parcels survive; skipped.",
                        fold,
                        gap_km,
                        arm,
                    )
                    continue
                if np.unique(train["ogf"].to_numpy()[keep_mask]).size < 2:
                    run_logger.warning(
                        "Fold %d gap %d km arm %s: surviving training parcels are single-class; "
                        "skipped.",
                        fold,
                        gap_km,
                        arm,
                    )
                    continue
                if gap_km <= 0 and shared is not None:
                    fold_result = shared
                else:
                    fold_result = _fold_result_for_exclusion(
                        config,
                        pixel_index,
                        memmap,
                        best_params[fold],
                        outer_fold=fold,
                        base_valid=base_valid,
                        parcel_ids=parcel_ids,
                        excluded_pids=train_pids[~keep_mask],
                    )
                    if gap_km <= 0:
                        shared = fold_result
                results.setdefault((arm, gap_km), []).append(fold_result)
                entries.append((arm, gap_km, fold_result))
        _write_fold_checkpoint(run_dir, fold, entries)
        run_logger.info(
            "Fold %d: swept %d gap distances x %d arm(s) (checkpointed).",
            fold,
            len(gap_km_values),
            len(arms),
        )

    frames: list[pd.DataFrame] = []
    for gap_km in gap_km_values:
        for arm in arms:
            fold_results = results.get((arm, gap_km), [])
            if not fold_results:
                continue
            aggregated = aggregate_pooled_metrics(fold_results)
            frame = tidy_summary("arm", arm, aggregated)
            frame.insert(0, "gap_km", int(gap_km))
            frames.append(frame)
            parcel = aggregated.pooled_parcel
            pr_auc = float(parcel.loc[parcel["metric"] == "pr_auc", "value"].iloc[0])
            run_logger.info(
                "Gap %d km arm %s (%d folds): pooled parcel PR-AUC %.4f.",
                gap_km,
                arm,
                len(fold_results),
                pr_auc,
            )

    summary = pd.concat(frames, ignore_index=True)[list(GAP_ANALYSIS_SCHEMA)]
    return summary, counts_table, results


def _inner_thresholds(source_run: Path) -> dict[int, float]:
    """Per-fold parcel inner threshold from the headline run (0.5 fallback if absent)."""
    per_fold_path = source_run / "per_fold_metrics.csv"
    if per_fold_path.is_file():
        return inner_thresholds_from_per_fold(pd.read_csv(per_fold_path))
    return dict.fromkeys(FOLD_IDS, 0.5)


def gap_detail(
    results: dict[tuple[str, int], list[FoldResult]],
    inner_thresholds: dict[int, float],
    block_assignment: pd.DataFrame,
    *,
    gap_km_values: Sequence[int],
    arms: Sequence[str],
    reps: int,
    seed: int,
    n_jobs: int,
) -> pd.DataFrame:
    """Long-form per-gap metric detail for the buffered and control arms (XGBoost + TESSERA).

    Every metric family (pr_auc, roc_auc, f1/precision/recall at the outer and inner
    thresholds) is scored with :func:`utils.reporting.metric_matrix_row`, giving the pooled
    estimate, its bootstrap interval, the per-fold values and their mean/sd at each gap.
    """
    rows: list[dict[str, object]] = []
    for gap_km in gap_km_values:
        for arm in arms:
            fold_results = results.get((arm, gap_km), [])
            if not fold_results:
                continue
            frames = {int(fr.outer_fold): fr.parcel_predictions for fr in fold_results}
            stats = metric_matrix_row(
                frames, inner_thresholds, block_assignment, reps=reps, seed=seed, n_jobs=n_jobs
            )
            for metric in METRIC_NAMES:
                for stat in STAT_NAMES:
                    rows.append(
                        {
                            "gap_km": int(gap_km),
                            "arm": arm,
                            "metric": metric,
                            "stat": stat,
                            "value": stats[f"{metric}__{stat}"],
                        }
                    )
    return pd.DataFrame(rows, columns=list(GAP_DETAIL_SCHEMA))


# --------------------------------------------------------------------------- #
# Shared buffered fixed-HP refit helpers, reused by scripts/run_performance_matrix.py
# for its 10 km buffered arm (the templates locate per-fold CNN hyperparameters).
# --------------------------------------------------------------------------- #

_TRAINING_RECIPE_TEMPLATE: Final[str] = "training_recipe_fold{fold}.json"
_PARCEL_PRED_TEMPLATE: Final[str] = "parcel_predictions_fold{fold}.parquet"


def _buffered_excluded_pids(
    labelled: gpd.GeoDataFrame, *, buffer_km: int, seed: int
) -> dict[int, np.ndarray]:
    """Buffered-arm excluded training parcel ids per outer fold at one buffer distance."""
    excluded: dict[int, np.ndarray] = {}
    for fold in FOLD_IDS:
        train, distances = _train_distances(labelled, fold)
        train_pids = train["parcel_id"].to_numpy().astype(np.int64)
        keep = _arm_keep_masks(distances, buffer_km, seed=seed, fold=fold)["buffered"]
        excluded[fold] = train_pids[~keep]
    return excluded


def _xgb_buffered_fold_results(
    config: NestedCVConfig,
    pixel_index: pd.DataFrame,
    memmap: np.ndarray,
    best_params: dict[int, dict[str, float | int]],
    excluded: dict[int, np.ndarray],
) -> list[FoldResult]:
    """Buffered fixed-HP XGBoost refit for every outer fold."""
    base_valid = pixel_index["valid"].to_numpy()
    parcel_ids = pixel_index["parcel_id"].to_numpy()
    return [
        _fold_result_for_exclusion(
            config,
            pixel_index,
            memmap,
            best_params[fold],
            outer_fold=fold,
            base_valid=base_valid,
            parcel_ids=parcel_ids,
            excluded_pids=excluded[fold],
        )
        for fold in FOLD_IDS
    ]


def _latest_cnn_run(source_root: Path, architecture: str, feature_set: str) -> Path:
    """Most recent run dir for one CNN architecture and feature set (any fold count).

    Unlike :func:`_latest_run`, completeness is not required: the matrix may still be
    running, and run_performance_matrix reports the buffered cells provisionally over
    whichever folds already carry hyperparameters.
    """
    suffix = f"__{architecture}__{feature_set}"
    candidates = sorted(
        (p for p in source_root.iterdir() if p.is_dir() and p.name.endswith(suffix)),
        key=lambda p: p.name,
        reverse=True,
    )
    if not candidates:
        raise FileNotFoundError(f"No {architecture} run for {feature_set!r} under {source_root}.")
    return candidates[0]


def _load_cnn_fold_hps(
    cnn_run: Path,
) -> tuple[dict[int, dict[str, float | int]], dict[int, int], list[Path]]:
    """Per-fold CNN ``best_params`` and ``cnn_epochs`` for the folds that have them."""
    best_params: dict[int, dict[str, float | int]] = {}
    cnn_epochs: dict[int, int] = {}
    inputs: list[Path] = []
    for fold in FOLD_IDS:
        bp_path = cnn_run / _BEST_PARAMS_TEMPLATE.format(fold=fold)
        recipe_path = cnn_run / _TRAINING_RECIPE_TEMPLATE.format(fold=fold)
        if not (bp_path.is_file() and recipe_path.is_file()):
            continue
        recipe = json.loads(recipe_path.read_text(encoding="utf-8"))
        if recipe.get("cnn_epochs") is None:
            continue
        best_params[fold] = json.loads(bp_path.read_text(encoding="utf-8"))
        cnn_epochs[fold] = int(recipe["cnn_epochs"])
        inputs += [bp_path, recipe_path]
    return best_params, cnn_epochs, inputs


def _cnn_buffered_fold_results(
    config: NestedCVConfig,
    pixel_index: pd.DataFrame,
    patch_cache: np.ndarray,
    best_params: dict[int, dict[str, float | int]],
    cnn_epochs: dict[int, int],
    excluded: dict[int, np.ndarray],
    *,
    patch_size: int,
    grid_shape: tuple[int, int],
    run_logger: logging.Logger,
    config_key: str,
) -> tuple[list[FoldResult], list[int]]:
    """Buffered fixed-HP CNN refit for every outer fold that has saved hyperparameters."""
    base_valid = pixel_index["valid"].to_numpy()
    parcel_ids = pixel_index["parcel_id"].to_numpy()
    results: list[FoldResult] = []
    folds_used: list[int] = []
    for fold in FOLD_IDS:
        if fold not in best_params or fold not in cnn_epochs:
            run_logger.warning("%s: fold %d has no saved CNN HPs yet; skipped.", config_key, fold)
            continue
        gap_valid = base_valid & ~np.isin(parcel_ids, excluded[fold])
        fold_index = pixel_index.assign(valid=gap_valid)
        results.append(
            fixed_hp_fold_result_cnn(
                config,
                fold_index,
                patch_cache,
                best_params[fold],
                patch_size=patch_size,
                grid_shape=grid_shape,
                cnn_epochs=cnn_epochs[fold],
                outer_fold=fold,
            )
        )
        folds_used.append(fold)
    run_logger.info("%s: %d/%d folds refit at the buffer.", config_key, len(results), len(FOLD_IDS))
    return results, folds_used


def _library_versions() -> dict[str, str]:
    """Return versions of the key libraries for the run metadata."""
    versions: dict[str, str] = {}
    for library in ("numpy", "pandas", "xgboost", "scikit-learn", "geopandas"):
        try:
            versions[library] = version(library)
        except PackageNotFoundError:
            versions[library] = "unknown"
    return versions


def _parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    """Parse the command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--feature-set",
        default=_DEFAULT_FEATURE_SET,
        choices=[*FEATURE_SETS, _COORD_FEATURE_SET],
        help=f"Feature set to sweep, or '{_COORD_FEATURE_SET}' for the coordinate-only control "
        "(sourced from results/spatial_baseline).",
    )
    parser.add_argument("--source-role", default=_SOURCE_ROLE)
    parser.add_argument(
        "--arms",
        default="both",
        choices=list(_ARM_SETS),
        help="'both' runs the buffered arm and its parcel-matched random control; "
        "'buffered' or 'control' runs just that one arm as a separate sweep (run both, per "
        "feature set, for the notebook to merge into a per-feature-set performance change).",
    )
    parser.add_argument(
        "--gap-km",
        type=int,
        nargs="+",
        default=list(_DEFAULT_GAP_KM),
        help="Buffer distances (km).",
    )
    parser.add_argument("--device", default="cuda", choices=["cpu", "cuda"])
    parser.add_argument("--n-jobs", type=int, default=_DEFAULT_N_JOBS)
    parser.add_argument("--n-per-parcel", type=int, default=N_PER_PARCEL)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument(
        "--resume",
        default=None,
        help="Reuse an existing results/spatial_buffer/<run_id>, continuing from its saved "
        "per-fold checkpoints instead of starting a fresh run.",
    )
    parser.add_argument(
        "--auto-resume",
        action="store_true",
        help="Resume the newest unfinished run for this feature set and --arms mode "
        "automatically (results/spatial_buffer/*__<feature_set> with per-fold checkpoints, no "
        "gap_analysis.csv, and a matching run_arms.json); start fresh if there is none. A "
        "control sweep never resumes a buffered sweep. Ignored when --resume is given.",
    )
    parser.add_argument(
        "--reps",
        type=int,
        default=BOOTSTRAP_REPS_CROSS_CONFIG,
        help="Bootstrap replicates for the gap-detail intervals (notebook-010 count).",
    )
    return parser.parse_args(argv)


def _load_coordinate_data(
    paths: ProjectPaths, source_run: Path
) -> tuple[pd.DataFrame, np.ndarray, list[Path], str]:
    """Pixel index + (easting, northing) feature matrix for the coordinate-only control.

    Rebuilds the inputs of the spatial-baseline run (scripts/run_coordinate_xgboost.py): the
    pixel index and validity mask of that run's ``valid_feature_set``, plus the two-column
    coordinate matrix from the reference grid. The validity feature set is returned to use as
    the config label only (:func:`fixed_hp_fold_result` reads the memmap, never the name).
    """
    from scripts.run_coordinate_xgboost import coordinate_features
    from utils.raster_io import open_reference_grid

    metadata = json.loads((source_run / "run_metadata.json").read_text("utf-8"))
    valid_feature_set = str(metadata["valid_feature_set"])
    pixel_index, _features, cache_inputs = _load_pixel_data(paths, valid_feature_set)
    memmap = coordinate_features(pixel_index, open_reference_grid())
    return pixel_index, memmap, cache_inputs, valid_feature_set


def _run(args: argparse.Namespace, paths: ProjectPaths, run_dir: Path) -> tuple[Path, dict]:
    """Resolve inputs, run the sweep, write the tables; return (summary path, metadata)."""
    run_logger = make_run_logger(_LOGGER_NAME, run_dir)
    is_coord = args.feature_set == _COORD_FEATURE_SET
    source_role = _COORD_SOURCE_ROLE if is_coord else args.source_role
    source_run = _latest_run(paths.results / source_role, args.feature_set)
    run_logger.info(
        "Feature set %s: source run %s; gaps %s km; device=%s, n_jobs=%d.",
        args.feature_set,
        source_run.relative_to(paths.repo_root),
        args.gap_km,
        args.device,
        args.n_jobs,
    )

    best_params = _load_best_params(source_run)
    if is_coord:
        pixel_index, memmap, cache_inputs, config_feature_set = _load_coordinate_data(
            paths, source_run
        )
        run_logger.info(
            "Coordinate-only control: (easting, northing) features; validity mask from %s.",
            config_feature_set,
        )
    else:
        pixel_index, memmap, cache_inputs = _load_pixel_data(paths, args.feature_set)
        config_feature_set = args.feature_set
    labels_path = paths.labels / _LABELS_FILENAME
    labelled = _load_labelled_geoms(labels_path)
    block_assignment = load_block_assignment(labels_path)
    inner_thresholds = _inner_thresholds(source_run)
    run_logger.info("Loaded %d pixels and %d labelled parcels.", len(pixel_index), len(labelled))

    summary, counts, results = run_gap_analysis(
        pixel_index,
        memmap,
        best_params,
        labelled,
        feature_set=config_feature_set,
        gap_km_values=args.gap_km,
        arms=_ARM_SETS[args.arms],
        device=args.device,
        n_jobs=args.n_jobs,
        seed=args.seed,
        n_per_parcel=args.n_per_parcel,
        run_logger=run_logger,
        run_dir=run_dir,
    )
    detail = gap_detail(
        results,
        inner_thresholds,
        block_assignment,
        gap_km_values=args.gap_km,
        arms=_ARM_SETS[args.arms],
        reps=args.reps,
        seed=args.seed,
        n_jobs=args.n_jobs,
    )
    summary_path = write_csv(summary, run_dir / "gap_analysis.csv", GAP_ANALYSIS_SCHEMA)
    counts_path = write_csv(counts, run_dir / "gap_parcel_counts.csv", GAP_COUNTS_SCHEMA)
    write_csv(detail, run_dir / "gap_detail.csv", GAP_DETAIL_SCHEMA)

    inputs = (
        cache_inputs
        + [labels_path]
        + [source_run / _BEST_PARAMS_TEMPLATE.format(fold=fold) for fold in FOLD_IDS]
    )
    write_input_manifest(inputs, run_dir / "input_manifest.json", repo_root=paths.repo_root)
    write_environment_snapshot(run_dir / "env_snapshot.txt")
    metadata = {
        "run_id": run_dir.name,
        "created_at": datetime.now(UTC).isoformat(),
        "script": "scripts/run_gap_analysis.py",
        "feature_set": args.feature_set,
        "config_feature_set": config_feature_set,
        "architecture": _ARCHITECTURE,
        "arms": list(_ARM_SETS[args.arms]),
        "source_run": str(source_run.relative_to(paths.repo_root)),
        "gap_km": list(args.gap_km),
        "device": args.device,
        "n_jobs": args.n_jobs,
        "n_per_parcel": args.n_per_parcel,
        "seed": args.seed,
        "method": "Ploton et al. (2020) buffered CV with a parcel-matched random control",
        "hp_reuse": "per-fold best_params from the headline run (no Optuna search)",
        "library_versions": _library_versions(),
        "outputs": {
            "gap_analysis": str(summary_path.relative_to(paths.repo_root)),
            "gap_parcel_counts": str(counts_path.relative_to(paths.repo_root)),
        },
    }
    return summary_path, metadata


def main(argv: Sequence[str] | None = None) -> int:
    """Run the spatial-gap analysis and write its artefacts.

    Args:
        argv: Argument list, or ``None`` to read from ``sys.argv``.

    Returns:
        Process exit code (0 on success).
    """
    args = _parse_args(argv)
    paths = get_project_paths()
    ensure_writable_dirs(paths)
    timestamp = f"{datetime.now(UTC):%Y%m%d_%H%M%S}"
    arms = _ARM_SETS[args.arms]
    resume_id = args.resume
    if resume_id is None and args.auto_resume:
        resume_id = _latest_unfinished_run(paths.results / _RESULTS_ROLE, args.feature_set, arms)
        if resume_id is None:
            logger.info(
                "--auto-resume: no unfinished %s %s run; starting fresh.",
                args.feature_set,
                args.arms,
            )
        else:
            logger.info("--auto-resume: continuing unfinished run %s.", resume_id)
    if resume_id is not None:
        run_id = resume_id
        run_dir = paths.results / _RESULTS_ROLE / run_id
        if not run_dir.is_dir():
            raise FileNotFoundError(f"--resume run directory not found: {run_dir}")
    else:
        run_id = f"{timestamp}__{_ARCHITECTURE}__{args.feature_set}"
        run_dir = paths.results / _RESULTS_ROLE / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    # Record this run's arms up front (before any training) so --auto-resume can distinguish a
    # buffered run from a control run; keep an existing marker so a resumed run keeps its arms.
    arms_marker = run_dir / _RUN_ARMS_FILE
    if not arms_marker.is_file():
        arms_marker.write_text(json.dumps({"arms": list(arms)}), encoding="utf-8")

    _output_path, metadata = _run(args, paths, run_dir)
    write_json(metadata, run_dir / "run_metadata.json")
    logger.info("Gap analysis complete: %s.", run_id)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
