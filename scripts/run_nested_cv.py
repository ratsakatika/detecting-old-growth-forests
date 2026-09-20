"""Run the nested cross-validation for one feature set and architecture.

This is the resumable, monitored driver around both nested-CV runners. It
dispatches on ``--architecture``: ``xgboost`` reads the flat per-pixel feature
memmap from ``scripts/build_pixel_table.py`` and runs
:func:`utils.cv.run_nested_cv`, while a ``cnn_*`` architecture builds (or reuses)
the patch cache with ``scripts/build_patch_cache.py`` and runs
:func:`utils.cv.run_nested_cv_cnn`. Everything else is single-copy: argument
parsing, the run directory and routed run logger, device detection, the
file-lock GPU semaphore, resume, the trials frame, pooled-table writing, run
metadata are shared by both architectures.

The two architectures differ in only three places: the feature array loaded, the
absence of a feature-importance table for the CNN, and the model runner invoked.
``torch`` is imported lazily inside the CNN branch alone, so an XGBoost run never
imports it and reproduces the certified results unchanged.

Parallelism is across feature sets, one operating-system process per feature set
(see ``scripts/launch_nested_cv_xgboost.sh`` for XGBoost and
``scripts/launch_nested_cv_cnn.sh`` for the CNN). Concurrent processes share the
OS page cache for the memmaps, so each matrix is read once and reused. An
optional file-lock semaphore caps the number of concurrent GPU fits.
"""

import argparse
import json
import logging
import os
import time
from collections.abc import Callable, Iterator, Sequence
from contextlib import AbstractContextManager, contextmanager
from dataclasses import dataclass
from datetime import UTC, datetime
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Final

import numpy as np
import pandas as pd
from filelock import FileLock, Timeout

from utils.config import config_to_dict
from utils.cv import (
    AggregatedMetrics,
    FoldResult,
    NestedCVConfig,
    TrialRecord,
    aggregate_pooled_metrics,
    run_nested_cv,
)
from utils.env_capture import write_environment_snapshot
from utils.io import (
    FEATURE_IMPORTANCE_SCHEMA,
    HP_BOUNDARY_SCHEMA,
    HP_TRIALS_SCHEMA,
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
from utils.paths import ProjectPaths, ensure_writable_dirs, get_project_paths
from utils.sampling import SAMPLING_MODES
from utils.terminology import (
    ARCHITECTURES,
    CNN_PATCH_SIZES,
    FEATURE_SETS,
    FOLD_IDS,
    MODEL_INPUT_BANDS,
    MODEL_INPUTS,
    N_HP_TRIALS,
    N_HP_TRIALS_XGBOOST,
    N_PER_PARCEL,
    SEED,
)

# The certified default architecture; also the only non-CNN architecture, so it
# alone writes a feature-importance table.
_DEFAULT_ARCHITECTURE: Final[str] = "xgboost"
_RESULTS_ROLE: Final[str] = "main_nested_cv"
_PIXEL_INDEX_FILENAME: Final[str] = "pixel_index.parquet"
_CACHE_SUBDIR: Final[Path] = Path("pixel_features")  # flat per-pixel memmaps (XGBoost)

# Total CPU thread budget split across concurrent feature-set processes.
_CORE_BUDGET: Final[int] = os.cpu_count() or 32

# CNN launch defaults. The patch loaders are I/O bound, so the loader-worker
# budget (n_jobs) defaults high; PyTorch intra-op threads are capped lower
# because a single CNN fit already saturates the GPU.
_CNN_MAX_N_JOBS: Final[int] = 16
_CNN_TORCH_THREADS: Final[int] = 8


# Inner-search budget per outer fold by architecture, as published: 50 trials for
# XGBoost, 30 for the CNNs (manuscript Section 2.7). ``--n-hp-trials`` overrides it.
def _default_n_hp_trials(architecture: str) -> int:
    """Return the published Optuna budget for ``architecture``."""
    return N_HP_TRIALS_XGBOOST if architecture == _DEFAULT_ARCHITECTURE else N_HP_TRIALS


# Smoke-run overrides (one feature set, one outer fold, 2 trials, 20 px/parcel).
_SMOKE_N_HP_TRIALS: Final[int] = 2
_SMOKE_N_PER_PARCEL: Final[int] = 20

# Decision threshold used when an inner threshold cannot be recovered on resume.
_FALLBACK_THRESHOLD: Final[float] = 0.5

# Per-fold training-recipe provenance (architecture, feature order, winning HPs,
# CNN epoch count). Bump the schema version when the recorded fields change.
TRAINING_RECIPE_SCHEMA_VERSION: Final[int] = 1
# The inner-search objective used to select hyperparameters, recorded for audit.
FINAL_RECIPE_SELECTION_METRIC: Final[str] = "mean_inner_parcel_pr_auc"


def _cache_paths(paths: ProjectPaths, feature_set: str) -> tuple[Path, Path]:
    """Return the ``(matrix .npy, validity parquet)`` cache paths for a feature set."""
    cache_dir = paths.cache / _CACHE_SUBDIR
    return cache_dir / f"{feature_set}.npy", cache_dir / f"{feature_set}_valid.parquet"


def _detect_device(requested: str, logger: logging.Logger) -> str:
    """Resolve the compute device, probing for CUDA when ``requested`` is ``auto``.

    Args:
        requested: ``"auto"``, ``"cuda"`` or ``"cpu"``.
        logger: Run logger for the probe outcome.

    Returns:
        ``"cuda"`` or ``"cpu"``.
    """
    if requested in ("cuda", "cpu"):
        return requested
    try:
        import xgboost as xgb

        probe = xgb.XGBClassifier(device="cuda", tree_method="hist", n_estimators=1, verbosity=0)
        probe.fit(np.zeros((4, 2), dtype=np.float32), np.array([0, 1, 0, 1], dtype=np.int8))
        logger.info("CUDA probe succeeded; using GPU.")
        return "cuda"
    except Exception as error:  # any failure (no GPU, no CUDA build) means CPU
        logger.info("CUDA probe failed (%s); using CPU.", error)
        return "cpu"


class _GpuSemaphore:
    """A file-lock semaphore capping concurrent GPU fits across processes.

    ``n_slots`` lock files live under ``data/cache``; each acquired slot permits
    one concurrent GPU fit. :meth:`slot` blocks (polling) until a slot is free,
    so the cap can be tuned per launch without code changes.
    """

    def __init__(self, lock_dir: Path, n_slots: int, *, poll_seconds: float = 0.5) -> None:
        lock_dir.mkdir(parents=True, exist_ok=True)
        self._locks = [
            FileLock(str(lock_dir / f"gpu_slots_{index}.lock")) for index in range(n_slots)
        ]
        self._poll_seconds = poll_seconds

    @contextmanager
    def slot(self) -> Iterator[None]:
        """Acquire one GPU slot for the duration of the ``with`` block."""
        while True:
            for lock in self._locks:
                try:
                    lock.acquire(timeout=0)
                except Timeout:
                    continue
                try:
                    yield
                finally:
                    lock.release()
                return
            time.sleep(self._poll_seconds)


@dataclass(frozen=True, slots=True)
class _FoldPaths:
    """Per-fold output paths under a run directory."""

    best_params: Path
    boundary: Path
    importance: Path
    pixel_predictions: Path
    parcel_predictions: Path
    hp_trials: Path
    training_recipe: Path

    @classmethod
    def for_fold(cls, run_dir: Path, fold: int) -> "_FoldPaths":
        return cls(
            best_params=run_dir / f"best_params_fold{fold}.json",
            boundary=run_dir / f"boundary_fold{fold}.csv",
            importance=run_dir / f"importance_fold{fold}.csv",
            pixel_predictions=run_dir / f"predictions_fold{fold}.parquet",
            parcel_predictions=run_dir / f"parcel_predictions_fold{fold}.parquet",
            hp_trials=run_dir / f"hp_trials_fold{fold}.csv",
            training_recipe=run_dir / f"training_recipe_fold{fold}.json",
        )

    def fold_complete(self, *, writes_importance: bool) -> bool:
        """Whether every output marking a finished fold is present.

        Args:
            writes_importance: Whether this architecture writes a feature
                importance table; the CNN does not, so the importance file is
                required only for XGBoost.
        """
        required = [
            self.best_params,
            self.boundary,
            self.pixel_predictions,
            self.parcel_predictions,
            self.hp_trials,
            self.training_recipe,
        ]
        if writes_importance:
            required.append(self.importance)
        return all(path.is_file() for path in required)


def _is_cnn(architecture: str) -> bool:
    """Whether an architecture name denotes a receptive-field CNN.

    Mirrors the ``cnn_`` prefix test used to derive
    :data:`utils.terminology.CNN_PATCH_SIZES`, so the two cannot drift.
    """
    return architecture.startswith("cnn_")


def _patch_size_from_architecture(architecture: str) -> int:
    """Receptive-field patch side length encoded in a ``cnn_<n>x<n>`` name.

    The architecture name is the single source of truth for the patch size, read
    here the same way :data:`utils.terminology.CNN_PATCH_SIZES` derives it.
    """
    return int(architecture.removeprefix("cnn_").split("x")[0])


def _validate_patch_size(architecture: str, patch_size: int | None) -> int | None:
    """Cross-check ``--patch-size`` against the architecture.

    Args:
        architecture: Canonical architecture name.
        patch_size: The value passed on the command line, or ``None``.

    Returns:
        The patch size for a CNN architecture, or ``None`` for XGBoost.

    Raises:
        ValueError: If a patch size is given for XGBoost, omitted for a CNN, or
            does not match the size encoded in the CNN architecture name.
    """
    if not _is_cnn(architecture):
        if patch_size is not None:
            raise ValueError(
                f"--patch-size is only valid for a CNN architecture, not {architecture!r}."
            )
        return None
    if patch_size is None:
        raise ValueError(f"--patch-size is required for the {architecture!r} architecture.")
    expected = _patch_size_from_architecture(architecture)
    if patch_size != expected:
        raise ValueError(
            f"--patch-size {patch_size} does not match architecture {architecture!r}, "
            f"which encodes patch size {expected}."
        )
    return patch_size


def _load_pixel_index(
    paths: ProjectPaths, feature_set: str, logger: logging.Logger
) -> tuple[pd.DataFrame, list[Path]]:
    """Load the pixel index and merge the feature-set validity mask.

    Shared by both architectures: only the feature array loaded afterwards
    differs (a flat per-pixel memmap for XGBoost, a grid memmap for the CNN). The
    per-pixel validity mask is the same for both, since it marks centre-pixel
    NoData.

    Args:
        paths: Resolved project paths.
        feature_set: Canonical feature-set name.
        logger: Run logger.

    Returns:
        A ``(pixel_index, inputs)`` pair: the pixel index with a boolean ``valid``
        column merged in, and the input files read (for the manifest).

    Raises:
        FileNotFoundError: If the index or validity cache is missing.
        ValueError: If the validity mask is not aligned with the index.
    """
    index_path = paths.results / _RESULTS_ROLE / _PIXEL_INDEX_FILENAME
    _, valid_path = _cache_paths(paths, feature_set)
    for path in (index_path, valid_path):
        if not path.is_file():
            raise FileNotFoundError(
                f"Missing cache file {path}. Run scripts/build_pixel_table.py "
                f"--feature-set {feature_set} first."
            )

    pixel_index = pd.read_parquet(index_path)
    # Validate the pixel cache's lineage, adopting a pre-lineage cache in place
    # (writing its manifest) rather than re-extracting a matrix that is already
    # correct; a genuinely stale or absent cache still raises.
    from scripts.build_pixel_table import ensure_pixel_cache_current

    ensure_pixel_cache_current(
        paths,
        feature_set,
        pixel_index_path=index_path,
        pixel_ids=pixel_index["pixel_id"].to_numpy(dtype=np.int64),
        logger=logger,
    )
    valid = pd.read_parquet(valid_path)
    if not pixel_index["pixel_id"].equals(valid["pixel_id"]):
        raise ValueError("pixel_index and validity mask are not aligned by pixel_id.")
    pixel_index = pixel_index.assign(valid=valid["valid"].to_numpy())
    logger.info(
        "Loaded pixel index: %d pixels, %d valid (%.1f%%).",
        len(pixel_index),
        int(pixel_index["valid"].sum()),
        100.0 * pixel_index["valid"].mean(),
    )
    return pixel_index, [index_path, valid_path]


def _run_one(
    config: NestedCVConfig,
    pixel_index: pd.DataFrame,
    paths: ProjectPaths,
    *,
    outer_folds: Sequence[int],
    on_trial_end: Callable[[int, TrialRecord], None],
    on_fold_end: Callable[[int, FoldResult], None],
    fit_guard: Callable[[], AbstractContextManager[object]] | None,
    torch_threads: int,
    logger: logging.Logger,
) -> tuple[list[FoldResult], Path]:
    """Load the architecture's feature array and run its nested cross-validation.

    The single dispatch point between the two architectures. XGBoost reads the
    flat ``(n_pixels, n_bands)`` per-pixel memmap from ``build_pixel_table`` and
    runs :func:`utils.cv.run_nested_cv`; a CNN builds or reuses the
    ``(n_pixels, n_bands, P, P)`` patch cache via ``build_patch_cache`` and runs
    :func:`utils.cv.run_nested_cv_cnn`. ``torch``, ``run_nested_cv_cnn`` and the
    patch-cache builder are imported on the CNN branch alone, so an XGBoost run
    never imports torch.

    Args:
        config: The run configuration; ``config.architecture`` selects the branch.
        pixel_index: Pixel index with a ``valid`` column already merged in.
        paths: Resolved project paths.
        outer_folds: Outer folds to execute.
        on_trial_end: Per-trial callback forwarded to the runner.
        on_fold_end: Per-fold callback forwarded to the runner.
        fit_guard: Optional GPU-fit guard forwarded to the runner.
        torch_threads: PyTorch intra-op thread count (CNN branch only).
        logger: Run logger.

    Returns:
        The per-fold results and the feature-array path read (for the manifest).

    Raises:
        FileNotFoundError: If the XGBoost feature memmap cache is missing.
        ValueError: If a feature array shape is inconsistent with the feature set,
            or (CNN) the grid memmap or pixel index needed to build the patch
            cache is missing.
    """
    n_bands = len(MODEL_INPUT_BANDS[config.feature_set])
    if not _is_cnn(config.architecture):
        memmap_path, _ = _cache_paths(paths, config.feature_set)
        if not memmap_path.is_file():
            raise FileNotFoundError(
                f"Missing feature memmap {memmap_path}. Run scripts/build_pixel_table.py "
                f"--feature-set {config.feature_set} first."
            )
        memmap = np.load(memmap_path, mmap_mode="r")
        if memmap.shape != (len(pixel_index), n_bands):
            raise ValueError(
                f"feature memmap shape {memmap.shape} does not match "
                f"({len(pixel_index)}, {n_bands}) for {config.feature_set!r}."
            )
        logger.info("Loaded flat feature memmap %s with shape %s.", memmap_path, memmap.shape)
        results = run_nested_cv(
            config,
            pixel_index,
            memmap,
            outer_folds=outer_folds,
            on_trial_end=on_trial_end,
            on_fold_end=on_fold_end,
            fit_guard=fit_guard,
        )
        return results, memmap_path

    # CNN branch: torch, the CNN runner and the patch-cache builder are imported
    # here only, so importing this module and running XGBoost never pull torch in.
    import torch

    from scripts.build_patch_cache import ensure_patch_cache
    from utils.cv import run_nested_cv_cnn

    torch.set_num_threads(int(torch_threads))
    patch_size = _patch_size_from_architecture(config.architecture)
    # Memoise the patch extraction once (build-if-absent), then read patches from
    # the cache rather than re-slicing the grid memmap every epoch. The grid memmap
    # is now opened only inside ensure_patch_cache.
    cache_path = ensure_patch_cache(config.feature_set, patch_size, paths=paths, logger=logger)
    patch_cache = np.load(cache_path, mmap_mode="r")
    expected = (len(pixel_index), n_bands, patch_size, patch_size)
    if patch_cache.ndim != 4 or patch_cache.shape != expected:
        raise ValueError(
            f"patch cache shape {patch_cache.shape} is not {expected} for "
            f"{config.feature_set!r} at patch size {patch_size}."
        )
    manifest = json.loads(cache_path.with_suffix(".json").read_text(encoding="utf-8"))
    grid_shape = (int(manifest["grid_height"]), int(manifest["grid_width"]))
    logger.info(
        "Loaded patch cache %s with shape %s (grid %dx%d).",
        cache_path,
        patch_cache.shape,
        grid_shape[0],
        grid_shape[1],
    )
    results = run_nested_cv_cnn(
        config,
        pixel_index,
        patch_cache,
        patch_size,
        grid_shape=grid_shape,
        outer_folds=outer_folds,
        on_trial_end=on_trial_end,
        on_fold_end=on_fold_end,
        fit_guard=fit_guard,
    )
    return results, cache_path


def _trials_frame(records: Sequence[TrialRecord]) -> pd.DataFrame:
    """Build a validated-ready trials frame from accumulated trial records."""
    frame = pd.DataFrame([record.to_row() for record in records])
    return frame.reindex(columns=list(HP_TRIALS_SCHEMA))


def _training_recipe(
    config: NestedCVConfig, result: FoldResult, *, patch_size: int | None
) -> dict[str, object]:
    """Return the complete fitted-fold recipe needed for later reconstruction."""
    return {
        "schema_version": TRAINING_RECIPE_SCHEMA_VERSION,
        "architecture": config.architecture,
        "feature_set": config.feature_set,
        "feature_names": list(MODEL_INPUT_BANDS[config.feature_set]),
        "patch_size": patch_size,
        "sampling_mode": config.sampling_mode,
        "n_per_parcel": config.n_per_parcel,
        "n_hp_trials": config.n_hp_trials,
        "seed": config.seed,
        "device": config.device,
        "n_jobs": config.n_jobs,
        "fold_ids": list(config.fold_ids),
        "outer_fold": result.outer_fold,
        "training_folds": [fold for fold in config.fold_ids if fold != result.outer_fold],
        "selection_metric": FINAL_RECIPE_SELECTION_METRIC,
        "best_inner_pr_auc": result.best_inner_pr_auc,
        "best_params": dict(result.best_params),
        "cnn_epochs": result.cnn_epochs,
        "n_train": result.n_train,
    }


def _validate_training_recipe(
    path: Path,
    config: NestedCVConfig,
    *,
    outer_fold: int,
    patch_size: int | None,
) -> None:
    """Raise if a persisted fold recipe is incompatible with the requested run."""
    recipe = json.loads(path.read_text(encoding="utf-8"))
    expected: dict[str, object] = {
        "schema_version": TRAINING_RECIPE_SCHEMA_VERSION,
        "architecture": config.architecture,
        "feature_set": config.feature_set,
        "feature_names": list(MODEL_INPUT_BANDS[config.feature_set]),
        "patch_size": patch_size,
        "sampling_mode": config.sampling_mode,
        "n_per_parcel": config.n_per_parcel,
        "n_hp_trials": config.n_hp_trials,
        "seed": config.seed,
        "fold_ids": list(config.fold_ids),
        "outer_fold": outer_fold,
        "training_folds": [fold for fold in config.fold_ids if fold != outer_fold],
    }
    # device and n_jobs are recorded in the recipe for provenance but are not
    # validated on resume: they are performance knobs (the launcher derives n_jobs
    # from the concurrency level, and a run may legitimately resume on cpu or cuda)
    # and the results are deterministic with respect to them.
    mismatches = [
        key for key, expected_value in expected.items() if recipe.get(key) != expected_value
    ]
    if mismatches:
        raise ValueError(
            f"Training recipe {path} is incompatible with the requested run; "
            f"mismatched fields: {mismatches}."
        )

    incomplete: list[str] = []
    if recipe.get("selection_metric") != FINAL_RECIPE_SELECTION_METRIC:
        incomplete.append("selection_metric")
    score = recipe.get("best_inner_pr_auc")
    if (
        isinstance(score, bool)
        or not isinstance(score, int | float)
        or not np.isfinite(float(score))
    ):
        incomplete.append("best_inner_pr_auc")
    best_params = recipe.get("best_params")
    if not isinstance(best_params, dict) or not best_params:
        incomplete.append("best_params")
    n_train = recipe.get("n_train")
    if isinstance(n_train, bool) or not isinstance(n_train, int) or n_train < 1:
        incomplete.append("n_train")
    cnn_epochs = recipe.get("cnn_epochs")
    if patch_size is None:
        if cnn_epochs is not None:
            incomplete.append("cnn_epochs")
    elif isinstance(cnn_epochs, bool) or not isinstance(cnn_epochs, int) or cnn_epochs < 1:
        incomplete.append("cnn_epochs")
    if incomplete:
        raise ValueError(f"Training recipe {path} has incomplete provenance fields: {incomplete}.")


def _persist_fold(
    run_dir: Path,
    result: FoldResult,
    logger: logging.Logger,
    *,
    config: NestedCVConfig,
    patch_size: int | None,
    writes_importance: bool,
) -> None:
    """Write all per-fold outputs for a completed outer fold.

    Predictions and diagnostics are written before ``fold_status.json`` is
    updated, so an interrupted write leaves the fold marked incomplete and it is
    recomputed on resume.

    Args:
        run_dir: The run directory.
        result: The completed fold result.
        logger: Run logger.
        config: Validated nested-CV configuration used for the fold.
        patch_size: CNN patch size, or ``None`` for XGBoost.
        writes_importance: Whether to write the feature importance table; the CNN
            has none, so it is written for XGBoost only.
    """
    fold = result.outer_fold
    fold_paths = _FoldPaths.for_fold(run_dir, fold)
    write_json(dict(result.best_params), fold_paths.best_params)
    write_csv(result.boundary, fold_paths.boundary, HP_BOUNDARY_SCHEMA)
    if writes_importance:
        write_csv(result.importance, fold_paths.importance, FEATURE_IMPORTANCE_SCHEMA)
    write_parquet(result.pixel_predictions, fold_paths.pixel_predictions, PIXEL_PREDICTIONS_SCHEMA)
    write_parquet(
        result.parcel_predictions, fold_paths.parcel_predictions, PARCEL_PREDICTIONS_SCHEMA
    )
    write_json(
        _training_recipe(config, result, patch_size=patch_size),
        fold_paths.training_recipe,
    )
    _update_fold_status(run_dir, result)
    logger.info("Persisted outputs for outer fold %d.", fold)


def _update_fold_status(run_dir: Path, result: FoldResult) -> None:
    """Record one fold's status and headline metrics in ``fold_status.json``."""
    status_path = run_dir / "fold_status.json"
    status: dict[str, object] = {}
    if status_path.is_file():
        status = json.loads(status_path.read_text(encoding="utf-8"))
    status[str(result.outer_fold)] = {
        "complete": True,
        "seconds": result.seconds,
        "n_train": result.n_train,
        "n_val": result.n_val,
        "pixel_pr_auc": result.pixel_metrics["pr_auc"],
        "parcel_pr_auc": result.parcel_metrics["pr_auc"],
    }
    write_json(status, status_path)


def _inner_thresholds_from_trials(hp_trials_path: Path) -> tuple[float, float]:
    """Recover the selected trial's inner thresholds from the persisted trials CSV.

    The selected trial is the one with the highest objective (mean parcel PR-AUC),
    matching ``study.best_trial``. Missing values fall back to 0.5.

    Args:
        hp_trials_path: Path to ``hp_trials_fold{k}.csv``.

    Returns:
        The ``(inner_pixel_threshold, inner_parcel_threshold)`` pair.
    """
    # round_trip parsing recovers the exact doubles written, so a resumed fold's
    # inner-threshold metrics are byte-identical to the original run.
    trials = pd.read_csv(hp_trials_path, float_precision="round_trip")
    scored = trials[trials["pr_auc"].notna()]
    if scored.empty:
        return _FALLBACK_THRESHOLD, _FALLBACK_THRESHOLD
    best = scored.loc[scored["pr_auc"].idxmax()]
    pixel = best["inner_pixel_threshold"]
    parcel = best["inner_parcel_threshold"]
    return (
        float(pixel) if pd.notna(pixel) else _FALLBACK_THRESHOLD,
        float(parcel) if pd.notna(parcel) else _FALLBACK_THRESHOLD,
    )


def _load_completed_fold(run_dir: Path, fold: int) -> FoldResult:
    """Rebuild a :class:`FoldResult` from a previously completed fold's files."""
    fold_paths = _FoldPaths.for_fold(run_dir, fold)
    pixel_predictions = pd.read_parquet(fold_paths.pixel_predictions)
    parcel_predictions = pd.read_parquet(fold_paths.parcel_predictions)
    best_params = json.loads(fold_paths.best_params.read_text(encoding="utf-8"))
    recipe = json.loads(fold_paths.training_recipe.read_text(encoding="utf-8"))
    if recipe.get("best_params") != best_params:
        raise ValueError(f"Training recipe and best-parameter file disagree for outer fold {fold}.")
    boundary = pd.read_csv(fold_paths.boundary)
    importance = (
        pd.read_csv(fold_paths.importance) if fold_paths.importance.is_file() else pd.DataFrame()
    )
    status_path = run_dir / "fold_status.json"
    status = (
        json.loads(status_path.read_text(encoding="utf-8")).get(str(fold), {})
        if status_path.is_file()
        else {}
    )
    inner_pixel_threshold, inner_parcel_threshold = _inner_thresholds_from_trials(
        fold_paths.hp_trials
    )
    return FoldResult.from_predictions(
        fold,
        pixel_predictions,
        parcel_predictions,
        best_params=best_params,
        boundary=boundary,
        importance=importance,
        inner_pixel_threshold=inner_pixel_threshold,
        inner_parcel_threshold=inner_parcel_threshold,
        best_inner_pr_auc=float(recipe.get("best_inner_pr_auc", float("nan"))),
        cnn_epochs=(int(recipe["cnn_epochs"]) if recipe.get("cnn_epochs") is not None else None),
        n_train=int(status.get("n_train", recipe.get("n_train", 0))),
        seconds=float(status.get("seconds", 0.0)),
    )


def _write_pooled_tables(
    run_dir: Path, aggregated: AggregatedMetrics, logger: logging.Logger
) -> dict[str, str]:
    """Write the per-fold, summary and pooled metric tables.

    Returns:
        A mapping of table name to its repo-relative path string.
    """
    paths_written = {
        "per_fold_metrics": write_csv(
            aggregated.per_fold, run_dir / "per_fold_metrics.csv", PER_FOLD_METRICS_SCHEMA
        ),
        "summary_metrics": write_csv(
            aggregated.summary, run_dir / "summary_metrics.csv", SUMMARY_METRICS_SCHEMA
        ),
        "pooled_pixel_metrics": write_csv(
            aggregated.pooled_pixel, run_dir / "pooled_pixel_metrics.csv", POOLED_METRICS_SCHEMA
        ),
        "pooled_parcel_metrics": write_csv(
            aggregated.pooled_parcel, run_dir / "pooled_parcel_metrics.csv", POOLED_METRICS_SCHEMA
        ),
    }
    logger.info("Wrote pooled metric tables to %s.", run_dir)
    return {name: str(path) for name, path in paths_written.items()}


def _library_versions(architecture: str) -> dict[str, str]:
    """Return versions of the key libraries for the run metadata.

    Records the modelling stack actually used: ``torch`` for a CNN run and
    ``xgboost`` (with scikit-learn and joblib) for an XGBoost run, alongside the
    libraries common to both.
    """
    libraries: tuple[str, ...]
    if _is_cnn(architecture):
        libraries = ("numpy", "pandas", "torch", "optuna")
    else:
        libraries = (
            "numpy",
            "pandas",
            "xgboost",
            "optuna",
            "scikit-learn",
            "joblib",
            "filelock",
        )
    versions: dict[str, str] = {}
    for library in libraries:
        try:
            versions[library] = version(library)
        except PackageNotFoundError:
            versions[library] = "unknown"
    return versions


def _parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    """Parse the command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--feature-set", required=True, choices=list(MODEL_INPUTS))
    parser.add_argument(
        "--architecture",
        default=_DEFAULT_ARCHITECTURE,
        choices=list(ARCHITECTURES),
        help="Model architecture; the CNN patch size is encoded in the cnn_* name.",
    )
    parser.add_argument(
        "--patch-size",
        type=int,
        default=None,
        choices=list(CNN_PATCH_SIZES),
        help="Receptive-field patch side; required for a CNN architecture, invalid for XGBoost.",
    )
    parser.add_argument("--sampling-mode", default="none", choices=list(SAMPLING_MODES))
    parser.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"])
    parser.add_argument(
        "--concurrency",
        type=int,
        default=len(FEATURE_SETS),
        help="Expected concurrent feature-set processes; sets the default --n-jobs.",
    )
    parser.add_argument(
        "--n-jobs",
        type=int,
        default=None,
        help=(
            "Per-fit CPU budget. XGBoost: threads per fit "
            f"(default {_CORE_BUDGET} // concurrency). CNN: patch-loader worker count, "
            f"loaders use n_jobs - 1 (default min({_CNN_MAX_N_JOBS}, cores))."
        ),
    )
    parser.add_argument(
        "--torch-threads",
        type=int,
        default=min(_CNN_TORCH_THREADS, _CORE_BUDGET),
        help=(
            f"PyTorch intra-op threads for CNN fits (default min({_CNN_TORCH_THREADS}, cores)); "
            "ignored for XGBoost."
        ),
    )
    parser.add_argument(
        "--outer-folds",
        type=int,
        nargs="+",
        default=None,
        choices=list(FOLD_IDS),
        help="Outer folds to run; default all folds.",
    )
    parser.add_argument(
        "--n-hp-trials",
        type=int,
        default=None,
        help=(
            f"Optuna trials per outer fold; default {N_HP_TRIALS_XGBOOST} for xgboost and "
            f"{N_HP_TRIALS} for the CNNs, as published."
        ),
    )
    parser.add_argument("--n-per-parcel", type=int, default=N_PER_PARCEL)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument(
        "--gpu-slots",
        type=int,
        default=None,
        help="Concurrent GPU fits permitted; default = --concurrency. Ignored on CPU.",
    )
    parser.add_argument(
        "--resume",
        default=None,
        help="Existing run id under results/main_nested_cv to resume.",
    )
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Tiny run: one outer fold, 2 trials, 20 pixels per parcel.",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    """Run the nested cross-validation for one feature set.

    Args:
        argv: Argument list, or ``None`` to read from ``sys.argv``.

    Returns:
        Process exit code (0 on success).
    """
    args = _parse_args(argv)
    architecture = args.architecture
    patch_size = _validate_patch_size(architecture, args.patch_size)
    # Only XGBoost has a feature-importance table to write or wait for.
    writes_importance = not _is_cnn(architecture)

    paths = get_project_paths()
    ensure_writable_dirs(paths)

    if args.n_jobs is not None:
        n_jobs = args.n_jobs
    elif _is_cnn(architecture):
        n_jobs = min(_CNN_MAX_N_JOBS, _CORE_BUDGET)
    else:
        n_jobs = max(1, _CORE_BUDGET // args.concurrency)
    if args.smoke:
        n_hp_trials = _SMOKE_N_HP_TRIALS
    elif args.n_hp_trials is not None:
        n_hp_trials = args.n_hp_trials
    else:
        n_hp_trials = _default_n_hp_trials(architecture)
    n_per_parcel = _SMOKE_N_PER_PARCEL if args.smoke else args.n_per_parcel
    requested_outer = (
        [FOLD_IDS[0]] if args.smoke else (args.outer_folds if args.outer_folds else list(FOLD_IDS))
    )

    results_root = paths.results / _RESULTS_ROLE
    if args.resume is not None:
        run_id = args.resume
        run_dir = results_root / run_id
        if not run_dir.is_dir():
            raise FileNotFoundError(f"--resume run directory does not exist: {run_dir}")
    else:
        run_id = f"{datetime.now(UTC):%Y%m%d_%H%M%S}__{architecture}__{args.feature_set}"
        run_dir = results_root / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    logger = make_run_logger("run_nested_cv", run_dir)

    # Route utils.cv's per-trial and per-fold logging through this run's handlers
    # so the detail reaches both stdout and the run log file.
    cv_logger = logging.getLogger("utils.cv")
    cv_logger.setLevel(logging.INFO)
    cv_logger.handlers = list(logger.handlers)
    cv_logger.propagate = False

    device = _detect_device(args.device, logger)
    logger.info(
        "Run %s: architecture=%s, patch_size=%s, feature_set=%s, sampling=%s, device=%s, "
        "n_jobs=%d, trials=%d, n_per_parcel=%d, outer_folds=%s.",
        run_id,
        architecture,
        patch_size,
        args.feature_set,
        args.sampling_mode,
        device,
        n_jobs,
        n_hp_trials,
        n_per_parcel,
        requested_outer,
    )

    config = NestedCVConfig(
        feature_set=args.feature_set,
        architecture=architecture,
        sampling_mode=args.sampling_mode,
        n_per_parcel=n_per_parcel,
        n_hp_trials=n_hp_trials,
        seed=args.seed,
        device=device,
        n_jobs=n_jobs,
        fold_ids=FOLD_IDS,
    )

    pixel_index, inputs = _load_pixel_index(paths, args.feature_set, logger)

    # Resume: skip outer folds whose validated outputs are already present.
    done = [
        fold
        for fold in requested_outer
        if _FoldPaths.for_fold(run_dir, fold).fold_complete(writes_importance=writes_importance)
    ]
    for fold in done:
        _validate_training_recipe(
            _FoldPaths.for_fold(run_dir, fold).training_recipe,
            config,
            outer_fold=fold,
            patch_size=patch_size,
        )
    todo = [fold for fold in requested_outer if fold not in done]
    if done:
        logger.info("Resuming: folds %s already complete; running %s.", done, todo)

    gpu_slots = args.gpu_slots if args.gpu_slots is not None else args.concurrency
    fit_guard = None
    if device == "cuda" and gpu_slots > 0:
        semaphore = _GpuSemaphore(paths.cache, gpu_slots)
        fit_guard = semaphore.slot
        logger.info("GPU semaphore active with %d slot(s).", gpu_slots)

    # Per-fold trial accumulators, rewritten on every trial for crash safety.
    trial_records: dict[int, list[TrialRecord]] = {fold: [] for fold in todo}

    def on_trial_end(fold: int, record: TrialRecord) -> None:
        trial_records[fold].append(record)
        write_csv(
            _trials_frame(trial_records[fold]),
            _FoldPaths.for_fold(run_dir, fold).hp_trials,
            HP_TRIALS_SCHEMA,
        )

    def on_fold_end(fold: int, result: FoldResult) -> None:
        _persist_fold(
            run_dir,
            result,
            logger,
            config=config,
            patch_size=patch_size,
            writes_importance=writes_importance,
        )

    started = time.perf_counter()
    new_results, feature_path = _run_one(
        config,
        pixel_index,
        paths,
        outer_folds=todo,
        on_trial_end=on_trial_end,
        on_fold_end=on_fold_end,
        fit_guard=fit_guard,
        torch_threads=args.torch_threads,
        logger=logger,
    )
    elapsed = time.perf_counter() - started
    inputs.append(feature_path)

    # Assemble every requested fold (freshly run plus resumed) and write pooled tables.
    fold_results = list(new_results) + [_load_completed_fold(run_dir, fold) for fold in done]
    aggregated = aggregate_pooled_metrics(fold_results)
    output_paths = _write_pooled_tables(run_dir, aggregated, logger)

    # Reproducibility floor.
    write_input_manifest(inputs, run_dir / "input_manifest.json", repo_root=paths.repo_root)
    write_environment_snapshot(run_dir / "env_snapshot.txt")
    run_metadata = {
        "run_id": run_id,
        "created_at": datetime.now(UTC).isoformat(),
        "script": "scripts/run_nested_cv.py",
        "architecture": architecture,
        "patch_size": patch_size,
        "torch_threads": args.torch_threads if _is_cnn(architecture) else None,
        "device": device,
        "n_jobs": n_jobs,
        "gpu_slots": gpu_slots,
        "concurrency": args.concurrency,
        "smoke": args.smoke,
        "requested_outer_folds": requested_outer,
        "config": config_to_dict(config),
        "wall_clock_seconds": elapsed,
        "folds": {
            str(result.outer_fold): {
                "best_params": result.best_params,
                "best_inner_pr_auc": result.best_inner_pr_auc,
                "cnn_epochs": result.cnn_epochs,
                "training_recipe": str(
                    _FoldPaths.for_fold(run_dir, result.outer_fold).training_recipe.relative_to(
                        paths.repo_root
                    )
                ),
                "pixel_metrics": result.pixel_metrics,
                "parcel_metrics": result.parcel_metrics,
                "pixel_metrics_inner": result.pixel_metrics_inner,
                "parcel_metrics_inner": result.parcel_metrics_inner,
                "seconds": result.seconds,
                "n_train": result.n_train,
                "n_val": result.n_val,
            }
            for result in sorted(fold_results, key=lambda r: r.outer_fold)
        },
        "pooled_pixel_metrics": dict(
            zip(aggregated.pooled_pixel["metric"], aggregated.pooled_pixel["value"], strict=True)
        ),
        "pooled_parcel_metrics": dict(
            zip(aggregated.pooled_parcel["metric"], aggregated.pooled_parcel["value"], strict=True)
        ),
        "library_versions": _library_versions(architecture),
        "outputs": output_paths,
    }
    write_json(run_metadata, run_dir / "run_metadata.json")
    logger.info("Nested cross-validation complete for %s in %.1fs.", run_id, elapsed)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
