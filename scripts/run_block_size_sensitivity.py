"""Block-size sensitivity of the headline parcel PR-AUC bootstrap interval.

The headline performance interval (``scripts/run_performance_matrix.py``) resamples
the labelled parcels in a single area-balanced block set of
:data:`utils.terminology.BOOTSTRAP_N_UNITS` blocks. This script checks that the
interval is stable to that choice: it recomputes the 95 per cent percentile
interval of the pooled parcel PR-AUC at several block counts for the selected
model only (XGBoost + ``baseline_tessera``).

No model is retrained; the saved out-of-fold parcel predictions are reused and
only the block assignment changes. Three kinds of block set are used:

  - ``6`` blocks are the six spatial cross-validation folds themselves
    (``fold_id``), the coarsest spatially coherent unit;
  - ``20`` blocks reuse the saved ``bootstrap_id`` partition (notebook 008), the
    exact set behind the headline interval;
  - every other count is a fresh area-balanced partition from
    :func:`utils.folds.assign_spatial_partition`.

The observed point estimate is identical across block counts (blocks only affect
the resampling); the interval is what moves. The output feeds
``notebooks/015_manuscript_tables.ipynb``.
"""

import argparse
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
from sklearn.metrics import average_precision_score

from utils.bootstrap import block_bootstrap_distribution, percentile_interval
from utils.env_capture import write_environment_snapshot
from utils.folds import assign_spatial_partition
from utils.io import ColumnSpec, Schema, write_csv, write_json
from utils.logging import make_run_logger
from utils.manifest import write_input_manifest
from utils.metrics import parcel_metrics
from utils.paths import ProjectPaths, ensure_writable_dirs, get_project_paths
from utils.terminology import (
    BOOTSTRAP_N_UNITS,
    BOOTSTRAP_REPS_CROSS_CONFIG,
    FEATURE_SETS,
    FOLD_IDS,
    SEED,
)

logger = logging.getLogger(__name__)

_LOGGER_NAME: Final[str] = "run_block_size_sensitivity"

_SOURCE_ROLE: Final[str] = "main_nested_cv"
_RESULTS_ROLE: Final[str] = "block_size_sensitivity"
_ARCHITECTURE: Final[str] = "xgboost"
_DEFAULT_FEATURE_SET: Final[str] = "baseline_tessera"

_LABELS_FILENAME: Final[str] = "ogf_reference_labels_partitioned.gpkg"
_PARCEL_PRED_TEMPLATE: Final[str] = "parcel_predictions_fold{fold}.parquet"
_POOLED_PARCEL_FILE: Final[str] = "pooled_parcel_metrics.csv"

# Block counts to sweep. 6 = the folds; 20 = the saved headline partition; the
# rest are fresh area-balanced partitions.
_DEFAULT_BLOCK_COUNTS: Final[tuple[int, ...]] = (6, 10, 15, 20, 25, 30)
_N_FOLD_BLOCKS: Final[int] = len(FOLD_IDS)

_PR_AUC_TOLERANCE: Final[float] = 1e-10
_CI_LEVEL: Final[float] = 0.95
_N_JOBS_CAP: Final[int] = 32

_FLOAT = ColumnSpec("float64")
_STR = ColumnSpec("object")
_INT = ColumnSpec("int64")

BLOCK_SIZE_SENSITIVITY_SCHEMA: Final[Schema] = {
    "n_blocks": _INT,
    "block_source": _STR,
    "n_distinct_blocks": _INT,
    "pr_auc": _FLOAT,
    "boot_mean": _FLOAT,
    "ci_lo": _FLOAT,
    "ci_hi": _FLOAT,
    "ci_width": _FLOAT,
}


def _is_complete_run(run_dir: Path) -> bool:
    """Return whether ``run_dir`` carries every parcel parquet and the pooled CSV."""
    if not (run_dir / _POOLED_PARCEL_FILE).is_file():
        return False
    return all((run_dir / _PARCEL_PRED_TEMPLATE.format(fold=fold)).is_file() for fold in FOLD_IDS)


def _latest_run(source_root: Path, feature_set: str) -> Path:
    """Return the most recent complete nested-CV run for one feature set."""
    suffix = f"__{_ARCHITECTURE}__{feature_set}"
    candidates = sorted(
        (p for p in source_root.iterdir() if p.is_dir() and p.name.endswith(suffix)),
        key=lambda p: p.name,
        reverse=True,
    )
    for candidate in candidates:
        if _is_complete_run(candidate):
            return candidate
    raise FileNotFoundError(
        f"No complete nested-CV run for {feature_set!r} under {source_root}; expected "
        "parcel_predictions_fold{1..6}.parquet plus pooled_parcel_metrics.csv."
    )


def _load_parcel_predictions(run_dir: Path) -> pd.DataFrame:
    """Concatenate the per-fold out-of-fold parcel predictions for one run."""
    frames = [
        pd.read_parquet(run_dir / _PARCEL_PRED_TEMPLATE.format(fold=fold)) for fold in FOLD_IDS
    ]
    combined = pd.concat(frames, ignore_index=True)
    combined = combined.drop_duplicates(subset=["parcel_id"]).reset_index(drop=True)
    combined["parcel_id"] = combined["parcel_id"].astype(np.int64)
    return combined.sort_values("parcel_id").reset_index(drop=True)


def _load_labelled_parcels(labels_path: Path) -> gpd.GeoDataFrame:
    """Load labelled parcels with geometry, fold id, saved block id and label."""
    frame = gpd.read_file(labels_path)
    labelled = frame[frame["ogf"].notna()].copy()
    labelled["parcel_id"] = labelled["parcel_id"].astype(np.int64)
    labelled["fold_id"] = labelled["fold_id"].astype(np.int64)
    labelled["bootstrap_id"] = labelled["bootstrap_id"].astype(np.int64)
    labelled["ogf"] = labelled["ogf"].astype(np.int64)
    return labelled.sort_values("parcel_id").reset_index(drop=True)


def _align(predictions: pd.DataFrame, labelled: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """Inner-join predictions onto labelled parcels, keeping geometry for partitioning.

    Raises:
        ValueError: If the prediction labels disagree with the labels file.
    """
    merged = labelled.merge(
        predictions[["parcel_id", "y_true", "p_mean"]], on="parcel_id", how="inner"
    )
    merged = merged.sort_values("parcel_id").reset_index(drop=True)
    if not np.array_equal(
        merged["y_true"].to_numpy().astype(np.int64), merged["ogf"].to_numpy().astype(np.int64)
    ):
        raise ValueError(
            "Prediction y_true disagrees with the labels file; the run used a different "
            "label table than ogf_reference_labels_partitioned.gpkg."
        )
    return merged


def _block_ids_for_count(aligned: gpd.GeoDataFrame, n_blocks: int) -> tuple[np.ndarray, str]:
    """Return the block id per parcel and a label for how it was derived."""
    if n_blocks == _N_FOLD_BLOCKS:
        return aligned["fold_id"].to_numpy().astype(np.int64), "folds"
    if n_blocks == BOOTSTRAP_N_UNITS:
        return aligned["bootstrap_id"].to_numpy().astype(np.int64), "saved_partition"
    units = assign_spatial_partition(aligned, n_blocks, group_connected=False, seed=SEED)
    return units.to_numpy().astype(np.int64), "spatial_partition"


def _verify_observed(y_true: np.ndarray, scores: np.ndarray, run_dir: Path) -> float:
    """Recompute pooled parcel PR-AUC and assert agreement with the run's CSV."""
    pr_auc = float(parcel_metrics(y_true, scores)["pr_auc"])
    saved = pd.read_csv(run_dir / _POOLED_PARCEL_FILE)
    saved_pr_auc = float(saved.loc[saved["metric"] == "pr_auc", "value"].iloc[0])
    if abs(pr_auc - saved_pr_auc) > _PR_AUC_TOLERANCE:
        raise ValueError(
            f"Recomputed pooled PR-AUC ({pr_auc:.6f}) disagrees with the run CSV "
            f"({saved_pr_auc:.6f})."
        )
    return pr_auc


def block_size_sensitivity(
    aligned: gpd.GeoDataFrame,
    *,
    block_counts: Sequence[int],
    observed_pr_auc: float,
    n_reps: int,
    seed: int,
    n_jobs: int,
    run_logger: logging.Logger,
) -> pd.DataFrame:
    """Compute the bootstrap PR-AUC interval at each block count."""
    y_true = aligned["y_true"].to_numpy().astype(np.int64)
    scores = aligned["p_mean"].to_numpy().astype(np.float64).reshape(-1, 1)

    rows: list[dict[str, object]] = []
    for n_blocks in block_counts:
        block_ids, block_source = _block_ids_for_count(aligned, n_blocks)
        n_distinct = int(np.unique(block_ids).size)
        distribution = block_bootstrap_distribution(
            y_true,
            scores,
            block_ids,
            average_precision_score,
            n_reps=n_reps,
            seed=seed,
            n_jobs=n_jobs,
            min_classes=2,
        )
        samples = distribution[:, 0]
        lo, hi = percentile_interval(samples, level=_CI_LEVEL)
        rows.append(
            {
                "n_blocks": int(n_blocks),
                "block_source": block_source,
                "n_distinct_blocks": n_distinct,
                "pr_auc": float(observed_pr_auc),
                "boot_mean": float(np.nanmean(samples)),
                "ci_lo": lo,
                "ci_hi": hi,
                "ci_width": float(hi - lo),
            }
        )
        run_logger.info(
            "Block count %d (%s, %d distinct): PR-AUC %.4f, 95%% CI [%.4f, %.4f], width %.4f.",
            n_blocks,
            block_source,
            n_distinct,
            observed_pr_auc,
            lo,
            hi,
            hi - lo,
        )
    return pd.DataFrame(rows, columns=list(BLOCK_SIZE_SENSITIVITY_SCHEMA))


def _library_versions() -> dict[str, str]:
    """Return versions of the key libraries for the run metadata."""
    versions: dict[str, str] = {}
    for library in ("numpy", "pandas", "scikit-learn", "geopandas", "joblib"):
        try:
            versions[library] = version(library)
        except PackageNotFoundError:
            versions[library] = "unknown"
    return versions


def _parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    """Parse the command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--feature-set", default=_DEFAULT_FEATURE_SET, choices=list(FEATURE_SETS))
    parser.add_argument("--source-role", default=_SOURCE_ROLE)
    parser.add_argument(
        "--block-counts",
        type=int,
        nargs="+",
        default=list(_DEFAULT_BLOCK_COUNTS),
        help="Block counts to sweep (default 6 10 15 20 25 30).",
    )
    parser.add_argument("--n-reps", type=int, default=BOOTSTRAP_REPS_CROSS_CONFIG)
    parser.add_argument("--seed", type=int, default=SEED)
    return parser.parse_args(argv)


def _run(args: argparse.Namespace, paths: ProjectPaths, run_dir: Path) -> tuple[Path, dict]:
    """Resolve inputs, run the sweep, write the table; return (table path, summary)."""
    run_logger = make_run_logger(_LOGGER_NAME, run_dir)
    source_run = _latest_run(paths.results / args.source_role, args.feature_set)
    run_logger.info(
        "Feature set %s: source run %s.",
        args.feature_set,
        source_run.relative_to(paths.repo_root),
    )

    labels_path = paths.labels / _LABELS_FILENAME
    predictions = _load_parcel_predictions(source_run)
    labelled = _load_labelled_parcels(labels_path)
    aligned = _align(predictions, labelled)
    run_logger.info("Aligned %d labelled parcels with predictions.", len(aligned))

    observed = _verify_observed(
        aligned["y_true"].to_numpy().astype(np.int64),
        aligned["p_mean"].to_numpy().astype(np.float64),
        source_run,
    )
    run_logger.info("Observed pooled parcel PR-AUC: %.6f.", observed)

    n_jobs = min(_N_JOBS_CAP, os.cpu_count() or 1)
    table = block_size_sensitivity(
        aligned,
        block_counts=args.block_counts,
        observed_pr_auc=observed,
        n_reps=args.n_reps,
        seed=args.seed,
        n_jobs=n_jobs,
        run_logger=run_logger,
    )
    table_path = write_csv(
        table, run_dir / "block_size_sensitivity.csv", BLOCK_SIZE_SENSITIVITY_SCHEMA
    )

    inputs = [labels_path, source_run / _POOLED_PARCEL_FILE] + [
        source_run / _PARCEL_PRED_TEMPLATE.format(fold=fold) for fold in FOLD_IDS
    ]
    write_input_manifest(inputs, run_dir / "input_manifest.json", repo_root=paths.repo_root)
    write_environment_snapshot(run_dir / "env_snapshot.txt")
    summary = {
        "run_id": datetime.now(UTC).strftime("%Y%m%d_%H%M%S"),
        "created_at": datetime.now(UTC).isoformat(),
        "script": "scripts/run_block_size_sensitivity.py",
        "feature_set": args.feature_set,
        "architecture": _ARCHITECTURE,
        "source_run": str(source_run.relative_to(paths.repo_root)),
        "block_counts": list(args.block_counts),
        "n_reps": args.n_reps,
        "seed": args.seed,
        "n_jobs": n_jobs,
        "ci_level": _CI_LEVEL,
        "observed_pr_auc": observed,
        "metric": "sklearn.metrics.average_precision_score (parcel-level)",
        "library_versions": _library_versions(),
        "outputs": {"block_size_sensitivity": str(table_path.relative_to(paths.repo_root))},
    }
    return table_path, summary


def main(argv: Sequence[str] | None = None) -> int:
    """Run the block-size sensitivity sweep and write its artefacts.

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

    _table_path, summary = _run(args, paths, run_dir)
    write_json(summary, run_dir / "run_metadata.json")
    logger.info("Block-size sensitivity complete.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
