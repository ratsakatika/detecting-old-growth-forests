"""Sampling-mode sensitivity for the selected model (XGBoost + baseline_tessera).

The headline run trains with no class or forest-type reweighting
(``sampling_mode="none"``). This script measures how much that choice matters by
re-evaluating the same selected model under each mode in
:data:`utils.sampling.SAMPLING_MODES` (``none``, ``inverse_prevalence``,
``stratified_forest_type``).

To isolate the sampling factor, the per-fold hyperparameters found by the
headline nested cross-validation are reused (``best_params_fold{k}.json``) and no
new Optuna search runs: for each outer fold the model is refit with those fixed
hyperparameters under the new sampling mode and used to predict the held-out
fold (:func:`utils.cv.fixed_hp_fold_result`). Switching mode changes only the
training sample weights, not which pixels are sampled, so ``mode="none"``
reproduces the headline fold metrics exactly (a built-in sanity check). Each
refit is a few seconds rather than a full nested-CV fold.

The model runs on GPU (CUDA) by default, much faster for these fits; pass
``--device cpu`` to stay off a GPU that is busy with other jobs. The output feeds
``notebooks/015_manuscript_tables.ipynb``.
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

import numpy as np
import pandas as pd

from utils.cv import NestedCVConfig, aggregate_pooled_metrics, fixed_hp_fold_result
from utils.env_capture import write_environment_snapshot
from utils.io import ColumnSpec, Schema, write_csv, write_json
from utils.logging import make_run_logger
from utils.manifest import write_input_manifest
from utils.paths import ProjectPaths, ensure_writable_dirs, get_project_paths
from utils.sampling import SAMPLING_MODES
from utils.terminology import (
    FEATURE_SET_BANDS,
    FEATURE_SETS,
    FOLD_IDS,
    N_HP_TRIALS,
    N_PER_PARCEL,
    SEED,
)

logger = logging.getLogger(__name__)

_LOGGER_NAME: Final[str] = "run_sampling_sensitivity"

_SOURCE_ROLE: Final[str] = "main_nested_cv"
_RESULTS_ROLE: Final[str] = "sampling_sensitivity"
_ARCHITECTURE: Final[str] = "xgboost"
_DEFAULT_FEATURE_SET: Final[str] = "baseline_tessera"

_PIXEL_INDEX_FILENAME: Final[str] = "pixel_index.parquet"
_CACHE_SUBDIR: Final[Path] = Path("pixel_features")
_BEST_PARAMS_TEMPLATE: Final[str] = "best_params_fold{fold}.json"
_PARCEL_PRED_TEMPLATE: Final[str] = "parcel_predictions_fold{fold}.parquet"
_POOLED_PARCEL_FILE: Final[str] = "pooled_parcel_metrics.csv"

_DEFAULT_N_JOBS: Final[int] = min(8, os.cpu_count() or 1)

_FLOAT = ColumnSpec("float64")
_NULLABLE_FLOAT = ColumnSpec("float64", nullable=True)
_STR = ColumnSpec("object")

# Tidy summary: across-fold mean[min-max] and the pooled value, per mode, level,
# metric. Reused (with a different first column) by the gap analysis.
SAMPLING_SENSITIVITY_SCHEMA: Final[Schema] = {
    "sampling_mode": _STR,
    "level": _STR,
    "metric": _STR,
    "mean": _FLOAT,
    "sd": _NULLABLE_FLOAT,
    "min": _FLOAT,
    "max": _FLOAT,
    "pooled": _NULLABLE_FLOAT,
}


def _is_complete_run(run_dir: Path) -> bool:
    """Return whether ``run_dir`` carries every parcel parquet, pooled CSV and best params."""
    if not (run_dir / _POOLED_PARCEL_FILE).is_file():
        return False
    return all(
        (run_dir / _PARCEL_PRED_TEMPLATE.format(fold=fold)).is_file()
        and (run_dir / _BEST_PARAMS_TEMPLATE.format(fold=fold)).is_file()
        for fold in FOLD_IDS
    )


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
        "best_params_fold{1..6}.json plus parcel_predictions and pooled_parcel_metrics.csv."
    )


def _load_best_params(run_dir: Path) -> dict[int, dict[str, float | int]]:
    """Load the per-fold best hyperparameters from the headline run."""
    return {
        fold: json.loads((run_dir / _BEST_PARAMS_TEMPLATE.format(fold=fold)).read_text("utf-8"))
        for fold in FOLD_IDS
    }


def _load_pixel_data(
    paths: ProjectPaths, feature_set: str
) -> tuple[pd.DataFrame, np.ndarray, list[Path]]:
    """Load the pixel index (with validity merged) and the flat feature memmap."""
    index_path = paths.results / _SOURCE_ROLE / _PIXEL_INDEX_FILENAME
    cache_dir = paths.cache / _CACHE_SUBDIR
    memmap_path = cache_dir / f"{feature_set}.npy"
    valid_path = cache_dir / f"{feature_set}_valid.parquet"
    for path in (index_path, memmap_path, valid_path):
        if not path.is_file():
            raise FileNotFoundError(
                f"Missing input {path}. Run scripts/build_pixel_table.py "
                f"--feature-set {feature_set} first."
            )
    pixel_index = pd.read_parquet(index_path)
    valid = pd.read_parquet(valid_path)
    if not pixel_index["pixel_id"].equals(valid["pixel_id"]):
        raise ValueError("pixel_index and validity mask are not aligned by pixel_id.")
    pixel_index = pixel_index.assign(valid=valid["valid"].to_numpy())
    n_bands = len(FEATURE_SET_BANDS[feature_set])
    memmap = np.load(memmap_path, mmap_mode="r")
    if memmap.shape != (len(pixel_index), n_bands):
        raise ValueError(
            f"feature memmap shape {memmap.shape} does not match "
            f"({len(pixel_index)}, {n_bands}) for {feature_set!r}."
        )
    return pixel_index, memmap, [index_path, valid_path, memmap_path]


def tidy_summary(label: str, label_value: str, aggregated: object) -> pd.DataFrame:
    """Join across-fold summary and pooled metrics into one tidy frame.

    Args:
        label: Name of the condition column (e.g. ``"sampling_mode"``).
        label_value: Its value for this condition.
        aggregated: An :class:`utils.cv.AggregatedMetrics`.

    Returns:
        Columns ``<label>, level, metric, mean, sd, min, max, pooled``.
    """
    summary = aggregated.summary.copy()  # type: ignore[attr-defined]
    pooled = pd.concat(
        [
            aggregated.pooled_pixel.assign(level="pixel"),  # type: ignore[attr-defined]
            aggregated.pooled_parcel.assign(level="parcel"),  # type: ignore[attr-defined]
        ],
        ignore_index=True,
    ).rename(columns={"value": "pooled"})
    merged = summary.merge(pooled, on=["level", "metric"], how="left")
    merged.insert(0, label, label_value)
    return merged


def run_sampling_sensitivity(
    pixel_index: pd.DataFrame,
    memmap: np.ndarray,
    best_params: dict[int, dict[str, float | int]],
    *,
    feature_set: str,
    sampling_modes: Sequence[str],
    device: str,
    n_jobs: int,
    seed: int,
    n_per_parcel: int,
    run_logger: logging.Logger,
) -> pd.DataFrame:
    """Refit every fold under each sampling mode and return the tidy summary table."""
    frames: list[pd.DataFrame] = []
    for mode in sampling_modes:
        config = NestedCVConfig(
            feature_set=feature_set,
            architecture=_ARCHITECTURE,
            sampling_mode=mode,
            n_per_parcel=n_per_parcel,
            n_hp_trials=N_HP_TRIALS,
            seed=seed,
            device=device,
            n_jobs=n_jobs,
            fold_ids=FOLD_IDS,
        )
        fold_results = [
            fixed_hp_fold_result(config, pixel_index, memmap, best_params[fold], outer_fold=fold)
            for fold in FOLD_IDS
        ]
        aggregated = aggregate_pooled_metrics(fold_results)
        frames.append(tidy_summary("sampling_mode", mode, aggregated))
        parcel = aggregated.pooled_parcel
        pr_auc = float(parcel.loc[parcel["metric"] == "pr_auc", "value"].iloc[0])
        run_logger.info("Sampling mode %r: pooled parcel PR-AUC %.4f.", mode, pr_auc)
    table = pd.concat(frames, ignore_index=True)
    return table[list(SAMPLING_SENSITIVITY_SCHEMA)]


def _library_versions() -> dict[str, str]:
    """Return versions of the key libraries for the run metadata."""
    versions: dict[str, str] = {}
    for library in ("numpy", "pandas", "xgboost", "scikit-learn", "joblib"):
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
        "--sampling-modes", nargs="+", default=list(SAMPLING_MODES), choices=list(SAMPLING_MODES)
    )
    parser.add_argument("--device", default="cuda", choices=["cpu", "cuda"])
    parser.add_argument("--n-jobs", type=int, default=_DEFAULT_N_JOBS)
    parser.add_argument("--n-per-parcel", type=int, default=N_PER_PARCEL)
    parser.add_argument("--seed", type=int, default=SEED)
    return parser.parse_args(argv)


def _run(args: argparse.Namespace, paths: ProjectPaths, run_dir: Path) -> tuple[Path, dict]:
    """Resolve inputs, run the sweep, write the table; return (table path, summary)."""
    run_logger = make_run_logger(_LOGGER_NAME, run_dir)
    source_run = _latest_run(paths.results / args.source_role, args.feature_set)
    run_logger.info(
        "Feature set %s: source run %s; device=%s, n_jobs=%d.",
        args.feature_set,
        source_run.relative_to(paths.repo_root),
        args.device,
        args.n_jobs,
    )

    best_params = _load_best_params(source_run)
    pixel_index, memmap, cache_inputs = _load_pixel_data(paths, args.feature_set)
    run_logger.info("Loaded %d pixels; refitting modes %s.", len(pixel_index), args.sampling_modes)

    table = run_sampling_sensitivity(
        pixel_index,
        memmap,
        best_params,
        feature_set=args.feature_set,
        sampling_modes=args.sampling_modes,
        device=args.device,
        n_jobs=args.n_jobs,
        seed=args.seed,
        n_per_parcel=args.n_per_parcel,
        run_logger=run_logger,
    )
    table_path = write_csv(table, run_dir / "sampling_sensitivity.csv", SAMPLING_SENSITIVITY_SCHEMA)

    inputs = cache_inputs + [
        source_run / _BEST_PARAMS_TEMPLATE.format(fold=fold) for fold in FOLD_IDS
    ]
    write_input_manifest(inputs, run_dir / "input_manifest.json", repo_root=paths.repo_root)
    write_environment_snapshot(run_dir / "env_snapshot.txt")
    summary = {
        "run_id": run_dir.name,
        "created_at": datetime.now(UTC).isoformat(),
        "script": "scripts/run_sampling_sensitivity.py",
        "feature_set": args.feature_set,
        "architecture": _ARCHITECTURE,
        "source_run": str(source_run.relative_to(paths.repo_root)),
        "sampling_modes": list(args.sampling_modes),
        "device": args.device,
        "n_jobs": args.n_jobs,
        "n_per_parcel": args.n_per_parcel,
        "seed": args.seed,
        "hp_reuse": "per-fold best_params from the headline run (no Optuna search)",
        "library_versions": _library_versions(),
        "outputs": {"sampling_sensitivity": str(table_path.relative_to(paths.repo_root))},
    }
    return table_path, summary


def main(argv: Sequence[str] | None = None) -> int:
    """Run the sampling-mode sensitivity analysis and write its artefacts.

    Args:
        argv: Argument list, or ``None`` to read from ``sys.argv``.

    Returns:
        Process exit code (0 on success).
    """
    args = _parse_args(argv)
    paths = get_project_paths()
    ensure_writable_dirs(paths)
    run_id = f"{datetime.now(UTC):%Y%m%d_%H%M%S}__{_ARCHITECTURE}__{args.feature_set}"
    run_dir = paths.results / _RESULTS_ROLE / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    _table_path, summary = _run(args, paths, run_dir)
    write_json(summary, run_dir / "run_metadata.json")
    logger.info("Sampling sensitivity complete: %s.", run_id)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
