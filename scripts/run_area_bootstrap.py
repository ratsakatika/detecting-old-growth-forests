"""Predicted old-growth area uncertainty by a block-retraining bootstrap (Stage 10).

The point estimate of the mapped old-growth area comes from the final model
(``scripts/run_final_inference.py``). Its uncertainty is propagated by resampling
the twenty spatial bootstrap blocks with replacement, retraining the model on the
resampled parcels and re-running wall-to-wall inference, then measuring the total
old-growth area at the fixed F1-max threshold on the uncalibrated predictions.

The hyperparameters are held fixed at the mapping run's selected values, so each
replicate is one retrain plus one inference pass, not a fresh hyperparameter
search. Replicates run on a pool of ``--n-parallel`` workers (default 4) to keep the
GPU busy, since a single fit-and-infer rarely saturates it; each worker loads the
memmaps once (the OS page cache shares the read-only data). The run stays exactly
reproducible because each replicate's RNG is seeded by ``[seed, rep_id]`` whichever
worker runs it. The parent process is the sole writer, appending each replicate to
``area_bootstrap.csv`` as it finishes, so the run is resumable (``--mapping-run``
re-enters the same directory and skips completed replicates) and the mapping notebook
renders whatever number is complete. The model runs on GPU (CUDA) by default. Output
is written into the mapping run directory under ``results/mapping/<run_id>/``.
"""

import argparse
import json
import logging
import multiprocessing
import os
from collections.abc import Iterator, Sequence
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Final

import geopandas as gpd
import numpy as np
import numpy.typing as npt
import pandas as pd

from scripts.run_final_inference import (
    _GRID_MEMMAP,
    _LABELS,
    _PARCEL_TEMPLATE,
    _config,
    _parcel_id_grid,
    _pooled_threshold,
    valid_pixel_ids,
    wall_to_wall_over_ids,
)
from scripts.run_sampling_sensitivity import _latest_run, _load_pixel_data
from utils.cv import NestedCVConfig, sample_train_rows
from utils.env_capture import write_environment_snapshot
from utils.inference import ogf_area_ha, parcel_consistent_area_ha
from utils.io import ColumnSpec, Schema, write_json
from utils.logging import make_run_logger
from utils.manifest import write_input_manifest
from utils.models.xgboost import make_xgb_classifier
from utils.paths import ProjectPaths, ensure_writable_dirs, get_project_paths
from utils.raster_io import open_reference_grid
from utils.sampling import build_sample_weights
from utils.terminology import (
    BOOTSTRAP_N_UNITS,
    BOOTSTRAP_REPS_INFERENCE,
    SEED,
)

logger = logging.getLogger(__name__)

_LOGGER_NAME: Final[str] = "run_area_bootstrap"
_FEATURE_SET: Final[str] = "baseline_tessera"
_ARCHITECTURE: Final[str] = "xgboost"
_RESULTS_ROLE: Final[str] = "mapping"
_SOURCE_ROLE: Final[str] = "main_nested_cv"
_BOOTSTRAP_FILE: Final[str] = "area_bootstrap.csv"

AREA_BOOTSTRAP_SCHEMA: Final[Schema] = {
    "rep_id": ColumnSpec("int64"),
    "pixel_area_ha": ColumnSpec("float64"),
    "parcel_area_ha": ColumnSpec("float64"),
    "seed": ColumnSpec("int64"),
}


def _bootstrap_positions(
    pixel_index: pd.DataFrame,
    bootstrap_id: npt.NDArray[np.int64],
    blocks: npt.NDArray[np.int64],
    n_per_parcel: int,
    rng: np.random.Generator,
) -> npt.NDArray[np.int64]:
    """Row positions for the resampled blocks (a block drawn twice contributes twice)."""
    valid = pixel_index["valid"].to_numpy()
    positions: list[npt.NDArray[np.int64]] = []
    for block in blocks:
        block_rows = np.flatnonzero((bootstrap_id == block) & valid)
        if block_rows.size == 0:
            continue
        sampled = sample_train_rows(pixel_index.iloc[block_rows], n_per_parcel, rng)
        positions.append(block_rows[sampled])
    return np.concatenate(positions) if positions else np.empty(0, dtype=np.int64)


def bootstrap_areas_ha(
    rep_id: int,
    pixel_index: pd.DataFrame,
    memmap: np.ndarray,
    bootstrap_id: npt.NDArray[np.int64],
    grid: npt.NDArray[np.float32],
    pixel_ids: npt.NDArray[np.int64],
    best_params: dict[str, float | int],
    *,
    config: NestedCVConfig,
    pixel_threshold: float,
    parcel_ids_per_pixel: npt.NDArray[np.int64],
    parcel_threshold: float,
    chunk_size: int,
) -> tuple[float, float]:
    """One replicate: resample blocks, retrain, infer, and measure both areas.

    Returns the per-pixel-threshold area and the parcel-consistent area (pixels in parcels
    whose mean probability reaches the parcel threshold), which differ because the two F1-max
    operating points differ.
    """
    unique_blocks = np.unique(bootstrap_id)
    rng = np.random.default_rng([config.seed, rep_id])
    blocks = rng.choice(unique_blocks, size=BOOTSTRAP_N_UNITS, replace=True)
    positions = _bootstrap_positions(pixel_index, bootstrap_id, blocks, config.n_per_parcel, rng)
    y_train = pixel_index["ogf"].to_numpy()[positions].astype(np.int8)
    weights = build_sample_weights(
        config.sampling_mode,
        labels=y_train,
        forest_type=pixel_index["forest_type_corine"].to_numpy()[positions],
    )
    model = make_xgb_classifier(
        best_params, device=config.device, n_jobs=config.n_jobs, seed=config.seed
    )
    model.fit(np.asarray(memmap[positions], dtype=np.float32), y_train, sample_weight=weights)
    probabilities = wall_to_wall_over_ids(model, grid, pixel_ids, chunk_size=chunk_size)
    pixel_area = ogf_area_ha(probabilities >= pixel_threshold)
    parcel_area = parcel_consistent_area_ha(
        parcel_ids_per_pixel, probabilities, threshold=parcel_threshold
    )
    return pixel_area, parcel_area


def _completed_reps(path: Path) -> set[int]:
    """Replicate ids already in the append-only CSV (for resuming)."""
    if not path.is_file():
        return set()
    return set(pd.read_csv(path)["rep_id"].astype(int))


def _append_replicate(
    path: Path, rep_id: int, pixel_area_ha: float, parcel_area_ha: float, seed: int
) -> None:
    """Append one replicate row, writing the header only for a new file."""
    pd.DataFrame(
        [
            {
                "rep_id": rep_id,
                "pixel_area_ha": pixel_area_ha,
                "parcel_area_ha": parcel_area_ha,
                "seed": seed,
            }
        ]
    ).to_csv(path, mode="a", header=not path.is_file(), index=False)


def _bootstrap_id_per_pixel(
    pixel_index: pd.DataFrame, labels: gpd.GeoDataFrame
) -> npt.NDArray[np.int64]:
    """Map each pixel's parcel to its bootstrap block id."""
    lookup = labels.set_index("parcel_id")["bootstrap_id"]
    mapped = pixel_index["parcel_id"].map(lookup)
    if mapped.isna().any():
        raise ValueError(
            "some pixels have no bootstrap_id; check that parcel_id has a consistent dtype "
            "(it is int64 in the pixel index but a zero-padded string in the labels gpkg)."
        )
    return mapped.to_numpy().astype(np.int64)


@dataclass(frozen=True)
class _WorkerArgs:
    """Picklable arguments to initialise a pool worker (it loads the heavy data itself)."""

    repo_root: Path
    mapping_run: str
    device: str
    n_jobs: int
    n_per_parcel: int
    seed: int
    chunk_size: int


@dataclass(frozen=True)
class _WorkerState:
    """The loaded data a worker reuses across replicates, populated once by ``_init_worker``."""

    pixel_index: pd.DataFrame
    memmap: np.ndarray
    bootstrap_id: npt.NDArray[np.int64]
    grid: npt.NDArray[np.float32]
    pixel_ids: npt.NDArray[np.int64]
    parcel_ids_per_pixel: npt.NDArray[np.int64]
    best_params: dict[str, float | int]
    config: NestedCVConfig
    pixel_threshold: float
    parcel_threshold: float
    chunk_size: int


# The pixel and grid memmaps are far too large to pickle per replicate, so each pool worker
# loads them once into this process-local state; the OS page cache shares the read-only
# memmaps across workers. Replicates remain reproducible because every replicate's RNG is
# seeded by [seed, rep_id] regardless of which worker runs it.
_STATE: _WorkerState | None = None


def _init_worker(worker_args: _WorkerArgs) -> None:
    """Load the pixel data, grid and block ids once into the worker's process state."""
    global _STATE
    paths = get_project_paths(worker_args.repo_root)
    run_dir = paths.results / _RESULTS_ROLE / worker_args.mapping_run
    best_params = json.loads((run_dir / "final_best_params.json").read_text("utf-8"))
    source_run = _latest_run(paths.results / _SOURCE_ROLE, _FEATURE_SET)
    pixel_index, memmap, _ = _load_pixel_data(paths, _FEATURE_SET)
    labels = gpd.read_file(paths.repo_root / _LABELS)
    labels["parcel_id"] = labels["parcel_id"].astype("int64")
    ref = open_reference_grid()
    # The bootstrap area uses the same domain as the map: the parcel-map area, not CORINE forest.
    template = gpd.read_file(paths.repo_root / _PARCEL_TEMPLATE)[["parcel_id", "geometry"]]
    template["parcel_id"] = template["parcel_id"].astype("int64")
    parcel_grid = _parcel_id_grid(template, ref)
    grid = np.load(paths.repo_root / _GRID_MEMMAP, mmap_mode="r")
    pixel_ids = valid_pixel_ids(grid, parcel_grid >= 0)
    _STATE = _WorkerState(
        pixel_index=pixel_index,
        memmap=memmap,
        bootstrap_id=_bootstrap_id_per_pixel(pixel_index, labels),
        grid=grid,
        pixel_ids=pixel_ids,
        parcel_ids_per_pixel=parcel_grid.reshape(-1)[pixel_ids],
        best_params=best_params,
        config=_config(
            worker_args.device, worker_args.n_jobs, worker_args.seed, worker_args.n_per_parcel
        ),
        pixel_threshold=_pooled_threshold(source_run, "pixel"),
        parcel_threshold=_pooled_threshold(source_run, "parcel"),
        chunk_size=worker_args.chunk_size,
    )


def _worker_replicate(rep_id: int) -> tuple[int, float, float]:
    """Compute one bootstrap replicate (pixel and parcel areas) from the worker's state."""
    if _STATE is None:  # pragma: no cover - the pool always runs the initializer first
        raise RuntimeError("worker state was not initialised.")
    pixel_area, parcel_area = bootstrap_areas_ha(
        rep_id,
        _STATE.pixel_index,
        _STATE.memmap,
        _STATE.bootstrap_id,
        _STATE.grid,
        _STATE.pixel_ids,
        _STATE.best_params,
        config=_STATE.config,
        pixel_threshold=_STATE.pixel_threshold,
        parcel_ids_per_pixel=_STATE.parcel_ids_per_pixel,
        parcel_threshold=_STATE.parcel_threshold,
        chunk_size=_STATE.chunk_size,
    )
    return rep_id, pixel_area, parcel_area


def _stream_replicates(
    executor: ProcessPoolExecutor, pending: Sequence[int]
) -> Iterator[tuple[int, float, float]]:
    """Submit the pending replicates; yield (rep_id, pixel_area, parcel_area) as each completes."""
    futures = {executor.submit(_worker_replicate, rep_id): rep_id for rep_id in pending}
    for future in as_completed(futures):
        yield future.result()


def _parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mapping-run", default=None, help="results/mapping/<run_id>; default newest."
    )
    parser.add_argument("--reps", type=int, default=BOOTSTRAP_REPS_INFERENCE)
    parser.add_argument(
        "--n-parallel", type=int, default=4, help="Replicates computed concurrently on the GPU."
    )
    parser.add_argument("--device", default="cuda", choices=["cpu", "cuda"])
    parser.add_argument(
        "--n-jobs",
        type=int,
        default=None,
        help="XGBoost CPU threads per worker (default: cores shared across --n-parallel).",
    )
    parser.add_argument("--n-per-parcel", type=int, default=500)
    parser.add_argument("--chunk-size", type=int, default=2_000_000)
    parser.add_argument("--seed", type=int, default=SEED)
    return parser.parse_args(argv)


def _resolve_mapping_run(paths: ProjectPaths, mapping_run: str | None) -> Path:
    """Resolve the mapping run directory carrying final_best_params.json."""
    root = paths.results / _RESULTS_ROLE
    if mapping_run is not None:
        return root / mapping_run
    candidates = sorted(
        (p for p in root.iterdir() if p.is_dir() and (p / "final_best_params.json").is_file()),
        reverse=True,
    )
    if not candidates:
        raise FileNotFoundError(f"no mapping run with final_best_params.json under {root}.")
    return candidates[0]


def _run(args: argparse.Namespace, paths: ProjectPaths, run_dir: Path) -> dict:
    """Run the outstanding area-bootstrap replicates and append them to the CSV."""
    run_logger = make_run_logger(_LOGGER_NAME, run_dir)
    n_jobs = args.n_jobs or max(1, (os.cpu_count() or 4) // max(1, args.n_parallel))
    source_run = _latest_run(paths.results / _SOURCE_ROLE, _FEATURE_SET)
    pixel_threshold = _pooled_threshold(source_run, "pixel")
    parcel_threshold = _pooled_threshold(source_run, "parcel")

    csv_path = run_dir / _BOOTSTRAP_FILE
    done = _completed_reps(csv_path)
    pending = [rep_id for rep_id in range(args.reps) if rep_id not in done]
    run_logger.info(
        "Area bootstrap: %d/%d replicates done; %d pending across %d workers (%d threads each).",
        len(done),
        args.reps,
        len(pending),
        args.n_parallel,
        n_jobs,
    )

    if pending:
        worker_args = _WorkerArgs(
            paths.repo_root,
            run_dir.name,
            args.device,
            n_jobs,
            args.n_per_parcel,
            args.seed,
            args.chunk_size,
        )
        # Spawn (not fork) so each worker starts a clean CUDA context. The parent stays
        # the sole writer of the CSV, so the
        # appends never race; a crash keeps every replicate written so far.
        context = multiprocessing.get_context("spawn")
        with ProcessPoolExecutor(
            max_workers=args.n_parallel,
            mp_context=context,
            initializer=_init_worker,
            initargs=(worker_args,),
        ) as executor:
            for rep_id, pixel_area, parcel_area in _stream_replicates(executor, pending):
                _append_replicate(csv_path, rep_id, pixel_area, parcel_area, args.seed)
                run_logger.info(
                    "Replicate %d: old-growth area pixel %.0f ha, parcel %.0f ha.",
                    rep_id,
                    pixel_area,
                    parcel_area,
                )

    write_input_manifest(
        [run_dir / "final_best_params.json", paths.repo_root / _LABELS],
        run_dir / "area_bootstrap_manifest.json",
        repo_root=paths.repo_root,
    )
    write_environment_snapshot(run_dir / "area_bootstrap_env.txt")
    completed = pd.read_csv(csv_path)

    # Carry the mapping run's calibrated thresholds, point estimates and selected calibrators into
    # this metadata so the pixel- and parcel-area stats (and their %-of-AOI) can be recreated from
    # ``area_bootstrap.csv`` (which stores both ``pixel_area_ha`` and ``parcel_area_ha`` per
    # replicate) without re-running the bootstrap. Areas are measured at the RAW F1-max thresholds
    # on each replicate's raw predictions; Platt calibration is monotonic, so it leaves the selected
    # parcel/pixel sets -- and therefore both areas -- unchanged (``parcel_area_ha`` already equals
    # the Platt parcel-verdict area).
    mapping_meta_path = run_dir / "run_metadata.json"
    mapping_meta = (
        json.loads(mapping_meta_path.read_text("utf-8")) if mapping_meta_path.is_file() else {}
    )
    carry = {
        key: mapping_meta[key]
        for key in (
            "pixel_threshold_calibrated",
            "parcel_threshold_calibrated",
            "predicted_ogf_area_ha",
            "predicted_ogf_area_parcel_ha",
            "selected_calibrator",
            "selected_calibrator_pixel",
        )
        if key in mapping_meta
    }
    return {
        "created_at": datetime.now(UTC).isoformat(),
        "script": "scripts/run_area_bootstrap.py",
        "mapping_run": run_dir.name,
        "reps_requested": args.reps,
        "reps_completed": int(len(completed)),
        "bootstrap_n_units": BOOTSTRAP_N_UNITS,
        "pixel_threshold": pixel_threshold,
        "parcel_threshold": parcel_threshold,
        "pixel_area_mean_ha": float(completed["pixel_area_ha"].mean()),
        "parcel_area_mean_ha": float(completed["parcel_area_ha"].mean()),
        "calibration_note": (
            "areas are at the raw F1-max thresholds; Platt is monotonic so parcel_area_ha equals "
            "the Platt parcel-verdict area (calibration does not change the selected set)"
        ),
        **carry,
        "device": args.device,
        "n_parallel": args.n_parallel,
        "seed": args.seed,
    }


def main(argv: Sequence[str] | None = None) -> int:
    """Run the predicted-area bootstrap and write its artefacts.

    Args:
        argv: Argument list, or ``None`` to read from ``sys.argv``.

    Returns:
        Process exit code (0 on success).
    """
    args = _parse_args(argv)
    paths = get_project_paths()
    ensure_writable_dirs(paths)
    run_dir = _resolve_mapping_run(paths, args.mapping_run)

    summary = _run(args, paths, run_dir)
    write_json(summary, run_dir / "area_bootstrap_metadata.json")
    logger.info("Area bootstrap complete: %s.", run_dir.name)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
