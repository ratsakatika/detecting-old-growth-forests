"""Functional smoke test for scripts/run_area_bootstrap.py.

Tiny synthetic pixel and grid data exercise the block resample, the fixed-HP
retrain, the area measurement and the resumable append loop on the CPU.
"""

from pathlib import Path

import numpy as np
import pandas as pd

from scripts import run_area_bootstrap as rab
from scripts import run_final_inference as rfi
from utils.terminology import FEATURE_SET_BANDS

_N_BANDS = len(FEATURE_SET_BANDS["baseline_tessera"])

_PARAMS: dict[str, float | int] = {
    "n_estimators": 20,
    "learning_rate": 0.3,
    "max_depth": 3,
    "min_child_weight": 1.0,
    "subsample": 1.0,
    "colsample_bytree": 1.0,
    "reg_lambda": 1.0,
    "reg_alpha": 0.0,
    "gamma": 0.0,
    "max_delta_step": 0,
}


def _pixel_data() -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    """Pixels with a separable signal and a bootstrap-block id per parcel."""
    rng = np.random.default_rng(0)
    rows: list[dict[str, object]] = []
    features: list[np.ndarray] = []
    blocks: list[int] = []
    pid = pixel_id = 0
    for fold in range(1, 7):
        for parcel in range(4):
            ogf = 1 if parcel < 2 else 0
            for _ in range(5):
                rows.append(
                    {
                        "pixel_id": pixel_id,
                        "parcel_id": pid,
                        "fold_id": np.int8(fold),
                        "ogf": np.int8(ogf),
                        "forest_type_corine": "broadleaf" if ogf else "coniferous",
                        "valid": True,
                    }
                )
                vector = rng.standard_normal(_N_BANDS).astype(np.float32)
                vector[0] = 4.0 * ogf + 0.3 * float(rng.standard_normal())
                features.append(vector)
                blocks.append(parcel % 4)  # four bootstrap blocks
                pixel_id += 1
            pid += 1
    index = pd.DataFrame(rows)
    return index, np.asarray(features, dtype=np.float32), np.asarray(blocks, dtype=np.int64)


def _grid() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    grid = np.zeros((_N_BANDS, 4, 5), dtype=np.float32)
    grid[0] = 3.0  # old-growth-leaning band-0 signal across a valid grid
    parcel_grid = np.zeros((4, 5), dtype=np.int64)
    parcel_grid[2:] = 1  # two parcels: top rows parcel 0, bottom rows parcel 1
    pixel_ids = rfi.valid_pixel_ids(grid, parcel_grid >= 0)
    parcel_ids_per_pixel = parcel_grid.reshape(-1)[pixel_ids]
    return grid, pixel_ids, parcel_ids_per_pixel


def test_bootstrap_positions_respect_block_multiplicity() -> None:
    index, _, blocks = _pixel_data()
    rng = np.random.default_rng(1)
    once = rab._bootstrap_positions(index, blocks, np.array([0]), 5, rng)
    twice = rab._bootstrap_positions(index, blocks, np.array([0, 0]), 5, rng)
    assert twice.size == 2 * once.size  # a block drawn twice contributes twice


def test_bootstrap_areas_ha_is_finite() -> None:
    index, memmap, blocks = _pixel_data()
    grid, pixel_ids, parcel_ids_per_pixel = _grid()
    config = rfi._config("cpu", 1, 42, n_per_parcel=5)
    pixel_area, parcel_area = rab.bootstrap_areas_ha(
        0,
        index,
        memmap,
        blocks,
        grid,
        pixel_ids,
        _PARAMS,
        config=config,
        pixel_threshold=0.5,
        parcel_ids_per_pixel=parcel_ids_per_pixel,
        parcel_threshold=0.5,
        chunk_size=8,
    )
    assert np.isfinite(pixel_area) and pixel_area >= 0.0
    assert np.isfinite(parcel_area) and parcel_area >= 0.0


def test_append_and_resume(tmp_path: Path) -> None:
    path = tmp_path / "area_bootstrap.csv"
    assert rab._completed_reps(path) == set()
    rab._append_replicate(path, 0, 1234.5, 1100.0, 42)
    rab._append_replicate(path, 1, 1300.0, 1150.0, 42)
    assert rab._completed_reps(path) == {0, 1}
    frame = pd.read_csv(path)
    assert list(frame.columns) == ["rep_id", "pixel_area_ha", "parcel_area_ha", "seed"]
    assert frame["pixel_area_ha"].iloc[0] == 1234.5
    assert frame["parcel_area_ha"].iloc[0] == 1100.0


def test_bootstrap_id_per_pixel() -> None:
    import geopandas as gpd
    from shapely.geometry import box

    index = pd.DataFrame({"parcel_id": [10, 10, 20]})
    labels = gpd.GeoDataFrame(
        {"parcel_id": [10, 20], "bootstrap_id": [3, 7]},
        geometry=[box(0, 0, 1, 1), box(1, 1, 2, 2)],
        crs="EPSG:3035",
    )
    assert list(rab._bootstrap_id_per_pixel(index, labels)) == [3, 3, 7]


def test_worker_replicate_uses_process_state() -> None:
    """The pool worker computes a replicate (both areas) from its initialised state."""
    index, memmap, blocks = _pixel_data()
    grid, pixel_ids, parcel_ids_per_pixel = _grid()
    config = rfi._config("cpu", 1, 42, n_per_parcel=5)
    rab._STATE = rab._WorkerState(
        pixel_index=index,
        memmap=memmap,
        bootstrap_id=blocks,
        grid=grid,
        pixel_ids=pixel_ids,
        parcel_ids_per_pixel=parcel_ids_per_pixel,
        best_params=_PARAMS,
        config=config,
        pixel_threshold=0.5,
        parcel_threshold=0.5,
        chunk_size=8,
    )
    try:
        rep_id, pixel_area, parcel_area = rab._worker_replicate(3)
        assert rep_id == 3
        assert np.isfinite(pixel_area) and np.isfinite(parcel_area)
        # Reproducible: the same replicate id gives the same areas.
        assert rab._worker_replicate(3) == (3, pixel_area, parcel_area)
    finally:
        rab._STATE = None


def test_n_jobs_default_shares_cores() -> None:
    args = rab._parse_args(["--n-parallel", "4"])
    assert args.n_jobs is None  # resolved in _run to cores // n_parallel
    assert args.n_parallel == 4
