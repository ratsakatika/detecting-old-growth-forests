"""Functional smoke test for scripts/run_final_inference.py core steps.

Tiny synthetic pixel and grid data exercise the CV-tuned search, the final fit,
the wall-to-wall pass, parcel aggregation, calibration and the mapped-frame join
on the CPU, without the multi-gigabyte grid memmap or the full file plumbing.
"""

import logging
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import pytest
from affine import Affine
from rasterio.crs import CRS
from shapely.geometry import box

from scripts import run_final_inference as rfi
from utils.models.xgboost import TUNABLE_PARAMS
from utils.raster_io import ReferenceGrid
from utils.terminology import FEATURE_SET_BANDS, NODATA

_N_BANDS = len(FEATURE_SET_BANDS["baseline_tessera"])  # 134
_LOG = logging.getLogger("test")


def _pixel_data() -> tuple[pd.DataFrame, np.ndarray]:
    """Six folds x four parcels x five pixels, with a separable old-growth signal."""
    rng = np.random.default_rng(0)
    rows: list[dict[str, object]] = []
    features: list[np.ndarray] = []
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
                pixel_id += 1
            pid += 1
    return pd.DataFrame(rows), np.asarray(features, dtype=np.float32)


def _ref() -> ReferenceGrid:
    return ReferenceGrid(
        crs=CRS.from_epsg(3035), transform=Affine(10, 0, 0, 0, -10, 40), width=5, height=4
    )


def test_cv_search_and_final_model() -> None:
    pixel_index, memmap = _pixel_data()
    config = rfi._config("cpu", 1, 42, n_per_parcel=5)
    best_params, value = rfi.cv_tuned_search(
        config, pixel_index, memmap, n_trials=2, run_logger=_LOG
    )
    assert set(TUNABLE_PARAMS).issubset(best_params)
    assert 0.0 <= value <= 1.0
    model = rfi.train_final_model(config, pixel_index, memmap, best_params)
    proba = model.predict_proba(memmap[:5])[:, 1]
    assert proba.shape == (5,) and proba.min() >= 0.0 and proba.max() <= 1.0


def test_wall_to_wall_and_parcel_aggregation() -> None:
    pixel_index, memmap = _pixel_data()
    config = rfi._config("cpu", 1, 42, n_per_parcel=5)
    model = rfi.train_final_model(
        config, pixel_index, memmap, rfi.suggest_xgb_params(_DummyTrial())
    )

    ref = _ref()
    grid = np.zeros((_N_BANDS, ref.height, ref.width), dtype=np.float32)
    grid[0] = 1.0  # band-0 valid everywhere
    grid[0, 0, 0] = NODATA  # one invalid cell
    forest = np.ones((ref.height, ref.width), dtype=bool)
    ids = rfi.valid_pixel_ids(grid, forest)
    assert ids.size == ref.height * ref.width - 1  # the NODATA cell is excluded

    probs = rfi.wall_to_wall_over_ids(model, grid, ids, chunk_size=8)
    assert probs.shape == ids.shape and probs.min() >= 0.0 and probs.max() <= 1.0

    template = gpd.GeoDataFrame(
        {"parcel_id": [10, 20]},
        geometry=[box(0, 20, 50, 40), box(0, 0, 50, 20)],
        crs="EPSG:3035",
    )
    parcel_grid = rfi._parcel_id_grid(template, ref)
    assert set(np.unique(parcel_grid)).issubset({-1, 10, 20})
    parcels = rfi.aggregate_to_parcels(ids, probs, parcel_grid)
    assert set(parcels["parcel_id"]).issubset({10, 20})
    assert parcels["p_mean"].between(0.0, 1.0).all()


def test_valid_pixel_ids_validation() -> None:
    grid = np.ones((_N_BANDS, 4, 5), dtype=np.float32)
    with pytest.raises(ValueError, match="domain mask shape"):
        rfi.valid_pixel_ids(grid, np.ones((3, 3), dtype=bool))


def test_calibration_and_mapped_frame(tmp_path: Path) -> None:
    rng = np.random.default_rng(1)
    p = rng.uniform(0.0, 1.0, 600)
    y = (rng.uniform(0.0, 1.0, 600) < p).astype(np.int64)
    oof = pd.DataFrame({"parcel_id": np.arange(600), "y_true": y, "p_mean": p})

    parcel_probs = pd.DataFrame({"parcel_id": [10, 20], "p_mean": [0.8, 0.2], "n_pixels": [4, 6]})
    method, comparison, calibrated, calibrator = rfi.calibrate_parcels(
        oof, parcel_probs["p_mean"].to_numpy(), seed=42
    )
    assert method in ("isotonic", "platt", "beta", "temperature", "spline")
    assert calibrated.shape == (2,) and calibrated.min() >= 0.0 and calibrated.max() <= 1.0
    assert "method" in comparison.columns
    assert hasattr(calibrator, "predict")

    # Forcing a method overrides the auto-selection but still computes the full comparison.
    forced, comparison_f, _, _ = rfi.calibrate_parcels(
        oof, parcel_probs["p_mean"].to_numpy(), seed=42, method="platt"
    )
    assert forced == "platt"
    assert set(comparison_f["method"]) >= {"isotonic", "platt", "beta", "temperature", "spline"}
    with pytest.raises(ValueError, match="unknown calibrator"):
        rfi.calibrate_parcels(oof, parcel_probs["p_mean"].to_numpy(), seed=42, method="nope")

    template = gpd.GeoDataFrame(
        {"parcel_id": [10, 20]}, geometry=[box(0, 20, 50, 40), box(0, 0, 50, 20)], crs="EPSG:3035"
    )
    labels = gpd.GeoDataFrame(
        {"parcel_id": [10], "ogf": [1]}, geometry=[box(1, 21, 49, 39)], crs="EPSG:3035"
    )
    mapped = rfi.build_mapped_frame(
        template, labels, parcel_probs, calibrated, threshold_calibrated=0.5
    )
    assert {
        "reference_label",
        "ogf_ratsakatika_probability_raw",
        "ogf_ratsakatika_probability",
        "ogf_ratsakatika",
    }.issubset(mapped.columns)
    assert mapped.loc[mapped["parcel_id"] == 10, "ogf_ratsakatika"].iloc[0]  # p_mean 0.8 >= 0.5


def test_pixel_calibration_preserves_binary() -> None:
    """Mapping the F1-max threshold through the monotonic pixel calibrator keeps the same pixels."""
    rng = np.random.default_rng(2)
    p = rng.uniform(0.0, 1.0, 2000)
    y = (rng.uniform(0.0, 1.0, 2000) < p).astype(np.int64)
    method, comparison, calibrator = rfi.fit_and_compare(y, p, seed=42, method="platt")
    assert method == "platt" and "method" in comparison.columns

    surface = rng.uniform(0.0, 1.0, 500)
    t_raw = 0.2829
    t_cal = float(calibrator.predict(np.asarray([t_raw]))[0])
    calibrated = calibrator.predict(surface)
    np.testing.assert_array_equal(calibrated >= t_cal, surface >= t_raw)


def test_pooled_threshold(tmp_path: Path) -> None:
    pd.DataFrame({"metric": ["pr_auc", "threshold"], "value": [0.8, 0.37]}).to_csv(
        tmp_path / "pooled_pixel_metrics.csv", index=False
    )
    assert rfi._pooled_threshold(tmp_path, "pixel") == pytest.approx(0.37)


class _DummyTrial:
    """Minimal Optuna-like trial that returns the low end of each search range."""

    def suggest_int(self, name: str, low: int, high: int, **_: object) -> int:
        return low

    def suggest_float(self, name: str, low: float, high: float, **_: object) -> float:
        return low
