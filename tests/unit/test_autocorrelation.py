"""Unit tests for utils.autocorrelation.

All tests are offline and synthetic: small multi-band rasters are written to
``tmp_path`` with rasterio, and the variogram and Moran fields are deterministic
functions of position plus seeded noise. None of them touch the real stacks or
the label products.
"""

import logging
from pathlib import Path

import numpy as np
import pytest
import rasterio
from rasterio.transform import from_origin

from utils.autocorrelation import (
    MORAN_PERMUTATIONS,
    MORAN_THRESHOLDS_M,
    N_SAMPLE,
    VGRAM_MODELS,
    VGRAM_USE_NUGGET,
    VariogramResult,
    fit_variograms,
    moran_correlogram,
    range_unresolved,
    sample_raster_points,
)
from utils.terminology import NODATA

_LOGGER_NAME = "utils.autocorrelation"

# A benign EPSG:3035 origin well inside the projection's valid European extent.
_ORIGIN_X = 4_000_000.0
_ORIGIN_Y = 2_800_000.0
_RES_M = 10.0


def _write_raster(
    path: Path,
    data: np.ndarray,
    *,
    nodata: float | None,
    crs: str = "EPSG:3035",
) -> Path:
    """Write a minimal multi-band GeoTIFF on a 10 m EPSG:3035 grid."""
    if data.ndim == 2:
        data = data[np.newaxis, :, :]
    count, height, width = data.shape
    transform = from_origin(_ORIGIN_X, _ORIGIN_Y, _RES_M, _RES_M)
    profile = {
        "driver": "GTiff",
        "height": height,
        "width": width,
        "count": count,
        "dtype": data.dtype.name,
        "crs": crs,
        "transform": transform,
    }
    if nodata is not None:
        profile["nodata"] = nodata
    with rasterio.open(path, "w", **profile) as dst:
        dst.write(data)
    return path


# --------------------------------------------------------------------------- #
# sample_raster_points
# --------------------------------------------------------------------------- #


def test_sample_raster_points_basic_draw(tmp_path: Path) -> None:
    # Two fully-valid bands on a 20x20 grid; draw a subset and check the contract.
    band1 = np.arange(400, dtype=np.float32).reshape(20, 20)
    band2 = (band1 * 2.0).astype(np.float32)
    path = _write_raster(tmp_path / "two_band.tif", np.stack([band1, band2]), nodata=NODATA)

    coords, values = sample_raster_points(path, bands=(1, 2), n=50, seed=42)

    assert coords.shape == (50, 2)
    assert values.shape == (50, 2)
    assert coords.dtype == np.float64
    assert values.dtype == np.float64

    # Coordinates fall within the raster footprint (pixel centres).
    with rasterio.open(path) as src:
        left, bottom, right, top = src.bounds
    assert np.all(coords[:, 0] > left) and np.all(coords[:, 0] < right)
    assert np.all(coords[:, 1] > bottom) and np.all(coords[:, 1] < top)

    # Band 2 is exactly twice band 1 wherever sampled.
    assert np.allclose(values[:, 1], values[:, 0] * 2.0)


def test_sample_raster_points_excludes_nodata(tmp_path: Path) -> None:
    # Band 1 NoData in its top half, band 2 NoData in its left half: a pixel is
    # valid only where neither band is NoData.
    band1 = np.ones((10, 10), dtype=np.float32)
    band1[:5, :] = NODATA
    band2 = np.full((10, 10), 3.0, dtype=np.float32)
    band2[:, :5] = NODATA
    path = _write_raster(tmp_path / "masked.tif", np.stack([band1, band2]), nodata=NODATA)

    coords, values = sample_raster_points(path, bands=(1, 2), n=N_SAMPLE, seed=0)

    # Only the bottom-right 5x5 block is valid in both bands: 25 pixels.
    assert coords.shape[0] == 25
    assert values.shape == (25, 2)
    assert not np.any(values == NODATA)


def test_sample_raster_points_reproducible(tmp_path: Path) -> None:
    rng = np.random.default_rng(1)
    data = rng.normal(size=(1, 40, 40)).astype(np.float32)
    path = _write_raster(tmp_path / "rand.tif", data, nodata=NODATA)

    first_coords, first_values = sample_raster_points(path, n=100, seed=7)
    second_coords, second_values = sample_raster_points(path, n=100, seed=7)
    other_coords, _ = sample_raster_points(path, n=100, seed=8)

    assert np.array_equal(first_coords, second_coords)
    assert np.array_equal(first_values, second_values)
    # A different seed draws a different sample.
    assert not np.array_equal(first_coords, other_coords)


def test_sample_raster_points_rejects_non_positive_n(tmp_path: Path) -> None:
    path = _write_raster(tmp_path / "tiny.tif", np.ones((1, 4, 4), dtype=np.float32), nodata=NODATA)

    with pytest.raises(ValueError, match="n must be positive"):
        sample_raster_points(path, n=0)


def test_sample_raster_points_warns_when_fewer_than_requested(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    # Almost all NoData: only three valid pixels remain.
    band = np.full((10, 10), NODATA, dtype=np.float32)
    band[0, 0] = 1.0
    band[1, 1] = 2.0
    band[2, 2] = 3.0
    path = _write_raster(tmp_path / "sparse.tif", band, nodata=NODATA)

    with caplog.at_level(logging.WARNING, logger=_LOGGER_NAME):
        coords, values = sample_raster_points(path, n=N_SAMPLE, seed=42)

    assert coords.shape == (3, 2)
    assert values.shape == (3, 1)
    warnings = [
        r for r in caplog.records if r.name == _LOGGER_NAME and r.levelno == logging.WARNING
    ]
    assert len(warnings) == 1
    assert "valid pixels" in warnings[0].getMessage()


def test_sample_raster_points_uses_project_nodata_when_unset(tmp_path: Path) -> None:
    # No NoData declared on the file: the project sentinel must still be excluded.
    band = np.ones((8, 8), dtype=np.float32)
    band[0, :] = NODATA
    path = _write_raster(tmp_path / "no_nodata.tif", band, nodata=None)

    coords, values = sample_raster_points(path, n=N_SAMPLE, seed=3)

    assert coords.shape[0] == 56  # 64 cells minus the 8 sentinel cells.
    assert not np.any(values == NODATA)


# --------------------------------------------------------------------------- #
# fit_variograms
# --------------------------------------------------------------------------- #


def _structured_field(
    n_side: int = 20, spacing_m: float = 1_000.0
) -> tuple[np.ndarray, np.ndarray]:
    """Build a point grid whose value rises smoothly with x plus small noise."""
    axis = np.arange(n_side) * spacing_m
    grid_x, grid_y = np.meshgrid(axis, axis)
    coords = np.column_stack([grid_x.ravel(), grid_y.ravel()])
    rng = np.random.default_rng(42)
    values = 0.01 * coords[:, 0] + rng.normal(0.0, 0.5, size=coords.shape[0])
    return coords, values


def test_fit_variograms_structure_and_shapes() -> None:
    coords, values = _structured_field()

    result = fit_variograms(coords, values, n_lags=10, maxlag=10_000.0, n_curve=100)

    assert isinstance(result, VariogramResult)
    # One row per model, in order.
    assert list(result.fits["model"]) == list(VGRAM_MODELS)
    assert len(result.fits) == len(VGRAM_MODELS)

    # A field with real spatial structure yields a positive range and sill and a
    # non-negative nugget for every model.
    assert (result.fits["effective_range_m"] > 0).all()
    assert (result.fits["sill"] > 0).all()
    assert (result.fits["nugget"] >= 0).all()
    # The ratio is nugget / (nugget + partial sill), so it is bounded in [0, 1].
    assert result.fits["nugget_sill_ratio"].between(0.0, 1.0).all()
    assert result.fits["range_hits_maxlag"].dtype == bool

    # Empirical arrays carry one entry per lag class; curves carry n_curve points.
    assert result.lags_m.shape == (10,)
    assert result.semivariance.shape == (10,)
    assert result.curve_lag_m.shape == (100,)
    assert set(result.fitted) == set(VGRAM_MODELS)
    for model in VGRAM_MODELS:
        assert result.fitted[model].shape == (100,)

    assert result.n == coords.shape[0]
    assert result.maxlag_m == 10_000.0
    assert result.n_lags == 10


def test_fit_variograms_lags_within_maxlag() -> None:
    coords, values = _structured_field()

    result = fit_variograms(coords, values, n_lags=10, maxlag=10_000.0)

    # Bin centres are strictly inside (0, maxlag] and ascending.
    assert result.lags_m[0] > 0.0
    assert result.lags_m[-1] <= 10_000.0
    assert np.all(np.diff(result.lags_m) > 0)


def test_fit_variograms_nugget_toggle() -> None:
    # A weakly-structured field carrying substantial white noise has a genuine
    # nugget: fitting one yields a positive nugget, while disabling the nugget
    # forces it to exactly zero for every model.
    n_side = 20
    axis = np.arange(n_side) * 1_000.0
    grid_x, grid_y = np.meshgrid(axis, axis)
    coords = np.column_stack([grid_x.ravel(), grid_y.ravel()])
    rng = np.random.default_rng(42)
    values = 0.001 * coords[:, 0] + rng.normal(0.0, 5.0, size=coords.shape[0])

    with_nugget = fit_variograms(coords, values, n_lags=10, maxlag=10_000.0, use_nugget=True)
    without_nugget = fit_variograms(coords, values, n_lags=10, maxlag=10_000.0, use_nugget=False)

    assert (with_nugget.fits["nugget"] > 0).all()
    assert (without_nugget.fits["nugget"] == 0).all()


def test_fit_variograms_rejects_too_few_points() -> None:
    coords = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
    values = np.array([1.0, 2.0, 3.0])

    with pytest.raises(ValueError, match="at least"):
        fit_variograms(coords, values, n_lags=10)


def test_fit_variograms_rejects_length_mismatch() -> None:
    coords = np.zeros((20, 2))
    values = np.zeros(19)

    with pytest.raises(ValueError, match="equal length"):
        fit_variograms(coords, values, n_lags=5)


def test_fit_variograms_records_nan_row_on_fit_failure(
    caplog: pytest.LogCaptureFixture,
) -> None:
    coords, values = _structured_field()

    # The first (valid) model builds the shared empirical variogram; the second
    # is an unknown model name, which skgstat rejects, so that model contributes
    # a NaN row rather than aborting the call.
    with caplog.at_level(logging.WARNING, logger=_LOGGER_NAME):
        result = fit_variograms(
            coords,
            values,
            models=("spherical", "not_a_model"),
            n_lags=10,
            maxlag=10_000.0,
            n_curve=50,
        )

    assert list(result.fits["model"]) == ["spherical", "not_a_model"]
    good = result.fits.iloc[0]
    bad = result.fits.iloc[1]
    assert good["effective_range_m"] > 0
    assert np.isnan(bad["effective_range_m"])
    assert np.isnan(bad["sill"])
    assert np.isnan(bad["nugget_sill_ratio"])
    assert bad["range_hits_maxlag"] == np.False_
    # The failed model still yields a (NaN-filled) curve of the right length.
    assert result.fitted["not_a_model"].shape == (50,)
    assert np.isnan(result.fitted["not_a_model"]).all()

    warnings = [
        r for r in caplog.records if r.name == _LOGGER_NAME and r.levelno == logging.WARNING
    ]
    assert len(warnings) == 1
    assert "Variogram fit failed" in warnings[0].getMessage()


# --------------------------------------------------------------------------- #
# range_unresolved
# --------------------------------------------------------------------------- #


def test_range_unresolved_flags_both_boundaries() -> None:
    # 40 km max lag over 40 lag classes -> first-bin width 1 km. A range inside
    # the first bin (lower boundary) and one at/above 90% of the max lag (upper
    # boundary) are both unresolved, as is a non-finite range; a mid-window range
    # is resolved.
    ranges = np.array([500.0, 6_700.0, 36_000.0, 39_000.0, np.nan])
    flags = range_unresolved(ranges, 40_000.0, n_lags=40)

    assert flags.tolist() == [True, False, True, True, True]


def test_range_unresolved_floor_scales_with_n_lags() -> None:
    # The floor is maxlag / n_lags, so a 1.5 km range is resolved at 40 bins
    # (1 km floor) but unresolved at 20 bins (2 km floor).
    assert not bool(range_unresolved(1_500.0, 40_000.0, n_lags=40))
    assert bool(range_unresolved(1_500.0, 40_000.0, n_lags=20))


# --------------------------------------------------------------------------- #
# moran_correlogram
# --------------------------------------------------------------------------- #


def _clustered_coords(n_side: int = 12, spacing_m: float = 1.0) -> np.ndarray:
    """Return a regular point grid for the Moran tests."""
    axis = np.arange(n_side) * spacing_m
    grid_x, grid_y = np.meshgrid(axis, axis)
    return np.column_stack([grid_x.ravel(), grid_y.ravel()])


def test_moran_correlogram_clustered_field_is_significant() -> None:
    coords = _clustered_coords()
    # A smooth spatial gradient is strongly positively autocorrelated.
    values = coords[:, 0] + coords[:, 1]

    frame = moran_correlogram(coords, values, thresholds_m=(2,), permutations=199, seed=42)

    assert list(frame.columns) == [
        "morans_i",
        "expected_i",
        "z_sim",
        "p_sim",
        "p_norm",
        "n",
        "mean_neighbours",
    ]
    assert frame.index.name == "threshold_m"
    assert frame.loc[2, "morans_i"] > 0.5
    assert frame.loc[2, "p_sim"] < 0.05
    # Both p-value columns are populated.
    assert frame["p_sim"].notna().all()
    assert frame["p_norm"].notna().all()


def test_moran_correlogram_random_field_near_zero() -> None:
    coords = _clustered_coords()
    rng = np.random.default_rng(7)
    values = rng.normal(size=coords.shape[0])

    frame = moran_correlogram(coords, values, thresholds_m=(2,), permutations=199, seed=42)

    assert abs(frame.loc[2, "morans_i"]) < 0.15


def test_moran_correlogram_skips_empty_graph(caplog: pytest.LogCaptureFixture) -> None:
    coords = _clustered_coords(spacing_m=1.0)
    values = coords[:, 0] + coords[:, 1]

    # 0.5 m is below the 1 m minimum spacing, so that graph has no neighbours and
    # is skipped; the 2 m graph survives.
    with caplog.at_level(logging.WARNING, logger=_LOGGER_NAME):
        frame = moran_correlogram(coords, values, thresholds_m=(0.5, 2), permutations=99, seed=42)

    assert list(frame.index) == [2]
    warnings = [
        r for r in caplog.records if r.name == _LOGGER_NAME and r.levelno == logging.WARNING
    ]
    assert len(warnings) == 1
    assert "no neighbours" in warnings[0].getMessage()


def test_module_constants_have_expected_defaults() -> None:
    # Guard the analysis parameters the script imports against accidental drift.
    assert N_SAMPLE == 5_000
    assert MORAN_THRESHOLDS_M == (5_000, 10_000, 15_000, 20_000)
    assert MORAN_PERMUTATIONS == 999
    assert VGRAM_MODELS == ("spherical", "exponential", "gaussian")
    assert VGRAM_USE_NUGGET is True
