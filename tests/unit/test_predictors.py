"""Unit tests for utils.predictors.

All tests are offline: they build small synthetic arrays and grids in memory or
``tmp_path``. None require network access.
"""

import math
from pathlib import Path

import numpy as np
import pytest
import rasterio
from pyproj import Transformer
from rasterio.crs import CRS
from rasterio.transform import from_origin

from utils.predictors import (
    assemble_stack,
    heat_load_index,
    horn_aspect,
    horn_slope,
    latitude_grid,
)
from utils.raster_io import ReferenceGrid
from utils.terminology import CRS as PROJECT_CRS
from utils.terminology import NODATA

_RES = 30.0  # native FABDEM-like metric resolution, metres


def _east_facing_plane(size: int, gradient: float, x_res: float) -> np.ndarray:
    """Return a plane whose elevation decreases eastward at a known gradient.

    A plane facing east (downhill to the east) has aspect 90 degrees and slope
    ``atan(gradient)``.
    """
    cols = np.arange(size, dtype=np.float64)
    return 100.0 - gradient * x_res * cols[np.newaxis, :].repeat(size, axis=0)


def _ref_grid(width: int, height: int) -> ReferenceGrid:
    """Build a small project-CRS reference grid at 10 m resolution."""
    return ReferenceGrid(
        crs=CRS.from_user_input(PROJECT_CRS),
        transform=from_origin(4_000_000.0, 2_800_000.0, 10.0, 10.0),
        width=width,
        height=height,
    )


# --------------------------------------------------------------------------- #
# horn_slope
# --------------------------------------------------------------------------- #


def test_horn_slope_flat_interior_and_border() -> None:
    elevation = np.zeros((5, 5), dtype=np.float64)

    slope = horn_slope(elevation, x_res=_RES, y_res=_RES)

    assert np.allclose(slope[1:-1, 1:-1], 0.0)
    # The whole border lacks a full 3x3 window.
    assert slope[0, 0] == NODATA
    assert slope[0, 2] == NODATA
    assert slope[2, 0] == NODATA
    assert slope[4, 4] == NODATA


def test_horn_slope_east_gradient() -> None:
    gradient = 0.5
    elevation = _east_facing_plane(5, gradient, _RES)

    slope = horn_slope(elevation, x_res=_RES, y_res=_RES)

    expected = np.degrees(np.arctan(gradient))
    assert np.allclose(slope[1:-1, 1:-1], expected)


def test_horn_slope_nodata_propagates_to_eight_neighbours() -> None:
    elevation = np.zeros((7, 7), dtype=np.float64)
    elevation[3, 3] = NODATA

    slope = horn_slope(elevation, x_res=_RES, y_res=_RES)

    # The flagged cell and its eight neighbours all have a window touching nodata.
    for dr in (-1, 0, 1):
        for dc in (-1, 0, 1):
            assert slope[3 + dr, 3 + dc] == NODATA
    # A far interior cell is unaffected and stays flat (0 degrees).
    assert slope[1, 1] != NODATA
    assert np.isclose(slope[1, 1], 0.0)


# --------------------------------------------------------------------------- #
# horn_aspect
# --------------------------------------------------------------------------- #


def test_horn_aspect_east_facing_is_90() -> None:
    elevation = _east_facing_plane(5, 0.5, _RES)

    aspect = horn_aspect(elevation, x_res=_RES, y_res=_RES)

    assert np.allclose(aspect[1:-1, 1:-1], 90.0, atol=1e-6)


def test_horn_aspect_flat_value_and_border() -> None:
    elevation = np.zeros((5, 5), dtype=np.float64)

    aspect = horn_aspect(elevation, x_res=_RES, y_res=_RES, flat_value=-7.0)

    assert np.all(aspect[1:-1, 1:-1] == -7.0)
    assert aspect[0, 0] == NODATA
    assert aspect[4, 2] == NODATA


def test_horn_aspect_nodata_propagation() -> None:
    elevation = np.zeros((7, 7), dtype=np.float64)
    elevation[3, 3] = NODATA

    aspect = horn_aspect(elevation, x_res=_RES, y_res=_RES)

    for dr in (-1, 0, 1):
        for dc in (-1, 0, 1):
            assert aspect[3 + dr, 3 + dc] == NODATA
    # Unaffected interior flat cell carries the default flat value.
    assert aspect[1, 1] == -1.0


# --------------------------------------------------------------------------- #
# heat_load_index
# --------------------------------------------------------------------------- #


def test_heat_load_index_sw_hotter_than_ne() -> None:
    slope = np.full((2, 2), 30.0)
    latitude = np.full((2, 2), 45.5)

    hli_sw = heat_load_index(slope, np.full((2, 2), 225.0), latitude)
    hli_ne = heat_load_index(slope, np.full((2, 2), 45.0), latitude)

    assert np.all(hli_sw > hli_ne)


def test_heat_load_index_flat_ground_independent_of_aspect() -> None:
    slope = np.zeros((2, 2))
    latitude = np.full((2, 2), 45.5)

    hli_ne = heat_load_index(slope, np.full((2, 2), 45.0), latitude)
    hli_sw = heat_load_index(slope, np.full((2, 2), 225.0), latitude)
    hli_flat = heat_load_index(slope, np.full((2, 2), -1.0), latitude)

    assert np.allclose(hli_ne, hli_sw)
    assert np.allclose(hli_ne, hli_flat)


def test_heat_load_index_matches_hand_computation() -> None:
    slope = np.array([[30.0]])
    aspect = np.array([[180.0]])
    latitude = np.array([[45.5]])

    hli = heat_load_index(slope, aspect, latitude)

    # Independently evaluate the documented McCune & Keon (2002) equation 3.
    big_l = math.radians(45.5)
    big_s = math.radians(30.0)
    reduced = ((180.0 - 225.0 + 180.0) % 360.0) - 180.0
    big_f = math.radians(180.0 - abs(reduced))
    expected = math.exp(
        -1.467
        + 1.582 * math.cos(big_l) * math.cos(big_s)
        - 1.500 * math.cos(big_f) * math.sin(big_s) * math.sin(big_l)
        - 0.262 * math.sin(big_l) * math.sin(big_s)
        + 0.607 * math.sin(big_f) * math.sin(big_s)
    )
    assert np.isclose(hli[0, 0], expected, rtol=1e-9)


def test_heat_load_index_nodata_propagation() -> None:
    slope = np.full((2, 2), 30.0)
    aspect = np.full((2, 2), 180.0)
    latitude = np.full((2, 2), 45.5)
    slope[0, 0] = NODATA
    aspect[1, 1] = NODATA

    hli = heat_load_index(slope, aspect, latitude)

    assert hli[0, 0] == NODATA
    assert hli[1, 1] == NODATA
    assert hli[0, 1] != NODATA
    assert np.isfinite(hli[0, 1])


# --------------------------------------------------------------------------- #
# latitude_grid
# --------------------------------------------------------------------------- #


def test_latitude_grid_over_fagaras_aoi() -> None:
    # Place a small 30 m EPSG:3035 grid with its top-left near the Fagaras AOI.
    forward = Transformer.from_crs("EPSG:4326", "EPSG:3035", always_xy=True)
    x0, y0 = forward.transform(24.7, 45.7)
    transform = from_origin(x0, y0, _RES, _RES)

    latitudes = latitude_grid(transform, CRS.from_epsg(3035), 5, 5)

    assert latitudes.shape == (5, 5)
    assert np.all((latitudes > 45.0) & (latitudes < 46.0))
    # Latitude decreases southward: the northern (top) row exceeds the bottom row.
    assert latitudes[0, 0] > latitudes[-1, 0]


# --------------------------------------------------------------------------- #
# assemble_stack
# --------------------------------------------------------------------------- #


def test_assemble_stack_writes_float32_with_descriptions(tmp_path: Path) -> None:
    ref = _ref_grid(width=3, height=4)
    band0 = np.arange(12, dtype=np.float64).reshape(4, 3)
    band1 = np.full((4, 3), 2.0)
    band1[0, 0] = np.nan  # must be written as the nodata sentinel
    out_path = tmp_path / "stack.tif"

    returned = assemble_stack(out_path, [band0, band1], ref, band_names=["elevation", "slope"])

    assert returned == out_path
    with rasterio.open(out_path) as src:
        assert src.count == 2
        assert src.dtypes[0] == "float32"
        assert src.nodata == NODATA
        assert src.descriptions == ("elevation", "slope")
        assert np.array_equal(src.read(1), band0.astype(np.float32))
        written_band1 = src.read(2)
        assert written_band1[0, 0] == NODATA
        assert written_band1[1, 1] == np.float32(2.0)


def test_assemble_stack_rejects_length_mismatch(tmp_path: Path) -> None:
    ref = _ref_grid(width=3, height=4)
    band0 = np.zeros((4, 3))
    band1 = np.zeros((4, 3))

    with pytest.raises(ValueError, match="one-to-one"):
        assemble_stack(tmp_path / "bad.tif", [band0, band1], ref, band_names=["only"])


def test_assemble_stack_rejects_shape_mismatch(tmp_path: Path) -> None:
    ref = _ref_grid(width=3, height=4)
    band0 = np.zeros((4, 3))
    band_bad = np.zeros((4, 4))  # wrong width

    with pytest.raises(ValueError, match="does not match"):
        assemble_stack(tmp_path / "bad.tif", [band0, band_bad], ref, band_names=["a", "b"])


def test_horn_slope_small_array_is_all_nodata() -> None:
    # An array smaller than a 3x3 Horn window has no defined gradients anywhere.
    slope = horn_slope(np.zeros((2, 2), dtype=np.float64), x_res=_RES, y_res=_RES)

    assert slope.shape == (2, 2)
    assert np.all(slope == NODATA)
