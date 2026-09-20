"""Synthetic smoke test for scripts/build_continuous_product_parcels.py."""

from pathlib import Path

import geopandas as gpd
import numpy as np
import pytest
import rasterio
from affine import Affine
from rasterio.crs import CRS
from shapely.geometry import box

from scripts import build_continuous_product_parcels as bcp
from utils.raster_io import ReferenceGrid

_ORIGIN_X, _ORIGIN_Y = 5_427_190.0, 2_634_920.0
_TRANSFORM = Affine(10.0, 0.0, _ORIGIN_X, 0.0, -10.0, _ORIGIN_Y)
_W = _H = 6


def _ref() -> ReferenceGrid:
    return ReferenceGrid(crs=CRS.from_epsg(3035), transform=_TRANSFORM, width=_W, height=_H)


def _cover(x0_frac: float, x1_frac: float) -> box:
    return box(
        _ORIGIN_X + _W * 10.0 * x0_frac,
        _ORIGIN_Y - _H * 10.0,
        _ORIGIN_X + _W * 10.0 * x1_frac,
        _ORIGIN_Y,
    )


def _template() -> gpd.GeoDataFrame:
    return gpd.GeoDataFrame(
        {"parcel_id": [10, 20]}, geometry=[_cover(0.0, 0.5), _cover(0.5, 1.0)], crs="EPSG:3035"
    )


def test_parcel_id_grid_burns_ids() -> None:
    grid = bcp.parcel_id_grid(_template(), _ref())
    assert grid.shape == (_H, _W)
    assert set(np.unique(grid)) == {10, 20}  # the two parcels tile the whole grid
    assert (grid[:, :3] == 10).all() and (grid[:, 3:] == 20).all()


def test_surface_parcel_means(tmp_path: Path) -> None:
    data = np.full((_H, _W), 0.8, dtype=np.float32)  # left half 0.8, right half 0.2
    data[:, 3:] = 0.2
    surface = tmp_path / "surface.tif"
    profile = {
        "driver": "GTiff",
        "height": _H,
        "width": _W,
        "count": 1,
        "dtype": "float32",
        "crs": CRS.from_epsg(3035),
        "transform": _TRANSFORM,
        "nodata": -3.4e38,
    }
    with rasterio.open(surface, "w", **profile) as dst:
        dst.write(data, 1)
    means = bcp.surface_parcel_means(surface, _ref(), bcp.parcel_id_grid(_template(), _ref()))
    assert means.loc[10] == pytest.approx(0.8, abs=1e-5)
    assert means.loc[20] == pytest.approx(0.2, abs=1e-5)


def _profile() -> dict:
    return {
        "driver": "GTiff",
        "height": _H,
        "width": _W,
        "count": 1,
        "dtype": "float32",
        "crs": CRS.from_epsg(3035),
        "transform": _TRANSFORM,
        "nodata": -3.4e38,
    }


def test_surface_parcel_means_nodata_counts_as_zero(tmp_path: Path) -> None:
    """NoData pixels inside a parcel count as 0 (non-old-growth), not dropped from the mean."""
    data = np.full((_H, _W), 0.8, dtype=np.float32)
    data[:, :3] = -3.4e38  # all of parcel 10 (left three columns) is NoData
    data[0, 3] = -3.4e38  # one NoData pixel in parcel 20 (right three columns)
    surface = tmp_path / "surface_nodata.tif"
    with rasterio.open(surface, "w", **_profile()) as dst:
        dst.write(data, 1)
    means = bcp.surface_parcel_means(surface, _ref(), bcp.parcel_id_grid(_template(), _ref()))
    assert means.loc[10] == pytest.approx(0.0, abs=1e-5)  # all NoData -> mean 0
    # parcel 20 has 18 pixels: one NoData counts as 0, the other 17 are 0.8.
    assert means.loc[20] == pytest.approx(0.8 * 17 / 18, abs=1e-4)
