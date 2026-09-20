"""Unit tests for utils.raster_io.

All tests are offline: they build small synthetic rasters in ``tmp_path`` with
rasterio. None of them depend on the real reference-grid file existing.
"""

import logging
from pathlib import Path

import numpy as np
import pytest
import rasterio
from rasterio.crs import CRS
from rasterio.transform import from_bounds, from_origin
from rasterio.warp import transform_bounds
from shapely.geometry import LineString, box

from utils.raster_io import (
    BandStats,
    RasterAudit,
    ReferenceGrid,
    audit_raster,
    distance_to_lines_on_ref,
    open_reference_grid,
    rasterize_mask,
    reproject_raster_to_ref,
    tiled_block_iterator,
    write_geotiff,
)
from utils.terminology import CRS as PROJECT_CRS
from utils.terminology import NODATA, REF_RESOLUTION_M

# A benign EPSG:3035 origin well inside the projection's valid European extent.
_ORIGIN_X = 4_000_000.0
_ORIGIN_Y = 2_800_000.0
_LOGGER_NAME = "utils.raster_io"


def _write_raster(
    path: Path,
    data: np.ndarray,
    *,
    crs: str | CRS,
    transform,
    nodata: float | None = None,
) -> Path:
    """Write a minimal GeoTIFF for use as test input."""
    if data.ndim == 2:
        data = data[np.newaxis, :, :]
    count, height, width = data.shape
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


def _ref_grid(width: int, height: int) -> ReferenceGrid:
    """Build a project-CRS reference grid at the reference resolution."""
    transform = from_origin(_ORIGIN_X, _ORIGIN_Y, REF_RESOLUTION_M, REF_RESOLUTION_M)
    return ReferenceGrid(
        crs=CRS.from_user_input(PROJECT_CRS),
        transform=transform,
        width=width,
        height=height,
    )


# --------------------------------------------------------------------------- #
# open_reference_grid
# --------------------------------------------------------------------------- #


def test_open_reference_grid_accepts_valid_grid(tmp_path: Path) -> None:
    transform = from_origin(_ORIGIN_X, _ORIGIN_Y, REF_RESOLUTION_M, REF_RESOLUTION_M)
    path = _write_raster(
        tmp_path / "grid.tif",
        np.zeros((8, 8), dtype=np.float32),
        crs=PROJECT_CRS,
        transform=transform,
    )

    grid = open_reference_grid(path)

    assert isinstance(grid, ReferenceGrid)
    assert grid.crs == CRS.from_user_input(PROJECT_CRS)
    assert grid.shape == (8, 8)
    assert grid.resolution == (REF_RESOLUTION_M, REF_RESOLUTION_M)


def test_open_reference_grid_rejects_wrong_crs(tmp_path: Path) -> None:
    # Correct 10 m pixels but the wrong CRS (UTM 35N) must be rejected.
    transform = from_origin(500_000.0, 4_900_000.0, REF_RESOLUTION_M, REF_RESOLUTION_M)
    path = _write_raster(
        tmp_path / "utm.tif",
        np.zeros((8, 8), dtype=np.float32),
        crs="EPSG:32635",
        transform=transform,
    )

    with pytest.raises(ValueError, match="CRS"):
        open_reference_grid(path)


def test_open_reference_grid_rejects_wrong_resolution(tmp_path: Path) -> None:
    # Correct CRS but 30 m pixels must be rejected.
    transform = from_origin(_ORIGIN_X, _ORIGIN_Y, 30.0, 30.0)
    path = _write_raster(
        tmp_path / "coarse.tif",
        np.zeros((8, 8), dtype=np.float32),
        crs=PROJECT_CRS,
        transform=transform,
    )

    with pytest.raises(ValueError, match="pixel size"):
        open_reference_grid(path)


# --------------------------------------------------------------------------- #
# reproject_raster_to_ref
# --------------------------------------------------------------------------- #


def test_reproject_raster_to_ref_from_other_crs(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    ref = _ref_grid(width=20, height=20)

    # Express the left ~45% of the reference extent as a lon/lat box and write a
    # constant-valued EPSG:4326 source covering only that strip. Reprojected back
    # to 3035, the right side of the grid then falls outside source coverage.
    left, bottom, right, top = ref.bounds
    west, south, east, north = transform_bounds(ref.crs, "EPSG:4326", left, bottom, right, top)
    east_partial = west + (east - west) * 0.45
    src_transform = from_bounds(west, south, east_partial, north, 30, 40)
    src_path = _write_raster(
        tmp_path / "src_4326.tif",
        np.full((1, 40, 30), 5.0, dtype=np.float32),
        crs="EPSG:4326",
        transform=src_transform,
    )

    with caplog.at_level(logging.INFO, logger=_LOGGER_NAME):
        out = reproject_raster_to_ref(src_path, ref)

    assert out.shape == (1, *ref.shape)
    # Some cells carry the source value; cells outside coverage carry dst_nodata.
    assert np.isclose(out, 5.0).any()
    assert (out == NODATA).any()

    info = [r for r in caplog.records if r.name == _LOGGER_NAME and r.levelno == logging.INFO]
    assert len(info) == 1
    message = info[0].getMessage()
    assert "EPSG:4326" in message
    assert "EPSG:3035" in message
    assert "bilinear" in message
    assert str(src_path) in message


def test_reproject_raster_to_ref_requires_source_crs(tmp_path: Path) -> None:
    ref = _ref_grid(width=8, height=8)
    transform = from_origin(_ORIGIN_X, _ORIGIN_Y, REF_RESOLUTION_M, REF_RESOLUTION_M)
    path = _write_raster(
        tmp_path / "no_crs.tif",
        np.zeros((1, 8, 8), dtype=np.float32),
        crs=None,  # type: ignore[arg-type]
        transform=transform,
    )

    with pytest.raises(ValueError, match="no CRS"):
        reproject_raster_to_ref(path, ref)


# --------------------------------------------------------------------------- #
# write_geotiff
# --------------------------------------------------------------------------- #


def test_write_geotiff_round_trips(tmp_path: Path) -> None:
    ref = _ref_grid(width=5, height=6)
    data = np.arange(2 * 6 * 5, dtype=np.float32).reshape(2, 6, 5)
    out_path = tmp_path / "nested" / "deep" / "out.tif"

    returned = write_geotiff(
        out_path,
        data,
        ref,
        band_descriptions=["alpha", "beta"],
    )

    assert returned == out_path
    assert out_path.parent.is_dir()
    with rasterio.open(out_path) as src:
        assert src.dtypes[0] == "float32"
        assert src.nodata == NODATA
        assert src.descriptions == ("alpha", "beta")
        assert src.crs == ref.crs
        assert src.transform == ref.transform
        assert np.array_equal(src.read(), data)


def test_write_geotiff_accepts_single_band_2d(tmp_path: Path) -> None:
    ref = _ref_grid(width=5, height=6)
    data = np.ones((6, 5), dtype=np.float32)

    out_path = write_geotiff(tmp_path / "single.tif", data, ref)

    with rasterio.open(out_path) as src:
        assert src.count == 1
        assert np.array_equal(src.read(1), data)


def test_write_geotiff_rejects_shape_mismatch(tmp_path: Path) -> None:
    ref = _ref_grid(width=5, height=6)
    data = np.zeros((6, 4), dtype=np.float32)  # wrong width

    with pytest.raises(ValueError, match="does not match"):
        write_geotiff(tmp_path / "bad.tif", data, ref)


def test_write_geotiff_rejects_band_description_mismatch(tmp_path: Path) -> None:
    ref = _ref_grid(width=5, height=6)
    data = np.zeros((2, 6, 5), dtype=np.float32)

    with pytest.raises(ValueError, match="band_descriptions"):
        write_geotiff(tmp_path / "bad.tif", data, ref, band_descriptions=["only_one"])


# --------------------------------------------------------------------------- #
# rasterize_mask
# --------------------------------------------------------------------------- #


def test_rasterize_mask_interior_and_exterior(tmp_path: Path) -> None:
    ref = _ref_grid(width=10, height=10)
    # Box covering columns 3..6 and rows 3..6 of the grid.
    minx = _ORIGIN_X + 3 * REF_RESOLUTION_M
    maxx = _ORIGIN_X + 7 * REF_RESOLUTION_M
    maxy = _ORIGIN_Y - 3 * REF_RESOLUTION_M
    miny = _ORIGIN_Y - 7 * REF_RESOLUTION_M
    polygon = box(minx, miny, maxx, maxy)

    mask = rasterize_mask([polygon], ref)

    assert mask.shape == ref.shape
    assert mask.dtype == np.uint8
    assert mask[5, 5] == 1  # interior
    assert mask[0, 0] == 0  # exterior
    assert mask[9, 9] == 0  # exterior


# --------------------------------------------------------------------------- #
# distance_to_lines_on_ref
# --------------------------------------------------------------------------- #


def test_distance_to_lines_on_ref_single_line(tmp_path: Path) -> None:
    ref = _ref_grid(width=11, height=11)
    # Vertical line through the centre of column 5 (avoids touching two columns).
    x_centre = _ORIGIN_X + 5 * REF_RESOLUTION_M + REF_RESOLUTION_M / 2
    top = _ORIGIN_Y
    bottom = _ORIGIN_Y - 11 * REF_RESOLUTION_M
    line = LineString([(x_centre, top), (x_centre, bottom)])

    distance = distance_to_lines_on_ref([line], ref)

    assert distance.dtype == np.float32
    assert distance[5, 5] == 0.0
    assert np.isclose(distance[5, 4], REF_RESOLUTION_M)
    assert np.isclose(distance[5, 6], REF_RESOLUTION_M)


# --------------------------------------------------------------------------- #
# tiled_block_iterator
# --------------------------------------------------------------------------- #


def test_tiled_block_iterator_covers_grid_exactly() -> None:
    ref = _ref_grid(width=25, height=15)
    coverage = np.zeros(ref.shape, dtype=np.int64)
    windows = list(tiled_block_iterator(ref, block_size=10))

    for window in windows:
        coverage[
            window.row_off : window.row_off + window.height,
            window.col_off : window.col_off + window.width,
        ] += 1

    # Every pixel covered exactly once: no gaps, no overlap.
    assert coverage.min() == 1
    assert coverage.max() == 1
    # 3 columns of blocks (10, 10, 5) x 2 rows of blocks (10, 5).
    assert len(windows) == 6
    last = windows[-1]
    assert (last.width, last.height) == (5, 5)


# --------------------------------------------------------------------------- #
# audit_raster
# --------------------------------------------------------------------------- #


def test_audit_raster_without_stats(tmp_path: Path) -> None:
    transform = from_origin(_ORIGIN_X, _ORIGIN_Y, REF_RESOLUTION_M, REF_RESOLUTION_M)
    data = np.zeros((2, 4, 4), dtype=np.float32)
    path = _write_raster(
        tmp_path / "audit.tif", data, crs=PROJECT_CRS, transform=transform, nodata=NODATA
    )

    audit = audit_raster(path)

    assert isinstance(audit, RasterAudit)
    assert audit.band_stats is None
    assert audit.crs.to_epsg() == 3035
    assert (audit.width, audit.height, audit.count) == (4, 4, 2)
    assert audit.dtype == "float32"
    assert audit.resolution == (REF_RESOLUTION_M, REF_RESOLUTION_M)
    assert audit.nodata == (NODATA, NODATA)
    # str(audit) is a single line and omits stats when band_stats is None.
    text = str(audit)
    assert "\n" not in text
    assert "audit.tif" in text


def test_audit_raster_with_stats(tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
    transform = from_origin(_ORIGIN_X, _ORIGIN_Y, REF_RESOLUTION_M, REF_RESOLUTION_M)
    # Band 1: values 0..15 with two cells set to NoData. Band 2: all valid.
    band1 = np.arange(16, dtype=np.float32).reshape(4, 4)
    band1[0, 0] = NODATA
    band1[3, 3] = NODATA
    band2 = np.full((4, 4), 7.0, dtype=np.float32)
    data = np.stack([band1, band2])
    path = _write_raster(
        tmp_path / "stats.tif", data, crs=PROJECT_CRS, transform=transform, nodata=NODATA
    )

    with caplog.at_level(logging.INFO, logger=_LOGGER_NAME):
        audit = audit_raster(path, with_stats=True)

    # audit_raster performs no logging itself.
    assert [r for r in caplog.records if r.name == _LOGGER_NAME] == []

    assert audit.band_stats is not None
    assert len(audit.band_stats) == 2

    expected_valid = np.array([v for v in range(16) if v not in (0, 15)], dtype=np.float32)
    b1 = audit.band_stats[0]
    assert isinstance(b1, BandStats)
    assert b1.index == 1
    assert b1.valid_count == 14
    assert b1.nodata_count == 2
    assert np.isclose(b1.minimum, expected_valid.min())
    assert np.isclose(b1.maximum, expected_valid.max())
    assert np.isclose(b1.mean, expected_valid.mean())
    assert np.isclose(b1.std, expected_valid.std())

    b2 = audit.band_stats[1]
    assert b2.valid_count == 16
    assert b2.nodata_count == 0
    assert np.isclose(b2.mean, 7.0)

    text = str(audit)
    assert "\n" not in text


def test_reproject_raster_to_ref_converts_dtype_with_threads(tmp_path: Path) -> None:
    ref = _ref_grid(width=8, height=8)
    transform = from_origin(_ORIGIN_X, _ORIGIN_Y, REF_RESOLUTION_M, REF_RESOLUTION_M)
    src_path = _write_raster(
        tmp_path / "f32.tif",
        np.ones((1, 8, 8), dtype=np.float32),
        crs=PROJECT_CRS,
        transform=transform,
    )

    out = reproject_raster_to_ref(src_path, ref, dtype="float64", num_threads=1)

    assert out.dtype == np.dtype("float64")
    assert out.shape == (1, *ref.shape)


def test_write_geotiff_rejects_one_dimensional_array(tmp_path: Path) -> None:
    ref = _ref_grid(width=5, height=6)

    with pytest.raises(ValueError, match="Expected a 2-D"):
        write_geotiff(tmp_path / "bad.tif", np.zeros(5, dtype=np.float32), ref)


def test_audit_raster_all_nodata_band_has_nan_stats(tmp_path: Path) -> None:
    transform = from_origin(_ORIGIN_X, _ORIGIN_Y, REF_RESOLUTION_M, REF_RESOLUTION_M)
    data = np.full((1, 4, 4), NODATA, dtype=np.float32)
    path = _write_raster(
        tmp_path / "all_nodata.tif", data, crs=PROJECT_CRS, transform=transform, nodata=NODATA
    )

    audit = audit_raster(path, with_stats=True)

    assert audit.band_stats is not None
    band = audit.band_stats[0]
    assert band.valid_count == 0
    assert band.nodata_count == 16
    assert np.isnan(band.minimum)
    assert np.isnan(band.mean)
