"""Synthetic integration check for scripts/build_pixel_table.py.

A tiny reference grid and two square parcels exercise the strict full-pixel
rasterisation, the pixel index construction and the cached feature-matrix
extraction without touching the real (multi-gigabyte) inputs.
"""

import json
import logging
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import pytest
import rasterio
from affine import Affine
from rasterio.crs import CRS
from shapely.geometry import box

from scripts import build_pixel_table as bpt
from utils.cache import CACHE_MANIFEST_VERSION, atomic_write_json, file_signature
from utils.io import PIXEL_INDEX_SCHEMA, validate_schema
from utils.paths import ProjectPaths, get_project_paths
from utils.raster_io import ReferenceGrid
from utils.terminology import FEATURE_SET_BANDS, NODATA

# Reference grid: 14 columns x 8 rows of 10 m pixels, origin at (0, 80).
_WIDTH = 14
_HEIGHT = 8
_TRANSFORM = Affine(10.0, 0.0, 0.0, 0.0, -10.0, 80.0)
# A 50 m square parcel erodes by half a pixel diagonal (~7.07 m) to a region
# whose contained pixel centres form a 3x3 interior block.
_PIXELS_PER_PARCEL = 9
# The NoData pixel sits inside parcel A's 3x3 block (rows 2-4, cols 2-4).
_NODATA_ROW, _NODATA_COL = 3, 3


def _ref() -> ReferenceGrid:
    return ReferenceGrid(
        crs=CRS.from_epsg(3035), transform=_TRANSFORM, width=_WIDTH, height=_HEIGHT
    )


def _parcels() -> gpd.GeoDataFrame:
    # CORINE and inventory forest types deliberately disagree so each column can
    # be checked independently: A is CORINE broadleaf / inventory coniferous,
    # B is CORINE mixed / inventory broadleaf.
    return gpd.GeoDataFrame(
        {
            "parcel_id": ["00001", "00002"],
            "ogf": [1.0, 0.0],
            "fold_id": [1.0, 2.0],
            "corine_forest_type": ["broadleaf", "mixed"],
            "class": ["OGF_Coniferous", "Non_OGF_Broadleaf"],
        },
        geometry=[box(10, 20, 60, 70), box(80, 20, 130, 70)],
        crs="EPSG:3035",
    )


def _band_data() -> np.ndarray:
    """Three bands whose values encode (band, row, col) so reads are checkable."""
    rows, cols = np.meshgrid(np.arange(_HEIGHT), np.arange(_WIDTH), indexing="ij")
    base = (rows * _WIDTH + cols).astype(np.float32)
    data = np.stack([base + 100.0 * band for band in range(3)]).astype(np.float32)
    data[0, _NODATA_ROW, _NODATA_COL] = NODATA  # one invalid pixel in band 0
    return data


def _write_stack(path: Path) -> np.ndarray:
    data = _band_data()
    profile = {
        "driver": "GTiff",
        "height": _HEIGHT,
        "width": _WIDTH,
        "count": 3,
        "dtype": "float32",
        "crs": CRS.from_epsg(3035),
        "transform": _TRANSFORM,
        "nodata": NODATA,
    }
    with rasterio.open(path, "w", **profile) as dst:
        dst.write(data)
    return data


def test_strict_full_pixel_buffer_is_half_diagonal() -> None:
    assert bpt.STRICT_FULL_PIXEL_BUFFER_M == pytest.approx(10.0 * np.sqrt(2.0) / 2.0)


def test_rasterise_parcel_index_keeps_full_pixels_only() -> None:
    index_raster = bpt.rasterise_parcel_index(
        _parcels(), _ref(), buffer_m=bpt.STRICT_FULL_PIXEL_BUFFER_M
    )
    assert index_raster.shape == (_HEIGHT, _WIDTH)
    assert index_raster.dtype == np.int32
    assert int(np.count_nonzero(index_raster)) == 2 * _PIXELS_PER_PARCEL
    assert set(np.unique(index_raster)) == {0, 1, 2}


def test_build_pixel_index_schema_and_contents() -> None:
    index = bpt.build_pixel_index(_parcels(), _ref())

    validate_schema(index, PIXEL_INDEX_SCHEMA)
    assert len(index) == 2 * _PIXELS_PER_PARCEL
    # Sorted by pixel_id and pixel_id == row * width + col.
    assert index["pixel_id"].is_monotonic_increasing
    expected_id = index["row"].astype(np.int64) * _WIDTH + index["col"].astype(np.int64)
    assert (index["pixel_id"] == expected_id).all()
    # Cell-centre coordinates.
    assert (index["x"] == index["col"] * 10.0 + 5.0).all()
    assert (index["y"] == 80.0 - (index["row"] * 10.0 + 5.0)).all()
    # Parcel A (broadleaf, ogf, fold 1) and parcel B (mixed, non-ogf, fold 2).
    parcel_a = index[index["parcel_id"] == 1]
    parcel_b = index[index["parcel_id"] == 2]
    assert len(parcel_a) == _PIXELS_PER_PARCEL
    assert len(parcel_b) == _PIXELS_PER_PARCEL
    assert set(parcel_a["fold_id"]) == {1} and set(parcel_a["ogf"]) == {1}
    # CORINE column comes straight from corine_forest_type.
    assert set(parcel_a["forest_type_corine"]) == {"broadleaf"}
    assert set(parcel_b["forest_type_corine"]) == {"mixed"}
    # Inventory column is the lower-cased suffix of the inventory class.
    assert set(parcel_a["forest_type_inventory"]) == {"coniferous"}
    assert set(parcel_b["forest_type_inventory"]) == {"broadleaf"}


def test_extract_features_to_memmap_values_and_validity(tmp_path: Path) -> None:
    stack_path = tmp_path / "stack.tif"
    data = _write_stack(stack_path)
    index = bpt.build_pixel_index(_parcels(), _ref())
    rows = index["row"].to_numpy(dtype=np.int32)
    cols = index["col"].to_numpy(dtype=np.int32)

    memmap_path = tmp_path / "features.npy"
    valid = bpt.extract_features_to_memmap(
        stack_path, rows, cols, n_bands=3, memmap_path=memmap_path, n_jobs=1
    )

    matrix = np.load(memmap_path, mmap_mode="r")
    assert matrix.shape == (2 * _PIXELS_PER_PARCEL, 3)
    for i in range(matrix.shape[0]):
        for band in range(3):
            assert matrix[i, band] == data[band, rows[i], cols[i]]

    # Exactly one invalid pixel: the NoData cell in band 0.
    assert valid.sum() == 2 * _PIXELS_PER_PARCEL - 1
    invalid_position = np.flatnonzero((rows == _NODATA_ROW) & (cols == _NODATA_COL))
    assert invalid_position.size == 1
    assert not valid[invalid_position[0]]


def test_pixel_cache_lineage_detects_same_shape_input_changes(tmp_path: Path) -> None:
    source = tmp_path / "stack.bin"
    source.write_bytes(b"source-v1")
    index_path = tmp_path / "pixel_index.parquet"
    pd.DataFrame({"pixel_id": pd.Series([10, 11], dtype="int64")}).to_parquet(index_path)
    pixel_ids = np.array([10, 11], dtype=np.int64)
    memmap_path = tmp_path / "baseline.npy"
    valid_path = tmp_path / "baseline_valid.parquet"
    manifest_path = tmp_path / "baseline.json"
    np.save(memmap_path, np.ones((2, 3), dtype=np.float32))
    pd.DataFrame(
        {
            "pixel_id": pd.Series(pixel_ids, dtype="int64"),
            "valid": pd.Series([True, True], dtype="bool"),
        }
    ).to_parquet(valid_path)
    source_signature = file_signature(source)
    index_signature = file_signature(index_path)
    atomic_write_json(
        {
            "cache_manifest_version": CACHE_MANIFEST_VERSION,
            "feature_set": "baseline",
            "shape": [2, 3],
            "dtype": "float32",
            "source_signature": source_signature,
            "pixel_index_signature": index_signature,
        },
        manifest_path,
    )

    assert bpt._pixel_cache_matches(
        memmap_path,
        valid_path,
        manifest_path,
        feature_set="baseline",
        source_signature=source_signature,
        pixel_index_signature=index_signature,
        pixel_ids=pixel_ids,
        n_bands=3,
    )

    source.write_bytes(b"source-v2")
    assert not bpt._pixel_cache_matches(
        memmap_path,
        valid_path,
        manifest_path,
        feature_set="baseline",
        source_signature=file_signature(source),
        pixel_index_signature=index_signature,
        pixel_ids=pixel_ids,
        n_bands=3,
    )

    pd.DataFrame({"pixel_id": pd.Series([11, 10], dtype="int64")}).to_parquet(index_path)
    assert not bpt._pixel_cache_matches(
        memmap_path,
        valid_path,
        manifest_path,
        feature_set="baseline",
        source_signature=source_signature,
        pixel_index_signature=file_signature(index_path),
        pixel_ids=np.array([11, 10], dtype=np.int64),
        n_bands=3,
    )


def _write_pre_lineage_pixel_cache(
    paths: ProjectPaths, *, n_rows: int, pixel_ids: np.ndarray
) -> tuple[Path, Path]:
    """Write a matrix, validity table, pixel index and source stack, but no manifest."""
    feature_set = "baseline"
    n_bands = len(FEATURE_SET_BANDS[feature_set])
    memmap_path, valid_path, _ = bpt._cache_paths(paths, feature_set)
    memmap_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(memmap_path, np.ones((n_rows, n_bands), dtype=np.float32))
    pd.DataFrame(
        {
            "pixel_id": pd.Series(pixel_ids, dtype="int64"),
            "valid": pd.Series([True] * len(pixel_ids), dtype="bool"),
        }
    ).to_parquet(valid_path)
    index_path = paths.results / "main_nested_cv" / "pixel_index.parquet"
    index_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"pixel_id": pd.Series(pixel_ids, dtype="int64")}).to_parquet(index_path)
    stack_path = bpt._stack_path(paths, feature_set)
    stack_path.parent.mkdir(parents=True, exist_ok=True)
    stack_path.write_bytes(b"synthetic stack")
    return memmap_path, index_path


def test_ensure_pixel_cache_adopts_pre_lineage_without_re_extraction(tmp_path: Path) -> None:
    # A pre-lineage pixel cache (no manifest) is adopted: its matrix is kept and a
    # current manifest is written, never re-extracting the features.
    paths = get_project_paths(tmp_path)
    pixel_ids = np.array([10, 11], dtype=np.int64)
    memmap_path, index_path = _write_pre_lineage_pixel_cache(paths, n_rows=2, pixel_ids=pixel_ids)
    manifest_path = bpt._cache_paths(paths, "baseline")[2]
    array_bytes = memmap_path.read_bytes()
    assert not manifest_path.exists()

    bpt.ensure_pixel_cache_current(
        paths,
        "baseline",
        pixel_index_path=index_path,
        pixel_ids=pixel_ids,
        logger=logging.getLogger("t"),
    )
    assert memmap_path.read_bytes() == array_bytes  # adopted, not re-extracted
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["cache_manifest_version"] == CACHE_MANIFEST_VERSION
    assert manifest["feature_set"] == "baseline"
    assert len(manifest["source_signature"]["sha256"]) == 64
    # Now current: a second call is a no-op (raises nothing).
    bpt.ensure_pixel_cache_current(
        paths,
        "baseline",
        pixel_index_path=index_path,
        pixel_ids=pixel_ids,
        logger=logging.getLogger("t"),
    )


def test_ensure_pixel_cache_absent_raises(tmp_path: Path) -> None:
    paths = get_project_paths(tmp_path)
    index_path = tmp_path / "pixel_index.parquet"
    pd.DataFrame({"pixel_id": pd.Series([0], dtype="int64")}).to_parquet(index_path)
    with pytest.raises(ValueError, match="absent"):
        bpt.ensure_pixel_cache_current(
            paths,
            "baseline",
            pixel_index_path=index_path,
            pixel_ids=np.array([0], dtype=np.int64),
            logger=logging.getLogger("t"),
        )


def test_ensure_pixel_cache_stale_array_raises(tmp_path: Path) -> None:
    # A pre-lineage cache whose matrix shape disagrees with the index is not
    # adopted; it raises so the caller rebuilds with build_pixel_table.
    paths = get_project_paths(tmp_path)
    pixel_ids = np.array([10, 11], dtype=np.int64)
    _, index_path = _write_pre_lineage_pixel_cache(paths, n_rows=3, pixel_ids=pixel_ids)
    with pytest.raises(ValueError, match="stale or invalid"):
        bpt.ensure_pixel_cache_current(
            paths,
            "baseline",
            pixel_index_path=index_path,
            pixel_ids=pixel_ids,
            logger=logging.getLogger("t"),
        )


def test_corine_unclassified_when_column_absent() -> None:
    parcels = _parcels().drop(columns=["corine_forest_type"])
    index = bpt.build_pixel_index(parcels, _ref())
    assert set(index["forest_type_corine"]) == {"unclassified"}
    # The inventory column is unaffected (still derived from class).
    assert set(index["forest_type_inventory"]) == {"coniferous", "broadleaf"}


def test_inventory_falls_back_to_composition_when_class_absent() -> None:
    parcels = _parcels().drop(columns=["class"])
    parcels["final_composition"] = ["10FA", "10MO"]  # pure beech, pure spruce
    index = bpt.build_pixel_index(parcels, _ref())
    parcel_a = index[index["parcel_id"] == 1]
    parcel_b = index[index["parcel_id"] == 2]
    assert set(parcel_a["forest_type_inventory"]) == {"broadleaf"}
    assert set(parcel_b["forest_type_inventory"]) == {"coniferous"}


def test_inventory_unclassified_when_no_source() -> None:
    parcels = _parcels().drop(columns=["class"])
    index = bpt.build_pixel_index(parcels, _ref())
    assert set(index["forest_type_inventory"]) == {"unclassified"}
