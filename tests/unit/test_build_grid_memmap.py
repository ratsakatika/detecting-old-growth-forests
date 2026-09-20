"""Synthetic unit checks for scripts/build_grid_memmap.py.

A tiny stack co-registered to a small reference grid exercises the band-by-band
memmap conversion, the manifest contents and every validation guard without
touching the real (multi-gigabyte) stacks.
"""

import json
import logging
from pathlib import Path

import numpy as np
import pytest
import rasterio
from affine import Affine
from rasterio.crs import CRS

from scripts import build_grid_memmap as bgm
from utils import terminology
from utils.cache import CACHE_MANIFEST_VERSION
from utils.paths import get_project_paths
from utils.raster_io import ReferenceGrid
from utils.terminology import FEATURE_SET_BANDS, NODATA

# Use a real feature set so the band-count check passes; baseline is the smallest.
_FEATURE_SET = "baseline"
_N_BANDS = len(FEATURE_SET_BANDS[_FEATURE_SET])
_HEIGHT = 4
_WIDTH = 5
# 10 m EPSG:3035 grid, origin at (0, 40); a matching reference grid for every test.
_TRANSFORM = Affine(10.0, 0.0, 0.0, 0.0, -10.0, 40.0)
# A 20 m grid: same origin and CRS, wrong pixel size.
_TRANSFORM_20M = Affine(20.0, 0.0, 0.0, 0.0, -20.0, 40.0)
# The cell forced to the NoData sentinel (in band 0).
_NODATA_ROW, _NODATA_COL = 1, 2


def _ref() -> ReferenceGrid:
    return ReferenceGrid(
        crs=CRS.from_epsg(3035), transform=_TRANSFORM, width=_WIDTH, height=_HEIGHT
    )


def _write_stack(
    path: Path,
    *,
    crs: CRS,
    transform: Affine,
    width: int,
    height: int,
    n_bands: int,
    nodata_cell: tuple[int, int] | None,
) -> np.ndarray:
    """Write a synthetic float32 stack; band b holds base values offset by 100*b."""
    base = np.arange(height * width, dtype=np.float32).reshape(height, width)
    data = np.stack([base + 100.0 * band for band in range(n_bands)]).astype(np.float32)
    if nodata_cell is not None:
        row, col = nodata_cell
        data[0, row, col] = NODATA
    profile = {
        "driver": "GTiff",
        "height": height,
        "width": width,
        "count": n_bands,
        "dtype": "float32",
        "crs": crs,
        "transform": transform,
        "nodata": NODATA,
    }
    with rasterio.open(path, "w", **profile) as dst:
        dst.write(data)
    return data


def test_convert_writes_memmap_and_manifest(tmp_path: Path) -> None:
    stack = tmp_path / "baseline_3035_10m.tif"
    data = _write_stack(
        stack,
        crs=CRS.from_epsg(3035),
        transform=_TRANSFORM,
        width=_WIDTH,
        height=_HEIGHT,
        n_bands=_N_BANDS,
        nodata_cell=(_NODATA_ROW, _NODATA_COL),
    )
    out_npy = tmp_path / "baseline.npy"
    out_json = tmp_path / "baseline.json"

    manifest = bgm.convert_stack_to_memmap(
        stack, out_npy, out_json, feature_set=_FEATURE_SET, reference_grid=_ref()
    )

    array = np.load(out_npy, mmap_mode="r")
    assert array.shape == (_N_BANDS, _HEIGHT, _WIDTH)
    assert array.dtype == np.float32
    # Values are copied verbatim, including the preserved NoData sentinel.
    np.testing.assert_array_equal(array[:], data)
    assert array[0, _NODATA_ROW, _NODATA_COL] == NODATA

    assert manifest["shape"] == [_N_BANDS, _HEIGHT, _WIDTH]
    assert manifest["n_bands"] == _N_BANDS
    assert manifest["dtype"] == "float32"
    assert manifest["nodata"] == NODATA
    assert manifest["crs"] == terminology.CRS
    assert manifest["cache_manifest_version"] == CACHE_MANIFEST_VERSION
    assert len(manifest["source_signature"]["sha256"]) == 64
    assert manifest["reference_grid"]["width"] == _WIDTH

    on_disk = json.loads(out_json.read_text(encoding="utf-8"))
    assert on_disk["shape"] == [_N_BANDS, _HEIGHT, _WIDTH]
    assert on_disk["n_bands"] == _N_BANDS
    assert on_disk["dtype"] == "float32"
    assert on_disk["crs"] == terminology.CRS
    assert bgm._manifest_matches_source(
        out_npy,
        out_json,
        stack,
        feature_set=_FEATURE_SET,
        reference_grid=_ref(),
    )


def test_same_shape_source_change_invalidates_cache(tmp_path: Path) -> None:
    stack = tmp_path / "baseline_3035_10m.tif"
    _write_stack(
        stack,
        crs=CRS.from_epsg(3035),
        transform=_TRANSFORM,
        width=_WIDTH,
        height=_HEIGHT,
        n_bands=_N_BANDS,
        nodata_cell=None,
    )
    out_npy = tmp_path / "baseline.npy"
    out_json = tmp_path / "baseline.json"
    bgm.convert_stack_to_memmap(
        stack, out_npy, out_json, feature_set=_FEATURE_SET, reference_grid=_ref()
    )

    with rasterio.open(stack, "r+") as dataset:
        band = dataset.read(1)
        band[0, 0] += np.float32(1.0)
        dataset.write(band, 1)

    assert not bgm._manifest_matches_source(
        out_npy,
        out_json,
        stack,
        feature_set=_FEATURE_SET,
        reference_grid=_ref(),
    )


def test_ensure_grid_cache_adopts_pre_lineage_without_re_extraction(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # A grid array built before lineage existed is re-described from its unchanged
    # source rather than re-extracted (the multi-gigabyte read is avoided).
    paths = get_project_paths(tmp_path)
    stack = bgm._stack_path(paths, _FEATURE_SET)
    stack.parent.mkdir(parents=True, exist_ok=True)
    _write_stack(
        stack,
        crs=CRS.from_epsg(3035),
        transform=_TRANSFORM,
        width=_WIDTH,
        height=_HEIGHT,
        n_bands=_N_BANDS,
        nodata_cell=None,
    )
    monkeypatch.setattr(bgm, "open_reference_grid", _ref)
    logger = logging.getLogger("test_build_grid_memmap")

    out = bgm.ensure_grid_cache(_FEATURE_SET, paths=paths, logger=logger)
    out_json = out.with_suffix(".json")
    array_bytes = out.read_bytes()
    manifest = json.loads(out_json.read_text(encoding="utf-8"))
    legacy = {
        key: value
        for key, value in manifest.items()
        if key not in ("cache_manifest_version", "source_signature", "reference_grid")
    }
    out_json.write_text(json.dumps(legacy), encoding="utf-8")

    monkeypatch.setattr(
        bgm, "convert_stack_to_memmap", lambda *a, **k: pytest.fail("re-extracted a valid cache")
    )
    again = bgm.ensure_grid_cache(_FEATURE_SET, paths=paths, logger=logger)
    assert again.read_bytes() == array_bytes  # adopted, not rebuilt
    restored = json.loads(out_json.read_text(encoding="utf-8"))
    assert restored["cache_manifest_version"] == CACHE_MANIFEST_VERSION
    assert len(restored["source_signature"]["sha256"]) == 64
    # The adopted manifest is current, so a further call simply skips (no rebuild).
    bgm.ensure_grid_cache(_FEATURE_SET, paths=paths, logger=logger)


def test_failed_rebuild_preserves_existing_cache(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    stack = tmp_path / "baseline_3035_10m.tif"
    _write_stack(
        stack,
        crs=CRS.from_epsg(3035),
        transform=_TRANSFORM,
        width=_WIDTH,
        height=_HEIGHT,
        n_bands=_N_BANDS,
        nodata_cell=None,
    )
    out_npy = tmp_path / "baseline.npy"
    out_json = tmp_path / "baseline.json"
    bgm.convert_stack_to_memmap(
        stack, out_npy, out_json, feature_set=_FEATURE_SET, reference_grid=_ref()
    )
    old_array = out_npy.read_bytes()
    old_manifest = out_json.read_bytes()

    def fail_open_memmap(*args: object, **kwargs: object) -> None:
        raise RuntimeError("synthetic rebuild failure")

    monkeypatch.setattr(bgm.np.lib.format, "open_memmap", fail_open_memmap)
    with pytest.raises(RuntimeError, match="synthetic rebuild failure"):
        bgm.convert_stack_to_memmap(
            stack, out_npy, out_json, feature_set=_FEATURE_SET, reference_grid=_ref()
        )

    assert out_npy.read_bytes() == old_array
    assert out_json.read_bytes() == old_manifest


def test_wrong_band_count_raises(tmp_path: Path) -> None:
    stack = tmp_path / "stack.tif"
    _write_stack(
        stack,
        crs=CRS.from_epsg(3035),
        transform=_TRANSFORM,
        width=_WIDTH,
        height=_HEIGHT,
        n_bands=_N_BANDS + 1,
        nodata_cell=None,
    )
    with pytest.raises(ValueError):
        bgm.convert_stack_to_memmap(
            stack,
            tmp_path / "out.npy",
            tmp_path / "out.json",
            feature_set=_FEATURE_SET,
            reference_grid=_ref(),
        )


def test_grid_mismatch_raises(tmp_path: Path) -> None:
    # Source is one pixel wider than the reference grid.
    stack = tmp_path / "stack.tif"
    _write_stack(
        stack,
        crs=CRS.from_epsg(3035),
        transform=_TRANSFORM,
        width=_WIDTH + 1,
        height=_HEIGHT,
        n_bands=_N_BANDS,
        nodata_cell=None,
    )
    with pytest.raises(ValueError):
        bgm.convert_stack_to_memmap(
            stack,
            tmp_path / "out.npy",
            tmp_path / "out.json",
            feature_set=_FEATURE_SET,
            reference_grid=_ref(),
        )


def test_wrong_crs_raises(tmp_path: Path) -> None:
    stack = tmp_path / "stack.tif"
    _write_stack(
        stack,
        crs=CRS.from_epsg(32635),
        transform=_TRANSFORM,
        width=_WIDTH,
        height=_HEIGHT,
        n_bands=_N_BANDS,
        nodata_cell=None,
    )
    with pytest.raises(ValueError):
        bgm.convert_stack_to_memmap(
            stack,
            tmp_path / "out.npy",
            tmp_path / "out.json",
            feature_set=_FEATURE_SET,
            reference_grid=_ref(),
        )


def test_wrong_resolution_raises(tmp_path: Path) -> None:
    stack = tmp_path / "stack.tif"
    _write_stack(
        stack,
        crs=CRS.from_epsg(3035),
        transform=_TRANSFORM_20M,
        width=_WIDTH,
        height=_HEIGHT,
        n_bands=_N_BANDS,
        nodata_cell=None,
    )
    with pytest.raises(ValueError):
        bgm.convert_stack_to_memmap(
            stack,
            tmp_path / "out.npy",
            tmp_path / "out.json",
            feature_set=_FEATURE_SET,
            reference_grid=_ref(),
        )
