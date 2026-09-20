"""Synthetic unit checks for scripts/build_patch_cache.py.

A tiny grid memmap and pixel index exercise the streamed cache build, the
manifest, idempotency, the force rebuild and the missing-grid guard without
touching the real (tens-of-gigabytes) caches. An equivalence check confirms the
cached-and-standardised patches reproduce the earlier mean-fill-then-z-score
extraction exactly, for interior and edge pixels.
"""

import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from scripts import build_patch_cache as bpc
from utils.cache import CACHE_MANIFEST_VERSION
from utils.patches import PatchDataset, compute_band_stats, extract_raw_patch
from utils.paths import ProjectPaths, get_project_paths
from utils.terminology import FEATURE_SET_BANDS, NODATA

_FS = "baseline"  # smallest real feature set, so the band-count guard passes
_N_BANDS = len(FEATURE_SET_BANDS[_FS])
_GRID_H, _GRID_W = 6, 8
_PATCH = 3
# Interior, two corners (edges) and the NoData-centre pixel, in cache order.
_PIXELS = [(1, 1), (2, 3), (0, 0), (_GRID_H - 1, _GRID_W - 1), (3, 5), (2, 4)]


def _logger() -> logging.Logger:
    return logging.getLogger("test_build_patch_cache")


def _make_grid(seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    grid = rng.standard_normal((_N_BANDS, _GRID_H, _GRID_W)).astype(np.float32)
    grid[0, 2, 3] = NODATA  # an in-grid NoData cell (also a patch centre)
    return grid


def _write_inputs(paths: ProjectPaths, grid: np.ndarray) -> None:
    grid_dir = paths.cache / "grid_memmap"
    grid_dir.mkdir(parents=True, exist_ok=True)
    np.save(grid_dir / f"{_FS}.npy", grid)
    (grid_dir / f"{_FS}.json").write_text(
        json.dumps(
            {
                "cache_manifest_version": CACHE_MANIFEST_VERSION,
                "feature_set": _FS,
                "shape": list(grid.shape),
                "dtype": "float32",
            }
        ),
        encoding="utf-8",
    )
    index_dir = paths.results / "main_nested_cv"
    index_dir.mkdir(parents=True, exist_ok=True)
    rows = [
        {
            "pixel_id": r * _GRID_W + c,
            "row": r,
            "col": c,
            "parcel_id": 1,
            "fold_id": 1,
            "ogf": 1,
            "forest_type_corine": "broadleaf",
        }
        for r, c in _PIXELS
    ]
    pd.DataFrame(rows).to_parquet(index_dir / "pixel_index.parquet")


@pytest.fixture(autouse=True)
def _keep_grid_build_local(monkeypatch: pytest.MonkeyPatch) -> None:
    """Use each test's synthetic grid rather than invoking the grid builder."""
    monkeypatch.setattr(
        bpc,
        "ensure_grid_cache",
        lambda feature_set, *, paths, logger: paths.cache / "grid_memmap" / f"{feature_set}.npy",
    )


def _setup(tmp_path: Path, seed: int = 0) -> tuple[ProjectPaths, np.ndarray]:
    paths = get_project_paths(tmp_path)
    grid = _make_grid(seed)
    _write_inputs(paths, grid)
    return paths, grid


def _reference_standardised(
    grid: np.ndarray, row: int, col: int, mean: np.ndarray, std: np.ndarray
) -> np.ndarray:
    """Independent mean-fill-then-z-score reference (the earlier extraction)."""
    n_bands, height, width = grid.shape
    half = (_PATCH - 1) // 2
    mean_r = mean.reshape(n_bands, 1, 1).astype(np.float32)
    std_r = std.reshape(n_bands, 1, 1).astype(np.float32)
    patch = np.empty((n_bands, _PATCH, _PATCH), dtype=np.float32)
    patch[:] = mean_r
    top, left = row - half, col - half
    g_top, g_bottom = max(0, top), min(height, row + half + 1)
    g_left, g_right = max(0, left), min(width, col + half + 1)
    if g_bottom > g_top and g_right > g_left:
        window = grid[:, g_top:g_bottom, g_left:g_right].astype(np.float32)
        p_top, p_left = g_top - top, g_left - left
        patch[:, p_top : p_top + (g_bottom - g_top), p_left : p_left + (g_right - g_left)] = window
    nodata_cells = patch == np.float32(NODATA)
    if nodata_cells.any():
        mean_full = np.broadcast_to(mean_r, patch.shape)
        patch[nodata_cells] = mean_full[nodata_cells]
    return ((patch - mean_r) / std_r).astype(np.float32)


def test_ensure_builds_cache_and_manifest(tmp_path: Path) -> None:
    paths, grid = _setup(tmp_path)
    out = bpc.ensure_patch_cache(_FS, _PATCH, paths=paths, logger=_logger())

    assert out.is_file()
    cache = np.load(out, mmap_mode="r")
    assert cache.shape == (len(_PIXELS), _N_BANDS, _PATCH, _PATCH)
    assert cache.dtype == np.float32
    # Each cached row equals the direct, sentinel-preserving extraction.
    for i, (r, c) in enumerate(_PIXELS):
        np.testing.assert_array_equal(cache[i], extract_raw_patch(grid, r, c, patch_size=_PATCH))

    manifest = json.loads(out.with_suffix(".json").read_text(encoding="utf-8"))
    assert manifest["feature_set"] == _FS
    assert manifest["patch_size"] == _PATCH
    assert manifest["shape"] == [len(_PIXELS), _N_BANDS, _PATCH, _PATCH]
    assert manifest["dtype"] == "float32"
    assert manifest["nodata"] == NODATA
    assert manifest["n_pixels"] == len(_PIXELS)
    assert manifest["grid_height"] == _GRID_H
    assert manifest["grid_width"] == _GRID_W
    assert manifest["cache_manifest_version"] == CACHE_MANIFEST_VERSION
    assert len(manifest["grid_manifest_signature"]["sha256"]) == 64
    assert len(manifest["pixel_index_signature"]["sha256"]) == 64


def test_cached_standardised_patches_match_reference(tmp_path: Path) -> None:
    # Equivalence lock: building the cache then standardising it in PatchDataset
    # reproduces the earlier on-the-fly extraction byte-for-byte.
    paths, grid = _setup(tmp_path)
    out = bpc.ensure_patch_cache(_FS, _PATCH, paths=paths, logger=_logger())
    cache = np.load(out, mmap_mode="r")
    positions = np.arange(len(_PIXELS))
    mean, std = compute_band_stats(cache, positions)
    dataset = PatchDataset(
        cache,
        positions,
        np.zeros(len(_PIXELS)),
        np.ones(len(_PIXELS)),
        band_means=mean,
        band_stds=std,
    )
    for i, (r, c) in enumerate(_PIXELS):
        patch, _, _ = dataset[i]
        np.testing.assert_array_equal(patch.numpy(), _reference_standardised(grid, r, c, mean, std))


def test_ensure_idempotent_skip(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    paths, _ = _setup(tmp_path)
    bpc.ensure_patch_cache(_FS, _PATCH, paths=paths, logger=_logger())
    # A second call must not rebuild: fail loudly if the builder is invoked.
    monkeypatch.setattr(
        bpc, "build_patch_cache", lambda *a, **k: pytest.fail("rebuilt despite a matching cache")
    )
    with caplog.at_level("INFO", logger="test_build_patch_cache"):
        out = bpc.ensure_patch_cache(_FS, _PATCH, paths=paths, logger=_logger())
    assert out.is_file()
    assert any("skipping" in record.getMessage() for record in caplog.records)


def test_ensure_force_rebuilds(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    paths, _ = _setup(tmp_path)
    bpc.ensure_patch_cache(_FS, _PATCH, paths=paths, logger=_logger())
    calls: list[int] = []
    real = bpc.build_patch_cache

    def spy(*args: object, **kwargs: object) -> dict[str, object]:
        calls.append(1)
        return real(*args, **kwargs)

    monkeypatch.setattr(bpc, "build_patch_cache", spy)
    bpc.ensure_patch_cache(_FS, _PATCH, paths=paths, logger=_logger(), force=True)
    assert calls == [1]  # force rebuilds even though a matching cache exists


def test_ensure_stale_manifest_rebuilds(tmp_path: Path) -> None:
    # A manifest for a different grid size must not be reused.
    paths, _ = _setup(tmp_path)
    out_npy, out_json = bpc._cache_paths(paths, _FS, _PATCH)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_npy.write_bytes(b"stale")
    out_json.write_text(json.dumps({"feature_set": _FS, "patch_size": _PATCH, "shape": [1]}))
    out = bpc.ensure_patch_cache(_FS, _PATCH, paths=paths, logger=_logger())
    cache = np.load(out, mmap_mode="r")
    assert cache.shape == (len(_PIXELS), _N_BANDS, _PATCH, _PATCH)


def test_grid_manifest_change_invalidates_same_shape_cache(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    paths, _ = _setup(tmp_path)
    bpc.ensure_patch_cache(_FS, _PATCH, paths=paths, logger=_logger())
    grid_manifest = paths.cache / "grid_memmap" / f"{_FS}.json"
    grid_manifest.write_text('{"changed": true}\n', encoding="utf-8")
    calls: list[int] = []
    real = bpc.build_patch_cache

    def spy(*args: object, **kwargs: object) -> dict[str, object]:
        calls.append(1)
        return real(*args, **kwargs)

    monkeypatch.setattr(bpc, "build_patch_cache", spy)
    bpc.ensure_patch_cache(_FS, _PATCH, paths=paths, logger=_logger())
    assert calls == [1]


def test_pixel_index_change_invalidates_same_shape_cache(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    paths, _ = _setup(tmp_path)
    bpc.ensure_patch_cache(_FS, _PATCH, paths=paths, logger=_logger())
    index_path = paths.results / "main_nested_cv" / "pixel_index.parquet"
    index = pd.read_parquet(index_path)
    index.loc[[0, 1], ["row", "col"]] = index.loc[[1, 0], ["row", "col"]].to_numpy()
    index.to_parquet(index_path)
    calls: list[int] = []
    real = bpc.build_patch_cache

    def spy(*args: object, **kwargs: object) -> dict[str, object]:
        calls.append(1)
        return real(*args, **kwargs)

    monkeypatch.setattr(bpc, "build_patch_cache", spy)
    bpc.ensure_patch_cache(_FS, _PATCH, paths=paths, logger=_logger())
    assert calls == [1]


def test_failed_patch_rebuild_preserves_existing_cache(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    paths, _ = _setup(tmp_path)
    out = bpc.ensure_patch_cache(_FS, _PATCH, paths=paths, logger=_logger())
    manifest_path = out.with_suffix(".json")
    old_array = out.read_bytes()
    old_manifest = manifest_path.read_bytes()

    def fail_fill(*args: object, **kwargs: object) -> None:
        raise RuntimeError("synthetic patch failure")

    monkeypatch.setattr(bpc, "_fill_chunk", fail_fill)
    with pytest.raises(RuntimeError, match="synthetic patch failure"):
        bpc.ensure_patch_cache(_FS, _PATCH, paths=paths, logger=_logger(), force=True)

    assert out.read_bytes() == old_array
    assert manifest_path.read_bytes() == old_manifest


def test_ensure_adopts_pre_lineage_cache_without_re_extraction(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # A cache built before lineage existed is re-described in place, never rebuilt.
    paths, _ = _setup(tmp_path)
    out = bpc.ensure_patch_cache(_FS, _PATCH, paths=paths, logger=_logger())
    manifest_path = out.with_suffix(".json")
    array_bytes = out.read_bytes()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    legacy = {
        key: value
        for key, value in manifest.items()
        if key not in ("cache_manifest_version", "grid_manifest_signature", "pixel_index_signature")
    }
    manifest_path.write_text(json.dumps(legacy), encoding="utf-8")

    monkeypatch.setattr(
        bpc, "build_patch_cache", lambda *a, **k: pytest.fail("re-extracted a valid cache")
    )
    again = bpc.ensure_patch_cache(_FS, _PATCH, paths=paths, logger=_logger())
    assert again.read_bytes() == array_bytes  # adopted, not rebuilt
    restored = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert restored["cache_manifest_version"] == CACHE_MANIFEST_VERSION
    assert len(restored["grid_manifest_signature"]["sha256"]) == 64
    assert len(restored["pixel_index_signature"]["sha256"]) == 64
    # The adopted manifest is current, so a further call simply skips (no rebuild).
    bpc.ensure_patch_cache(_FS, _PATCH, paths=paths, logger=_logger())


def test_ensure_missing_grid_raises(tmp_path: Path) -> None:
    paths = get_project_paths(tmp_path)
    with pytest.raises(ValueError, match="Missing grid memmap"):
        bpc.ensure_patch_cache(_FS, _PATCH, paths=paths, logger=_logger())


def test_ensure_missing_index_raises(tmp_path: Path) -> None:
    paths = get_project_paths(tmp_path)
    grid_dir = paths.cache / "grid_memmap"
    grid_dir.mkdir(parents=True, exist_ok=True)
    np.save(grid_dir / f"{_FS}.npy", _make_grid())
    with pytest.raises(ValueError, match="Missing pixel index"):
        bpc.ensure_patch_cache(_FS, _PATCH, paths=paths, logger=_logger())


def test_ensure_wrong_band_count_raises(tmp_path: Path) -> None:
    paths = get_project_paths(tmp_path)
    grid_dir = paths.cache / "grid_memmap"
    grid_dir.mkdir(parents=True, exist_ok=True)
    # One extra band relative to the feature set's declared count.
    np.save(grid_dir / f"{_FS}.npy", np.zeros((_N_BANDS + 1, _GRID_H, _GRID_W), dtype=np.float32))
    index_dir = paths.results / "main_nested_cv"
    index_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([{"row": 0, "col": 0}]).to_parquet(index_dir / "pixel_index.parquet")
    with pytest.raises(ValueError, match="bands but feature set"):
        bpc.ensure_patch_cache(_FS, _PATCH, paths=paths, logger=_logger())


def test_main_builds_requested(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    paths, _ = _setup(tmp_path)
    monkeypatch.setattr(bpc, "get_project_paths", lambda: paths)
    code = bpc.main(["--feature-sets", _FS, "--patch-sizes", str(_PATCH)])
    assert code == 0
    out_npy, out_json = bpc._cache_paths(paths, _FS, _PATCH)
    assert out_npy.is_file() and out_json.is_file()
