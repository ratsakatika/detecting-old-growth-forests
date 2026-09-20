"""Unit tests for utils.patches (cached patch supply)."""

import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch
from torch.utils.data import DataLoader

from utils import patches
from utils.patches import (
    PatchDataset,
    build_patch_loaders,
    compute_band_stats,
    extract_raw_patch,
)
from utils.terminology import NODATA


def _grid(height: int = 6, width: int = 7) -> np.ndarray:
    """A 2-band float32 grid: band 0 = 0..H*W-1, band 1 = band 0 + 100."""
    base = np.arange(height * width, dtype=np.float32).reshape(height, width)
    return np.stack([base, base + 100.0]).astype(np.float32)


def _build_cache(
    grid: np.ndarray, rows: np.ndarray, cols: np.ndarray, patch_size: int
) -> np.ndarray:
    """Raw patch cache (n, n_bands, P, P) for the given centre pixels, in order."""
    rows = np.asarray(rows, dtype=np.int64)
    cols = np.asarray(cols, dtype=np.int64)
    cache = np.empty((rows.shape[0], grid.shape[0], patch_size, patch_size), dtype=np.float32)
    for i in range(rows.shape[0]):
        cache[i] = extract_raw_patch(grid, int(rows[i]), int(cols[i]), patch_size=patch_size)
    return cache


def _cache_from_index(grid: np.ndarray, index: pd.DataFrame, patch_size: int) -> np.ndarray:
    """Raw patch cache aligned row-for-row with ``index`` (using its row/col)."""
    return _build_cache(grid, index["row"].to_numpy(), index["col"].to_numpy(), patch_size)


def _reference_standardised(
    grid: np.ndarray,
    row: int,
    col: int,
    patch_size: int,
    mean: np.ndarray,
    std: np.ndarray,
    nodata: float = NODATA,
) -> np.ndarray:
    """Independent mean-fill-then-z-score reference (the earlier extraction)."""
    n_bands, height, width = grid.shape
    half = (patch_size - 1) // 2
    mean_r = mean.reshape(n_bands, 1, 1).astype(np.float32)
    std_r = std.reshape(n_bands, 1, 1).astype(np.float32)
    patch = np.empty((n_bands, patch_size, patch_size), dtype=np.float32)
    patch[:] = mean_r
    top, left = row - half, col - half
    g_top, g_bottom = max(0, top), min(height, row + half + 1)
    g_left, g_right = max(0, left), min(width, col + half + 1)
    if g_bottom > g_top and g_right > g_left:
        window = grid[:, g_top:g_bottom, g_left:g_right].astype(np.float32)
        p_top, p_left = g_top - top, g_left - left
        patch[:, p_top : p_top + (g_bottom - g_top), p_left : p_left + (g_right - g_left)] = window
    nodata_cells = patch == np.float32(nodata)
    if nodata_cells.any():
        mean_full = np.broadcast_to(mean_r, patch.shape)
        patch[nodata_cells] = mean_full[nodata_cells]
    return ((patch - mean_r) / std_r).astype(np.float32)


# --------------------------------------------------------------------------- #
# Import side effects.
# --------------------------------------------------------------------------- #


def test_import_writes_no_files(tmp_path: Path) -> None:
    repo_root = Path(patches.__file__).resolve().parents[1]
    result = subprocess.run(
        [sys.executable, "-c", "import utils.patches"],
        cwd=tmp_path,
        env={**os.environ, "PYTHONPATH": str(repo_root)},
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert list(tmp_path.iterdir()) == []


# --------------------------------------------------------------------------- #
# extract_raw_patch (sentinel-preserving, fold-independent).
# --------------------------------------------------------------------------- #


def test_extract_raw_patch_interior_copies_grid() -> None:
    grid = _grid()
    patch = extract_raw_patch(grid, 2, 3, patch_size=3)
    assert patch.shape == (2, 3, 3) and patch.dtype == np.float32
    np.testing.assert_array_equal(patch, grid[:, 1:4, 2:5].astype(np.float32))


def test_extract_raw_patch_edge_fills_sentinel() -> None:
    grid = _grid()
    patch = extract_raw_patch(grid, 0, 0, patch_size=3)
    # The top and left ring fall outside the grid: kept as the NoData sentinel.
    assert patch[0, 0, 0] == NODATA
    assert patch[0, 0, 1] == NODATA
    assert patch[0, 1, 0] == NODATA
    # The in-grid centre carries the raw value, unmodified.
    assert patch[0, 1, 1] == grid[0, 0, 0]


def test_extract_raw_patch_preserves_nodata_cell() -> None:
    grid = _grid()
    grid[0, 2, 3] = NODATA
    patch = extract_raw_patch(grid, 2, 3, patch_size=3)
    assert patch[0, 1, 1] == NODATA  # in-grid NoData kept, not mean-filled


# --------------------------------------------------------------------------- #
# compute_band_stats.
# --------------------------------------------------------------------------- #


def test_compute_band_stats_matches_centre_pixels() -> None:
    grid = _grid()
    rows = np.array([1, 2, 3], dtype=np.int64)
    cols = np.array([1, 2, 3], dtype=np.int64)
    cache = _build_cache(grid, rows, cols, patch_size=3)
    mean, std = compute_band_stats(cache, np.arange(3))
    centres = grid[0, rows, cols]
    assert mean.dtype == np.float32 and std.dtype == np.float32
    assert mean[0] == pytest.approx(centres.mean())
    assert mean[1] == pytest.approx(centres.mean() + 100.0)
    assert std[0] == pytest.approx(centres.std())


def test_compute_band_stats_floors_constant_band() -> None:
    height, width = 6, 7
    constant = np.full((height, width), 5.0, dtype=np.float32)
    grid = np.stack([constant, np.arange(height * width, dtype=np.float32).reshape(height, width)])
    cache = _build_cache(
        grid, np.array([1, 2, 3], dtype=np.int64), np.array([1, 2, 3], dtype=np.int64), patch_size=3
    )
    _, std = compute_band_stats(cache, np.arange(3))
    assert std[0] == 1.0  # zero variance -> divisor 1, not infinity


def test_compute_band_stats_excludes_nodata() -> None:
    grid = _grid()
    grid[0, 2, 2] = NODATA  # one centre pixel is NoData in band 0
    rows = np.array([1, 2, 3], dtype=np.int64)
    cols = np.array([1, 2, 3], dtype=np.int64)
    cache = _build_cache(grid, rows, cols, patch_size=3)
    mean, _ = compute_band_stats(cache, np.arange(3))
    assert mean[0] == pytest.approx(np.array([8.0, 24.0]).mean())  # 16 excluded


def test_compute_band_stats_rejects_non_4d_cache() -> None:
    with pytest.raises(ValueError, match="patch_cache must be"):
        compute_band_stats(np.zeros((2, 3, 3), dtype=np.float32), np.array([0]))


# --------------------------------------------------------------------------- #
# PatchDataset.
# --------------------------------------------------------------------------- #


def _dataset(patch_size: int = 3) -> PatchDataset:
    grid = _grid()
    rows = np.array([1, 2, 3, 4], dtype=np.int64)
    cols = np.array([1, 2, 3, 4], dtype=np.int64)
    labels = np.array([0, 1, 1, 0], dtype=np.int64)
    weights = np.array([1.0, 2.0, 0.5, 1.0], dtype=np.float64)
    cache = _build_cache(grid, rows, cols, patch_size)
    mean, std = compute_band_stats(cache, np.arange(4))
    return PatchDataset(
        cache,
        np.arange(4),
        labels,
        weights,
        band_means=mean,
        band_stds=std,
    )


def test_patch_dataset_length_and_item_shapes() -> None:
    dataset = _dataset()
    assert len(dataset) == 4
    patch, label, weight = dataset[1]
    assert tuple(patch.shape) == (2, 3, 3)
    assert patch.dtype == torch.float32
    assert label.item() == pytest.approx(1.0)
    assert weight.item() == pytest.approx(2.0)


def test_patch_dataset_batches_with_loader() -> None:
    dataset = _dataset()
    loader = DataLoader(dataset, batch_size=4, shuffle=False)
    patches_b, labels_b, weights_b = next(iter(loader))
    assert tuple(patches_b.shape) == (4, 2, 3, 3)
    assert tuple(labels_b.shape) == (4,)
    assert tuple(weights_b.shape) == (4,)
    np.testing.assert_allclose(weights_b.numpy(), [1.0, 2.0, 0.5, 1.0])


def test_patch_dataset_standardises_and_zeros_sentinels() -> None:
    grid = _grid()
    grid[0, 2, 3] = NODATA  # NoData centre at pixel (2, 3)
    rows = np.array([2, 0], dtype=np.int64)  # a NoData centre, then a corner (edge)
    cols = np.array([3, 0], dtype=np.int64)
    cache = _build_cache(grid, rows, cols, patch_size=3)
    mean = np.array([16.0, 116.0], dtype=np.float32)
    std = np.array([1.0, 1.0], dtype=np.float32)
    dataset = PatchDataset(cache, np.arange(2), [0, 1], [1.0, 1.0], band_means=mean, band_stds=std)
    nodata_patch, _, _ = dataset[0]
    assert nodata_patch[0, 1, 1].item() == 0.0  # NoData centre -> neutral 0
    edge_patch, _, _ = dataset[1]
    assert edge_patch[0, 0, 0].item() == 0.0  # out-of-grid -> neutral 0
    assert edge_patch[0, 1, 1].item() == pytest.approx(grid[0, 0, 0] - mean[0])


def test_patch_dataset_matches_reference_standardisation() -> None:
    # Byte-identity lock: the cache-plus-dataset path must reproduce the earlier
    # mean-fill-then-z-score extraction exactly, for interior, edge and NoData.
    grid = _grid()
    grid[0, 3, 3] = NODATA
    rows = np.array([2, 0, 3], dtype=np.int64)  # interior, corner edge, NoData centre
    cols = np.array([3, 0, 3], dtype=np.int64)
    cache = _build_cache(grid, rows, cols, patch_size=3)
    mean, std = compute_band_stats(cache, np.arange(3))
    dataset = PatchDataset(
        cache, np.arange(3), [0, 1, 0], [1.0, 1.0, 1.0], band_means=mean, band_stds=std
    )
    for i in range(3):
        patch, _, _ = dataset[i]
        reference = _reference_standardised(grid, int(rows[i]), int(cols[i]), 3, mean, std)
        np.testing.assert_array_equal(patch.numpy(), reference)


def test_patch_dataset_rejects_non_4d_cache() -> None:
    with pytest.raises(ValueError, match="patch_cache must be"):
        PatchDataset(
            np.zeros((4, 4, 4), dtype=np.float32),
            [0],
            [0],
            [1.0],
            band_means=np.zeros(4, dtype=np.float32),
            band_stds=np.ones(4, dtype=np.float32),
        )


def test_patch_dataset_rejects_length_mismatch() -> None:
    grid = _grid()
    cache = _build_cache(grid, np.array([1]), np.array([1]), patch_size=3)
    mean, std = compute_band_stats(cache, np.arange(1))
    with pytest.raises(ValueError, match="equal length"):
        PatchDataset(cache, [0, 1], [0], [1.0], band_means=mean, band_stds=std)


def test_patch_dataset_rejects_wrong_stats_length() -> None:
    grid = _grid()
    cache = _build_cache(grid, np.array([1]), np.array([1]), patch_size=3)
    with pytest.raises(ValueError, match="length 2"):
        PatchDataset(
            cache,
            [0],
            [0],
            [1.0],
            band_means=np.array([0.0], dtype=np.float32),
            band_stds=np.array([1.0], dtype=np.float32),
        )


# --------------------------------------------------------------------------- #
# build_patch_loaders.
# --------------------------------------------------------------------------- #


def _loader_problem(patch_size: int = 3) -> tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    """A 2-band 8x8 grid, its patch cache and a pixel index spanning two folds.

    Band 0 holds ``pixel_id = row * 8 + col`` and band 1 holds ``pixel_id + 1000``,
    so a patch centre's value reveals which pixel it was cut from. Each fold has
    one old-growth and one non-old-growth parcel; fold 1 carries one invalid pixel,
    and each fold carries one edge pixel whose patch would overrun the grid.
    """
    width = 8
    base = np.arange(width * width, dtype=np.float32).reshape(width, width)
    grid = np.stack([base, base + 1000.0]).astype(np.float32)
    pixels = [
        # row, col, fold, parcel, ogf, valid
        (1, 1, 1, 100, 1, True),
        (1, 2, 1, 100, 1, True),
        (2, 1, 1, 100, 1, True),
        (2, 2, 1, 100, 1, True),
        (3, 3, 1, 100, 1, False),  # invalid
        (4, 4, 1, 101, 0, True),
        (4, 5, 1, 101, 0, True),
        (5, 4, 1, 101, 0, True),
        (0, 4, 1, 101, 0, True),  # edge (row 0)
        (1, 5, 2, 200, 1, True),
        (2, 5, 2, 200, 1, True),
        (3, 5, 2, 200, 1, True),
        (5, 1, 2, 201, 0, True),
        (5, 2, 2, 201, 0, True),
        (6, 6, 2, 201, 0, True),
        (7, 6, 2, 201, 0, True),  # edge (row 7)
    ]
    rows = [
        {
            "pixel_id": row * width + col,
            "row": row,
            "col": col,
            "parcel_id": parcel,
            "fold_id": fold,
            "ogf": ogf,
            "forest_type_corine": "broadleaf" if ogf else "coniferous",
            "valid": valid,
        }
        for row, col, fold, parcel, ogf, valid in pixels
    ]
    index = pd.DataFrame(rows)
    cache = _cache_from_index(grid, index, patch_size)
    return grid, cache, index


def test_build_patch_loaders_shapes_and_val_index() -> None:
    grid, cache, index = _loader_problem()
    train_loader, val_loader, val_index = build_patch_loaders(
        cache,
        index,
        grid_shape=grid.shape[1:],
        train_folds=(1,),
        val_fold=2,
        sampling_mode="none",
        n_per_parcel=100,
        patch_size=3,
        batch_size=4,
        rng=np.random.default_rng(0),
        num_workers=0,
        seed=0,
    )
    # Training: fold 1 valid, non-edge pixels (parcel 100 has 4 after the invalid
    # pixel; parcel 101 has 3 after the edge pixel) -> 7.
    assert len(train_loader.dataset) == 7
    # Validation: fold 2 valid, non-edge pixels (3 + 3) -> 6.
    assert len(val_loader.dataset) == 6
    patches_b, labels_b, weights_b = next(iter(train_loader))
    assert tuple(patches_b.shape[1:]) == (2, 3, 3)
    assert patches_b.dtype == torch.float32
    assert tuple(labels_b.shape) == (4,) and tuple(weights_b.shape) == (4,)
    # Val pixel index carries pixel_id, parcel_id, y_true in pixel order.
    assert list(val_index.columns) == ["pixel_id", "parcel_id", "y_true"]
    assert val_index["pixel_id"].tolist() == [13, 21, 29, 41, 42, 54]
    assert val_index["parcel_id"].tolist() == [200, 200, 200, 201, 201, 201]
    assert val_index["y_true"].tolist() == [1, 1, 1, 0, 0, 0]
    # The (7, 6) edge pixel (pixel_id 62) never enters the validation set.
    assert 62 not in val_index["pixel_id"].tolist()


def test_build_patch_loaders_drops_only_singleton_final_training_batch() -> None:
    grid, cache, index = _loader_problem()
    train_loader, _, _ = build_patch_loaders(
        cache,
        index,
        grid_shape=grid.shape[1:],
        train_folds=(1,),
        val_fold=2,
        sampling_mode="none",
        n_per_parcel=100,
        patch_size=3,
        batch_size=3,
        rng=np.random.default_rng(0),
        num_workers=0,
        seed=0,
    )

    assert len(train_loader.dataset) == 7
    assert train_loader.drop_last is True
    assert [len(batch[1]) for batch in train_loader] == [3, 3]


def test_build_patch_loaders_keeps_non_singleton_remainder() -> None:
    grid, cache, index = _loader_problem()
    train_loader, _, _ = build_patch_loaders(
        cache,
        index,
        grid_shape=grid.shape[1:],
        train_folds=(1,),
        val_fold=2,
        sampling_mode="none",
        n_per_parcel=100,
        patch_size=3,
        batch_size=4,
        rng=np.random.default_rng(0),
        num_workers=0,
        seed=0,
    )

    assert train_loader.drop_last is False
    assert sorted(len(batch[1]) for batch in train_loader) == [3, 4]


def test_build_patch_loaders_rejects_batch_size_one() -> None:
    grid, cache, index = _loader_problem()
    with pytest.raises(ValueError, match="batch_size must be at least 2"):
        build_patch_loaders(
            cache,
            index,
            grid_shape=grid.shape[1:],
            train_folds=(1,),
            val_fold=2,
            sampling_mode="none",
            n_per_parcel=100,
            patch_size=3,
            batch_size=1,
            rng=np.random.default_rng(0),
            num_workers=0,
            seed=0,
        )


def test_build_patch_loaders_rejects_single_training_pixel() -> None:
    grid, cache, index = _loader_problem()
    train_rows = index.index[index["fold_id"] == 1]
    index.loc[train_rows, "valid"] = False
    index.loc[train_rows[0], "valid"] = True

    with pytest.raises(ValueError, match="At least two training pixels"):
        build_patch_loaders(
            cache,
            index,
            grid_shape=grid.shape[1:],
            train_folds=(1,),
            val_fold=2,
            sampling_mode="none",
            n_per_parcel=100,
            patch_size=3,
            batch_size=4,
            rng=np.random.default_rng(0),
            num_workers=0,
            seed=0,
        )


def test_build_patch_loaders_centre_maps_to_row_col(monkeypatch: pytest.MonkeyPatch) -> None:
    grid, cache, index = _loader_problem()
    n_bands = grid.shape[0]
    # Identity standardisation so each patch carries the raw grid values.
    monkeypatch.setattr(
        patches,
        "compute_band_stats",
        lambda *a, **k: (np.zeros(n_bands, dtype=np.float32), np.ones(n_bands, dtype=np.float32)),
    )
    _, val_loader, val_index = build_patch_loaders(
        cache,
        index,
        grid_shape=grid.shape[1:],
        train_folds=(1,),
        val_fold=2,
        sampling_mode="none",
        n_per_parcel=100,
        patch_size=3,
        batch_size=64,
        rng=np.random.default_rng(0),
        num_workers=0,
        seed=0,
    )
    patches_b, _, _ = next(iter(val_loader))
    # Band 0 centre equals pixel_id (= row * 8 + col) of each validation pixel,
    # confirming the position-to-pixel mapping and the preserved pixel order.
    centre_band0 = patches_b[:, 0, 1, 1].numpy()
    np.testing.assert_array_equal(centre_band0, np.array(val_index["pixel_id"], dtype=np.float32))


def test_build_patch_loaders_drops_edge_pixels(caplog: pytest.LogCaptureFixture) -> None:
    grid, cache, index = _loader_problem()
    with caplog.at_level("INFO", logger="utils.patches"):
        _, _, val_index = build_patch_loaders(
            cache,
            index,
            grid_shape=grid.shape[1:],
            train_folds=(1,),
            val_fold=2,
            sampling_mode="none",
            n_per_parcel=100,
            patch_size=3,
            batch_size=4,
            rng=np.random.default_rng(0),
            num_workers=0,
            seed=0,
        )
    messages = [record.getMessage() for record in caplog.records]
    assert any("training edge" in message for message in messages)
    assert any("validation edge" in message for message in messages)
    assert 62 not in val_index["pixel_id"].tolist()


def test_build_patch_loaders_standardisation_uses_training_stats() -> None:
    grid, cache, index = _loader_problem()
    _, val_loader, val_index = build_patch_loaders(
        cache,
        index,
        grid_shape=grid.shape[1:],
        train_folds=(1,),
        val_fold=2,
        sampling_mode="none",
        n_per_parcel=100,  # keep all training pixels (deterministic train set)
        patch_size=3,
        batch_size=64,
        rng=np.random.default_rng(0),
        num_workers=0,
        seed=0,
    )
    # The training positions are the valid, non-edge fold-1 rows of the index.
    train_positions = np.array([0, 1, 2, 3, 5, 6, 7], dtype=np.int64)
    mean, std = compute_band_stats(cache, train_positions)
    patches_b, _, _ = next(iter(val_loader))
    first_pixel = int(val_index["pixel_id"].iloc[0])  # 13 -> (1, 5)
    row, col = divmod(first_pixel, grid.shape[2])
    expected = (grid[:, row, col] - mean) / std
    np.testing.assert_allclose(patches_b[0, :, 1, 1].numpy(), expected, rtol=1e-5, atol=1e-6)


def test_build_patch_loaders_nodata_centre_reads_zero() -> None:
    grid, _, index = _loader_problem()
    grid[0, 1, 5] = NODATA  # NoData at validation pixel (1, 5) in band 0
    cache = _cache_from_index(grid, index, 3)  # rebuild the cache from the edited grid
    _, val_loader, val_index = build_patch_loaders(
        cache,
        index,
        grid_shape=grid.shape[1:],
        train_folds=(1,),
        val_fold=2,
        sampling_mode="none",
        n_per_parcel=100,
        patch_size=3,
        batch_size=64,
        rng=np.random.default_rng(0),
        num_workers=0,
        seed=0,
    )
    patches_b, _, _ = next(iter(val_loader))
    assert int(val_index["pixel_id"].iloc[0]) == 13
    # The NoData centre is neutral-filled and reads as 0 after standardisation.
    assert patches_b[0, 0, 1, 1].item() == 0.0


def test_build_patch_loaders_caps_training_per_parcel() -> None:
    grid, cache, index = _loader_problem()
    train_loader, _, _ = build_patch_loaders(
        cache,
        index,
        grid_shape=grid.shape[1:],
        train_folds=(1,),
        val_fold=2,
        sampling_mode="none",
        n_per_parcel=2,  # cap each parcel at 2 training pixels
        patch_size=3,
        batch_size=4,
        rng=np.random.default_rng(0),
        num_workers=0,
        seed=0,
    )
    dataset = train_loader.dataset
    pixel_ids = index["pixel_id"].to_numpy()[dataset._positions]
    parcel_of = dict(zip(index["pixel_id"], index["parcel_id"], strict=True))
    counts: dict[int, int] = {}
    for pixel_id in pixel_ids.tolist():
        parcel = int(parcel_of[int(pixel_id)])
        counts[parcel] = counts.get(parcel, 0) + 1
    # sample_train_rows reuse caps each parcel; edge drop can only lower a count.
    assert set(counts) <= {100, 101}
    assert all(count <= 2 for count in counts.values())


def test_build_patch_loaders_is_deterministic_for_seed() -> None:
    grid, cache, index = _loader_problem()
    pixel_ids = index["pixel_id"].to_numpy()

    def run() -> tuple[list[int], list[int]]:
        train_loader, _, val_index = build_patch_loaders(
            cache,
            index,
            grid_shape=grid.shape[1:],
            train_folds=(1,),
            val_fold=2,
            sampling_mode="none",
            n_per_parcel=2,
            patch_size=3,
            batch_size=4,
            rng=np.random.default_rng(7),
            num_workers=0,
            seed=11,
        )
        dataset = train_loader.dataset
        train_ids = sorted(pixel_ids[dataset._positions].tolist())
        return train_ids, val_index["pixel_id"].tolist()

    assert run() == run()
