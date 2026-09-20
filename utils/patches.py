"""Cached patch supply for the receptive-field CNN.

The CNN classifies the centre pixel of a small ``(C, P, P)`` patch. Re-cutting
every patch from the full-grid feature array on every epoch is the dominant cost
of CNN training: a single configuration must gather millions of strided windows
per epoch from a multi-gigabyte grid memmap. To remove that cost the patches are
memoised once by ``scripts/build_patch_cache.py`` into a contiguous
``(n_pixels, n_bands, P, P)`` float32 ``.npy`` memmap, row ``i`` holding the raw
patch for row ``i`` of the shared pixel index. Training then reads patches
directly from the cache and never touches the grid.

The memoisation is split so the cache is fold-independent:

  - :func:`extract_raw_patch` cuts one ``(C, P, P)`` window and is
    *sentinel-preserving*: cells outside the grid, and NoData cells inside it,
    keep the NoData sentinel. It applies no per-band mean-fill and no
    standardisation, so its output depends only on the grid, not on any fold.
    This is the function the cache builder calls for every pixel.
  - :func:`compute_band_stats` computes per-band mean and standard deviation from
    the training fold's centre pixels only (read from the cache), so the
    normalisation carries no information from the validation or test pixels.
  - :class:`PatchDataset` reads a raw patch from the cache and applies the
    per-fold standardisation: it z-scores with the fold's band statistics and
    then sets the sentinel cells to zero (equivalently, fills them with the band
    mean before z-scoring, the neutral value that does not bias the
    convolution). This reproduces the earlier on-the-fly extraction exactly while
    reading from the cache rather than re-slicing the grid.

:class:`PatchDataset` does not sample pixels or compute weights; it is given the
already selected cache positions, labels and weights, mirroring how the tabular
pipeline hands :class:`utils.cv.FoldArrays` to the model. The fold-level builder
:func:`build_patch_loaders` performs that selection: it is the patch analogue of
:func:`utils.cv.build_fold_arrays` and reuses the same sampling and validity
semantics so the CNN trains on the very pixels the XGBoost path would.

Importing this module binds names and a module logger only: it performs no
input/output and configures no logging handlers.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from typing import Final

import numpy as np
import numpy.typing as npt
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset

from utils.cv import sample_train_rows
from utils.sampling import build_sample_weights
from utils.terminology import NODATA

# Module logger. No handlers are configured here; callers own logging setup.
logger = logging.getLogger(__name__)

# Smallest standard deviation treated as non-zero; a constant band is left
# unscaled (divisor 1) rather than producing infinities.
_STD_EPS: Final[float] = 1e-12


def extract_raw_patch(
    grid: npt.NDArray[np.float32],
    row: int,
    col: int,
    *,
    patch_size: int,
    nodata: float = NODATA,
) -> npt.NDArray[np.float32]:
    """Cut one raw ``(n_bands, patch_size, patch_size)`` patch, preserving sentinels.

    Fold-independent: cells that fall outside the grid (when the window overruns
    an edge) keep the NoData sentinel, and NoData cells inside the grid keep their
    sentinel value. No per-band mean-fill and no standardisation are applied, so
    the result depends only on ``grid`` and can be cached once and reused by every
    fold. :class:`PatchDataset` applies the per-fold standardisation later.

    Args:
        grid: Feature grid of shape ``(n_bands, height, width)``.
        row: Centre-pixel row index.
        col: Centre-pixel column index.
        patch_size: Odd patch side length.
        nodata: NoData sentinel used to fill out-of-grid cells.

    Returns:
        The raw patch as a ``float32`` array, sentinel-preserving.
    """
    n_bands, height, width = grid.shape
    half = (patch_size - 1) // 2

    patch = np.full((n_bands, patch_size, patch_size), np.float32(nodata), dtype=np.float32)

    top, left = row - half, col - half
    g_top, g_bottom = max(0, top), min(height, row + half + 1)
    g_left, g_right = max(0, left), min(width, col + half + 1)
    if g_bottom > g_top and g_right > g_left:
        window = np.asarray(grid[:, g_top:g_bottom, g_left:g_right], dtype=np.float32)
        p_top, p_left = g_top - top, g_left - left
        patch[:, p_top : p_top + (g_bottom - g_top), p_left : p_left + (g_right - g_left)] = window

    return patch


def compute_band_stats(
    patch_cache: npt.NDArray[np.float32],
    positions: npt.ArrayLike,
    *,
    nodata: float = NODATA,
) -> tuple[npt.NDArray[np.float32], npt.NDArray[np.float32]]:
    """Per-band mean and standard deviation over a set of centre pixels.

    The statistics are fitted on the centre pixel of each selected patch, read
    straight from the cache (``patch_cache[positions, :, radius, radius]``), which
    equals the grid value at that pixel because :func:`extract_raw_patch`
    preserves it. Statistics are computed in float64 for accuracy and returned as
    float32. NoData values are excluded per band. A band whose standard deviation
    is effectively zero is given a standard deviation of 1, so z-scoring leaves it
    at zero rather than dividing by zero.

    Args:
        patch_cache: Patch cache of shape ``(n_pixels, n_bands, patch_size,
            patch_size)``.
        positions: Cache row positions of the centre pixels, length ``n``.
        nodata: NoData sentinel to exclude.

    Returns:
        ``(mean, std)``, each a ``float32`` array of length ``n_bands``.

    Raises:
        ValueError: If ``patch_cache`` is not four-dimensional.
    """
    if patch_cache.ndim != 4:
        raise ValueError(
            "patch_cache must be (n_pixels, n_bands, patch_size, patch_size); "
            f"got shape {patch_cache.shape}."
        )
    positions = np.asarray(positions, dtype=np.int64)
    radius = (patch_cache.shape[-1] - 1) // 2

    # Centre pixel of each selected patch, transposed to (n_bands, n) so the
    # per-band reduction below is identical to the earlier grid-based version.
    centres = np.asarray(patch_cache[positions, :, radius, radius], dtype=np.float64)
    values = centres.T  # (n_bands, n)
    valid = values != float(nodata)
    counts = valid.sum(axis=1)
    safe_counts = np.maximum(counts, 1)

    sums = np.where(valid, values, 0.0).sum(axis=1)
    mean = sums / safe_counts
    sq = np.where(valid, (values - mean[:, None]) ** 2, 0.0).sum(axis=1)
    std = np.sqrt(sq / safe_counts)
    std = np.where(std < _STD_EPS, 1.0, std)
    return mean.astype(np.float32), std.astype(np.float32)


class PatchDataset(Dataset):
    """Cached patches standardised on the fly for a fixed set of centre pixels.

    Each item is ``(patch, label, weight)`` where ``patch`` is a ``float32``
    tensor of shape ``(n_bands, patch_size, patch_size)``, and ``label`` and
    ``weight`` are scalar ``float32`` tensors. The dataset reads the raw patch for
    a position from ``patch_cache`` and standardises it: it z-scores with the
    supplied per-band statistics, then sets the sentinel cells (out-of-grid and
    NoData) to zero, the neutral value after standardisation. Order follows the
    position array, so a non-shuffling loader yields predictions aligned to the
    caller's pixel ids.

    Args:
        patch_cache: Patch cache of shape ``(n_pixels, n_bands, patch_size,
            patch_size)``, typically a memmap from ``numpy.load(..., mmap_mode="r")``.
        positions: Cache row positions for this split, length ``n``.
        labels: Binary labels per pixel, length ``n``.
        weights: Per-pixel sample weights, length ``n``.
        band_means: Per-band means (from :func:`compute_band_stats`).
        band_stds: Per-band standard deviations.
        nodata: NoData sentinel marking cells to zero after standardisation.

    Raises:
        ValueError: If ``patch_cache`` is not four-dimensional, the
            band-statistics lengths do not match the cache, or
            ``positions``/``labels``/``weights`` differ in length.
    """

    def __init__(
        self,
        patch_cache: npt.NDArray[np.float32],
        positions: npt.ArrayLike,
        labels: npt.ArrayLike,
        weights: npt.ArrayLike,
        *,
        band_means: npt.NDArray[np.float32],
        band_stds: npt.NDArray[np.float32],
        nodata: float = NODATA,
    ) -> None:
        """Initialise the dataset and validate the cache, positions and statistics."""
        if patch_cache.ndim != 4:
            raise ValueError(
                "patch_cache must be (n_pixels, n_bands, patch_size, patch_size); "
                f"got shape {patch_cache.shape}."
            )

        self._positions = np.asarray(positions, dtype=np.int64)
        self._labels = np.asarray(labels, dtype=np.float32)
        self._weights = np.asarray(weights, dtype=np.float32)
        lengths = {
            self._positions.shape[0],
            self._labels.shape[0],
            self._weights.shape[0],
        }
        if len(lengths) != 1:
            raise ValueError(
                "positions, labels and weights must have equal length; got "
                f"{self._positions.shape[0]}, {self._labels.shape[0]}, {self._weights.shape[0]}."
            )

        n_bands = int(patch_cache.shape[1])
        band_means = np.asarray(band_means, dtype=np.float32)
        band_stds = np.asarray(band_stds, dtype=np.float32)
        if band_means.shape != (n_bands,) or band_stds.shape != (n_bands,):
            raise ValueError(
                f"band_means and band_stds must have length {n_bands}; got "
                f"{band_means.shape} and {band_stds.shape}."
            )

        self._cache = patch_cache
        # Reshaped once so each item just broadcasts; matches the earlier
        # extract-and-standardise arithmetic exactly.
        self._mean = band_means.reshape(n_bands, 1, 1)
        self._std = band_stds.reshape(n_bands, 1, 1)
        self._nodata = float(nodata)

    def __len__(self) -> int:
        """Return the number of patches in the dataset."""
        return int(self._positions.shape[0])

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return the (patch, label, weight) tensors for the patch at ``index``."""
        position = int(self._positions[index])
        raw = np.asarray(self._cache[position], dtype=np.float32)
        sentinel = raw == np.float32(self._nodata)
        try:
            patch = (raw - self._mean) / self._std
        except ValueError as exc:
            # Diagnostic for an intermittent broadcast failure seen only on some
            # (patch size, fold) combinations: report the operands that failed so the
            # cause is visible rather than inferred. Costs nothing on the happy path.
            raise ValueError(
                "patch normalisation failed: "
                f"index={index!r} position={position!r} "
                f"raw.shape={raw.shape} raw.ndim={raw.ndim} raw.dtype={raw.dtype} "
                f"mean.shape={self._mean.shape} std.shape={self._std.shape} "
                f"cache.shape={getattr(self._cache, 'shape', None)} "
                f"cache.dtype={getattr(self._cache, 'dtype', None)} "
                f"n_positions={self._positions.shape[0]}"
            ) from exc
        if sentinel.any():
            patch[sentinel] = np.float32(0.0)
        patch = np.ascontiguousarray(patch, dtype=np.float32)
        return (
            torch.from_numpy(patch),
            torch.tensor(float(self._labels[index]), dtype=torch.float32),
            torch.tensor(float(self._weights[index]), dtype=torch.float32),
        )


def _drop_edge_positions(
    positions: npt.NDArray[np.int64],
    rows: npt.NDArray[np.int64],
    cols: npt.NDArray[np.int64],
    height: int,
    width: int,
    radius: int,
    *,
    kind: str,
) -> npt.NDArray[np.int64]:
    """Drop positions whose patch would overrun the grid edge, logging the count.

    Args:
        positions: Row positions into the pixel index (and cache) to filter.
        rows: Grid row for every row of the pixel index.
        cols: Grid column for every row of the pixel index.
        height: Grid height in pixels.
        width: Grid width in pixels.
        radius: Patch half-width (``patch_size // 2``).
        kind: Label for the log line (for example ``"training"``).

    Returns:
        The subset of ``positions`` whose full patch lies inside the grid.
    """
    r = rows[positions]
    c = cols[positions]
    interior = (r >= radius) & (r < height - radius) & (c >= radius) & (c < width - radius)
    dropped = int(positions.shape[0]) - int(interior.sum())
    if dropped:
        logger.info(
            "Dropped %d/%d %s edge pixels (patch radius %d).",
            dropped,
            int(positions.shape[0]),
            kind,
            radius,
        )
    return positions[interior]


def build_patch_loaders(
    patch_cache: npt.NDArray[np.float32],
    pixel_index: pd.DataFrame,
    *,
    grid_shape: tuple[int, int],
    train_folds: Sequence[int],
    val_fold: int,
    sampling_mode: str,
    n_per_parcel: int,
    patch_size: int,
    batch_size: int,
    rng: np.random.Generator,
    num_workers: int,
    seed: int,
) -> tuple[DataLoader, DataLoader, pd.DataFrame]:
    """Build train and validation patch loaders for one ``(train, val)`` split.

    This is the patch-based analogue of :func:`utils.cv.build_fold_arrays`, and
    deliberately mirrors its validity and sampling semantics so the CNN trains on
    the *same* pixels the XGBoost path would, for a fair comparison:

      - invalid pixels (``valid`` is ``False``) are excluded from both sides;
      - training rows are sampled per parcel by reusing
        :func:`utils.cv.sample_train_rows` with the caller's ``rng``, so a shared
        seed selects identical training pixels to ``build_fold_arrays``;
      - every valid pixel of the held-out fold is kept for validation.

    Patches are read from ``patch_cache`` whose row ``i`` is the raw patch for row
    ``i`` of ``pixel_index``, so a selected position indexes the cache directly.
    Pixels whose patch would overrun the grid edge are dropped from both sets (the
    count is logged), using each pixel's ``row``/``col`` and the grid height and
    width from ``grid_shape``, because a centre pixel that close to the border has
    no genuine neighbourhood. Per-band standardisation statistics come from the
    training centre pixels only (NoData excluded), so no validation information
    leaks into the normalisation. Training weights reuse
    :func:`utils.sampling.build_sample_weights` exactly as the XGBoost path does;
    validation weights are uniform, since ``build_fold_arrays`` carries no
    validation weighting and the early-stopping signal is the plain held-out
    cross-entropy.

    Args:
        patch_cache: Patch cache of shape ``(n_pixels, n_bands, patch_size,
            patch_size)``, typically a memmap from ``numpy.load(..., mmap_mode="r")``.
        pixel_index: Pixel index carrying ``pixel_id``, ``row``, ``col``,
            ``parcel_id``, ``fold_id``, ``ogf``, ``forest_type_corine`` and a
            boolean ``valid`` column.
        grid_shape: Source grid ``(height, width)``, used for the edge drop.
        train_folds: Fold ids contributing training pixels.
        val_fold: The single fold id whose pixels form the validation set.
        sampling_mode: One of :data:`utils.sampling.SAMPLING_MODES`.
        n_per_parcel: Maximum training pixels sampled per parcel.
        patch_size: Odd patch side length; one of
            :data:`utils.terminology.CNN_PATCH_SIZES`.
        batch_size: Loader batch size.
        rng: Seeded generator for the per-parcel sampling (shared with the
            XGBoost path for identical pixel selection).
        num_workers: DataLoader worker-process count.
        seed: Seed for the training loader's shuffling generator.

    Returns:
        ``(train_loader, val_loader, val_pixel_index)``: ``train_loader``
        shuffles, ``val_loader`` does not (so validation probabilities keep pixel
        order), and ``val_pixel_index`` has columns ``pixel_id``, ``parcel_id``
        and ``y_true`` for parcel aggregation.

    Raises:
        ValueError: If ``batch_size`` is less than two or fewer than two
            training pixels remain after filtering. BatchNorm requires at least
            two training observations when the final feature map is 1x1.
    """
    if int(batch_size) < 2:
        raise ValueError(f"batch_size must be at least 2 for CNN BatchNorm; got {batch_size}.")

    valid = pixel_index["valid"].to_numpy()
    fold = pixel_index["fold_id"].to_numpy()
    ogf = pixel_index["ogf"].to_numpy()
    parcel_ids = pixel_index["parcel_id"].to_numpy()
    pixel_ids = pixel_index["pixel_id"].to_numpy()
    rows = pixel_index["row"].to_numpy().astype(np.int64)
    cols = pixel_index["col"].to_numpy().astype(np.int64)
    # CORINE forest type drives the stratified_forest_type weighting (coverage).
    forest_type = pixel_index["forest_type_corine"].to_numpy()

    height, width = int(grid_shape[0]), int(grid_shape[1])

    # Selection and per-parcel sampling identical to build_fold_arrays, so a
    # shared rng draws the same training pixels for the CNN and the XGBoost path.
    train_all = np.flatnonzero(np.isin(fold, np.asarray(train_folds)) & valid)
    sampled_local = sample_train_rows(pixel_index.iloc[train_all], n_per_parcel, rng)
    train_positions = train_all[sampled_local]
    val_positions = np.flatnonzero((fold == val_fold) & valid)

    radius = patch_size // 2
    train_positions = _drop_edge_positions(
        train_positions, rows, cols, height, width, radius, kind="training"
    )
    val_positions = _drop_edge_positions(
        val_positions, rows, cols, height, width, radius, kind="validation"
    )

    y_train = ogf[train_positions].astype(np.int8)
    train_weight = build_sample_weights(
        sampling_mode, labels=y_train, forest_type=forest_type[train_positions]
    )
    # Standardisation fitted on training centre pixels only (no leakage).
    band_means, band_stds = compute_band_stats(patch_cache, train_positions, nodata=NODATA)

    train_dataset = PatchDataset(
        patch_cache,
        train_positions,
        y_train,
        train_weight,
        band_means=band_means,
        band_stds=band_stds,
    )
    n_train = len(train_dataset)
    if n_train < 2:
        raise ValueError(
            "At least two training pixels are required for CNN BatchNorm after "
            f"validity, sampling and edge filtering; got {n_train}."
        )
    y_val = ogf[val_positions].astype(np.int8)
    # Validation weights are uniform; build_fold_arrays carries no val weighting.
    val_weight = np.ones(int(val_positions.shape[0]), dtype=np.float64)
    val_dataset = PatchDataset(
        patch_cache,
        val_positions,
        y_val,
        val_weight,
        band_means=band_means,
        band_stds=band_stds,
    )

    # Keep workers alive across epochs (only legal with workers) and pin host
    # memory so the cached patches stage straight into a CUDA transfer.
    persistent = int(num_workers) > 0
    generator = torch.Generator()
    generator.manual_seed(int(seed))
    # The receptive-field stack ends at (B, C, 1, 1), so BatchNorm cannot
    # estimate training statistics for a final batch of one. Drop only that
    # singleton; every other remainder is retained.
    drop_singleton = n_train % int(batch_size) == 1
    train_loader = DataLoader(
        train_dataset,
        batch_size=int(batch_size),
        shuffle=True,
        num_workers=int(num_workers),
        generator=generator,
        persistent_workers=persistent,
        pin_memory=True,
        drop_last=drop_singleton,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=int(batch_size),
        shuffle=False,
        num_workers=int(num_workers),
        persistent_workers=persistent,
        pin_memory=True,
    )

    val_pixel_index = pd.DataFrame(
        {
            "pixel_id": pixel_ids[val_positions].astype(np.int64),
            "parcel_id": parcel_ids[val_positions].astype(np.int64),
            "y_true": y_val,
        }
    )
    return train_loader, val_loader, val_pixel_index


__all__ = [
    "PatchDataset",
    "build_patch_loaders",
    "compute_band_stats",
    "extract_raw_patch",
]
