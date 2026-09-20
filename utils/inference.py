"""Wall-to-wall inference building blocks for the final old-growth map.

The selected model (XGBoost + ``baseline_tessera``) is applied to every valid
pixel of the reference grid by slicing the co-registered grid memmap
(``data/cache/grid_memmap/<fs>.npy``, shape ``(n_bands, height, width)``) whose
flat column index is ``pixel_id = row * width + col``. These helpers are the pure,
data-agnostic core: gather a pixel's feature vector by id, predict in memory-bounded
chunks, and convert a binary surface to an area. The orchestrating script owns the
data plumbing (the valid mask, the rasterised parcel template, GeoTIFF and GeoPackage
writing) so these functions stay testable on a tiny synthetic grid.

Importing this module binds names only; the functions are pure and perform no
input/output or logging.
"""

from __future__ import annotations

from typing import Final

import numpy as np
import numpy.typing as npt
from xgboost import XGBClassifier

from utils.models.xgboost import predict_ogf_proba
from utils.terminology import REF_RESOLUTION_M

# Area of one reference-grid cell in hectares (10 m cell -> 0.01 ha).
PIXEL_AREA_HA: Final[float] = (REF_RESOLUTION_M**2) / 10_000.0


def gather_pixel_features(
    memmap: npt.NDArray[np.float32], pixel_ids: npt.ArrayLike
) -> npt.NDArray[np.float32]:
    """Gather a feature matrix from the grid memmap by flat pixel id.

    Args:
        memmap: The co-registered grid array of shape ``(n_bands, height, width)``
            (typically opened with ``numpy.load(..., mmap_mode="r")``).
        pixel_ids: One-dimensional flat pixel ids (``row * width + col``).

    Returns:
        A C-contiguous ``(len(pixel_ids), n_bands)`` ``float32`` matrix, row ``i``
        holding every band of ``pixel_ids[i]``.

    Raises:
        ValueError: If ``memmap`` is not three-dimensional, ``pixel_ids`` is not
            one-dimensional, or an id is out of range.
    """
    if memmap.ndim != 3:
        raise ValueError(f"memmap must be (n_bands, height, width); got shape {memmap.shape}.")
    ids = np.asarray(pixel_ids, dtype=np.int64)
    if ids.ndim != 1:
        raise ValueError("pixel_ids must be one-dimensional.")
    n_bands = memmap.shape[0]
    n_cells = memmap.shape[1] * memmap.shape[2]
    if ids.size and (ids.min() < 0 or ids.max() >= n_cells):
        raise ValueError(f"pixel_ids out of range [0, {n_cells}).")
    flat = memmap.reshape(n_bands, n_cells)
    return np.ascontiguousarray(flat[:, ids].T)


def predict_pixels(
    model: XGBClassifier, features: npt.ArrayLike, *, chunk_size: int = 1_000_000
) -> npt.NDArray[np.float64]:
    """Positive-class (old-growth) probability for each row, in memory-bounded chunks.

    Args:
        model: A fitted classifier.
        features: A ``(n_pixels, n_bands)`` feature matrix.
        chunk_size: Maximum rows scored per ``predict_proba`` call (bounds GPU
            memory for a wall-to-wall pass).

    Returns:
        A one-dimensional ``float64`` array of probabilities, one per row.

    Raises:
        ValueError: If ``features`` is not two-dimensional or ``chunk_size`` < 1.
    """
    matrix = np.asarray(features)
    if matrix.ndim != 2:
        raise ValueError("features must be a two-dimensional matrix.")
    if chunk_size < 1:
        raise ValueError("chunk_size must be a positive integer.")
    out = np.empty(matrix.shape[0], dtype=np.float64)
    for start in range(0, matrix.shape[0], chunk_size):
        stop = min(start + chunk_size, matrix.shape[0])
        out[start:stop] = predict_ogf_proba(model, matrix[start:stop])
    return out


def ogf_area_ha(binary_mask: npt.ArrayLike, *, pixel_area_ha: float = PIXEL_AREA_HA) -> float:
    """Total area (hectares) of the positive cells of a binary surface.

    This is the per-pixel operating point: every pixel whose probability cleared the
    pixel threshold counts. See :func:`parcel_consistent_area_ha` for the parcel-level one.

    Args:
        binary_mask: A boolean (or 0/1) array of old-growth predictions.
        pixel_area_ha: Area of one cell in hectares (defaults to the 10 m grid).

    Returns:
        The summed positive-cell area in hectares.
    """
    mask = np.asarray(binary_mask)
    return float(np.count_nonzero(mask)) * float(pixel_area_ha)


def parcel_consistent_area_ha(
    parcel_ids: npt.ArrayLike,
    probabilities: npt.ArrayLike,
    *,
    threshold: float,
    pixel_area_ha: float = PIXEL_AREA_HA,
) -> float:
    """Old-growth area from parcels whose mean pixel probability reaches the threshold.

    The parcel-level operating point, consistent with the parcel product: pixels are grouped
    by parcel, a parcel is old-growth when its mean probability is at least ``threshold``, and
    every pixel of such parcels contributes to the area. Because the parcel and pixel F1-max
    thresholds differ, this generally differs from :func:`ogf_area_ha`.

    Args:
        parcel_ids: Parcel id for each pixel (same length as ``probabilities``).
        probabilities: Per-pixel old-growth probabilities.
        threshold: Parcel mean-probability threshold for an old-growth verdict.
        pixel_area_ha: Area of one cell in hectares (defaults to the 10 m grid).

    Returns:
        The summed area (hectares) of the pixels in old-growth parcels.

    Raises:
        ValueError: If ``parcel_ids`` and ``probabilities`` differ in length.
    """
    ids = np.asarray(parcel_ids, dtype=np.int64)
    probs = np.asarray(probabilities, dtype=np.float64)
    if ids.shape != probs.shape:
        raise ValueError("parcel_ids and probabilities must have the same shape.")
    if ids.size == 0:
        return 0.0
    unique, inverse = np.unique(ids, return_inverse=True)
    counts = np.bincount(inverse, minlength=unique.size)
    means = np.bincount(inverse, weights=probs, minlength=unique.size) / counts
    return float(counts[means >= threshold].sum()) * float(pixel_area_ha)


__all__ = [
    "PIXEL_AREA_HA",
    "gather_pixel_features",
    "ogf_area_ha",
    "parcel_consistent_area_ha",
    "predict_pixels",
]
