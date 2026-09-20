"""Training-sample weighting schemes for the old-growth models.

Builds per-row sample weights for model training and the sampling-mode
sensitivity analysis. Three schemes are supported:

  - ``none``: every row weighted equally.
  - ``inverse_prevalence``: each row is weighted by the inverse of its class
    frequency, so the two classes carry equal total weight and the rare
    old-growth class is upweighted. This is the balanced class weighting used
    widely under imbalance (the same n / (n_classes * class_count) idea as
    scikit-learn's ``class_weight="balanced"``).
  - ``stratified_forest_type``: each row is weighted by the inverse of its
    (class, forest-type) cell frequency, so every cell carries equal total
    weight. This removes the confounding of class prevalence with forest type
    when probing whether performance is uniform across forest types.

All schemes are normalised so the mean weight is 1.0 (the weights sum to the row
count). That keeps the effective sample size, and therefore the learning-rate
and regularisation scale, comparable across schemes, so a sensitivity
comparison varies only the weighting and not the overall weight mass.

Forest-type categories are taken from the data, not hard-coded, so any label
set (including an unclassified category) is handled. Importing this module binds
names only; the function is pure and performs no input/output or logging.
"""

from __future__ import annotations

from typing import Final

import numpy as np
import numpy.typing as npt
import pandas as pd

# Canonical sampling-mode names. This module owns them; other code (the
# sensitivity script, any sampling config) imports them rather than re-declaring.
SAMPLING_MODES: Final[tuple[str, ...]] = (
    "none",
    "inverse_prevalence",
    "stratified_forest_type",
)


def _validate_labels(labels: npt.ArrayLike) -> npt.NDArray[np.int64]:
    """Coerce and check a binary label vector.

    Args:
        labels: One-dimensional binary labels (0/1 or boolean).

    Returns:
        The labels as ``int64``.

    Raises:
        ValueError: If the labels are not one-dimensional, empty, or not binary.
    """
    yt = np.asarray(labels)
    if yt.ndim != 1:
        raise ValueError("labels must be one-dimensional.")
    if yt.shape[0] == 0:
        raise ValueError("labels are empty.")
    unique = np.unique(yt)
    if not np.isin(unique, (0, 1)).all():
        raise ValueError(f"labels must be binary 0/1; found {unique.tolist()}.")
    return yt.astype(np.int64)


def build_sample_weights(
    mode: str,
    *,
    labels: npt.ArrayLike,
    forest_type: npt.ArrayLike | None = None,
) -> npt.NDArray[np.float64]:
    """Build per-row sample weights for a training fold.

    Args:
        mode: One of :data:`SAMPLING_MODES`.
        labels: One-dimensional binary old-growth labels, one per row.
        forest_type: One-dimensional forest-type category per row. Required for
            ``stratified_forest_type``; ignored otherwise.

    Returns:
        A ``float64`` weight per row, normalised so the mean weight is 1.0 (the
        weights sum to the row count), in the input row order.

    Raises:
        ValueError: If ``mode`` is unknown; if ``labels`` fail validation; or if
            ``mode`` is ``stratified_forest_type`` and ``forest_type`` is missing,
            not one-dimensional, or a different length from ``labels``.
    """
    if mode not in SAMPLING_MODES:
        raise ValueError(f"unknown mode {mode!r}; expected one of {list(SAMPLING_MODES)}.")

    yt = _validate_labels(labels)
    n = yt.shape[0]

    if mode == "none":
        raw = np.ones(n, dtype=np.float64)
    elif mode == "inverse_prevalence":
        class_size = (
            pd.DataFrame({"label": yt, "_one": 1})
            .groupby("label")["_one"]
            .transform("size")
            .to_numpy()
        )
        raw = 1.0 / class_size
    else:  # stratified_forest_type
        if forest_type is None:
            raise ValueError("forest_type is required for the 'stratified_forest_type' mode.")
        ft = np.asarray(forest_type)
        if ft.ndim != 1:
            raise ValueError("forest_type must be one-dimensional.")
        if ft.shape[0] != n:
            raise ValueError(f"length mismatch: labels has {n}, forest_type has {ft.shape[0]}.")
        cell_size = (
            pd.DataFrame({"label": yt, "ft": ft, "_one": 1})
            .groupby(["label", "ft"])["_one"]
            .transform("size")
            .to_numpy()
        )
        raw = 1.0 / cell_size

    return raw * (n / raw.sum())


__all__ = [
    "SAMPLING_MODES",
    "build_sample_weights",
]
