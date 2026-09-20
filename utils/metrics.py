"""Discrimination metrics and operating point for the old-growth model.

Pixel- and parcel-level metrics for the binary task (positive class =
old-growth), used to rank model configurations in the nested cross-validation.
The choices follow the operational problem, detecting a rare, valuable class:

  - Average precision (the step-summed area under the precision-recall curve)
    is the primary score. Under class imbalance it reflects positive-class
    performance better than ROC-AUC, which abundant true negatives flatter
    (Saito and Rehmsmeier, 2015, doi:10.1371/journal.pone.0118432). It is the
    step sum rather than a trapezoidal area because linear interpolation in PR
    space is invalid (Davis and Goadrich, 2006, doi:10.1145/1143844.1143874).
    Its no-skill baseline is the positive prevalence, so read it against
    ``prevalence``.
  - ROC-AUC is a secondary, prevalence-invariant discrimination summary for
    cross-study comparison, carrying the imbalance caveat above.
  - The F1-maximising threshold, with the precision and recall there, gives a
    single operating point for maps and confusion matrices.

Calibration is a separate final step, so calibration metrics (reliability,
expected calibration error, the Brier decomposition) live with that step:
average precision and ROC-AUC rank by discrimination and are invariant to any
monotonic recalibration, so calibration quality does not bear on the selection
this module serves.

Parcel scores are the unweighted mean of their pixel probabilities (AGENTS.md,
"Parcel aggregation"). Importing this module binds names only; the functions
are pure and perform no input/output or logging.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)


@dataclass(frozen=True, slots=True)
class ThresholdMetrics:
    """F1-maximising operating point.

    Attributes:
        threshold: Probability cut-off that maximises F1.
        f1: F1 at that threshold.
        precision: Precision at that threshold.
        recall: Recall at that threshold.
    """

    threshold: float
    f1: float
    precision: float
    recall: float


def _validate_binary_probabilities(
    y_true: npt.ArrayLike, y_prob: npt.ArrayLike
) -> tuple[npt.NDArray[np.int64], npt.NDArray[np.float64]]:
    """Coerce and check a binary label vector and its probabilities.

    Both classes are not required here; the ranking metrics add that check.

    Args:
        y_true: One-dimensional binary labels (0/1 or boolean).
        y_prob: One-dimensional positive-class probabilities in ``[0, 1]``.

    Returns:
        Labels as ``int64`` and probabilities as ``float64``.

    Raises:
        ValueError: On dimension, length, emptiness, non-finite, range or
            non-binary violations.
    """
    yt = np.asarray(y_true)
    yp = np.asarray(y_prob, dtype=np.float64)
    if yt.ndim != 1 or yp.ndim != 1:
        raise ValueError("y_true and y_prob must be one-dimensional.")
    if yt.shape[0] != yp.shape[0]:
        raise ValueError(f"length mismatch: y_true has {yt.shape[0]}, y_prob has {yp.shape[0]}.")
    if yt.shape[0] == 0:
        raise ValueError("inputs are empty.")
    if not np.isfinite(yp).all():
        raise ValueError("y_prob contains non-finite values.")
    if yp.min() < 0.0 or yp.max() > 1.0:
        raise ValueError("y_prob must lie in [0, 1].")
    labels = np.unique(yt)
    if not np.isin(labels, (0, 1)).all():
        raise ValueError(f"y_true must be binary 0/1; found {labels.tolist()}.")
    return yt.astype(np.int64), yp


def best_f1_threshold(y_true: npt.ArrayLike, y_prob: npt.ArrayLike) -> ThresholdMetrics:
    """Probability threshold that maximises F1.

    Sweeps the precision-recall curve and returns the highest-F1 operating
    point, guarding the degenerate point where precision and recall are zero.

    Args:
        y_true: One-dimensional binary labels; both classes must be present.
        y_prob: One-dimensional positive-class probabilities in ``[0, 1]``.

    Returns:
        The :class:`ThresholdMetrics` at the F1-maximising threshold.

    Raises:
        ValueError: If inputs fail validation or only one class is present.
    """
    yt, yp = _validate_binary_probabilities(y_true, y_prob)
    if np.unique(yt).shape[0] < 2:
        raise ValueError("best_f1_threshold requires both classes present in y_true.")

    precision, recall, thresholds = precision_recall_curve(yt, yp)
    # Drop the trailing (precision=1, recall=0) point, which has no threshold.
    p = precision[:-1]
    r = recall[:-1]
    denominator = p + r
    f1 = np.zeros_like(denominator)
    nonzero = denominator > 0.0
    f1[nonzero] = 2.0 * p[nonzero] * r[nonzero] / denominator[nonzero]

    best = int(np.argmax(f1))
    return ThresholdMetrics(
        threshold=float(thresholds[best]),
        f1=float(f1[best]),
        precision=float(p[best]),
        recall=float(r[best]),
    )


def metrics_at_threshold(
    y_true: npt.ArrayLike, y_prob: npt.ArrayLike, threshold: float
) -> dict[str, float]:
    """F1, precision and recall at a fixed decision threshold.

    For applying a threshold chosen on other data (for example inner folds) to a
    held-out set, so the operating-point metrics are not tuned in-sample. Both
    classes need not be present, since no ranking metric is computed.

    Args:
        y_true: One-dimensional binary labels (0/1 or boolean).
        y_prob: One-dimensional positive-class probabilities in ``[0, 1]``.
        threshold: Decision cut-off in ``[0, 1]``; predict positive if
            ``p >= threshold``.

    Returns:
        A dict with ``threshold``, ``f1``, ``precision`` and ``recall``.

    Raises:
        ValueError: If the inputs fail validation, or ``threshold`` is outside
            ``[0, 1]``.
    """
    yt, yp = _validate_binary_probabilities(y_true, y_prob)
    if not 0.0 <= threshold <= 1.0:
        raise ValueError(f"threshold must lie in [0, 1], got {threshold}.")
    yhat = (yp >= threshold).astype(np.int64)
    return {
        "threshold": float(threshold),
        "f1": float(f1_score(yt, yhat, zero_division=0.0)),
        "precision": float(precision_score(yt, yhat, zero_division=0.0)),
        "recall": float(recall_score(yt, yhat, zero_division=0.0)),
    }


def _classification_metrics(y_true: npt.ArrayLike, y_prob: npt.ArrayLike) -> dict[str, float]:
    """Discrimination metric core shared by pixel and parcel evaluation.

    Args:
        y_true: One-dimensional binary labels; both classes must be present.
        y_prob: One-dimensional positive-class probabilities in ``[0, 1]``.

    Returns:
        ``n``, ``prevalence``, ``pr_auc``, ``roc_auc``, ``threshold``, ``f1``,
        ``precision`` and ``recall``.

    Raises:
        ValueError: If inputs fail validation or only one class is present.
    """
    yt, yp = _validate_binary_probabilities(y_true, y_prob)
    if np.unique(yt).shape[0] < 2:
        raise ValueError("ranking metrics require both classes present in y_true.")

    operating_point = best_f1_threshold(yt, yp)
    return {
        "n": int(yt.shape[0]),
        "prevalence": float(yt.mean()),
        "pr_auc": float(average_precision_score(yt, yp)),
        "roc_auc": float(roc_auc_score(yt, yp)),
        "threshold": operating_point.threshold,
        "f1": operating_point.f1,
        "precision": operating_point.precision,
        "recall": operating_point.recall,
    }


def pixel_metrics(y_true: npt.ArrayLike, y_prob: npt.ArrayLike) -> dict[str, float]:
    """Discrimination metrics on pixel-level predictions.

    Args:
        y_true: Per-pixel binary labels; both classes must be present.
        y_prob: Per-pixel positive-class probabilities in ``[0, 1]``.

    Returns:
        The metric dict from :func:`_classification_metrics`.
    """
    return _classification_metrics(y_true, y_prob)


def parcel_metrics(y_true: npt.ArrayLike, p_mean: npt.ArrayLike) -> dict[str, float]:
    """Discrimination metrics on parcel-aggregated predictions.

    Args:
        y_true: Per-parcel binary labels; both classes must be present.
        p_mean: Per-parcel mean probability from :func:`parcel_aggregations`.

    Returns:
        The metric dict from :func:`_classification_metrics`.
    """
    return _classification_metrics(y_true, p_mean)


def parcel_aggregations(parcel_ids: npt.ArrayLike, probabilities: npt.ArrayLike) -> pd.DataFrame:
    """Aggregate pixel probabilities to parcels by their unweighted mean.

    Args:
        parcel_ids: One-dimensional integer parcel identifiers, one per pixel.
        probabilities: One-dimensional pixel probabilities in ``[0, 1]``.

    Returns:
        Columns ``parcel_id`` (int64), ``p_mean`` (float64) and ``n_pixels``
        (int64), sorted ascending by ``parcel_id``.

    Raises:
        ValueError: On dimension, length, emptiness, non-finite or range
            violations.
    """
    ids = np.asarray(parcel_ids)
    p = np.asarray(probabilities, dtype=np.float64)
    if ids.ndim != 1 or p.ndim != 1:
        raise ValueError("parcel_ids and probabilities must be one-dimensional.")
    if ids.shape[0] != p.shape[0]:
        raise ValueError(
            f"length mismatch: parcel_ids has {ids.shape[0]}, " f"probabilities has {p.shape[0]}."
        )
    if ids.shape[0] == 0:
        raise ValueError("inputs are empty.")
    if not np.isfinite(p).all():
        raise ValueError("probabilities contain non-finite values.")
    if p.min() < 0.0 or p.max() > 1.0:
        raise ValueError("probabilities must lie in [0, 1].")

    frame = pd.DataFrame({"parcel_id": ids.astype(np.int64), "probability": p})
    aggregated = (
        frame.groupby("parcel_id", sort=True)["probability"]
        .agg(p_mean="mean", n_pixels="size")
        .reset_index()
    )
    aggregated["p_mean"] = aggregated["p_mean"].astype("float64")
    aggregated["n_pixels"] = aggregated["n_pixels"].astype("int64")
    return aggregated[["parcel_id", "p_mean", "n_pixels"]]


__all__ = [
    "ThresholdMetrics",
    "best_f1_threshold",
    "metrics_at_threshold",
    "parcel_aggregations",
    "parcel_metrics",
    "pixel_metrics",
]
