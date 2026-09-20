"""Meyer-Pebesma (2021) area-of-applicability primitives.

Implements the dissimilarity index (DI) in the deployed model's feature space:
predictors are standardised by the training mean and standard deviation,
multiplied by the model's variable importance (gain), and compared by Euclidean
distance. ``DI = distance to the nearest training sample / mean pairwise
distance among training samples``. The AOA threshold is the paper's
outlier-removed maximum: the largest cross-validated training DI at or below
the upper whisker (Q3 + 1.5 x IQR). Cross-validated training DIs respect the
spatial folds — each training point's nearest neighbour is sought among points
outside its own fold, mirroring what the nested CV actually tested.

The absolute scale of the importance weights cancels in the DI (numerator and
denominator share it); only the relative weighting between features matters.

Raster sentinels (documented, per AGENTS.md): DI rasters are float32 with the
project NoData sentinel; AOA and nearest-label rasters are uint8 with
``AOA_INSIDE = 1``, ``AOA_OUTSIDE = 0`` and ``AOA_UINT8_NODATA = 255``.

Nearest-neighbour searches run blocked on the requested torch device (``cuda``
for real runs per AGENTS.md; the unit tests use small problems on ``cpu``).
Importing this module performs no I/O and does not import torch; torch is
imported lazily inside the distance functions.

Reference: Meyer, H., & Pebesma, E. (2021). Predicting into unknown space?
Estimating the area of applicability of spatial prediction models. Methods in
Ecology and Evolution, 12, 1620-1633. doi:10.1111/2041-210X.13650.
"""

import itertools
import logging
from dataclasses import dataclass
from typing import Final

import numpy as np
import pandas as pd

# Module logger. No handlers are configured here; callers own logging setup.
logger = logging.getLogger(__name__)

# Integer sentinels for the uint8 AOA and nearest-label rasters.
AOA_INSIDE: Final[int] = 1
AOA_OUTSIDE: Final[int] = 0
AOA_UINT8_NODATA: Final[int] = 255


@dataclass(frozen=True, slots=True)
class FeatureScaling:
    """Standardisation and importance weighting fitted on the training matrix.

    Attributes:
        means: Per-feature training means.
        sds: Per-feature training standard deviations (zeros replaced by one).
        weights: Per-feature importance weights (non-standardised, per paper).
    """

    means: np.ndarray
    sds: np.ndarray
    weights: np.ndarray

    def transform(self, features: np.ndarray) -> np.ndarray:
        """Return the standardised, importance-weighted matrix as float32."""
        scaled = (np.asarray(features, dtype=np.float64) - self.means) / self.sds
        return (scaled * self.weights).astype(np.float32)


def fit_feature_scaling(train: np.ndarray, importances: np.ndarray) -> FeatureScaling:
    """Fit standardisation and importance weights on the training matrix.

    Args:
        train: Training feature matrix, shape ``(n_samples, n_features)``.
        importances: Per-feature importance (e.g. XGBoost gain), same order.

    Returns:
        The fitted :class:`FeatureScaling`.

    Raises:
        ValueError: On shape mismatch or negative importances.
    """
    train = np.asarray(train, dtype=np.float64)
    importances = np.asarray(importances, dtype=np.float64)
    if train.ndim != 2 or importances.shape != (train.shape[1],):
        raise ValueError(
            f"Expected (n, p) features with p importances; got {train.shape} and "
            f"{importances.shape}."
        )
    if (importances < 0).any():
        raise ValueError("Importances must be non-negative.")

    means = train.mean(axis=0)
    sds = train.std(axis=0)
    constant = sds == 0
    if constant.any():
        logger.info("Guarding %d constant feature(s) with unit sd.", int(constant.sum()))
        sds = np.where(constant, 1.0, sds)
    return FeatureScaling(means=means, sds=sds, weights=importances)


def nearest_distances(
    query: np.ndarray,
    reference: np.ndarray,
    *,
    device: str = "cuda",
    query_block: int = 16_384,
    reference_block: int = 65_536,
    return_indices: bool = False,
) -> tuple[np.ndarray, np.ndarray | None]:
    """Blocked nearest-neighbour Euclidean distances from query to reference.

    Args:
        query: Query matrix ``(n_query, p)`` (weighted feature space).
        reference: Reference matrix ``(n_reference, p)``; held resident on the
            device, so it must fit in device memory as float32.
        device: torch device (``"cuda"`` for real runs; tests use ``"cpu"``).
        query_block: Query rows per block.
        reference_block: Reference rows per inner block (bounds the distance
            matrix at ``query_block x reference_block`` float32).
        return_indices: Also return the argmin index into ``reference``.

    Returns:
        ``(distances, indices)`` where ``indices`` is ``None`` unless requested.
    """
    import torch

    if len(query) == 0:
        empty_idx = np.empty(0, dtype=np.int64) if return_indices else None
        return np.empty(0, dtype=np.float32), empty_idx
    if len(reference) == 0:
        raise ValueError("Reference matrix is empty.")

    dev = torch.device(device)
    ref = torch.as_tensor(np.ascontiguousarray(reference, dtype=np.float32), device=dev)
    out = np.empty(len(query), dtype=np.float32)
    out_idx = np.empty(len(query), dtype=np.int64) if return_indices else None

    with torch.no_grad():
        for q_start in range(0, len(query), query_block):
            q_stop = min(q_start + query_block, len(query))
            block = torch.as_tensor(
                np.ascontiguousarray(query[q_start:q_stop], dtype=np.float32), device=dev
            )
            best = torch.full((q_stop - q_start,), torch.inf, device=dev)
            best_idx = torch.zeros(q_stop - q_start, dtype=torch.int64, device=dev)
            for r_start in range(0, len(ref), reference_block):
                r_stop = min(r_start + reference_block, len(ref))
                distances = torch.cdist(block, ref[r_start:r_stop])
                chunk_min, chunk_arg = distances.min(dim=1)
                improved = chunk_min < best
                best = torch.where(improved, chunk_min, best)
                best_idx = torch.where(improved, chunk_arg + r_start, best_idx)
            out[q_start:q_stop] = best.cpu().numpy()
            if out_idx is not None:
                out_idx[q_start:q_stop] = best_idx.cpu().numpy()
    return out, out_idx


def cv_nearest_distances(
    weighted: np.ndarray,
    fold_ids: np.ndarray,
    *,
    device: str = "cuda",
    query_block: int = 16_384,
    reference_block: int = 65_536,
) -> np.ndarray:
    """Distance from each training point to its nearest neighbour outside its fold.

    Args:
        weighted: Weighted training matrix ``(n, p)``.
        fold_ids: Per-row spatial fold identifier ``(n,)``.
        device: torch device (``"cuda"`` for real runs; tests use ``"cpu"``).
        query_block: Query rows per block.
        reference_block: Reference rows per inner block.

    Returns:
        Per-row nearest cross-fold distance ``(n,)`` float32.

    Raises:
        ValueError: If fewer than two folds are present.
    """
    fold_ids = np.asarray(fold_ids)
    folds = np.unique(fold_ids)
    if folds.size < 2:
        raise ValueError("Cross-validated DI needs at least two folds.")

    out = np.empty(len(weighted), dtype=np.float32)
    for fold in folds:
        in_fold = fold_ids == fold
        distances, _ = nearest_distances(
            weighted[in_fold],
            weighted[~in_fold],
            device=device,
            query_block=query_block,
            reference_block=reference_block,
        )
        out[in_fold] = distances
        logger.info(
            "Fold %s: %d points, nearest cross-fold distance median %.4f",
            fold,
            int(in_fold.sum()),
            float(np.median(distances)),
        )
    return out


def mean_pairwise_distance(
    weighted: np.ndarray,
    *,
    sample_size: int,
    seed: int,
    device: str = "cuda",
    block: int = 8_192,
) -> tuple[float, int]:
    """Mean pairwise Euclidean distance over a random training subset.

    The exact all-pairs mean over the full training set is quadratic in ``n``;
    a random subset gives an unbiased estimate of the paper's denominator (the
    subset-size sensitivity is reported by the analysis script).

    Args:
        weighted: Weighted training matrix ``(n, p)``.
        sample_size: Subset size (capped at ``n``).
        seed: Seed for the subset draw.
        device: torch device (``"cuda"`` for real runs; tests use ``"cpu"``).
        block: Rows per pairwise block.

    Returns:
        ``(mean_distance, subset_size_used)``.
    """
    import torch

    n = len(weighted)
    if n < 2:
        raise ValueError("At least two training samples are required.")
    size = min(sample_size, n)
    rng = np.random.default_rng(seed)
    subset_rows = rng.choice(n, size=size, replace=False) if size < n else np.arange(n)
    subset = np.ascontiguousarray(weighted[subset_rows], dtype=np.float32)

    dev = torch.device(device)
    tensor = torch.as_tensor(subset, device=dev)
    total = 0.0
    with torch.no_grad():
        for start in range(0, size, block):
            stop = min(start + block, size)
            total += float(torch.cdist(tensor[start:stop], tensor).sum().item())
    # The double sum counts every ordered pair once and the zero diagonal.
    mean = total / (size * (size - 1))
    if not np.isfinite(mean) or mean <= 0:
        raise ValueError(f"Invalid DI denominator: {mean!r}.")
    return mean, size


def di_threshold(train_di: np.ndarray) -> dict[str, float | int]:
    """Derive the AOA threshold from cross-validated training DIs.

    The paper's rule: outliers are DIs above the upper whisker
    (Q3 + 1.5 x IQR); the threshold is the outlier-removed maximum. The 0.95
    quantile is reported alongside as a sensitivity value only.

    Args:
        train_di: Cross-validated training DI values.

    Returns:
        Threshold statistics (quartiles, whisker, threshold, counts, q95).
    """
    di = np.asarray(train_di, dtype=np.float64)
    if di.size < 4:
        raise ValueError("At least four training DI values are required.")
    q1, q3 = np.quantile(di, [0.25, 0.75])
    iqr = q3 - q1
    upper_whisker = q3 + 1.5 * iqr
    kept = di[di <= upper_whisker]
    threshold = float(kept.max()) if kept.size else float(upper_whisker)
    return {
        "n": int(di.size),
        "q1": float(q1),
        "q3": float(q3),
        "iqr": float(iqr),
        "upper_whisker": float(upper_whisker),
        "aoa_threshold": threshold,
        "n_above_whisker": int((di > upper_whisker).sum()),
        "pct_above_whisker": float(100.0 * (di > upper_whisker).mean()),
        "q95_sensitivity": float(np.quantile(di, 0.95)),
    }


def _window_metrics(
    window_y: np.ndarray,
    window_p: np.ndarray,
    window_thresholds: np.ndarray | None,
) -> dict[str, float]:
    """Set-level metrics for one group of samples (window or bin)."""
    from utils.metrics import average_precision_score, f1_score, roc_auc_score

    prevalence = float(window_y.mean())
    both_classes = 0.0 < prevalence < 1.0
    pr_auc = float(average_precision_score(window_y, window_p)) if both_classes else float("nan")
    roc_auc = float(roc_auc_score(window_y, window_p)) if both_classes else float("nan")
    if window_thresholds is not None:
        predicted = (window_p >= window_thresholds).astype(np.int8)
        f1 = float(f1_score(window_y, predicted, zero_division=0.0))
    else:
        f1 = float("nan")
    return {
        "prevalence": prevalence,
        "pr_auc": pr_auc,
        "pr_auc_lift": pr_auc - prevalence if np.isfinite(pr_auc) else float("nan"),
        "roc_auc": roc_auc,
        "f1": f1,
        "brier": float(np.mean((window_p - window_y) ** 2)),
    }


def performance_by_di_window(
    di: np.ndarray,
    y_true: np.ndarray,
    p: np.ndarray,
    *,
    thresholds: np.ndarray | None = None,
    window: int = 20_000,
    step: int | None = None,
) -> pd.DataFrame:
    """Windowed performance along the DI axis with monotone fits.

    Ranking metrics (PR-AUC, ROC-AUC) are set-level quantities: a single sample
    has no AUC, so performance conditional on DI can only be estimated over
    groups of samples with similar DI — hence the sliding windows. The frame
    carries PR-AUC (primary), its lift over the window prevalence (PR-AUC's
    chance level is the prevalence, which varies between windows, so raw
    PR-AUC values are not directly comparable across windows), ROC-AUC (whose
    chance level is 0.5 regardless of prevalence), F1 at the supplied
    per-sample classification thresholds, and the Brier score (the one metric
    that is a per-sample average). Isotonic-regression fits are added as the
    monotone calibration curves the paper suggests; the fits IMPOSE
    monotonicity — flat stretches mean no decline was observed there, and any
    observed decline should be read from the raw windows.

    Args:
        di: Cross-validated DI per sample.
        y_true: Binary reference labels.
        p: Out-of-fold predicted probabilities.
        thresholds: Optional per-sample classification thresholds (e.g. each
            sample's own fold's inner-CV threshold, the leakage-free operating
            point); when ``None`` the F1 columns are NaN.
        window: Samples per window.
        step: Window step; defaults to ``window // 2``.

    Returns:
        One row per window: ``di_lo, di_mid, di_hi, n, prevalence, pr_auc,
        pr_auc_lift, roc_auc, f1, brier, pr_auc_fit, roc_auc_fit, f1_fit,
        brier_fit``.
    """
    di = np.asarray(di, dtype=np.float64)
    y_true = np.asarray(y_true)
    p = np.asarray(p, dtype=np.float64)
    if not (len(di) == len(y_true) == len(p)):
        raise ValueError("di, y_true and p must have equal length.")
    if thresholds is not None:
        thresholds = np.asarray(thresholds, dtype=np.float64)
        if len(thresholds) != len(di):
            raise ValueError("thresholds must have the same length as di.")
    if len(di) < window:
        raise ValueError(f"Need at least one full window of {window} samples; got {len(di)}.")
    step = window // 2 if step is None else step

    order = np.argsort(di, kind="stable")
    rows = []
    for start in range(0, len(order) - window + 1, step):
        picked = order[start : start + window]
        window_di = di[picked]
        row = {
            "di_lo": float(window_di.min()),
            "di_mid": float(np.median(window_di)),
            "di_hi": float(window_di.max()),
            "n": int(window),
        }
        row.update(
            _window_metrics(
                y_true[picked], p[picked], thresholds[picked] if thresholds is not None else None
            )
        )
        rows.append(row)
    frame = pd.DataFrame(rows)
    for column, increasing in (
        ("pr_auc", False),
        ("roc_auc", False),
        ("f1", False),
        ("brier", True),
    ):
        frame[f"{column}_fit"] = _monotone_fit(frame["di_mid"], frame[column], increasing)
    return frame


def _monotone_fit(di_mid: pd.Series, values: pd.Series, increasing: bool) -> pd.Series:
    """Isotonic fit of a windowed metric against DI (NaN-safe)."""
    from sklearn.isotonic import IsotonicRegression

    out = pd.Series(np.nan, index=values.index)
    usable = values.notna()
    if usable.sum() >= 2:
        fit = IsotonicRegression(increasing=increasing, out_of_bounds="clip")
        out.loc[usable] = fit.fit_transform(di_mid.loc[usable], values.loc[usable])
    return out


def performance_by_di_bins(
    di: np.ndarray,
    y_true: np.ndarray,
    p: np.ndarray,
    *,
    edges: np.ndarray,
    thresholds: np.ndarray | None = None,
) -> pd.DataFrame:
    """Set-level performance in non-overlapping DI bins (for tabular reporting).

    Unlike the sliding windows (which share samples and serve the curve), the
    bins partition the samples, so each row is an independent estimate.

    Args:
        di: Cross-validated DI per sample.
        y_true: Binary reference labels.
        p: Out-of-fold predicted probabilities.
        edges: Ascending bin edges; samples outside ``[edges[0], edges[-1])``
            are dropped.
        thresholds: Optional per-sample classification thresholds for F1.

    Returns:
        One row per bin: ``di_bin, di_lo, di_hi, n, prevalence, pr_auc,
        pr_auc_lift, roc_auc, f1, brier``.
    """
    di = np.asarray(di, dtype=np.float64)
    y_true = np.asarray(y_true)
    p = np.asarray(p, dtype=np.float64)
    edges = np.asarray(edges, dtype=np.float64)
    if not (len(di) == len(y_true) == len(p)):
        raise ValueError("di, y_true and p must have equal length.")
    if edges.ndim != 1 or len(edges) < 2 or (np.diff(edges) <= 0).any():
        raise ValueError("edges must be an ascending 1-D array of at least two values.")

    rows = []
    for lo, hi in itertools.pairwise(edges):
        mask = (di >= lo) & (di < hi)
        if not mask.any():
            continue
        row = {
            "di_bin": f"{lo:.2f}-{hi:.2f}",
            "di_lo": float(lo),
            "di_hi": float(hi),
            "n": int(mask.sum()),
        }
        row.update(
            _window_metrics(
                y_true[mask], p[mask], thresholds[mask] if thresholds is not None else None
            )
        )
        rows.append(row)
    return pd.DataFrame(rows)


__all__ = [
    "AOA_INSIDE",
    "AOA_OUTSIDE",
    "AOA_UINT8_NODATA",
    "FeatureScaling",
    "cv_nearest_distances",
    "di_threshold",
    "fit_feature_scaling",
    "mean_pairwise_distance",
    "nearest_distances",
    "performance_by_di_bins",
    "performance_by_di_window",
]
