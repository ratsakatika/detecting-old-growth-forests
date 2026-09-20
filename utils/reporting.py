"""Parcel-level performance rows: eight metric families x eleven statistics.

Turns saved out-of-fold parcel predictions (headline nested CV or a fixed-HP
buffered refit) into the reporting row used by ``scripts/run_performance_matrix.py``
and the gap detail in ``scripts/run_gap_analysis.py``. Eight metric families -
``pr_auc`` and ``roc_auc`` (threshold-free), and ``f1``/``precision``/``recall``
at both the *outer* test-fold F1-maximising threshold and the *inner* nested-CV
threshold - each receive eleven statistics: the pooled out-of-fold point estimate,
its block-bootstrap 95% interval, the six per-fold values, and the across-fold
mean and sd. :func:`metric_matrix_row` builds one config's row; :func:`paired_row`
differences two configs on the *same* block resamples (a paired contrast).

This is an evaluation bootstrap over saved predictions (:mod:`utils.bootstrap`); it
never retrains and reuses :mod:`utils.metrics` for the operating points. The inner
threshold is supplied per fold by the caller (the deployable threshold selected
during training) rather than re-derived, so a buffered arm reuses the headline
operating point. Importing this module binds names only: no input/output, no
logging, no ``scripts`` dependency (callers pass the block assignment in).
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Final

import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from utils.bootstrap import block_bootstrap_distribution, percentile_interval, two_sided_bootstrap_p
from utils.metrics import best_f1_threshold

# --------------------------------------------------------------------------- #
# Metric families and the statistics reported for each.
# --------------------------------------------------------------------------- #


def _f1(y_true: np.ndarray, prediction: np.ndarray) -> float:
    return float(f1_score(y_true, prediction >= 0.5, zero_division=0.0))


def _precision(y_true: np.ndarray, prediction: np.ndarray) -> float:
    return float(precision_score(y_true, prediction >= 0.5, zero_division=0.0))


def _recall(y_true: np.ndarray, prediction: np.ndarray) -> float:
    return float(recall_score(y_true, prediction >= 0.5, zero_division=0.0))


@dataclass(frozen=True, slots=True)
class MetricSpec:
    """One reported metric family.

    Attributes:
        name: Output name (for example ``"f1_outer"``).
        kind: ``"ranking"`` scores the probability directly; ``"outer"`` scores a
            0/1 prediction thresholded at the pooled outer F1-maximising threshold;
            ``"inner"`` scores a 0/1 prediction thresholded at each fold's inner
            (nested-CV) threshold.
        func: ``(y_true, score_column) -> float`` applied per fold, pooled, and on
            every bootstrap resample. Ranking metrics receive the probability;
            threshold metrics receive a 0/1 prediction.
    """

    name: str
    kind: str
    func: Callable[[np.ndarray, np.ndarray], float]


METRIC_SPECS: Final[tuple[MetricSpec, ...]] = (
    MetricSpec("pr_auc", "ranking", lambda y, p: float(average_precision_score(y, p))),
    MetricSpec("roc_auc", "ranking", lambda y, p: float(roc_auc_score(y, p))),
    MetricSpec("f1_outer", "outer", _f1),
    MetricSpec("precision_outer", "outer", _precision),
    MetricSpec("recall_outer", "outer", _recall),
    MetricSpec("f1_inner", "inner", _f1),
    MetricSpec("precision_inner", "inner", _precision),
    MetricSpec("recall_inner", "inner", _recall),
)

METRIC_NAMES: Final[tuple[str, ...]] = tuple(spec.name for spec in METRIC_SPECS)

# The eleven statistics per metric family, in output order. ``fold1``..``fold6``
# carry the six per-fold values (NaN where a fold is absent in a provisional run).
STAT_NAMES: Final[tuple[str, ...]] = (
    "pooled",
    "ci_lo",
    "ci_hi",
    "fold1",
    "fold2",
    "fold3",
    "fold4",
    "fold5",
    "fold6",
    "fold_mean",
    "fold_sd",
)

_FOLD_IDS: Final[tuple[int, ...]] = (1, 2, 3, 4, 5, 6)


def matrix_columns() -> list[str]:
    """The 88 metric column names, ``{metric}__{stat}`` in family-major order."""
    return [f"{name}__{stat}" for name in METRIC_NAMES for stat in STAT_NAMES]


def _nan_row() -> dict[str, float]:
    """An all-NaN 88-value row (a pending config with no completed folds)."""
    return dict.fromkeys(matrix_columns(), float("nan"))


# --------------------------------------------------------------------------- #
# Score columns: probability, outer-threshold prediction, inner-threshold prediction.
# --------------------------------------------------------------------------- #


def _pooled_frame(fold_frames: Mapping[int, pd.DataFrame]) -> pd.DataFrame:
    """Concatenate per-fold parcel predictions, keeping the owning fold id."""
    parts = [frame.assign(outer_fold=int(fold)) for fold, frame in sorted(fold_frames.items())]
    return pd.concat(parts, ignore_index=True)


def _score_columns(
    p_mean: np.ndarray,
    y_true: np.ndarray,
    outer_fold: np.ndarray,
    inner_thresholds: Mapping[int, float],
) -> tuple[dict[str, np.ndarray], float]:
    """Probability and the two 0/1 prediction vectors, plus the pooled outer threshold.

    The outer prediction thresholds the probability at the pooled F1-maximising
    cut-off; the inner prediction thresholds each parcel at its own fold's inner
    threshold (the deployable operating point selected during training).
    """
    outer_tau = best_f1_threshold(y_true, p_mean).threshold
    per_parcel_inner = np.array(
        [float(inner_thresholds[int(fold)]) for fold in outer_fold], dtype=np.float64
    )
    columns = {
        "ranking": p_mean,
        "outer": (p_mean >= outer_tau).astype(np.float64),
        "inner": (p_mean >= per_parcel_inner).astype(np.float64),
    }
    return columns, float(outer_tau)


def _bootstrap_intervals(
    y_true: np.ndarray,
    columns: Mapping[str, np.ndarray],
    block_ids: np.ndarray,
    *,
    reps: int,
    seed: int,
    n_jobs: int,
) -> dict[str, tuple[float, float]]:
    """Block-bootstrap 95% interval for every metric family.

    Metrics sharing a callable and a score column kind are bootstrapped together so
    each ``(metric, kind)`` group is one resampling pass. Returns ``name -> (lo, hi)``.
    """
    intervals: dict[str, tuple[float, float]] = {}
    # Group specs by their callable so one bootstrap pass covers, for example, f1 at
    # both the outer and inner threshold (each on its own score column).
    groups: dict[int, list[MetricSpec]] = {}
    for spec in METRIC_SPECS:
        groups.setdefault(id(spec.func), []).append(spec)
    for specs in groups.values():
        score = np.column_stack([columns[spec.kind] for spec in specs])
        distribution = block_bootstrap_distribution(
            y_true,
            score,
            block_ids,
            specs[0].func,
            n_reps=reps,
            seed=seed,
            n_jobs=n_jobs,
            min_classes=2,
        )
        for column, spec in enumerate(specs):
            intervals[spec.name] = percentile_interval(distribution[:, column], level=0.95)
    return intervals


# --------------------------------------------------------------------------- #
# Per-fold and pooled point estimates.
# --------------------------------------------------------------------------- #


def _fold_values(
    fold_frames: Mapping[int, pd.DataFrame], inner_thresholds: Mapping[int, float]
) -> dict[str, dict[int, float]]:
    """Per-fold value of every metric family, keyed ``name -> {fold: value}``."""
    values: dict[str, dict[int, float]] = {name: {} for name in METRIC_NAMES}
    for fold, frame in fold_frames.items():
        y_true = frame["y_true"].to_numpy().astype(np.int64)
        p_mean = frame["p_mean"].to_numpy().astype(np.float64)
        columns, _ = _score_columns(
            p_mean, y_true, np.full(p_mean.shape, int(fold)), inner_thresholds
        )
        for spec in METRIC_SPECS:
            values[spec.name][int(fold)] = spec.func(y_true, columns[spec.kind])
    return values


def metric_matrix_row(
    fold_frames: Mapping[int, pd.DataFrame],
    inner_thresholds: Mapping[int, float],
    block_assignment: pd.DataFrame,
    *,
    reps: int = 5000,
    seed: int = 42,
    n_jobs: int = 1,
) -> dict[str, float]:
    """Build one config's 88-value reporting row from saved parcel predictions.

    Args:
        fold_frames: ``fold id -> parcel-prediction frame`` with columns
            ``parcel_id``, ``y_true`` and ``p_mean`` (the saved out-of-fold
            predictions for that fold). May be a subset of the six folds.
        inner_thresholds: ``fold id -> parcel inner threshold`` (the nested-CV
            operating point). A buffered arm passes the headline thresholds.
        block_assignment: Frame with ``parcel_id`` and ``bootstrap_id``; the pooled
            interval resamples these area-balanced blocks.
        reps: Bootstrap replicates for the pooled interval.
        seed: Seed for the block bootstrap.
        n_jobs: Joblib workers for the bootstrap.

    Returns:
        A flat dict over :func:`matrix_columns` (88 entries) plus ``folds_used``.
        A config with no folds returns an all-NaN row and ``folds_used = 0``.
    """
    row = _nan_row()
    folds = sorted(int(f) for f in fold_frames)
    row["folds_used"] = float(len(folds))
    if not folds:
        return row

    fold_values = _fold_values(fold_frames, inner_thresholds)
    pooled = _pooled_frame(fold_frames)
    y_true = pooled["y_true"].to_numpy().astype(np.int64)
    p_mean = pooled["p_mean"].to_numpy().astype(np.float64)
    columns, _ = _score_columns(
        p_mean, y_true, pooled["outer_fold"].to_numpy().astype(np.int64), inner_thresholds
    )

    # The interval resamples only parcels carrying a block id (every labelled parcel
    # in practice); point estimates use the whole pool. The binarisation is shared,
    # so the interval brackets the same operating point.
    block_series = pooled.merge(
        block_assignment[["parcel_id", "bootstrap_id"]], on="parcel_id", how="left"
    )["bootstrap_id"]
    has_block = block_series.notna().to_numpy()
    intervals = _bootstrap_intervals(
        y_true[has_block],
        {kind: column[has_block] for kind, column in columns.items()},
        block_series.to_numpy()[has_block].astype(np.int64),
        reps=reps,
        seed=seed,
        n_jobs=n_jobs,
    )

    for spec in METRIC_SPECS:
        per_fold = fold_values[spec.name]
        fold_array = np.array([per_fold[f] for f in folds], dtype=np.float64)
        row[f"{spec.name}__pooled"] = spec.func(y_true, columns[spec.kind])
        row[f"{spec.name}__ci_lo"], row[f"{spec.name}__ci_hi"] = intervals[spec.name]
        for fold in _FOLD_IDS:
            row[f"{spec.name}__fold{fold}"] = per_fold.get(fold, float("nan"))
        row[f"{spec.name}__fold_mean"] = float(np.mean(fold_array))
        row[f"{spec.name}__fold_sd"] = (
            float(np.std(fold_array, ddof=1)) if fold_array.size > 1 else float("nan")
        )
    return row


# --------------------------------------------------------------------------- #
# Paired contrast (config A minus config B on the same resamples).
# --------------------------------------------------------------------------- #


def _aligned_difference_columns(
    pooled_a: pd.DataFrame,
    pooled_b: pd.DataFrame,
    inner_a: Mapping[int, float],
    inner_b: Mapping[int, float],
    block_assignment: pd.DataFrame,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, tuple[np.ndarray, np.ndarray]]]:
    """Inner-join two pooled frames on parcels; return ``(y_true, has_block, block_ids, columns)``.

    ``columns[kind]`` is the ``(a, b)`` pair of aligned score vectors for that kind.
    Both configs share the same parcels and labels, so a single resample pairs them.
    ``has_block`` masks the parcels carrying a block id (those the interval resamples).
    """
    merged = pooled_a.merge(pooled_b, on="parcel_id", how="inner", suffixes=("_a", "_b")).merge(
        block_assignment[["parcel_id", "bootstrap_id"]], on="parcel_id", how="left"
    )
    y_true = merged["y_true_a"].to_numpy().astype(np.int64)
    has_block = merged["bootstrap_id"].notna().to_numpy()
    block_ids = merged["bootstrap_id"].to_numpy()[has_block].astype(np.int64)
    cols_a, _ = _score_columns(
        merged["p_mean_a"].to_numpy().astype(np.float64),
        y_true,
        merged["outer_fold_a"].to_numpy().astype(np.int64),
        inner_a,
    )
    cols_b, _ = _score_columns(
        merged["p_mean_b"].to_numpy().astype(np.float64),
        y_true,
        merged["outer_fold_b"].to_numpy().astype(np.int64),
        inner_b,
    )
    columns = {kind: (cols_a[kind], cols_b[kind]) for kind in cols_a}
    return y_true, has_block, block_ids, columns


def paired_row(
    frames_a: Mapping[int, pd.DataFrame],
    frames_b: Mapping[int, pd.DataFrame],
    inner_a: Mapping[int, float],
    inner_b: Mapping[int, float],
    block_assignment: pd.DataFrame,
    *,
    reps: int = 5000,
    seed: int = 42,
    n_jobs: int = 1,
) -> dict[str, float]:
    """Build the A-minus-B contrast row (88 values) on paired block resamples.

    Both configs are scored on the same parcels, so each metric's pooled difference
    has a paired-bootstrap interval and a two-sided p-value (``{metric}__p``). Folds
    common to both arms supply the per-fold differences. Returns an all-NaN row with
    ``folds_used = 0`` when the two arms share no folds.
    """
    row = _nan_row()
    for name in METRIC_NAMES:
        row[f"{name}__p"] = float("nan")
    common_folds = sorted(set(map(int, frames_a)) & set(map(int, frames_b)))
    row["folds_used"] = float(len(common_folds))
    if not common_folds:
        return row

    sub_a = {f: frames_a[f] for f in common_folds}
    sub_b = {f: frames_b[f] for f in common_folds}
    values_a = _fold_values(sub_a, inner_a)
    values_b = _fold_values(sub_b, inner_b)
    y_true, has_block, block_ids, columns = _aligned_difference_columns(
        _pooled_frame(sub_a), _pooled_frame(sub_b), inner_a, inner_b, block_assignment
    )

    # One paired bootstrap per (callable, kind) group, then difference the columns.
    groups: dict[tuple[int, str], list[MetricSpec]] = {}
    for spec in METRIC_SPECS:
        groups.setdefault((id(spec.func), spec.kind), []).append(spec)
    diff_distribution: dict[str, np.ndarray] = {}
    for (_, kind), specs in groups.items():
        score_a, score_b = columns[kind]
        score = np.column_stack([score_a[has_block], score_b[has_block]])
        distribution = block_bootstrap_distribution(
            y_true[has_block],
            score,
            block_ids,
            specs[0].func,
            n_reps=reps,
            seed=seed,
            n_jobs=n_jobs,
        )
        for spec in specs:
            diff_distribution[spec.name] = distribution[:, 0] - distribution[:, 1]

    for spec in METRIC_SPECS:
        pooled_a = spec.func(y_true, columns[spec.kind][0])
        pooled_b = spec.func(y_true, columns[spec.kind][1])
        diff = diff_distribution[spec.name]
        lo, hi = percentile_interval(diff, level=0.95)
        fold_diffs = np.array(
            [values_a[spec.name][f] - values_b[spec.name][f] for f in common_folds],
            dtype=np.float64,
        )
        row[f"{spec.name}__pooled"] = float(pooled_a - pooled_b)
        row[f"{spec.name}__ci_lo"] = lo
        row[f"{spec.name}__ci_hi"] = hi
        row[f"{spec.name}__p"] = two_sided_bootstrap_p(diff)
        for fold in _FOLD_IDS:
            idx = common_folds.index(fold) if fold in common_folds else None
            row[f"{spec.name}__fold{fold}"] = (
                float(fold_diffs[idx]) if idx is not None else float("nan")
            )
        row[f"{spec.name}__fold_mean"] = float(np.mean(fold_diffs))
        row[f"{spec.name}__fold_sd"] = (
            float(np.std(fold_diffs, ddof=1)) if fold_diffs.size > 1 else float("nan")
        )
    return row


def inner_thresholds_from_per_fold(per_fold_metrics: pd.DataFrame) -> dict[int, float]:
    """Map ``outer_fold -> parcel inner threshold`` from a ``per_fold_metrics`` frame."""
    parcel = per_fold_metrics[per_fold_metrics["level"] == "parcel"]
    return {
        int(fold): float(threshold)
        for fold, threshold in zip(parcel["outer_fold"], parcel["inner_threshold"], strict=True)
    }


__all__ = [
    "METRIC_NAMES",
    "METRIC_SPECS",
    "STAT_NAMES",
    "MetricSpec",
    "inner_thresholds_from_per_fold",
    "matrix_columns",
    "metric_matrix_row",
    "paired_row",
]
