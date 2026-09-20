"""Unit tests for utils.reporting (the 8-metric x 11-stat performance row)."""

import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import average_precision_score, roc_auc_score

from utils.reporting import (
    METRIC_NAMES,
    STAT_NAMES,
    inner_thresholds_from_per_fold,
    matrix_columns,
    metric_matrix_row,
    paired_row,
)


def _fold_frame(fold: int, n: int, rng: np.random.Generator, *, shift: float = 0.0) -> pd.DataFrame:
    """Parcel predictions for one fold: balanced classes with a separable signal."""
    y = np.tile([0, 1], n // 2).astype(np.int64)
    base = fold * 1000
    p = np.clip(0.5 + (y - 0.5) * 0.6 + shift + rng.normal(0, 0.1, y.size), 0.01, 0.99)
    return pd.DataFrame(
        {"parcel_id": np.arange(base, base + y.size, dtype=np.int64), "y_true": y, "p_mean": p}
    )


def _blocks(fold_frames: dict[int, pd.DataFrame], n_blocks: int = 5) -> pd.DataFrame:
    ids = np.concatenate([frame["parcel_id"].to_numpy() for frame in fold_frames.values()])
    rng = np.random.default_rng(7)
    return pd.DataFrame({"parcel_id": ids, "bootstrap_id": rng.integers(1, n_blocks + 1, ids.size)})


def test_matrix_columns_count() -> None:
    columns = matrix_columns()
    assert len(columns) == len(METRIC_NAMES) * len(STAT_NAMES) == 88
    assert "pr_auc__pooled" in columns
    assert "recall_inner__fold_sd" in columns


def test_metric_matrix_row_reproduces_points_and_fold_summaries() -> None:
    rng = np.random.default_rng(0)
    frames = {fold: _fold_frame(fold, 40, rng) for fold in (1, 2, 3)}
    blocks = _blocks(frames)
    row = metric_matrix_row(frames, {1: 0.5, 2: 0.45, 3: 0.55}, blocks, reps=64, seed=42)

    assert row["folds_used"] == 3.0
    assert set(matrix_columns()).issubset(row)

    pooled = pd.concat(frames.values())
    y = pooled["y_true"].to_numpy()
    p = pooled["p_mean"].to_numpy()
    assert row["pr_auc__pooled"] == pytest.approx(average_precision_score(y, p))
    assert row["roc_auc__pooled"] == pytest.approx(roc_auc_score(y, p))
    assert np.isfinite(row["f1_inner__pooled"])

    assert row["pr_auc__ci_lo"] <= row["pr_auc__ci_hi"]
    assert np.isfinite(row["pr_auc__ci_lo"])
    assert row["pr_auc__fold_mean"] == pytest.approx(
        np.mean([row[f"pr_auc__fold{k}"] for k in (1, 2, 3)])
    )
    assert np.isnan(row["pr_auc__fold4"])  # only three folds present


def test_inner_threshold_zero_predicts_all_positive() -> None:
    rng = np.random.default_rng(1)
    frames = {fold: _fold_frame(fold, 30, rng) for fold in (1, 2)}
    blocks = _blocks(frames)
    row = metric_matrix_row(frames, {1: 0.0, 2: 0.0}, blocks, reps=32, seed=1)

    prevalence = float(pd.concat(frames.values())["y_true"].mean())
    assert row["recall_inner__pooled"] == pytest.approx(1.0)
    assert row["precision_inner__pooled"] == pytest.approx(prevalence)


def test_metric_matrix_row_pending_when_no_folds() -> None:
    blocks = pd.DataFrame({"parcel_id": [1], "bootstrap_id": [1]})
    row = metric_matrix_row({}, {}, blocks)
    assert row["folds_used"] == 0.0
    assert all(np.isnan(row[column]) for column in matrix_columns())


def test_single_fold_has_nan_fold_sd() -> None:
    rng = np.random.default_rng(2)
    frames = {1: _fold_frame(1, 40, rng)}
    row = metric_matrix_row(frames, {1: 0.5}, _blocks(frames), reps=32, seed=1)
    assert row["folds_used"] == 1.0
    assert np.isfinite(row["pr_auc__fold1"])
    assert np.isnan(row["pr_auc__fold2"])
    assert np.isnan(row["pr_auc__fold_sd"])


def test_paired_row_zero_for_identical_configs() -> None:
    rng = np.random.default_rng(3)
    frames = {fold: _fold_frame(fold, 40, rng) for fold in (1, 2, 3)}
    inner = {1: 0.5, 2: 0.5, 3: 0.5}
    row = paired_row(frames, frames, inner, inner, _blocks(frames), reps=64, seed=5)

    assert row["folds_used"] == 3.0
    for name in METRIC_NAMES:
        assert row[f"{name}__pooled"] == pytest.approx(0.0, abs=1e-9)
        assert row[f"{name}__fold_mean"] == pytest.approx(0.0, abs=1e-9)
        assert row[f"{name}__ci_lo"] == pytest.approx(0.0, abs=1e-9)
        assert row[f"{name}__ci_hi"] == pytest.approx(0.0, abs=1e-9)
        assert np.isfinite(row[f"{name}__p"])


def test_paired_row_pending_when_no_common_folds() -> None:
    rng = np.random.default_rng(4)
    arm_a = {1: _fold_frame(1, 20, rng)}
    arm_b = {2: _fold_frame(2, 20, rng)}
    blocks = _blocks({**arm_a, **arm_b})
    row = paired_row(arm_a, arm_b, {1: 0.5}, {2: 0.5}, blocks)
    assert row["folds_used"] == 0.0
    assert np.isnan(row["pr_auc__pooled"])
    assert np.isnan(row["pr_auc__p"])


def test_paired_row_single_common_fold_nan_sd() -> None:
    rng = np.random.default_rng(6)
    arm_a = {1: _fold_frame(1, 30, rng), 2: _fold_frame(2, 30, rng)}
    arm_b = {1: _fold_frame(1, 30, rng)}  # same fold-1 parcels, different probabilities
    row = paired_row(arm_a, arm_b, {1: 0.5, 2: 0.5}, {1: 0.5}, _blocks(arm_a), reps=32, seed=1)
    assert row["folds_used"] == 1.0
    assert np.isfinite(row["pr_auc__fold1"])
    assert np.isnan(row["pr_auc__fold_sd"])


def test_inner_thresholds_from_per_fold() -> None:
    per_fold = pd.DataFrame(
        {
            "outer_fold": [1, 2, 1, 2],
            "level": ["parcel", "parcel", "pixel", "pixel"],
            "inner_threshold": [0.4, 0.6, 0.1, 0.2],
        }
    )
    assert inner_thresholds_from_per_fold(per_fold) == {1: 0.4, 2: 0.6}
