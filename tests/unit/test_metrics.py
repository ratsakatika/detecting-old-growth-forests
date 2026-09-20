"""Unit tests for utils.metrics."""

import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from utils import metrics
from utils.metrics import (
    ThresholdMetrics,
    best_f1_threshold,
    parcel_aggregations,
    parcel_metrics,
    pixel_metrics,
)

_METRIC_KEYS = {
    "n",
    "prevalence",
    "pr_auc",
    "roc_auc",
    "threshold",
    "f1",
    "precision",
    "recall",
}


def test_import_writes_no_files(tmp_path: Path) -> None:
    # A fresh interpreter importing the module in an empty directory must create
    # nothing. Run in a subprocess so the check cannot perturb this session's
    # module state; importlib.reload would rebind the dataclass and is a footgun
    # once other modules import metrics.
    repo_root = Path(metrics.__file__).resolve().parents[1]
    result = subprocess.run(
        [sys.executable, "-c", "import utils.metrics"],
        cwd=tmp_path,
        env={**os.environ, "PYTHONPATH": str(repo_root)},
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert list(tmp_path.iterdir()) == []


# --- best_f1_threshold ---


def test_best_f1_threshold_separable() -> None:
    result = best_f1_threshold([0, 0, 1, 1], [0.1, 0.2, 0.8, 0.9])
    assert isinstance(result, ThresholdMetrics)
    assert result.threshold == pytest.approx(0.8)
    assert result.f1 == pytest.approx(1.0)
    assert result.precision == pytest.approx(1.0)
    assert result.recall == pytest.approx(1.0)


def test_best_f1_threshold_zero_precision_recall_point() -> None:
    # Top-scored sample is a negative, so at the highest threshold precision and
    # recall are both zero; the guard avoids a divide-by-zero and that point is
    # not chosen. Determinate optimum: threshold 0.4, F1 0.8, precision 2/3.
    result = best_f1_threshold([0, 1, 1], [0.9, 0.5, 0.4])
    assert result.threshold == pytest.approx(0.4)
    assert result.f1 == pytest.approx(0.8)
    assert result.precision == pytest.approx(2.0 / 3.0)
    assert result.recall == pytest.approx(1.0)


def test_best_f1_threshold_rejects_single_class() -> None:
    with pytest.raises(ValueError, match="both classes"):
        best_f1_threshold([1, 1, 1], [0.4, 0.5, 0.6])


# --- pixel_metrics / parcel_metrics ---


def test_pixel_metrics_perfect_classifier() -> None:
    result = pixel_metrics([0, 0, 1, 1], [0.05, 0.1, 0.9, 0.95])
    assert set(result) == _METRIC_KEYS
    assert result["n"] == 4
    assert result["prevalence"] == pytest.approx(0.5)
    assert result["pr_auc"] == pytest.approx(1.0)
    assert result["roc_auc"] == pytest.approx(1.0)
    assert result["f1"] == pytest.approx(1.0)
    assert result["precision"] == pytest.approx(1.0)
    assert result["recall"] == pytest.approx(1.0)


def test_pixel_metrics_known_imperfect_values() -> None:
    # Verified against scikit-learn: ROC-AUC 0.75, average precision 0.8333...
    result = pixel_metrics([0, 1, 0, 1], [0.6, 0.4, 0.3, 0.8])
    assert result["roc_auc"] == pytest.approx(0.75)
    assert result["pr_auc"] == pytest.approx(0.8333333, abs=1e-6)


def test_metric_value_types_are_plain_python() -> None:
    result = pixel_metrics([0, 1, 0, 1], [0.2, 0.8, 0.3, 0.7])
    assert type(result["n"]) is int
    for key in _METRIC_KEYS - {"n"}:
        assert type(result[key]) is float


def test_parcel_metrics_equals_pixel_metrics_on_identical_arrays() -> None:
    y_true = [0, 1, 0, 1, 1]
    scores = [0.2, 0.6, 0.4, 0.8, 0.55]
    assert parcel_metrics(y_true, scores) == pixel_metrics(y_true, scores)


def test_metrics_reject_single_class() -> None:
    with pytest.raises(ValueError, match="both classes"):
        pixel_metrics([0, 0, 0], [0.1, 0.2, 0.3])


def test_metrics_accept_boolean_labels() -> None:
    result = pixel_metrics(np.array([False, False, True, True]), [0.1, 0.2, 0.8, 0.9])
    assert result["prevalence"] == pytest.approx(0.5)


# --- input validation (shared core, exercised through pixel_metrics) ---


@pytest.mark.parametrize(
    ("y_true", "y_prob", "match"),
    [
        ([0, 1, 1], [0.1, 0.2], "length mismatch"),
        ([], [], "empty"),
        ([0, 1], [0.5, np.inf], "non-finite"),
        ([0, 1], [0.5, 1.5], r"\[0, 1\]"),
        ([0, 1], [-0.1, 0.5], r"\[0, 1\]"),
        ([0, 2], [0.5, 0.5], "binary"),
    ],
)
def test_validation_errors(y_true, y_prob, match) -> None:
    with pytest.raises(ValueError, match=match):
        pixel_metrics(y_true, y_prob)


def test_validation_rejects_two_dimensional_inputs() -> None:
    with pytest.raises(ValueError, match="one-dimensional"):
        pixel_metrics(np.zeros((2, 2)), np.zeros((2, 2)))


# --- parcel_aggregations ---


def test_parcel_aggregations_means_and_counts() -> None:
    out = parcel_aggregations([7, 7, 3, 3, 3], [0.2, 0.4, 0.9, 0.7, 0.8])
    assert list(out.columns) == ["parcel_id", "p_mean", "n_pixels"]
    assert out["parcel_id"].tolist() == [3, 7]
    np.testing.assert_allclose(out["p_mean"].tolist(), [0.8, 0.3])
    assert out["n_pixels"].tolist() == [3, 2]
    assert str(out["parcel_id"].dtype) == "int64"
    assert str(out["p_mean"].dtype) == "float64"
    assert str(out["n_pixels"].dtype) == "int64"


def test_parcel_aggregations_feeds_parcel_metrics() -> None:
    pixels = pd.DataFrame({"parcel_id": [1, 1, 2, 2, 3, 3], "prob": [0.1, 0.3, 0.6, 0.8, 0.2, 0.4]})
    label_by_parcel = {1: 0, 2: 1, 3: 0}
    agg = parcel_aggregations(pixels["parcel_id"], pixels["prob"])
    labels = agg["parcel_id"].map(label_by_parcel).to_numpy()
    result = parcel_metrics(labels, agg["p_mean"].to_numpy())
    assert set(result) == _METRIC_KEYS


@pytest.mark.parametrize(
    ("ids", "probs", "match"),
    [
        ([1, 1, 2], [0.2, 0.4], "length mismatch"),
        ([], [], "empty"),
        ([1, 2], [0.5, np.nan], "non-finite"),
        ([1, 2], [0.5, 2.0], r"\[0, 1\]"),
    ],
)
def test_parcel_aggregations_validation(ids, probs, match) -> None:
    with pytest.raises(ValueError, match=match):
        parcel_aggregations(ids, probs)


def test_parcel_aggregations_rejects_two_dimensional() -> None:
    with pytest.raises(ValueError, match="one-dimensional"):
        parcel_aggregations(np.zeros((2, 2)), np.zeros((2, 2)))


def test_metrics_at_threshold_perfect_at_half() -> None:
    out = metrics.metrics_at_threshold([0, 0, 1, 1], [0.1, 0.2, 0.8, 0.9], 0.5)
    assert out == {"threshold": 0.5, "f1": 1.0, "precision": 1.0, "recall": 1.0}


def test_metrics_at_threshold_partial() -> None:
    # threshold 0.85: only the 0.9 sample is positive -> TP=1, FN=1, FP=0.
    out = metrics.metrics_at_threshold([0, 0, 1, 1], [0.1, 0.2, 0.8, 0.9], 0.85)
    assert out["precision"] == pytest.approx(1.0)
    assert out["recall"] == pytest.approx(0.5)
    assert out["f1"] == pytest.approx(2.0 / 3.0)


def test_metrics_at_threshold_zero_division_all_false_positive() -> None:
    # Single-class negatives, all predicted positive: precision/recall 0, no error.
    out = metrics.metrics_at_threshold([0, 0], [0.6, 0.7], 0.5)
    assert out["precision"] == 0.0
    assert out["recall"] == 0.0
    assert out["f1"] == 0.0


@pytest.mark.parametrize("bad", [-0.01, 1.01])
def test_metrics_at_threshold_rejects_out_of_range(bad: float) -> None:
    with pytest.raises(ValueError, match=r"threshold must lie in \[0, 1\]"):
        metrics.metrics_at_threshold([0, 1], [0.2, 0.8], bad)
