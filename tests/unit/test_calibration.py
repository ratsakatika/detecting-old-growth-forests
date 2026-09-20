"""Unit tests for utils.calibration (the five calibrators and their selection)."""

import numpy as np
import pandas as pd
import pytest

from utils.calibration import (
    CALIBRATION_METHODS,
    CALIBRATION_METRICS,
    calibration_metrics,
    compare_calibrators,
    fit_calibrator,
    select_best_calibrator,
)


def _overconfident(n: int, seed: int, *, sharpen: float = 2.0) -> tuple[np.ndarray, np.ndarray]:
    """Labels drawn from a true probability, and an overconfident raw score of it."""
    rng = np.random.default_rng(seed)
    p_true = np.clip(rng.uniform(0.02, 0.98, n), 0.02, 0.98)
    y = (rng.uniform(0.0, 1.0, n) < p_true).astype(np.int64)
    logit = np.log(p_true / (1.0 - p_true))
    raw = 1.0 / (1.0 + np.exp(-sharpen * logit))  # temperature distortion with T = 1/sharpen
    return y, raw


@pytest.mark.parametrize("method", CALIBRATION_METHODS)
def test_each_calibrator_returns_valid_probabilities(method: str) -> None:
    y, raw = _overconfident(1500, 0)
    calibrator = fit_calibrator(method, y, raw)
    out = calibrator.predict(raw)
    assert calibrator.method == method
    assert out.shape == raw.shape
    assert np.isfinite(out).all()
    assert out.min() >= 0.0 and out.max() <= 1.0
    # Extremes are clipped, not extrapolated out of range.
    edge = calibrator.predict(np.array([0.0, 1.0]))
    assert edge.min() >= 0.0 and edge.max() <= 1.0


@pytest.mark.parametrize("method", ["isotonic", "platt", "beta", "temperature"])
def test_monotone_calibrators_are_non_decreasing(method: str) -> None:
    y, raw = _overconfident(2000, 1)
    calibrator = fit_calibrator(method, y, raw)
    grid = np.linspace(0.0, 1.0, 101)
    mapped = calibrator.predict(grid)
    assert np.all(np.diff(mapped) >= -1e-9)


@pytest.mark.parametrize("method", CALIBRATION_METHODS)
def test_calibration_reduces_held_out_ece(method: str) -> None:
    y_tr, raw_tr = _overconfident(4000, 2)
    y_te, raw_te = _overconfident(4000, 3)
    calibrator = fit_calibrator(method, y_tr, raw_tr)
    raw_ece = calibration_metrics(y_te, raw_te)["ece"]
    cal_ece = calibration_metrics(y_te, calibrator.predict(raw_te))["ece"]
    assert cal_ece < raw_ece


def test_calibration_metrics_finite_and_low_for_calibrated_input() -> None:
    rng = np.random.default_rng(4)
    p = rng.uniform(0.0, 1.0, 5000)
    y = (rng.uniform(0.0, 1.0, 5000) < p).astype(np.int64)  # already calibrated
    metrics = calibration_metrics(y, p)
    assert set(metrics) == set(CALIBRATION_METRICS)
    assert all(np.isfinite(v) for v in metrics.values())
    assert metrics["ece"] < 0.05


def test_compare_calibrators_frame_shape() -> None:
    y, raw = _overconfident(2000, 5)
    frame = compare_calibrators(y, raw, n_splits=4)
    assert list(frame.columns) == ["method", *CALIBRATION_METRICS]
    assert list(frame["method"]) == [*CALIBRATION_METHODS, "uncalibrated"]
    assert frame[list(CALIBRATION_METRICS)].to_numpy().dtype.kind == "f"
    assert np.isfinite(frame[list(CALIBRATION_METRICS)].to_numpy()).all()


def test_select_best_returns_method_and_comparison() -> None:
    y, raw = _overconfident(3000, 6)
    best, comparison = select_best_calibrator(y, raw, n_splits=4)
    assert best in CALIBRATION_METHODS
    assert isinstance(comparison, pd.DataFrame)
    # The selected method has the lowest cross-validated ECE among the candidates.
    candidates = comparison[comparison["method"].isin(CALIBRATION_METHODS)]
    assert best == candidates.sort_values(["ece", "brier"], kind="stable").iloc[0]["method"]
    # Calibration improves on the raw scores for this clearly miscalibrated input.
    raw_ece = float(comparison.loc[comparison["method"] == "uncalibrated", "ece"].iloc[0])
    best_ece = float(candidates.loc[candidates["method"] == best, "ece"].iloc[0])
    assert best_ece <= raw_ece


def test_fit_calibrator_rejects_unknown_method() -> None:
    y, raw = _overconfident(200, 7)
    with pytest.raises(ValueError, match="unknown calibration method"):
        fit_calibrator("histogram", y, raw)


@pytest.mark.parametrize(
    "y_true, y_prob, match",
    [
        ([0, 1], [0.5], "length mismatch"),
        ([], [], "empty"),
        ([0, 1], [0.5, 1.5], r"\[0, 1\]"),
        ([0, 2], [0.5, 0.5], "binary"),
        ([[0, 1]], [[0.5, 0.5]], "one-dimensional"),
        ([0, 1], [0.5, np.nan], "non-finite"),
    ],
)
def test_validation_errors(y_true: list, y_prob: list, match: str) -> None:
    with pytest.raises(ValueError, match=match):
        calibration_metrics(y_true, y_prob)


def test_spline_falls_back_with_few_distinct_scores() -> None:
    rng = np.random.default_rng(8)
    raw = rng.choice([0.2, 0.5, 0.8], size=300)  # only three distinct scores
    y = (rng.uniform(0.0, 1.0, 300) < raw).astype(np.int64)
    out = fit_calibrator("spline", y, raw).predict(raw)
    assert out.min() >= 0.0 and out.max() <= 1.0 and np.isfinite(out).all()
