"""Unit tests for utils.bootstrap."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from utils import bootstrap
from utils.bootstrap import (
    block_bootstrap_distribution,
    holm_adjusted_p_values,
    load_block_assignment,
    percentile_interval,
    two_sided_bootstrap_p,
)


def test_import_writes_no_files(tmp_path: Path) -> None:
    # A fresh interpreter importing the module in an empty directory must create
    # nothing on disk. Mirrors the pattern used for other utils modules.
    repo_root = Path(bootstrap.__file__).resolve().parents[1]
    result = subprocess.run(
        [sys.executable, "-c", "import utils.bootstrap"],
        cwd=tmp_path,
        env={**os.environ, "PYTHONPATH": str(repo_root)},
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert list(tmp_path.iterdir()) == []


# --- holm_adjusted_p_values ---


def test_holm_known_vector() -> None:
    # Worked example: p = (0.01, 0.04, 0.03, 0.005).
    # Sorted ascending: 0.005, 0.01, 0.03, 0.04 with weights 4, 3, 2, 1.
    # Raw weighted: 0.02, 0.03, 0.06, 0.04.
    # Running maximum -> 0.02, 0.03, 0.06, 0.06.
    # Map back to input order: 0.03, 0.06, 0.06, 0.02.
    result = holm_adjusted_p_values([0.01, 0.04, 0.03, 0.005])
    assert result == pytest.approx([0.03, 0.06, 0.06, 0.02])


def test_holm_clip_to_one() -> None:
    # Largest weighted value exceeds 1 and must be clipped before the running max.
    # Sorted: 0.6, 0.6. Weights 2, 1. Raw min(1, 1.2)=1.0, min(1, 0.6)=0.6.
    # Running max: 1.0, 1.0.
    assert holm_adjusted_p_values([0.6, 0.6]) == pytest.approx([1.0, 1.0])


def test_holm_monotone_in_sorted_order() -> None:
    rng = np.random.default_rng(0)
    p = rng.uniform(0.0, 1.0, size=50)
    adjusted = np.array(holm_adjusted_p_values(p))
    order = np.argsort(p, kind="stable")
    sorted_adjusted = adjusted[order]
    assert np.all(np.diff(sorted_adjusted) >= -1e-12)


def test_holm_rejects_empty() -> None:
    with pytest.raises(ValueError, match="empty"):
        holm_adjusted_p_values([])


def test_holm_rejects_out_of_range() -> None:
    with pytest.raises(ValueError, match=r"\[0, 1\]"):
        holm_adjusted_p_values([0.1, 1.5])
    with pytest.raises(ValueError, match=r"\[0, 1\]"):
        holm_adjusted_p_values([-0.01, 0.1])


def test_holm_rejects_non_finite() -> None:
    with pytest.raises(ValueError, match="non-finite"):
        holm_adjusted_p_values([0.1, float("nan")])


def test_holm_rejects_multidim() -> None:
    with pytest.raises(ValueError, match="one-dimensional"):
        holm_adjusted_p_values(np.array([[0.1, 0.2]]))


# --- percentile_interval ---


def test_percentile_interval_values() -> None:
    samples = np.arange(101, dtype=np.float64)  # 0..100
    lo, hi = percentile_interval(samples, level=0.95)
    assert lo == pytest.approx(2.5)
    assert hi == pytest.approx(97.5)


def test_percentile_interval_ignores_nan() -> None:
    samples = np.array([np.nan, 1.0, 2.0, 3.0, 4.0, np.nan])
    lo, hi = percentile_interval(samples, level=0.5)
    # Finite quartiles of [1, 2, 3, 4]: 1.75 and 3.25.
    assert lo == pytest.approx(1.75)
    assert hi == pytest.approx(3.25)


def test_percentile_interval_level_guard() -> None:
    with pytest.raises(ValueError, match=r"\(0, 1\)"):
        percentile_interval([1.0, 2.0], level=1.0)
    with pytest.raises(ValueError, match=r"\(0, 1\)"):
        percentile_interval([1.0, 2.0], level=0.0)


def test_percentile_interval_no_finite_values() -> None:
    with pytest.raises(ValueError, match="finite"):
        percentile_interval([float("nan"), float("nan")])


# --- two_sided_bootstrap_p ---


def test_two_sided_bootstrap_p_symmetric_about_zero() -> None:
    # 1001 evenly spaced points from -1 to 1: a count of 501 lies on each side
    # (one is exactly zero, counted in both halves under the >= 0 / <= 0 rule).
    samples = np.linspace(-1.0, 1.0, 1001)
    p = two_sided_bootstrap_p(samples)
    # min((501 + 1) / 1002, (501 + 1) / 1002) = 501 / 1001; doubled gives ~1.0.
    assert p == pytest.approx(min(1.0, 2.0 * (501 + 1) / (1001 + 1)))


def test_two_sided_bootstrap_p_all_positive() -> None:
    # All d > 0: #(d <= 0) = 0; smaller side is (0 + 1) / (n + 1).
    samples = np.array([0.1, 0.2, 0.3, 0.4, 0.5])
    p = two_sided_bootstrap_p(samples)
    assert p == pytest.approx(2.0 / (samples.size + 1))


def test_two_sided_bootstrap_p_clipped_to_one() -> None:
    # All exact zeros: both sides count every replicate plus one.
    # 2 * (n + 1) / (n + 1) = 2.0 -> clipped to 1.0.
    samples = np.zeros(7)
    assert two_sided_bootstrap_p(samples) == 1.0


def test_two_sided_bootstrap_p_ignores_nan() -> None:
    samples = np.array([np.nan, 0.1, 0.2, np.nan, 0.3])
    p = two_sided_bootstrap_p(samples)
    assert p == pytest.approx(2.0 / (3 + 1))


def test_two_sided_bootstrap_p_no_finite_values() -> None:
    with pytest.raises(ValueError, match="finite"):
        two_sided_bootstrap_p([float("nan"), float("nan")])


# --- block_bootstrap_distribution ---


def _make_dataset(
    rng_seed: int = 0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Synthetic dataset: 60 observations, 4 blocks of 15, two score columns."""
    rng = np.random.default_rng(rng_seed)
    n_obs = 60
    y_true = rng.integers(0, 2, size=n_obs).astype(np.int64)
    scores = rng.random((n_obs, 2))
    block_ids = np.repeat(np.arange(4, dtype=np.int64), n_obs // 4)
    return y_true, scores, block_ids


def _mean_score(y_true: np.ndarray, score: np.ndarray) -> float:
    """Toy metric: mean score on positives minus mean score on negatives."""
    pos = score[y_true == 1].mean() if (y_true == 1).any() else 0.0
    neg = score[y_true == 0].mean() if (y_true == 0).any() else 0.0
    return float(pos - neg)


def test_block_bootstrap_distribution_shape_and_dtype() -> None:
    y_true, scores, block_ids = _make_dataset()
    out = block_bootstrap_distribution(y_true, scores, block_ids, _mean_score, n_reps=25, seed=123)
    assert out.shape == (25, 2)
    assert out.dtype == np.float64


def test_block_bootstrap_determinism_serial() -> None:
    y_true, scores, block_ids = _make_dataset()
    a = block_bootstrap_distribution(y_true, scores, block_ids, _mean_score, n_reps=50, seed=7)
    b = block_bootstrap_distribution(y_true, scores, block_ids, _mean_score, n_reps=50, seed=7)
    np.testing.assert_array_equal(a, b)


def test_block_bootstrap_serial_matches_parallel() -> None:
    y_true, scores, block_ids = _make_dataset()
    serial = block_bootstrap_distribution(
        y_true, scores, block_ids, _mean_score, n_reps=40, seed=11, n_jobs=1
    )
    parallel = block_bootstrap_distribution(
        y_true, scores, block_ids, _mean_score, n_reps=40, seed=11, n_jobs=2
    )
    np.testing.assert_array_equal(serial, parallel)


def test_block_bootstrap_paired_columns() -> None:
    # When two score columns are identical, the per-replicate difference must be
    # exactly zero on every replicate, which is the paired-resample guarantee.
    y_true, scores_single, block_ids = _make_dataset()
    scores = np.column_stack([scores_single[:, 0], scores_single[:, 0]])
    out = block_bootstrap_distribution(y_true, scores, block_ids, _mean_score, n_reps=30, seed=3)
    np.testing.assert_array_equal(out[:, 0], out[:, 1])


def test_block_bootstrap_min_classes_nan_path() -> None:
    # Configure a dataset where each block is single-class so every resample
    # carries only one class; with min_classes=2 every replicate is NaN.
    y_true = np.concatenate([np.zeros(10, dtype=np.int64), np.zeros(10, dtype=np.int64)])
    block_ids = np.concatenate([np.zeros(10, dtype=np.int64), np.ones(10, dtype=np.int64)])
    scores = np.random.default_rng(0).random((20, 2))
    out = block_bootstrap_distribution(
        y_true, scores, block_ids, _mean_score, n_reps=10, seed=1, min_classes=2
    )
    assert np.all(np.isnan(out))


def test_block_bootstrap_accepts_1d_scores() -> None:
    # A single configuration may pass a 1-D scores array; it is reshaped to a
    # single column for the paired-metric path.
    y_true, scores, block_ids = _make_dataset()
    out = block_bootstrap_distribution(
        y_true, scores[:, 0], block_ids, _mean_score, n_reps=5, seed=1
    )
    assert out.shape == (5, 1)


def test_block_bootstrap_rejects_non_positive_reps() -> None:
    y_true, scores, block_ids = _make_dataset()
    with pytest.raises(ValueError, match="n_reps"):
        block_bootstrap_distribution(y_true, scores, block_ids, _mean_score, n_reps=0, seed=1)


def test_block_bootstrap_rejects_non_positive_min_classes() -> None:
    y_true, scores, block_ids = _make_dataset()
    with pytest.raises(ValueError, match="min_classes"):
        block_bootstrap_distribution(
            y_true, scores, block_ids, _mean_score, n_reps=2, seed=1, min_classes=0
        )


def test_block_bootstrap_rejects_score_length_mismatch() -> None:
    y_true, scores, block_ids = _make_dataset()
    with pytest.raises(ValueError, match="length mismatch"):
        block_bootstrap_distribution(y_true, scores[:-1], block_ids, _mean_score, n_reps=2, seed=1)


def test_block_bootstrap_rejects_block_length_mismatch() -> None:
    y_true, scores, block_ids = _make_dataset()
    with pytest.raises(ValueError, match="length mismatch"):
        block_bootstrap_distribution(y_true, scores, block_ids[:-1], _mean_score, n_reps=2, seed=1)


def test_block_bootstrap_rejects_high_dim_scores() -> None:
    y_true, _, block_ids = _make_dataset()
    bad = np.zeros((y_true.size, 1, 1))
    with pytest.raises(ValueError, match="1- or 2-dimensional"):
        block_bootstrap_distribution(y_true, bad, block_ids, _mean_score, n_reps=2, seed=1)


def test_block_bootstrap_rejects_empty_blocks() -> None:
    y_true = np.empty(0, dtype=np.int64)
    scores = np.empty((0, 1), dtype=np.float64)
    block_ids = np.empty(0, dtype=np.int64)
    with pytest.raises(ValueError, match="at least one block"):
        block_bootstrap_distribution(y_true, scores, block_ids, _mean_score, n_reps=2, seed=1)


# --------------------------------------------------------------------------- #
# load_block_assignment
# --------------------------------------------------------------------------- #


def _partitioned_labels(tmp_path: Path) -> Path:
    import geopandas as gpd
    from shapely.geometry import box

    frame = gpd.GeoDataFrame(
        {
            "parcel_id": ["00003", "00001", "00002"],
            "ogf": [1.0, np.nan, 0.0],
            "bootstrap_id": [2.0, 1.0, 3.0],
            "fold_id": [1.0, 2.0, 3.0],
        },
        geometry=[box(0, 0, 1, 1), box(1, 1, 2, 2), box(2, 2, 3, 3)],
        crs="EPSG:3035",
    )
    path = tmp_path / "ogf_reference_labels_partitioned.gpkg"
    frame.to_file(path, driver="GPKG")
    return path


def test_load_block_assignment_casts_sorts_and_drops_unlabelled(tmp_path: Path) -> None:
    out = load_block_assignment(_partitioned_labels(tmp_path))
    assert list(out.columns) == ["parcel_id", "bootstrap_id", "fold_id", "ogf"]
    assert out["parcel_id"].tolist() == [2, 3]  # the unlabelled parcel 1 is dropped
    assert out["bootstrap_id"].tolist() == [3, 2]
    assert out["fold_id"].tolist() == [3, 1]
    assert out["ogf"].tolist() == [0, 1]
    assert all(out[column].dtype == np.int64 for column in out.columns)
    assert out.index.tolist() == [0, 1]
