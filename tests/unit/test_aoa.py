"""Unit tests for utils.aoa.

All tests run tiny synthetic problems on the CPU: nearest-neighbour results are
checked against brute force, threshold statistics against hand-computed
quartiles, and the windowed performance curve against its documented columns
and monotonicity guarantees.
"""

import numpy as np
import pytest

from utils.aoa import (
    AOA_INSIDE,
    AOA_OUTSIDE,
    AOA_UINT8_NODATA,
    cv_nearest_distances,
    di_threshold,
    fit_feature_scaling,
    mean_pairwise_distance,
    nearest_distances,
    performance_by_di_bins,
    performance_by_di_window,
)

RNG = np.random.default_rng(42)


def _brute_nearest(query: np.ndarray, reference: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    d = np.linalg.norm(query[:, None, :] - reference[None, :, :], axis=2)
    return d.min(axis=1), d.argmin(axis=1)


class TestSentinels:
    def test_values(self) -> None:
        assert (AOA_INSIDE, AOA_OUTSIDE, AOA_UINT8_NODATA) == (1, 0, 255)


class TestFeatureScaling:
    def test_transform_matches_manual(self) -> None:
        train = RNG.normal(size=(50, 3)) * [1.0, 10.0, 100.0] + [0.0, 5.0, -20.0]
        weights = np.array([2.0, 0.5, 0.0])
        scaling = fit_feature_scaling(train, weights)
        transformed = scaling.transform(train)
        manual = (train - train.mean(axis=0)) / train.std(axis=0) * weights
        np.testing.assert_allclose(transformed, manual.astype(np.float32), rtol=1e-6)
        # Zero-weight features contribute nothing to distances.
        assert np.all(transformed[:, 2] == 0.0)

    def test_constant_feature_guard(self) -> None:
        train = np.column_stack([RNG.normal(size=20), np.full(20, 7.0)])
        scaling = fit_feature_scaling(train, np.array([1.0, 1.0]))
        assert scaling.sds[1] == 1.0
        assert np.isfinite(scaling.transform(train)).all()

    def test_negative_importance_raises(self) -> None:
        with pytest.raises(ValueError, match="non-negative"):
            fit_feature_scaling(RNG.normal(size=(10, 2)), np.array([1.0, -0.1]))

    def test_shape_mismatch_raises(self) -> None:
        with pytest.raises(ValueError, match="importances"):
            fit_feature_scaling(RNG.normal(size=(10, 2)), np.array([1.0]))


class TestNearestDistances:
    def test_matches_brute_force(self) -> None:
        query = RNG.normal(size=(37, 5)).astype(np.float32)
        reference = RNG.normal(size=(53, 5)).astype(np.float32)
        distances, indices = nearest_distances(
            query, reference, device="cpu", query_block=8, reference_block=16, return_indices=True
        )
        expected_d, expected_i = _brute_nearest(query, reference)
        np.testing.assert_allclose(distances, expected_d, rtol=1e-4, atol=1e-5)
        np.testing.assert_array_equal(indices, expected_i)

    def test_empty_query(self) -> None:
        distances, indices = nearest_distances(
            np.empty((0, 3), dtype=np.float32), RNG.normal(size=(5, 3)), device="cpu"
        )
        assert distances.size == 0 and indices is None

    def test_empty_reference_raises(self) -> None:
        with pytest.raises(ValueError, match="Reference"):
            nearest_distances(RNG.normal(size=(3, 2)), np.empty((0, 2)), device="cpu")


class TestCvNearestDistances:
    def test_excludes_own_fold(self) -> None:
        # Two tight clusters; fold 1 = cluster A, fold 2 = cluster B. With the
        # own fold excluded, every nearest distance is the between-cluster gap.
        cluster_a = RNG.normal(scale=0.01, size=(10, 2)) + np.array([0.0, 0.0])
        cluster_b = RNG.normal(scale=0.01, size=(10, 2)) + np.array([10.0, 0.0])
        weighted = np.vstack([cluster_a, cluster_b]).astype(np.float32)
        folds = np.array([1] * 10 + [2] * 10)
        distances = cv_nearest_distances(weighted, folds, device="cpu")
        assert distances.min() > 9.0  # within-cluster (~0.01) would fail this

    def test_single_fold_raises(self) -> None:
        with pytest.raises(ValueError, match="two folds"):
            cv_nearest_distances(RNG.normal(size=(5, 2)).astype(np.float32), np.ones(5))


class TestMeanPairwiseDistance:
    def test_exact_on_full_set(self) -> None:
        points = RNG.normal(size=(12, 3)).astype(np.float32)
        mean, used = mean_pairwise_distance(points, sample_size=12, seed=0, device="cpu", block=5)
        d = np.linalg.norm(points[:, None] - points[None, :], axis=2)
        expected = d.sum() / (12 * 11)
        assert used == 12
        assert mean == pytest.approx(expected, rel=1e-5)

    def test_subset_is_deterministic(self) -> None:
        points = RNG.normal(size=(200, 4)).astype(np.float32)
        first, _ = mean_pairwise_distance(points, sample_size=50, seed=7, device="cpu")
        second, _ = mean_pairwise_distance(points, sample_size=50, seed=7, device="cpu")
        assert first == second

    def test_too_few_raises(self) -> None:
        with pytest.raises(ValueError, match="two training samples"):
            mean_pairwise_distance(
                np.ones((1, 2), dtype=np.float32), sample_size=5, seed=0, device="cpu"
            )


class TestDiThreshold:
    def test_known_quartiles_and_outlier_removed_max(self) -> None:
        di = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 5.0])
        stats = di_threshold(di)
        q1, q3 = np.quantile(di, [0.25, 0.75])
        whisker = q3 + 1.5 * (q3 - q1)
        assert stats["q1"] == pytest.approx(q1)
        assert stats["upper_whisker"] == pytest.approx(whisker)
        # 5.0 is the only value above the whisker; the threshold is the
        # outlier-removed maximum, i.e. 0.8.
        assert stats["n_above_whisker"] == 1
        assert stats["aoa_threshold"] == pytest.approx(0.8)
        assert stats["q95_sensitivity"] == pytest.approx(np.quantile(di, 0.95))

    def test_too_few_values_raises(self) -> None:
        with pytest.raises(ValueError, match="four"):
            di_threshold(np.array([0.1, 0.2, 0.3]))


class TestPerformanceByDiWindow:
    def _synthetic(self, n: int = 2_000) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        di = np.sort(RNG.uniform(0, 1, size=n))
        # Signal degrades with DI: probabilities become noisier at high DI.
        y = RNG.binomial(1, 0.3, size=n)
        noise = RNG.uniform(0, 1, size=n)
        p = np.where(RNG.uniform(size=n) < 1 - di, y * 0.8 + 0.1, noise)
        return di, y, p

    def test_columns_and_monotone_fits(self) -> None:
        di, y, p = self._synthetic()
        frame = performance_by_di_window(di, y, p, window=400, step=200)
        assert list(frame.columns) == [
            "di_lo",
            "di_mid",
            "di_hi",
            "n",
            "prevalence",
            "pr_auc",
            "pr_auc_lift",
            "roc_auc",
            "f1",
            "brier",
            "pr_auc_fit",
            "roc_auc_fit",
            "f1_fit",
            "brier_fit",
        ]
        assert (frame["di_lo"] <= frame["di_mid"]).all()
        assert frame["f1"].isna().all()  # no thresholds supplied
        for col, sign in (("pr_auc_fit", -1), ("roc_auc_fit", -1), ("brier_fit", 1)):
            fits = frame[col].dropna().to_numpy()
            assert (sign * np.diff(fits) >= -1e-9).all()

    def test_f1_with_per_sample_thresholds(self) -> None:
        di, y, p = self._synthetic()
        thresholds = np.full(len(di), 0.5)
        frame = performance_by_di_window(di, y, p, thresholds=thresholds, window=400, step=200)
        assert frame["f1"].notna().all()
        assert ((frame["f1"] >= 0) & (frame["f1"] <= 1)).all()

    def test_threshold_length_mismatch_raises(self) -> None:
        di, y, p = self._synthetic()
        with pytest.raises(ValueError, match="thresholds"):
            performance_by_di_window(di, y, p, thresholds=np.ones(3), window=400)

    def test_bins_partition_and_columns(self) -> None:
        di, y, p = self._synthetic()
        table = performance_by_di_bins(
            di, y, p, edges=np.array([0.0, 0.3, 0.6, 1.01]), thresholds=np.full(len(di), 0.5)
        )
        assert list(table.columns) == [
            "di_bin",
            "di_lo",
            "di_hi",
            "n",
            "prevalence",
            "pr_auc",
            "pr_auc_lift",
            "roc_auc",
            "f1",
            "brier",
        ]
        assert table["n"].sum() == len(di)

    def test_bins_bad_edges_raise(self) -> None:
        di, y, p = self._synthetic(200)
        with pytest.raises(ValueError, match="edges"):
            performance_by_di_bins(di, y, p, edges=np.array([0.5, 0.2]))

    def test_window_larger_than_data_raises(self) -> None:
        di, y, p = self._synthetic(100)
        with pytest.raises(ValueError, match="full window"):
            performance_by_di_window(di, y, p, window=500)

    def test_length_mismatch_raises(self) -> None:
        with pytest.raises(ValueError, match="equal length"):
            performance_by_di_window(np.ones(5), np.ones(4), np.ones(5), window=2)
