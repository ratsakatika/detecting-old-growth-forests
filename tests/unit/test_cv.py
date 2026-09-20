"""Unit tests for utils.cv (nested cross-validation core).

The pure helpers (sampling, fold-array building, aggregation) are tested in
detail on synthetic arrays; two small end-to-end runs on a tiny separable
problem exercise the Optuna driver on CPU in a couple of seconds.
"""

import os
import subprocess
import sys
from contextlib import nullcontext
from pathlib import Path

import numpy as np
import optuna
import pandas as pd
import pytest

import utils.cv as cv
from utils.cv import (
    DEVICES,
    AggregatedMetrics,
    FoldResult,
    NestedCVConfig,
    TrialRecord,
    aggregate_pooled_metrics,
    build_fold_arrays,
    run_nested_cv,
    run_nested_cv_cnn,
    sample_train_rows,
)
from utils.hp_search import CNN_SEARCH_SPACE, XGB_SEARCH_SPACE
from utils.io import (
    FEATURE_IMPORTANCE_SCHEMA,
    HP_BOUNDARY_SCHEMA,
    PARCEL_PREDICTIONS_SCHEMA,
    PER_FOLD_METRICS_SCHEMA,
    PIXEL_PREDICTIONS_SCHEMA,
    POOLED_METRICS_SCHEMA,
    SUMMARY_METRICS_SCHEMA,
    validate_schema,
)
from utils.terminology import FEATURE_SET_BANDS

# --------------------------------------------------------------------------- #
# No import side effects.
# --------------------------------------------------------------------------- #


def test_import_writes_no_files(tmp_path: Path) -> None:
    repo_root = Path(cv.__file__).resolve().parents[1]
    result = subprocess.run(
        [sys.executable, "-c", "import utils.cv"],
        cwd=tmp_path,
        env={**os.environ, "PYTHONPATH": str(repo_root)},
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert list(tmp_path.iterdir()) == []


# --------------------------------------------------------------------------- #
# NestedCVConfig validation.
# --------------------------------------------------------------------------- #


def test_config_defaults_are_valid() -> None:
    config = NestedCVConfig(feature_set="baseline")
    assert config.architecture == "xgboost"
    assert config.sampling_mode == "none"
    assert config.device == "cpu"


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"feature_set": "nope"}, "unknown feature set"),
        ({"feature_set": "baseline", "architecture": "nope"}, "unknown architecture"),
        ({"feature_set": "baseline", "sampling_mode": "nope"}, "unknown sampling mode"),
        ({"feature_set": "baseline", "device": "tpu"}, "device must be one of"),
        ({"feature_set": "baseline", "n_per_parcel": 0}, "n_per_parcel must be positive"),
        ({"feature_set": "baseline", "n_hp_trials": 0}, "n_hp_trials must be positive"),
        ({"feature_set": "baseline", "seed": -1}, "seed must be non-negative"),
        ({"feature_set": "baseline", "n_jobs": 0}, "n_jobs must be positive"),
        ({"feature_set": "baseline", "fold_ids": ()}, "fold_ids must not be empty"),
        ({"feature_set": "baseline", "fold_ids": (1, 7)}, "outside FOLD_IDS"),
    ],
)
def test_config_rejects_bad_fields(kwargs: dict, match: str) -> None:
    with pytest.raises(ValueError, match=match):
        NestedCVConfig(**kwargs)


def test_devices_constant() -> None:
    assert DEVICES == ("cuda", "cpu")


# --------------------------------------------------------------------------- #
# sample_train_rows.
# --------------------------------------------------------------------------- #


def _subset(parcel_ids: list[int]) -> pd.DataFrame:
    return pd.DataFrame({"parcel_id": parcel_ids})


def test_sample_train_rows_caps_per_parcel_without_replacement() -> None:
    # Parcel 1 has 5 rows, parcel 2 has 3 rows; cap at 2 per parcel.
    subset = _subset([1, 1, 1, 1, 1, 2, 2, 2])
    rng = np.random.default_rng(0)
    positions = sample_train_rows(subset, n_per_parcel=2, rng=rng)

    assert positions.dtype == np.int64
    # Sorted, unique, two from each parcel.
    assert np.all(np.diff(positions) > 0)
    chosen = subset.iloc[positions]["parcel_id"].value_counts()
    assert chosen[1] == 2 and chosen[2] == 2


def test_sample_train_rows_keeps_all_when_under_cap() -> None:
    subset = _subset([1, 1, 2])
    positions = sample_train_rows(subset, n_per_parcel=10, rng=np.random.default_rng(1))
    assert positions.tolist() == [0, 1, 2]


def test_sample_train_rows_is_deterministic_for_seed() -> None:
    subset = _subset([1] * 10 + [2] * 10)
    first = sample_train_rows(subset, 3, np.random.default_rng(7))
    second = sample_train_rows(subset, 3, np.random.default_rng(7))
    assert first.tolist() == second.tolist()


def test_sample_train_rows_empty_subset() -> None:
    positions = sample_train_rows(_subset([]), 5, np.random.default_rng(0))
    assert positions.tolist() == []


# --------------------------------------------------------------------------- #
# build_fold_arrays.
# --------------------------------------------------------------------------- #


def _toy_index() -> tuple[pd.DataFrame, np.ndarray]:
    """Three folds, two parcels each, three pixels per parcel; one invalid pixel."""
    rows = []
    pixel_id = 0
    for fold in (1, 2, 3):
        for parcel in (fold * 10 + 1, fold * 10 + 2):
            ogf = 1 if parcel % 2 == 1 else 0
            for _ in range(3):
                rows.append(
                    {
                        "pixel_id": pixel_id,
                        "parcel_id": parcel,
                        "fold_id": fold,
                        "ogf": ogf,
                        "forest_type_corine": "broadleaf" if ogf else "mixed",
                        "valid": True,
                    }
                )
                pixel_id += 1
    index = pd.DataFrame(rows)
    index.loc[0, "valid"] = False  # one invalid pixel in fold 1
    memmap = np.arange(len(index) * 2, dtype=np.float32).reshape(len(index), 2)
    return index, memmap


def test_build_fold_arrays_masks_invalid_and_samples() -> None:
    index, memmap = _toy_index()
    arrays = build_fold_arrays(
        memmap,
        index,
        train_folds=(1, 2),
        val_fold=3,
        sampling_mode="none",
        n_per_parcel=2,
        rng=np.random.default_rng(0),
    )
    # Validation is every pixel of fold 3 (2 parcels x 3 pixels = 6), no sampling.
    assert arrays.X_val.shape == (6, 2)
    assert set(arrays.val_parcel_ids) == {31, 32}
    assert arrays.val_pixel_ids.shape == (6,)
    # Training: folds 1 and 2 have 4 parcels; fold 1 parcel 11 lost one invalid
    # pixel (2 valid), the rest have 3; capped at 2 each -> 8 rows.
    assert arrays.X_train.shape == (8, 2)
    assert arrays.y_train.shape == (8,)
    assert arrays.train_sample_weight.shape == (8,)
    # The invalid pixel (pixel_id 0) must never appear in training features.
    assert not np.any(np.all(arrays.X_train == memmap[0], axis=1))


def test_build_fold_arrays_inverse_prevalence_weights() -> None:
    index, memmap = _toy_index()
    arrays = build_fold_arrays(
        memmap,
        index,
        train_folds=(1, 2),
        val_fold=3,
        sampling_mode="inverse_prevalence",
        n_per_parcel=10,
        rng=np.random.default_rng(0),
    )
    # Mean weight normalised to 1.0 (utils.sampling contract).
    assert arrays.train_sample_weight.mean() == pytest.approx(1.0)


def test_build_fold_arrays_uses_corine_for_stratified_weights(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    index, memmap = _toy_index()
    # An inventory column with a sentinel value: it must never reach the weighter.
    index["forest_type_inventory"] = "sentinel"
    captured: dict[str, np.ndarray] = {}

    def spy(mode: str, *, labels: np.ndarray, forest_type: np.ndarray | None = None) -> np.ndarray:
        captured["forest_type"] = np.asarray(forest_type)
        return np.ones(len(labels), dtype=np.float64)

    monkeypatch.setattr(cv, "build_sample_weights", spy)
    build_fold_arrays(
        memmap,
        index,
        train_folds=(1, 2),
        val_fold=3,
        sampling_mode="stratified_forest_type",
        n_per_parcel=10,
        rng=np.random.default_rng(0),
    )
    # The weighter received the CORINE column, not the inventory sentinel.
    assert set(captured["forest_type"].tolist()) <= {"broadleaf", "mixed"}
    assert "sentinel" not in captured["forest_type"].tolist()


# --------------------------------------------------------------------------- #
# aggregate_pooled_metrics.
# --------------------------------------------------------------------------- #


def _fold_result(outer_fold: int, seed: int) -> FoldResult:
    rng = np.random.default_rng(seed)
    n = 40
    y = np.array([0, 1] * (n // 2), dtype=np.int8)
    # Probabilities correlated with the label so both classes are present and
    # the metrics are well defined.
    p = np.clip(y * 0.5 + rng.uniform(0, 0.5, n), 0, 1).astype(np.float64)
    pixel_preds = pd.DataFrame(
        {
            "pixel_id": np.arange(n, dtype=np.int64) + outer_fold * 1000,
            "outer_fold": np.full(n, outer_fold, dtype=np.int8),
            "y_true": y,
            "p": p,
        }
    )
    parcel_ids = np.arange(n, dtype=np.int64) + outer_fold * 1000
    parcel_preds = pd.DataFrame(
        {
            "parcel_id": parcel_ids,
            "outer_fold": np.full(n, outer_fold, dtype=np.int8),
            "y_true": y,
            "p_mean": p,
            "n_pixels": np.full(n, 5, dtype=np.int64),
        }
    )
    return FoldResult.from_predictions(outer_fold, pixel_preds, parcel_preds)


def test_aggregate_pooled_metrics_tables_and_schemas() -> None:
    results = [_fold_result(1, 1), _fold_result(2, 2), _fold_result(3, 3)]
    aggregated = aggregate_pooled_metrics(results)

    assert isinstance(aggregated, AggregatedMetrics)
    # Per-fold: one row per fold and level.
    assert len(aggregated.per_fold) == 3 * 2
    assert set(aggregated.per_fold["level"]) == {"pixel", "parcel"}
    validate_schema(aggregated.per_fold, PER_FOLD_METRICS_SCHEMA)
    # The honest inner-threshold operating point columns are present and in range.
    inner_cols = [
        "inner_threshold",
        "f1_inner_threshold",
        "precision_inner_threshold",
        "recall_inner_threshold",
    ]
    inner = aggregated.per_fold[inner_cols]
    assert ((inner >= 0) & (inner <= 1)).all().all()
    # Summary covers both operating points per level (long form, rows not columns).
    assert set(aggregated.summary["metric"]) >= {"f1", "f1_inner_threshold", "inner_threshold"}
    assert len(aggregated.summary) == 2 * len(cv._SUMMARY_METRICS)
    validate_schema(aggregated.summary, SUMMARY_METRICS_SCHEMA)
    validate_schema(aggregated.pooled_pixel, POOLED_METRICS_SCHEMA)
    validate_schema(aggregated.pooled_parcel, POOLED_METRICS_SCHEMA)
    # Pooled tables keep only the threshold-free + in-sample max-F1 metrics.
    assert "f1_inner_threshold" not in set(aggregated.pooled_pixel["metric"])
    # All ranking metrics in [0, 1].
    ranking = aggregated.per_fold[["pr_auc", "roc_auc", "f1", "precision", "recall"]]
    assert ((ranking >= 0) & (ranking <= 1)).all().all()


def test_aggregate_single_fold_has_nan_sd() -> None:
    aggregated = aggregate_pooled_metrics([_fold_result(1, 1)])
    assert aggregated.summary["sd"].isna().all()


def test_aggregate_empty_raises() -> None:
    with pytest.raises(ValueError, match="fold_results is empty"):
        aggregate_pooled_metrics([])


# --------------------------------------------------------------------------- #
# TrialRecord.
# --------------------------------------------------------------------------- #


def test_trial_record_to_row_flattens_params() -> None:
    record = TrialRecord(
        trial_number=3,
        state="COMPLETE",
        pr_auc=0.8,
        roc_auc=0.9,
        f1=0.7,
        precision=0.6,
        recall=0.85,
        inner_pixel_threshold=0.3,
        inner_parcel_threshold=0.4,
        params={"max_depth": 5, "learning_rate": 0.05},
    )
    row = record.to_row()
    assert row["trial_number"] == 3
    assert row["state"] == "COMPLETE"
    assert row["inner_pixel_threshold"] == 0.3
    assert row["inner_parcel_threshold"] == 0.4
    assert row["max_depth"] == 5
    assert row["learning_rate"] == 0.05
    assert np.isnan(row["cnn_epochs"])


def test_trial_record_to_row_carries_cnn_epochs() -> None:
    record = TrialRecord(
        trial_number=1,
        state="COMPLETE",
        pr_auc=0.8,
        roc_auc=0.9,
        f1=0.7,
        precision=0.6,
        recall=0.85,
        inner_pixel_threshold=0.3,
        inner_parcel_threshold=0.4,
        params={"n_channels": 64, "batch_size": 4096},
        cnn_epochs=12,
    )

    row = record.to_row()

    assert row["n_channels"] == 64
    assert row["batch_size"] == 4096
    assert row["cnn_epochs"] == 12.0


def test_pooled_f1_threshold_two_class_and_single_class() -> None:
    rng = np.random.default_rng(0)
    y = np.array([0, 0, 1, 1, 0, 1], dtype=np.int8)
    p = rng.uniform(size=6)
    threshold = cv._pooled_f1_threshold(y, p)
    assert 0.0 <= threshold <= 1.0
    # Single-class pool falls back to 0.5.
    single = cv._pooled_f1_threshold(np.zeros(5, dtype=np.int8), rng.uniform(size=5))
    assert single == 0.5


# --------------------------------------------------------------------------- #
# End-to-end driver (small synthetic problem).
# --------------------------------------------------------------------------- #


def _synthetic_problem(
    feature_set: str = "baseline", seed: int = 0
) -> tuple[pd.DataFrame, np.ndarray]:
    """Three folds of separable pixels, aligned with a feature memmap."""
    n_bands = len(FEATURE_SET_BANDS[feature_set])
    rng = np.random.default_rng(seed)
    rows = []
    features = []
    pixel_id = 0
    for fold in (1, 2, 3):
        for parcel in range(4):  # 2 old-growth, 2 non-old-growth per fold
            ogf = 1 if parcel < 2 else 0
            centre = 2.0 if ogf else -2.0
            for _ in range(30):
                rows.append(
                    {
                        "pixel_id": pixel_id,
                        "parcel_id": fold * 100 + parcel,
                        "fold_id": fold,
                        "ogf": ogf,
                        "forest_type_corine": "broadleaf" if ogf else "coniferous",
                        "valid": True,
                    }
                )
                features.append(rng.normal(centre, 1.0, n_bands))
                pixel_id += 1
    index = pd.DataFrame(rows)
    memmap = np.asarray(features, dtype=np.float32)
    return index, memmap


def test_run_nested_cv_end_to_end_cpu() -> None:
    index, memmap = _synthetic_problem()
    config = NestedCVConfig(
        feature_set="baseline",
        sampling_mode="none",
        n_per_parcel=20,
        n_hp_trials=2,
        device="cpu",
        n_jobs=1,
        fold_ids=(1, 2, 3),
    )

    trial_calls: list[tuple[int, TrialRecord]] = []
    fold_calls: list[tuple[int, FoldResult]] = []
    results = run_nested_cv(
        config,
        index,
        memmap,
        on_trial_end=lambda fold, record: trial_calls.append((fold, record)),
        on_fold_end=lambda fold, result: fold_calls.append((fold, result)),
    )

    assert [r.outer_fold for r in results] == [1, 2, 3]
    # One trial record per (fold, trial); one fold record per fold.
    assert len(trial_calls) == 3 * 2
    assert len(fold_calls) == 3
    for _, result in fold_calls:
        validate_schema(result.pixel_predictions, PIXEL_PREDICTIONS_SCHEMA)
        validate_schema(result.parcel_predictions, PARCEL_PREDICTIONS_SCHEMA)
        validate_schema(result.boundary, HP_BOUNDARY_SCHEMA)
        validate_schema(result.importance, FEATURE_IMPORTANCE_SCHEMA)
        # Separable problem: discrimination should be strong.
        assert result.parcel_metrics["pr_auc"] > 0.8
        # Selected params are exactly the tunable search space.
        assert set(result.best_params) == set(XGB_SEARCH_SPACE)
        # Both F1 operating points are populated and finite in [0, 1].
        for metrics in (result.pixel_metrics_inner, result.parcel_metrics_inner):
            assert set(metrics) == {"threshold", "f1", "precision", "recall"}
            assert all(0.0 <= metrics[key] <= 1.0 for key in metrics)
    # Each trial recorded its two inner thresholds.
    for _, record in trial_calls:
        assert 0.0 <= record.inner_pixel_threshold <= 1.0
        assert 0.0 <= record.inner_parcel_threshold <= 1.0
        assert record.cnn_epochs is None
    # The aggregation pipeline runs on the real results, with both operating points.
    aggregated = aggregate_pooled_metrics(results)
    validate_schema(aggregated.per_fold, PER_FOLD_METRICS_SCHEMA)
    assert aggregated.per_fold["f1_inner_threshold"].between(0.0, 1.0).all()
    assert aggregated.pooled_parcel["value"].notna().all()


def test_run_nested_cv_outer_folds_subset_keeps_full_inner() -> None:
    index, memmap = _synthetic_problem()
    config = NestedCVConfig(
        feature_set="baseline",
        n_per_parcel=20,
        n_hp_trials=1,
        device="cpu",
        n_jobs=1,
        fold_ids=(1, 2, 3),
    )
    fold_calls: list[int] = []
    results = run_nested_cv(
        config,
        index,
        memmap,
        outer_folds=[2],
        on_fold_end=lambda fold, result: fold_calls.append(fold),
    )
    # Only outer fold 2 runs, but it still trains on inner folds 1 and 3.
    assert [r.outer_fold for r in results] == [2]
    assert fold_calls == [2]
    assert results[0].n_train > 0


def test_run_nested_cv_rejects_unknown_outer_fold() -> None:
    index, memmap = _synthetic_problem()
    config = NestedCVConfig(feature_set="baseline", fold_ids=(1, 2, 3))
    with pytest.raises(ValueError, match="not in config.fold_ids"):
        run_nested_cv(config, index, memmap, outer_folds=[5])


def test_fixed_hp_fold_result_reproduces_run_nested_cv_fold() -> None:
    # The sensitivity-analysis primitive must reproduce the headline outer refit
    # exactly when given that fold's best params and the same sampling/seed.
    index, memmap = _synthetic_problem()
    config = NestedCVConfig(
        feature_set="baseline",
        n_per_parcel=20,
        n_hp_trials=2,
        device="cpu",
        n_jobs=1,
        fold_ids=(1, 2, 3),
    )
    headline = run_nested_cv(config, index, memmap, outer_folds=[2])[0]
    refit = cv.fixed_hp_fold_result(config, index, memmap, headline.best_params, outer_fold=2)

    np.testing.assert_array_equal(
        refit.pixel_predictions["p"].to_numpy(), headline.pixel_predictions["p"].to_numpy()
    )
    np.testing.assert_array_equal(
        refit.parcel_predictions["p_mean"].to_numpy(),
        headline.parcel_predictions["p_mean"].to_numpy(),
    )
    assert refit.parcel_metrics["pr_auc"] == headline.parcel_metrics["pr_auc"]
    assert refit.n_train == headline.n_train


# --------------------------------------------------------------------------- #
# Receptive-field CNN path (additive; reuses the shared scoring helpers).
# --------------------------------------------------------------------------- #


def test_importing_cv_does_not_import_torch(tmp_path: Path) -> None:
    # The additive isolation guarantee: utils.cv stays torch-free because the
    # torch-dependent imports live inside the CNN functions, not at module top.
    repo_root = Path(cv.__file__).resolve().parents[1]
    result = subprocess.run(
        [sys.executable, "-c", "import sys; import utils.cv; assert 'torch' not in sys.modules"],
        cwd=tmp_path,
        env={**os.environ, "PYTHONPATH": str(repo_root)},
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr


def test_loader_seed_is_deterministic_and_varies() -> None:
    assert cv._loader_seed(42, 1, 0, 2) == cv._loader_seed(42, 1, 0, 2)
    assert cv._loader_seed(42, 1, 0, 2) != cv._loader_seed(42, 1, 0, 3)
    seed = cv._loader_seed(42, 1)
    assert isinstance(seed, int) and seed >= 0


def test_cnn_epochs_from_fits_mean_excludes_unimproved() -> None:
    # best_epoch + 1 over improved folds [3, 5, 7] -> mean 5.
    assert cv._cnn_epochs_from_fits([2, 4, 6], warmup_epochs=1) == 5
    # Folds that never improved (best_epoch == -1) are excluded; mean of [3] -> 3.
    assert cv._cnn_epochs_from_fits([-1, 2, -1], warmup_epochs=1) == 3
    # Lower clamp at warmup + 1 when the rounded mean is below it.
    assert cv._cnn_epochs_from_fits([0, 0], warmup_epochs=5) == 6
    # No usable fold falls back to the floor (warmup + 1).
    assert cv._cnn_epochs_from_fits([-1, -1], warmup_epochs=4) == 5


def _synthetic_cnn_problem(
    feature_set: str = "baseline", *, signal: float = 5.0, seed: int = 0
) -> tuple[pd.DataFrame, np.ndarray]:
    """A small grid memmap and pixel index with class signal in the centre pixel.

    Three folds, four parcels each (two old-growth, two not), six interior pixels
    per parcel laid out on a 16x16 two-dimensional grid. Band 0's centre pixel
    carries ``signal * ogf`` plus noise, so the receptive-field CNN can separate
    the classes; every other cell is noise.
    """
    n_bands = len(FEATURE_SET_BANDS[feature_set])
    rng = np.random.default_rng(seed)
    height = width = 16
    grid = rng.standard_normal((n_bands, height, width)).astype(np.float32)
    interior = [(row, col) for row in range(1, height - 1) for col in range(1, width - 1)]

    rows: list[dict[str, object]] = []
    cursor = 0
    for fold in (1, 2, 3):
        for parcel in range(4):
            ogf = 1 if parcel < 2 else 0
            for _ in range(6):
                row, col = interior[cursor]
                cursor += 1
                grid[0, row, col] = np.float32(signal * ogf + 0.3 * float(rng.standard_normal()))
                rows.append(
                    {
                        "pixel_id": row * width + col,
                        "row": row,
                        "col": col,
                        "parcel_id": fold * 100 + parcel,
                        "fold_id": fold,
                        "ogf": ogf,
                        "forest_type_corine": "broadleaf" if ogf else "coniferous",
                        "valid": True,
                    }
                )
    return pd.DataFrame(rows), grid


def _cnn_patch_cache(index: pd.DataFrame, grid: np.ndarray, patch_size: int) -> np.ndarray:
    """In-memory patch cache (n_pixels, n_bands, P, P) aligned with ``index`` rows."""
    from utils.patches import extract_raw_patch

    rows = index["row"].to_numpy()
    cols = index["col"].to_numpy()
    cache = np.empty((len(index), grid.shape[0], patch_size, patch_size), dtype=np.float32)
    for i in range(len(index)):
        cache[i] = extract_raw_patch(grid, int(rows[i]), int(cols[i]), patch_size=patch_size)
    return cache


def _cnn_config(**overrides: object) -> NestedCVConfig:
    params: dict[str, object] = {
        "feature_set": "baseline",
        "architecture": "cnn_3x3",
        "sampling_mode": "none",
        "n_per_parcel": 6,
        "n_hp_trials": 1,
        "device": "cpu",
        "n_jobs": 1,
        "fold_ids": (1, 2, 3),
    }
    params.update(overrides)
    return NestedCVConfig(**params)  # type: ignore[arg-type]


def test_inner_objective_cnn_records_cnn_epochs_and_metrics() -> None:
    index, grid = _synthetic_cnn_problem()
    cache = _cnn_patch_cache(index, grid, 3)
    config = _cnn_config()
    parcel_label = cv._parcel_labels(index)
    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=0))

    def objective(trial: optuna.trial.Trial) -> float:
        return cv._inner_objective_cnn(
            trial,
            config=config,
            pixel_index=index,
            patch_cache=cache,
            patch_size=3,
            grid_shape=grid.shape[1:],
            outer_fold=1,
            inner_folds=(2, 3),
            parcel_label=parcel_label,
            fit_guard=nullcontext,
        )

    study.optimize(objective, n_trials=1)
    attrs = study.best_trial.user_attrs
    # The outer refit reads this training length: a positive int.
    assert isinstance(attrs["cnn_epochs"], int)
    assert attrs["cnn_epochs"] > 0
    assert np.isfinite(attrs["pr_auc"])
    assert 0.0 <= attrs["inner_pixel_threshold"] <= 1.0
    assert 0.0 <= attrs["inner_parcel_threshold"] <= 1.0
    assert set(study.best_params) == set(CNN_SEARCH_SPACE)
    assert np.isfinite(study.best_value)


def test_run_nested_cv_cnn_end_to_end_cpu() -> None:
    index, grid = _synthetic_cnn_problem()
    cache = _cnn_patch_cache(index, grid, 3)
    config = _cnn_config()

    trial_calls: list[tuple[int, TrialRecord]] = []
    fold_calls: list[tuple[int, FoldResult]] = []
    results = run_nested_cv_cnn(
        config,
        index,
        cache,
        3,
        grid_shape=grid.shape[1:],
        on_trial_end=lambda fold, record: trial_calls.append((fold, record)),
        on_fold_end=lambda fold, result: fold_calls.append((fold, result)),
    )

    assert [r.outer_fold for r in results] == [1, 2, 3]
    assert len(trial_calls) == 3 * 1
    assert len(fold_calls) == 3
    for _, result in fold_calls:
        validate_schema(result.pixel_predictions, PIXEL_PREDICTIONS_SCHEMA)
        validate_schema(result.parcel_predictions, PARCEL_PREDICTIONS_SCHEMA)
        validate_schema(result.boundary, HP_BOUNDARY_SCHEMA)
        # No feature-importance table for the CNN (not fabricated).
        assert result.importance.empty
        # Selected params are exactly the CNN search space.
        assert set(result.best_params) == set(CNN_SEARCH_SPACE)
        # Parcel PR-AUC is finite and in range.
        assert np.isfinite(result.parcel_metrics["pr_auc"])
        assert 0.0 <= result.parcel_metrics["pr_auc"] <= 1.0
        assert result.n_train > 0
        assert result.cnn_epochs is not None and result.cnn_epochs > 0
        # Both F1 operating points are populated and finite in [0, 1].
        for metrics in (result.pixel_metrics_inner, result.parcel_metrics_inner):
            assert set(metrics) == {"threshold", "f1", "precision", "recall"}
            assert all(0.0 <= metrics[key] <= 1.0 for key in metrics)
    for _, record in trial_calls:
        assert record.cnn_epochs is not None and record.cnn_epochs > 0
    # The aggregation pipeline runs on the real CNN results, validating schemas.
    aggregated = aggregate_pooled_metrics(results)
    validate_schema(aggregated.per_fold, PER_FOLD_METRICS_SCHEMA)
    validate_schema(aggregated.summary, SUMMARY_METRICS_SCHEMA)
    validate_schema(aggregated.pooled_pixel, POOLED_METRICS_SCHEMA)
    validate_schema(aggregated.pooled_parcel, POOLED_METRICS_SCHEMA)
    assert aggregated.pooled_parcel["value"].notna().all()


def test_run_nested_cv_cnn_outer_folds_subset_keeps_full_inner() -> None:
    index, grid = _synthetic_cnn_problem()
    cache = _cnn_patch_cache(index, grid, 3)
    config = _cnn_config()
    fold_calls: list[int] = []
    results = run_nested_cv_cnn(
        config,
        index,
        cache,
        3,
        grid_shape=grid.shape[1:],
        outer_folds=[2],
        on_fold_end=lambda fold, result: fold_calls.append(fold),
    )
    # Only outer fold 2 runs, but it still trains on inner folds 1 and 3.
    assert [r.outer_fold for r in results] == [2]
    assert fold_calls == [2]
    assert results[0].n_train > 0


def test_run_nested_cv_cnn_rejects_unknown_outer_fold() -> None:
    index, grid = _synthetic_cnn_problem()
    cache = _cnn_patch_cache(index, grid, 3)
    config = _cnn_config()
    with pytest.raises(ValueError, match="not in config.fold_ids"):
        run_nested_cv_cnn(config, index, cache, 3, grid_shape=grid.shape[1:], outer_folds=[5])


def test_fixed_hp_fold_result_cnn_reproduces_run_nested_cv_cnn_fold() -> None:
    # The CNN sensitivity primitive must reproduce the CNN outer refit exactly when
    # given that fold's best params, cnn_epochs and the same sampling/seed. train_cnn_fixed
    # reseeds torch globally (make_cnn -> _seed_everything), so the refit is independent of
    # the inner search's RNG use and the predictions are byte-identical.
    index, grid = _synthetic_cnn_problem()
    cache = _cnn_patch_cache(index, grid, 3)
    config = _cnn_config()
    headline = run_nested_cv_cnn(
        config, index, cache, 3, grid_shape=grid.shape[1:], outer_folds=[2]
    )[0]
    assert headline.cnn_epochs is not None
    refit = cv.fixed_hp_fold_result_cnn(
        config,
        index,
        cache,
        headline.best_params,
        patch_size=3,
        grid_shape=grid.shape[1:],
        cnn_epochs=headline.cnn_epochs,
        outer_fold=2,
    )

    np.testing.assert_array_equal(
        refit.pixel_predictions["p"].to_numpy(), headline.pixel_predictions["p"].to_numpy()
    )
    np.testing.assert_array_equal(
        refit.parcel_predictions["p_mean"].to_numpy(),
        headline.parcel_predictions["p_mean"].to_numpy(),
    )
    assert refit.parcel_metrics["pr_auc"] == headline.parcel_metrics["pr_auc"]
    assert refit.n_train == headline.n_train
    assert refit.cnn_epochs == headline.cnn_epochs
