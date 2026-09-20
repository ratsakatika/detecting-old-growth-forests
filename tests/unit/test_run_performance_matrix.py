"""Synthetic smoke test for scripts/run_performance_matrix.py."""

import json

import geopandas as gpd
import numpy as np
import pandas as pd
import pytest
from shapely.geometry import box

from scripts import run_performance_matrix as rpm
from utils.io import validate_schema
from utils.paths import ProjectPaths, get_project_paths
from utils.terminology import FEATURE_SET_BANDS, FEATURE_SETS, FOLD_IDS

_BEST_PARAMS = {
    "n_estimators": 15,
    "learning_rate": 0.3,
    "max_depth": 3,
    "min_child_weight": 1.0,
    "subsample": 1.0,
    "colsample_bytree": 1.0,
    "reg_lambda": 1.0,
    "reg_alpha": 0.0,
    "gamma": 0.0,
    "max_delta_step": 0,
}


def _build(paths: ProjectPaths) -> None:
    """Synthetic pixel table, feature caches, labels and one XGBoost run per feature set."""
    feature_sets = list(FEATURE_SETS)
    rng = np.random.default_rng(1)
    rows: list[dict[str, object]] = []
    feats: dict[str, list[np.ndarray]] = {fs: [] for fs in feature_sets}
    geoms: list[object] = []
    meta: list[dict[str, object]] = []
    pid = pixel_id = 0
    for fold in FOLD_IDS:
        for parcel in range(4):
            ogf = 1 if parcel < 2 else 0
            # Folds 15 km apart: the 10 km buffer keeps every other fold, the 20 km buffer
            # drops only the neighbouring fold, so no buffered arm runs out of training data.
            x0 = (fold - 1) * 15_000.0
            geoms.append(box(x0, parcel * 1000.0, x0 + 500.0, parcel * 1000.0 + 500.0))
            meta.append(
                {
                    "parcel_id": pid,
                    "fold_id": float(fold),
                    "ogf": float(ogf),
                    "bootstrap_id": float((pid % 5) + 1),
                }
            )
            for _ in range(6):
                rows.append(
                    {
                        "pixel_id": pixel_id,
                        "parcel_id": pid,
                        "fold_id": np.int8(fold),
                        "ogf": np.int8(ogf),
                        "forest_type_corine": "broadleaf" if ogf else "coniferous",
                    }
                )
                for fs in feature_sets:
                    vector = rng.standard_normal(len(FEATURE_SET_BANDS[fs])).astype(np.float32)
                    vector[0] = 4.0 * ogf + 0.3 * float(rng.standard_normal())
                    feats[fs].append(vector)
                pixel_id += 1
            pid += 1
    index = pd.DataFrame(rows)

    index_path = paths.results / "main_nested_cv" / "pixel_index.parquet"
    index_path.parent.mkdir(parents=True, exist_ok=True)
    index.to_parquet(index_path)

    cache = paths.cache / "pixel_features"
    cache.mkdir(parents=True, exist_ok=True)
    for fs in feature_sets:
        np.save(cache / f"{fs}.npy", np.asarray(feats[fs], dtype=np.float32))
        pd.DataFrame({"pixel_id": index["pixel_id"], "valid": True}).to_parquet(
            cache / f"{fs}_valid.parquet"
        )

    paths.labels.mkdir(parents=True, exist_ok=True)
    gpd.GeoDataFrame(meta, geometry=geoms, crs="EPSG:3035").to_file(
        paths.labels / "ogf_reference_labels_partitioned.gpkg"
    )

    for fs in feature_sets:
        run_dir = paths.results / "main_nested_cv" / f"20260101_000000__xgboost__{fs}"
        run_dir.mkdir(parents=True, exist_ok=True)
        for fold in FOLD_IDS:
            (run_dir / f"best_params_fold{fold}.json").write_text(json.dumps(_BEST_PARAMS))
            parcels = index[index["fold_id"] == fold].drop_duplicates("parcel_id")
            pd.DataFrame(
                {
                    "parcel_id": parcels["parcel_id"].astype(np.int64).to_numpy(),
                    "outer_fold": np.full(len(parcels), fold, dtype=np.int8),
                    "y_true": parcels["ogf"].astype(np.int8).to_numpy(),
                    "p_mean": parcels["ogf"].astype(np.float64).to_numpy() * 0.8 + 0.1,
                    "n_pixels": np.full(len(parcels), 6, dtype=np.int64),
                }
            ).to_parquet(run_dir / f"parcel_predictions_fold{fold}.parquet")
            # Minimal hp_trials so the best trial supplies the inner threshold.
            pd.DataFrame(
                {
                    "trial_number": [0, 1],
                    "state": ["COMPLETE", "COMPLETE"],
                    "pr_auc": [0.5, 0.8],
                    "inner_parcel_threshold": [0.30, 0.45],
                }
            ).to_csv(run_dir / f"hp_trials_fold{fold}.csv", index=False)
        # Marks the run complete for _latest_run's completeness check.
        pd.DataFrame({"metric": ["pr_auc"], "value": [0.9]}).to_csv(
            run_dir / "pooled_parcel_metrics.csv", index=False
        )


def test_performance_matrix_writes_tables(tmp_path, monkeypatch: pytest.MonkeyPatch) -> None:
    """The matrix builds XGBoost cells, leaves CNN cells pending, and contrasts validate."""
    paths = get_project_paths(tmp_path)
    monkeypatch.setattr(rpm, "get_project_paths", lambda: paths)
    _build(paths)

    assert rpm.main(["--device", "cpu", "--n-jobs", "1", "--reps", "32"]) == 0
    run = next((paths.results / "performance_matrix").iterdir())

    matrix = pd.read_csv(run / "performance_matrix.csv")
    validate_schema(matrix, rpm.MATRIX_SCHEMA)
    assert len(matrix) == len(FEATURE_SETS) * 4 * len(rpm._BUFFERS)  # fs x arch x buffers

    xgb0 = matrix[(matrix["architecture"] == "xgboost") & (matrix["buffer_km"] == 0)]
    assert (xgb0["folds_used"] == len(FOLD_IDS)).all()
    assert xgb0["pr_auc__pooled"].notna().all()
    assert xgb0["f1_inner__pooled"].notna().all()

    # No synthetic CNN runs, so every CNN cell is pending (NaN, zero folds).
    cnn = matrix[matrix["architecture"] == "cnn_5x5"]
    assert (cnn["folds_used"] == 0).all()
    assert cnn["pr_auc__pooled"].isna().all()

    fs_contrasts = pd.read_csv(run / "feature_set_contrasts.csv")
    validate_schema(fs_contrasts, rpm.CONTRAST_SCHEMA)
    assert len(fs_contrasts) == 3 * len(rpm._BUFFERS)  # three EO feature sets minus baseline
    assert {
        "baseline_tessera - baseline",
        "baseline_alphaearth - baseline",
    } <= set(fs_contrasts["contrast"])
    assert "pr_auc__p_holm" in fs_contrasts.columns

    embedding_contrasts = pd.read_csv(run / "embedding_contrasts.csv")
    validate_schema(embedding_contrasts, rpm.CONTRAST_SCHEMA)
    assert len(embedding_contrasts) == 3 * len(rpm._BUFFERS)  # three EO pairs per buffer
    assert {"baseline_tessera - baseline_alphaearth"} <= set(embedding_contrasts["contrast"])

    arch_contrasts = pd.read_csv(run / "architecture_contrasts.csv")
    validate_schema(arch_contrasts, rpm.CONTRAST_SCHEMA)
    assert len(arch_contrasts) == 3 * len(FEATURE_SETS) * len(rpm._BUFFERS)  # all pending
    assert (arch_contrasts["folds_used"] == 0).all()


def test_buffered_refits_are_cached(tmp_path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A re-run reuses the cached buffered refits; --refresh-buffered forces a recompute."""
    paths = get_project_paths(tmp_path)
    monkeypatch.setattr(rpm, "get_project_paths", lambda: paths)
    _build(paths)
    common = [
        "--device",
        "cpu",
        "--n-jobs",
        "1",
        "--reps",
        "8",
        "--architectures",
        "xgboost",
        "--feature-sets",
        "baseline",
        "baseline_tessera",
    ]
    assert rpm.main(common) == 0
    assert any((paths.cache / rpm._BUFFERED_CACHE).glob("xgboost__*"))  # cache populated

    def _fail(*args: object, **kwargs: object) -> object:
        raise AssertionError("buffered refit recomputed despite a warm cache")

    monkeypatch.setattr(rpm, "_buffered_frames", _fail)
    assert rpm.main(common) == 0  # reuse: _buffered_frames is never called

    with pytest.raises(AssertionError, match="recomputed"):
        rpm.main([*common, "--refresh-buffered"])  # refresh bypasses the cache
