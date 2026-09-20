"""Synthetic smoke test for scripts/run_seed_sweep.py."""

import json

import geopandas as gpd
import numpy as np
import pandas as pd
import pytest
from shapely.geometry import box

from scripts import run_seed_sweep as rss
from utils.io import validate_schema
from utils.paths import ProjectPaths, get_project_paths
from utils.terminology import FEATURE_SET_BANDS, FOLD_IDS

_FEATURE_SETS = ("baseline", "baseline_tessera")
_BEST_PARAMS = {
    "n_estimators": 15,
    "learning_rate": 0.3,
    "max_depth": 3,
    "min_child_weight": 1.0,
    "subsample": 0.6,
    "colsample_bytree": 0.6,
    "reg_lambda": 1.0,
    "reg_alpha": 0.0,
    "gamma": 0.0,
    "max_delta_step": 0,
}


def _build(paths: ProjectPaths) -> None:
    """Synthetic pixel table, feature caches, labels and one XGBoost run per feature set."""
    rng = np.random.default_rng(1)
    rows: list[dict[str, object]] = []
    feats: dict[str, list[np.ndarray]] = {fs: [] for fs in _FEATURE_SETS}
    geoms: list[object] = []
    meta: list[dict[str, object]] = []
    pid = pixel_id = 0
    for fold in FOLD_IDS:
        for parcel in range(4):
            ogf = 1 if parcel < 2 else 0
            x0 = (fold - 1) * 5000.0
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
                for fs in _FEATURE_SETS:
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
    for fs in _FEATURE_SETS:
        np.save(cache / f"{fs}.npy", np.asarray(feats[fs], dtype=np.float32))
        pd.DataFrame({"pixel_id": index["pixel_id"], "valid": True}).to_parquet(
            cache / f"{fs}_valid.parquet"
        )

    paths.labels.mkdir(parents=True, exist_ok=True)
    gpd.GeoDataFrame(meta, geometry=geoms, crs="EPSG:3035").to_file(
        paths.labels / "ogf_reference_labels_partitioned.gpkg"
    )

    for fs in _FEATURE_SETS:
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
        # Marks the run complete for _latest_run's completeness check.
        pd.DataFrame({"metric": ["pr_auc"], "value": [0.9]}).to_csv(
            run_dir / "pooled_parcel_metrics.csv", index=False
        )


def test_seed_sweep_writes_metrics_and_contrasts(tmp_path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Two seeds x two buffers x two feature sets: metrics, contrasts and checkpoints."""
    paths = get_project_paths(tmp_path)
    monkeypatch.setattr(rss, "get_project_paths", lambda: paths)
    _build(paths)

    args = [
        "--device",
        "cpu",
        "--n-jobs",
        "1",
        "--feature-sets",
        *_FEATURE_SETS,
        "--seeds",
        "42",
        "7",
        "--buffer-kms",
        "0",
        "10",
        "--n-per-parcel",
        "6",
    ]
    assert rss.main(args) == 0
    run = next((paths.results / rss._RESULTS_ROLE).iterdir())

    metrics = pd.read_csv(run / "seed_sweep_metrics.csv")
    validate_schema(metrics, rss.METRICS_SCHEMA)
    assert len(metrics) == 2 * 2 * 2  # feature sets x buffers x seeds
    assert metrics["pr_auc__pooled"].notna().all()
    assert metrics["roc_auc__pooled"].between(0.0, 1.0).all()

    contrasts = pd.read_csv(run / "seed_sweep_contrasts.csv")
    validate_schema(contrasts, rss.CONTRASTS_SCHEMA)
    assert len(contrasts) == 2 * 2  # (tessera - baseline) x buffers x seeds
    assert set(contrasts["contrast"]) == {"baseline_tessera - baseline"}

    # One checkpoint parquet per (feature set, buffer, seed, fold).
    checkpoints = list((run / rss._PREDICTIONS_DIR).glob("*.parquet"))
    assert len(checkpoints) == 2 * 2 * 2 * len(FOLD_IDS)

    metadata = json.loads((run / "run_metadata.json").read_text(encoding="utf-8"))
    assert metadata["seeds"] == [42, 7]
    assert metadata["integrity_check_paper_seed_0km"]  # paper seed at 0 km was checked


def test_seed_sweep_resume_skips_saved_fits(tmp_path, monkeypatch: pytest.MonkeyPatch) -> None:
    """--resume reuses saved per-fold predictions instead of refitting."""
    paths = get_project_paths(tmp_path)
    monkeypatch.setattr(rss, "get_project_paths", lambda: paths)
    _build(paths)

    args = [
        "--device",
        "cpu",
        "--n-jobs",
        "1",
        "--feature-sets",
        "baseline",
        "--seeds",
        "42",
        "--buffer-kms",
        "0",
        "--n-per-parcel",
        "6",
    ]
    assert rss.main(args) == 0
    run = next((paths.results / rss._RESULTS_ROLE).iterdir())
    first = pd.read_csv(run / "seed_sweep_metrics.csv")

    def _fail(*args: object, **kwargs: object) -> object:
        raise AssertionError("refit despite saved checkpoints")

    monkeypatch.setattr(rss, "_fold_result_for_exclusion", _fail)
    assert rss.main([*args, "--resume", run.name]) == 0
    resumed = pd.read_csv(run / "seed_sweep_metrics.csv")
    pd.testing.assert_frame_equal(first, resumed)
