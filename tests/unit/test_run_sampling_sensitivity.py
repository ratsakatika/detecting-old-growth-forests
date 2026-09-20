"""Synthetic smoke test for scripts/run_sampling_sensitivity.py."""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from scripts import run_sampling_sensitivity as rss
from utils.io import validate_schema
from utils.paths import ProjectPaths, get_project_paths
from utils.sampling import SAMPLING_MODES
from utils.terminology import FEATURE_SET_BANDS, FOLD_IDS

_FS = "baseline"
_SIGNAL = 4.0
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


def build_caches(paths: ProjectPaths, feature_set: str = _FS) -> pd.DataFrame:
    """Write a synthetic pixel table, feature memmap and a complete headline run."""
    n_bands = len(FEATURE_SET_BANDS[feature_set])
    rng = np.random.default_rng(0)
    rows: list[dict[str, object]] = []
    features: list[np.ndarray] = []
    pid, pixel_id = 1, 0
    for fold in FOLD_IDS:
        for parcel in range(2):
            ogf = 1 - parcel
            for _ in range(12):
                rows.append(
                    {
                        "pixel_id": pixel_id,
                        "parcel_id": pid,
                        "fold_id": np.int8(fold),
                        "ogf": np.int8(ogf),
                        "forest_type_corine": "broadleaf" if ogf else "coniferous",
                    }
                )
                vector = rng.standard_normal(n_bands)
                vector[0] = _SIGNAL * ogf + 0.3 * rng.standard_normal()
                features.append(vector.astype(np.float32))
                pixel_id += 1
            pid += 1
    index = pd.DataFrame(rows)

    index_path = paths.results / "main_nested_cv" / "pixel_index.parquet"
    index_path.parent.mkdir(parents=True, exist_ok=True)
    index.to_parquet(index_path)

    cache = paths.cache / "pixel_features"
    cache.mkdir(parents=True, exist_ok=True)
    np.save(cache / f"{feature_set}.npy", np.asarray(features, dtype=np.float32))
    pd.DataFrame({"pixel_id": index["pixel_id"], "valid": True}).to_parquet(
        cache / f"{feature_set}_valid.parquet"
    )

    run_dir = paths.results / "main_nested_cv" / f"20260101_000000__xgboost__{feature_set}"
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
                "n_pixels": np.full(len(parcels), 12, dtype=np.int64),
            }
        ).to_parquet(run_dir / f"parcel_predictions_fold{fold}.parquet")
    pd.DataFrame({"metric": ["pr_auc"], "value": [0.9]}).to_csv(
        run_dir / "pooled_parcel_metrics.csv", index=False
    )
    return index


def test_main_writes_sampling_table(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    paths = get_project_paths(tmp_path)
    monkeypatch.setattr(rss, "get_project_paths", lambda: paths)
    build_caches(paths)

    code = rss.main(["--feature-set", _FS, "--device", "cpu", "--n-jobs", "1"])
    assert code == 0

    run_dirs = [p for p in (paths.results / "sampling_sensitivity").iterdir() if p.is_dir()]
    assert len(run_dirs) == 1
    table = pd.read_csv(run_dirs[0] / "sampling_sensitivity.csv")
    validate_schema(table, rss.SAMPLING_SENSITIVITY_SCHEMA)
    assert set(table["sampling_mode"]) == set(SAMPLING_MODES)
    assert set(table["level"]) == {"pixel", "parcel"}
    assert "pr_auc" in set(table["metric"])
    # The separable signal should give strong parcel PR-AUC under every mode.
    parcel_pr = table[(table["level"] == "parcel") & (table["metric"] == "pr_auc")]
    assert (parcel_pr["pooled"] > 0.7).all()
    assert (run_dirs[0] / "run_metadata.json").is_file()
