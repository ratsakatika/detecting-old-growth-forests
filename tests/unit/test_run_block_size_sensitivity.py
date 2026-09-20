"""Synthetic smoke test for scripts/run_block_size_sensitivity.py."""

from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import pytest
from shapely.geometry import box

from scripts import run_block_size_sensitivity as rbss
from utils.io import validate_schema
from utils.metrics import parcel_metrics
from utils.paths import ProjectPaths, get_project_paths

_FS = "baseline_tessera"
_RUN = "20260101_000000__xgboost__baseline_tessera"


def _make_inputs(paths: ProjectPaths, n_per_fold: int = 10) -> None:
    """Write a synthetic labels GeoPackage and a complete fake nested-CV run."""
    rng = np.random.default_rng(0)
    rows: list[dict[str, object]] = []
    geoms = []
    pid = 1
    for fold in range(1, 7):
        for i in range(n_per_fold):
            col, row = (pid - 1) % 10, (pid - 1) // 10
            geoms.append(box(col * 1000.0, row * 1000.0, col * 1000.0 + 800, row * 1000.0 + 800))
            rows.append(
                {
                    "parcel_id": pid,
                    "fold_id": float(fold),
                    "ogf": float(i % 2),
                    "bootstrap_id": float(((pid - 1) % 20) + 1),
                }
            )
            pid += 1
    gdf = gpd.GeoDataFrame(rows, geometry=geoms, crs="EPSG:3035")
    paths.labels.mkdir(parents=True, exist_ok=True)
    gdf.to_file(paths.labels / "ogf_reference_labels_partitioned.gpkg")

    run_dir = paths.results / "main_nested_cv" / _RUN
    run_dir.mkdir(parents=True, exist_ok=True)
    y_all, p_all = [], []
    for fold in range(1, 7):
        sub = gdf[gdf["fold_id"] == fold]
        y = sub["ogf"].astype(np.int8).to_numpy()
        p = np.clip(0.2 + 0.6 * y + 0.1 * rng.standard_normal(len(y)), 0.0, 1.0)
        pd.DataFrame(
            {
                "parcel_id": sub["parcel_id"].astype(np.int64).to_numpy(),
                "outer_fold": np.full(len(y), fold, dtype=np.int8),
                "y_true": y,
                "p_mean": p.astype(np.float64),
                "n_pixels": np.full(len(y), 5, dtype=np.int64),
            }
        ).to_parquet(run_dir / f"parcel_predictions_fold{fold}.parquet")
        y_all.append(y)
        p_all.append(p)
    y = np.concatenate(y_all).astype(np.int64)
    p = np.concatenate(p_all)
    pr_auc = parcel_metrics(y, p)["pr_auc"]
    pd.DataFrame({"metric": ["pr_auc"], "value": [pr_auc]}).to_csv(
        run_dir / "pooled_parcel_metrics.csv", index=False
    )


def test_main_writes_block_size_table(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    paths = get_project_paths(tmp_path)
    monkeypatch.setattr(rbss, "get_project_paths", lambda: paths)
    _make_inputs(paths)

    code = rbss.main(
        ["--feature-set", _FS, "--block-counts", "6", "8", "20", "--n-reps", "60", "--seed", "42"]
    )
    assert code == 0

    out = paths.results / "block_size_sensitivity" / "block_size_sensitivity.csv"
    table = pd.read_csv(out)
    validate_schema(table, rbss.BLOCK_SIZE_SENSITIVITY_SCHEMA)
    assert table["n_blocks"].tolist() == [6, 8, 20]
    assert table.loc[table["n_blocks"] == 6, "block_source"].iloc[0] == "folds"
    assert table.loc[table["n_blocks"] == 8, "block_source"].iloc[0] == "spatial_partition"
    assert table.loc[table["n_blocks"] == 20, "block_source"].iloc[0] == "saved_partition"
    # Blocks only affect the resampling, so the point estimate is constant.
    assert table["pr_auc"].nunique() == 1
    assert (table["ci_hi"] >= table["ci_lo"]).all()
    assert (paths.results / "block_size_sensitivity" / "run_metadata.json").is_file()


def test_align_rejects_label_mismatch(tmp_path: Path) -> None:
    paths = get_project_paths(tmp_path)
    _make_inputs(paths)
    labelled = rbss._load_labelled_parcels(paths.labels / "ogf_reference_labels_partitioned.gpkg")
    preds = pd.DataFrame(
        {
            "parcel_id": labelled["parcel_id"].to_numpy(),
            "y_true": 1 - labelled["ogf"].to_numpy(),  # flipped -> mismatch
            "p_mean": np.full(len(labelled), 0.5),
        }
    )
    with pytest.raises(ValueError, match="disagrees with the labels"):
        rbss._align(preds, labelled)
