"""Synthetic smoke test for scripts/run_coordinate_xgboost.py."""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from affine import Affine
from rasterio.crs import CRS

from scripts import run_coordinate_xgboost as rsb
from utils.io import (
    PARCEL_PREDICTIONS_SCHEMA,
    PER_FOLD_METRICS_SCHEMA,
    validate_schema,
)
from utils.paths import ProjectPaths, get_project_paths
from utils.raster_io import ReferenceGrid
from utils.terminology import FEATURE_SET_BANDS, FOLD_IDS

_FS = "baseline"
_GRID_WIDTH = 16
_GRID_HEIGHT = 16


def _synthetic_grid() -> ReferenceGrid:
    """A tiny EPSG:3035 reference grid that decodes the synthetic pixel ids."""
    transform = Affine(10.0, 0.0, 4_000_000.0, 0.0, -10.0, 3_000_000.0)
    return ReferenceGrid(
        crs=CRS.from_epsg(3035), transform=transform, width=_GRID_WIDTH, height=_GRID_HEIGHT
    )


def _build_caches(paths: ProjectPaths, feature_set: str = _FS) -> pd.DataFrame:
    """Write a synthetic pixel index, validity mask and feature memmap (no headline run)."""
    n_bands = len(FEATURE_SET_BANDS[feature_set])
    rng = np.random.default_rng(0)
    rows: list[dict[str, object]] = []
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
                pixel_id += 1
            pid += 1
    index = pd.DataFrame(rows)

    index_path = paths.results / "main_nested_cv" / "pixel_index.parquet"
    index_path.parent.mkdir(parents=True, exist_ok=True)
    index.to_parquet(index_path)

    cache = paths.cache / "pixel_features"
    cache.mkdir(parents=True, exist_ok=True)
    np.save(
        cache / f"{feature_set}.npy",
        rng.standard_normal((len(index), n_bands)).astype(np.float32),
    )
    pd.DataFrame({"pixel_id": index["pixel_id"], "valid": True}).to_parquet(
        cache / f"{feature_set}_valid.parquet"
    )
    return index


def test_coordinate_features_decode_pixel_ids() -> None:
    """Coordinates decode from pixel ids and the grid transform, shifted to the min corner."""
    grid = ReferenceGrid(
        crs=CRS.from_epsg(3035),
        transform=Affine(10.0, 0.0, 500_000.0, 0.0, -10.0, 200_000.0),
        width=10,
        height=10,
    )
    index = pd.DataFrame({"pixel_id": [0, 1, 10, 11]})  # (row, col): (0,0) (0,1) (1,0) (1,1)
    features = rsb.coordinate_features(index, grid)
    expected = np.array([[0.0, 10.0], [10.0, 10.0], [0.0, 0.0], [10.0, 0.0]], dtype=np.float32)
    np.testing.assert_allclose(features, expected, atol=1e-3)
    assert features.dtype == np.float32


def test_main_writes_spatial_baseline_run(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    paths = get_project_paths(tmp_path)
    monkeypatch.setattr(rsb, "get_project_paths", lambda: paths)
    monkeypatch.setattr(rsb, "open_reference_grid", _synthetic_grid)
    _build_caches(paths)

    code = rsb.main(
        [
            "--valid-feature-set",
            _FS,
            "--device",
            "cpu",
            "--n-jobs",
            "1",
            "--n-trials",
            "1",
            "--n-per-parcel",
            "8",
            "--outer-folds",
            "1",
            "2",
        ]
    )
    assert code == 0

    run_dirs = [p for p in (paths.results / "spatial_baseline").iterdir() if p.is_dir()]
    assert len(run_dirs) == 1
    run_dir = run_dirs[0]
    assert run_dir.name.endswith("__xgboost__xy_coords")

    # Per-fold out-of-fold predictions in the main_nested_cv layout, schema-valid.
    for fold in (1, 2):
        parcels = pd.read_parquet(run_dir / f"parcel_predictions_fold{fold}.parquet")
        validate_schema(parcels, PARCEL_PREDICTIONS_SCHEMA)
    per_fold = pd.read_csv(run_dir / "per_fold_metrics.csv")
    validate_schema(per_fold, PER_FOLD_METRICS_SCHEMA)

    # The pooled parcel PR-AUC is a finite probability.
    pooled = pd.read_csv(run_dir / "pooled_parcel_metrics.csv")
    pr_auc = float(pooled.loc[pooled["metric"] == "pr_auc", "value"].iloc[0])
    assert 0.0 <= pr_auc <= 1.0

    # Importance covers exactly the two coordinate features.
    importance = pd.read_csv(run_dir / "importance_fold1.csv")
    assert set(importance["feature"]) == set(rsb._FEATURE_NAMES)

    metadata = json.loads((run_dir / "run_metadata.json").read_text())
    assert metadata["feature_label"] == "xy_coords"
    assert metadata["valid_feature_set"] == _FS
    assert metadata["n_labelled_parcels"] == 12  # 6 folds x 2 parcels
