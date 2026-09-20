"""Synthetic smoke test for scripts/run_gap_analysis.py."""

import json
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import pytest
from shapely.geometry import box

from scripts import run_gap_analysis as rga
from utils.io import validate_schema
from utils.paths import ProjectPaths, get_project_paths
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


def _build(paths: ProjectPaths, feature_set: str = _FS) -> None:
    """Write synthetic caches, a headline run and a labels GeoPackage with geometry."""
    n_bands = len(FEATURE_SET_BANDS[feature_set])
    rng = np.random.default_rng(0)
    rows: list[dict[str, object]] = []
    features: list[np.ndarray] = []
    geoms: list[object] = []
    parcel_meta: list[dict[str, object]] = []
    pid, pixel_id = 1, 0
    for fold in FOLD_IDS:
        for parcel in range(2):
            ogf = 1 - parcel
            # Parcels of fold f sit at x = (f-1) * 5 km, so distances vary by fold.
            x0 = (fold - 1) * 5000.0
            y0 = parcel * 1000.0
            geoms.append(box(x0, y0, x0 + 500.0, y0 + 500.0))
            parcel_meta.append(
                {
                    "parcel_id": pid,
                    "fold_id": float(fold),
                    "ogf": float(ogf),
                    "bootstrap_id": float(((pid - 1) % 5) + 1),
                }
            )
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

    paths.labels.mkdir(parents=True, exist_ok=True)
    gpd.GeoDataFrame(parcel_meta, geometry=geoms, crs="EPSG:3035").to_file(
        paths.labels / "ogf_reference_labels_partitioned.gpkg"
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


def test_main_writes_gap_tables(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    paths = get_project_paths(tmp_path)
    monkeypatch.setattr(rga, "get_project_paths", lambda: paths)
    _build(paths)

    code = rga.main(
        [
            "--feature-set",
            _FS,
            "--gap-km",
            "0",
            "5",
            "--device",
            "cpu",
            "--n-jobs",
            "1",
            "--reps",
            "32",
        ]
    )
    assert code == 0

    run_dirs = [p for p in (paths.results / "spatial_buffer").iterdir() if p.is_dir()]
    assert len(run_dirs) == 1
    table = pd.read_csv(run_dirs[0] / "gap_analysis.csv")
    validate_schema(table, rga.GAP_ANALYSIS_SCHEMA)
    assert set(table["gap_km"]) <= {0, 5}
    assert set(table["arm"]) <= {"buffered", "control"}

    # gap_detail carries the eight metric families x eleven statistics per (gap, arm).
    detail = pd.read_csv(run_dirs[0] / "gap_detail.csv")
    validate_schema(detail, rga.GAP_DETAIL_SCHEMA)
    assert detail["metric"].nunique() == 8
    assert detail["stat"].nunique() == 11
    pooled_pr_auc = detail[(detail["metric"] == "pr_auc") & (detail["stat"] == "pooled")]
    assert pooled_pr_auc["value"].notna().all()

    counts = pd.read_csv(run_dirs[0] / "gap_parcel_counts.csv")
    validate_schema(counts, rga.GAP_COUNTS_SCHEMA)
    # At gap 0 nothing is excluded; both arms keep the whole training set.
    gap0 = counts[counts["gap_km"] == 0]
    assert (gap0["n_excluded"] == 0).all()
    assert (gap0["arm"].value_counts() == len(FOLD_IDS)).all()
    # Retained-prevalence is populated; at gap 0 it is the full-training prevalence
    # (each synthetic fold holds one OGF and one non-OGF parcel, so 0.5 overall).
    assert counts["n_train_parcels_prevalence"].notna().all()
    assert np.allclose(gap0["n_train_parcels_prevalence"], 0.5)
    # The buffered and control arms must coincide at gap 0 (same metrics).
    base = table[table["gap_km"] == 0]
    buffered = base[base["arm"] == "buffered"].set_index(["level", "metric"])["pooled"]
    control = base[base["arm"] == "control"].set_index(["level", "metric"])["pooled"]
    pd.testing.assert_series_equal(buffered, control, check_names=False)


@pytest.mark.parametrize("arm", ["buffered", "control"])
def test_main_single_arm_runs_only_that_arm(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, arm: str
) -> None:
    """``--arms buffered`` / ``--arms control`` run (and score) only that one arm, so a
    feature set's buffered and control curves can be produced as separate merge-able sweeps."""
    paths = get_project_paths(tmp_path)
    monkeypatch.setattr(rga, "get_project_paths", lambda: paths)
    _build(paths)

    code = rga.main(
        [
            "--feature-set",
            _FS,
            "--gap-km",
            "0",
            "5",
            "--arms",
            arm,
            "--device",
            "cpu",
            "--n-jobs",
            "1",
            "--reps",
            "32",
        ]
    )
    assert code == 0

    run_dir = next(p for p in (paths.results / "spatial_buffer").iterdir() if p.is_dir())
    table = pd.read_csv(run_dir / "gap_analysis.csv")
    detail = pd.read_csv(run_dir / "gap_detail.csv")
    assert set(table["arm"]) == {arm}
    assert set(detail["arm"]) == {arm}


def test_latest_unfinished_run_matches_arm_and_newest(tmp_path: Path) -> None:
    """--auto-resume targets the newest unfinished run with checkpoints AND a matching arm
    marker, so a control sweep never resumes a buffered sweep."""
    root = tmp_path / "spatial_buffer"
    fs = "baseline_tessera"

    def make(
        run_id: str,
        *,
        checkpoint: bool,
        finished: bool,
        arms: tuple[str, ...] = ("buffered",),
        marker: bool = True,
    ) -> None:
        run = root / run_id
        run.mkdir(parents=True)
        if checkpoint:
            (run / rga._CHECKPOINT_DIR).mkdir()
            (run / rga._CHECKPOINT_DIR / "fold1_parcel.parquet").write_bytes(b"x")
        if finished:
            (run / "gap_analysis.csv").write_text("gap_km\n")
        if marker:
            (run / rga._RUN_ARMS_FILE).write_text(json.dumps({"arms": list(arms)}))

    # Ignored: finished run, checkpoint-less run, marker-less run, wrong feature set. The
    # control-arm run is only a candidate when control is requested.
    make(f"20260101_000000__xgboost__{fs}", checkpoint=True, finished=True)
    make(f"20260102_000000__xgboost__{fs}", checkpoint=False, finished=False)
    make(f"20260103_000000__xgboost__{fs}", checkpoint=True, finished=False, marker=False)
    make(f"20260104_000000__xgboost__{fs}", checkpoint=True, finished=False, arms=("control",))
    make(f"20260105_000000__xgboost__{fs}", checkpoint=True, finished=False)
    make(f"20260106_000000__xgboost__{fs}", checkpoint=True, finished=False)
    make("20260107_000000__xgboost__baseline_alphaearth", checkpoint=True, finished=False)

    # Buffered request: newest buffered-with-marker run wins; the control run is not eligible.
    assert rga._latest_unfinished_run(root, fs, ("buffered",)) == f"20260106_000000__xgboost__{fs}"
    # Control request: only the control-arm run matches.
    assert rga._latest_unfinished_run(root, fs, ("control",)) == f"20260104_000000__xgboost__{fs}"
    assert rga._latest_unfinished_run(root, "baseline_conventional_eo", ("buffered",)) is None
    assert rga._latest_unfinished_run(tmp_path / "missing", fs, ("buffered",)) is None


def test_gap_parcel_counts_matched_and_prevalence(tmp_path: Path) -> None:
    """gap_parcel_counts derives matched counts and the retained prevalence, no model."""
    paths = get_project_paths(tmp_path)
    _build(paths)
    labelled = rga._load_labelled_geoms(paths.labels / "ogf_reference_labels_partitioned.gpkg")

    counts = rga.gap_parcel_counts(labelled, gap_km_values=[0, 5], seed=42)
    validate_schema(counts, rga.GAP_COUNTS_SCHEMA)

    # The control removes the same number of parcels as the buffer at every gap.
    matched = counts.pivot_table(index=["gap_km", "outer_fold"], columns="arm", values="n_retained")
    assert (matched["buffered"] == matched["control"]).all()

    # At gap 0 nothing is excluded and the two arms coincide exactly.
    gap0 = counts[counts["gap_km"] == 0]
    assert (gap0["n_excluded"] == 0).all()
    prev0 = gap0.pivot_table(index="outer_fold", columns="arm", values="n_train_parcels_prevalence")
    assert np.allclose(prev0["buffered"], prev0["control"])
    assert np.allclose(prev0["buffered"], 0.5)


def test_load_cnn_fold_hps_uses_only_folds_with_both_files(tmp_path: Path) -> None:
    """Provisional CNN handling: a fold needs best_params and a recipe with cnn_epochs."""
    run = tmp_path / "20260101_000000__cnn_3x3__baseline_tessera"
    run.mkdir()
    for fold in (1, 3):
        (run / f"best_params_fold{fold}.json").write_text(json.dumps({"n_channels": 8}))
        (run / f"training_recipe_fold{fold}.json").write_text(json.dumps({"cnn_epochs": 5 + fold}))
    # Fold 2 has hyperparameters but no recipe, so it is skipped.
    (run / "best_params_fold2.json").write_text(json.dumps({"n_channels": 8}))

    best_params, cnn_epochs, inputs = rga._load_cnn_fold_hps(run)
    assert sorted(best_params) == [1, 3]
    assert cnn_epochs == {1: 6, 3: 8}
    assert len(inputs) == 4  # two files per usable fold


def test_latest_cnn_run_picks_newest_for_feature_set(tmp_path: Path) -> None:
    (tmp_path / "20260101_000000__cnn_3x3__baseline_tessera").mkdir()
    (tmp_path / "20260202_000000__cnn_3x3__baseline_tessera").mkdir()
    (tmp_path / "20260303_000000__cnn_3x3__baseline").mkdir()  # other feature set, ignored
    latest = rga._latest_cnn_run(tmp_path, "cnn_3x3", "baseline_tessera")
    assert latest.name == "20260202_000000__cnn_3x3__baseline_tessera"


def test_fold_checkpoint_round_trip(tmp_path: Path) -> None:
    """A fold's per-(arm, gap) predictions survive a checkpoint write and reload."""
    from utils.cv import FoldResult

    pixel = pd.DataFrame(
        {
            "pixel_id": [0, 1, 2, 3],
            "outer_fold": np.int8(1),
            "y_true": np.array([0, 1, 0, 1], dtype=np.int8),
            "p": [0.2, 0.8, 0.3, 0.7],
        }
    )
    parcel = pd.DataFrame(
        {
            "parcel_id": [10, 11],
            "outer_fold": np.int8(1),
            "y_true": np.array([0, 1], dtype=np.int8),
            "p_mean": [0.25, 0.75],
            "n_pixels": [2, 2],
        }
    )
    fold_result = FoldResult.from_predictions(
        1, pixel, parcel, best_params={"n_estimators": 10}, n_train=4
    )
    entries = [("buffered", 0, fold_result), ("control", 5, fold_result)]

    rga._write_fold_checkpoint(tmp_path, 1, entries)
    parcel_path, pixel_path = rga._fold_checkpoint_paths(tmp_path, 1)
    assert parcel_path.is_file() and pixel_path.is_file()

    rebuilt = rga._read_fold_checkpoint(tmp_path, 1, {"n_estimators": 10})
    assert set(rebuilt) == {("buffered", 0), ("control", 5)}
    pd.testing.assert_frame_equal(
        rebuilt[("control", 5)].parcel_predictions.reset_index(drop=True), parcel
    )
    assert rebuilt[("buffered", 0)].n_train == 4
