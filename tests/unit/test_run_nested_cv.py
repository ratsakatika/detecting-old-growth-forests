"""Synthetic dispatch checks for scripts/run_nested_cv.py.

Tiny CPU runs of the architecture-aware driver. An XGBoost run writes the full
per-fold output set including feature importance; a receptive-field CNN run
writes the same outputs minus importance. Argument validation rejects an
inconsistent ``--patch-size``, and a subprocess check locks in that the XGBoost
code path never imports torch.
"""

import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from scripts import build_patch_cache, build_pixel_table, run_nested_cv
from utils.cache import CACHE_MANIFEST_VERSION, atomic_write_json, file_signature
from utils.hp_search import CNN_SEARCH_SPACE, XGB_SEARCH_SPACE
from utils.paths import ProjectPaths, get_project_paths
from utils.terminology import FEATURE_SET_BANDS

_FEATURE_SET = "baseline"
_N_BANDS = len(FEATURE_SET_BANDS[_FEATURE_SET])
_FOLDS = (1, 2, 3)  # the real six folds are shrunk to three to keep the run tiny
_GRID = 16  # square side of the CNN patch memmap
_SIGNAL = 5.0  # class signal placed in band 0 so the models can separate parcels


def _problem_index(width: int) -> pd.DataFrame:
    """A pixel index over folds 1-3, two parcels each, on grid-encoded pixel ids."""
    interior = [(r, c) for r in range(1, width - 1) for c in range(1, width - 1)]
    rows: list[dict[str, object]] = []
    cursor = 0
    for fold in _FOLDS:
        for parcel in range(2):  # one old-growth parcel, one not
            ogf = 1 if parcel == 0 else 0
            for _ in range(8):
                r, c = interior[cursor]
                cursor += 1
                rows.append(
                    {
                        "pixel_id": r * width + c,
                        "row": r,
                        "col": c,
                        "parcel_id": fold * 100 + parcel,
                        "fold_id": fold,
                        "ogf": ogf,
                        "forest_type_corine": "broadleaf" if ogf else "coniferous",
                    }
                )
    return pd.DataFrame(rows)


def _write_shared(paths: ProjectPaths, index: pd.DataFrame) -> None:
    """Write the pixel index and an all-valid mask shared by both architectures."""
    index_path = paths.results / "main_nested_cv" / "pixel_index.parquet"
    index_path.parent.mkdir(parents=True, exist_ok=True)
    index.to_parquet(index_path)
    valid = pd.DataFrame({"pixel_id": index["pixel_id"], "valid": True})
    valid_dir = paths.cache / "pixel_features"
    valid_dir.mkdir(parents=True, exist_ok=True)
    valid.to_parquet(valid_dir / f"{_FEATURE_SET}_valid.parquet")
    np.save(valid_dir / f"{_FEATURE_SET}.npy", np.zeros((len(index), _N_BANDS), dtype=np.float32))
    stack = paths.processed / "synthetic_stack.bin"
    stack.parent.mkdir(parents=True, exist_ok=True)
    stack.write_bytes(b"synthetic stack")
    atomic_write_json(
        {
            "cache_manifest_version": CACHE_MANIFEST_VERSION,
            "feature_set": _FEATURE_SET,
            "shape": [len(index), _N_BANDS],
            "dtype": "float32",
            "source_signature": file_signature(stack),
            "pixel_index_signature": file_signature(index_path),
        },
        valid_dir / f"{_FEATURE_SET}.json",
    )


def _build_flat(paths: ProjectPaths, seed: int = 0) -> pd.DataFrame:
    """Build the XGBoost caches: pixel index, validity mask and flat memmap."""
    rng = np.random.default_rng(seed)
    index = _problem_index(_GRID)
    _write_shared(paths, index)
    features = rng.standard_normal((len(index), _N_BANDS)).astype(np.float32)
    features[:, 0] = (
        _SIGNAL * index["ogf"].to_numpy() + 0.3 * rng.standard_normal(len(index))
    ).astype(np.float32)
    np.save(paths.cache / "pixel_features" / f"{_FEATURE_SET}.npy", features)
    return index


def _build_grid(paths: ProjectPaths, seed: int = 0) -> pd.DataFrame:
    """Build the CNN caches: pixel index, validity mask and a (bands, H, W) grid."""
    rng = np.random.default_rng(seed)
    index = _problem_index(_GRID)
    _write_shared(paths, index)
    grid = rng.standard_normal((_N_BANDS, _GRID, _GRID)).astype(np.float32)
    for pixel_id, ogf in zip(index["pixel_id"], index["ogf"], strict=True):
        r, c = divmod(int(pixel_id), _GRID)
        grid[0, r, c] = np.float32(_SIGNAL * int(ogf) + 0.3 * float(rng.standard_normal()))
    grid_dir = paths.cache / "grid_memmap"
    grid_dir.mkdir(parents=True, exist_ok=True)
    np.save(grid_dir / f"{_FEATURE_SET}.npy", grid)
    atomic_write_json(
        {
            "cache_manifest_version": CACHE_MANIFEST_VERSION,
            "feature_set": _FEATURE_SET,
            "shape": list(grid.shape),
            "dtype": "float32",
        },
        grid_dir / f"{_FEATURE_SET}.json",
    )
    return index


def _patch_paths(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> ProjectPaths:
    """Point the script at a tmp project root and shrink the fold set for speed."""
    paths = get_project_paths(tmp_path)
    monkeypatch.setattr(run_nested_cv, "get_project_paths", lambda: paths)
    monkeypatch.setattr(run_nested_cv, "FOLD_IDS", _FOLDS)
    monkeypatch.setattr(build_pixel_table, "ensure_pixel_cache_current", lambda *a, **k: None)
    monkeypatch.setattr(
        build_patch_cache,
        "ensure_grid_cache",
        lambda feature_set, *, paths, logger: paths.cache / "grid_memmap" / f"{feature_set}.npy",
    )
    return paths


def _only_run_dir(paths: ProjectPaths) -> Path:
    """The single run directory created under ``main_nested_cv``."""
    root = paths.results / "main_nested_cv"
    run_dirs = [path for path in root.iterdir() if path.is_dir()]
    assert len(run_dirs) == 1, run_dirs
    return run_dirs[0]


_SHARED_OUTPUTS = (
    "best_params_fold1.json",
    "boundary_fold1.csv",
    "predictions_fold1.parquet",
    "parcel_predictions_fold1.parquet",
    "hp_trials_fold1.csv",
    "training_recipe_fold1.json",
    "fold_status.json",
    "per_fold_metrics.csv",
    "summary_metrics.csv",
    "pooled_pixel_metrics.csv",
    "pooled_parcel_metrics.csv",
    "run_metadata.json",
)


def test_main_xgboost_smoke_writes_outputs_with_importance(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    paths = _patch_paths(monkeypatch, tmp_path)
    _build_flat(paths)

    code = run_nested_cv.main(
        [
            "--feature-set",
            _FEATURE_SET,
            "--architecture",
            "xgboost",
            "--smoke",
            "--device",
            "cpu",
            "--n-jobs",
            "1",
        ]
    )
    assert code == 0

    run_dir = _only_run_dir(paths)
    assert "__xgboost__" in run_dir.name
    for name in _SHARED_OUTPUTS:
        assert (run_dir / name).is_file(), name
    # XGBoost alone writes the feature-importance table.
    assert (run_dir / "importance_fold1.csv").is_file()
    meta = json.loads((run_dir / "run_metadata.json").read_text(encoding="utf-8"))
    assert meta["architecture"] == "xgboost"
    assert meta["patch_size"] is None
    assert "xgboost" in meta["library_versions"]
    recipe = json.loads((run_dir / "training_recipe_fold1.json").read_text(encoding="utf-8"))
    assert recipe["architecture"] == "xgboost"
    assert recipe["feature_names"] == list(FEATURE_SET_BANDS[_FEATURE_SET])
    assert recipe["cnn_epochs"] is None
    trials = pd.read_csv(run_dir / "hp_trials_fold1.csv")
    assert trials[list(XGB_SEARCH_SPACE)].notna().all().all()
    assert trials[[*CNN_SEARCH_SPACE, "cnn_epochs"]].isna().all().all()
    loaded = run_nested_cv._load_completed_fold(run_dir, 1)
    assert loaded.best_params == recipe["best_params"]
    assert loaded.cnn_epochs is None
    assert loaded.n_train > 0
    assert loaded.seconds > 0
    assert not loaded.boundary.empty
    assert not loaded.importance.empty


def test_main_cnn_smoke_writes_outputs_without_importance(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    paths = _patch_paths(monkeypatch, tmp_path)
    _build_grid(paths)

    code = run_nested_cv.main(
        [
            "--feature-set",
            _FEATURE_SET,
            "--architecture",
            "cnn_3x3",
            "--patch-size",
            "3",
            "--smoke",
            "--device",
            "cpu",
            # n_jobs 1 means zero patch-loader workers, so the tiny run stays in
            # the main process (no per-loader worker forks).
            "--n-jobs",
            "1",
        ]
    )
    assert code == 0

    run_dir = _only_run_dir(paths)
    assert "__cnn_3x3__" in run_dir.name
    for name in _SHARED_OUTPUTS:
        assert (run_dir / name).is_file(), name
    # The CNN has no feature importance, so it neither writes nor waits for one.
    assert not (run_dir / "importance_fold1.csv").exists()
    meta = json.loads((run_dir / "run_metadata.json").read_text(encoding="utf-8"))
    assert meta["architecture"] == "cnn_3x3"
    assert meta["patch_size"] == 3
    assert "torch" in meta["library_versions"]
    assert "xgboost" not in meta["library_versions"]
    recipe = json.loads((run_dir / "training_recipe_fold1.json").read_text(encoding="utf-8"))
    assert recipe["architecture"] == "cnn_3x3"
    assert recipe["patch_size"] == 3
    assert recipe["cnn_epochs"] > 0
    trials = pd.read_csv(run_dir / "hp_trials_fold1.csv")
    cnn_columns = ["n_channels", "lr", "weight_decay", "dropout", "batch_size", "cnn_epochs"]
    assert trials[cnn_columns].notna().all().all()
    assert trials[list(XGB_SEARCH_SPACE)].isna().all().all()
    loaded = run_nested_cv._load_completed_fold(run_dir, 1)
    assert loaded.best_params == recipe["best_params"]
    assert loaded.cnn_epochs == recipe["cnn_epochs"]
    assert loaded.n_train > 0
    assert loaded.seconds > 0
    assert not loaded.boundary.empty
    assert loaded.importance.empty


def test_fold_completion_requires_training_recipe(tmp_path: Path) -> None:
    fold_paths = run_nested_cv._FoldPaths.for_fold(tmp_path, 1)
    for path in (
        fold_paths.best_params,
        fold_paths.boundary,
        fold_paths.pixel_predictions,
        fold_paths.parcel_predictions,
        fold_paths.hp_trials,
    ):
        path.touch()

    assert not fold_paths.fold_complete(writes_importance=False)
    fold_paths.training_recipe.touch()
    assert fold_paths.fold_complete(writes_importance=False)


def test_training_recipe_validation_rejects_config_mismatch(tmp_path: Path) -> None:
    config = run_nested_cv.NestedCVConfig(
        feature_set=_FEATURE_SET,
        architecture="cnn_3x3",
        sampling_mode="none",
        n_per_parcel=20,
        n_hp_trials=2,
        seed=42,
        device="cpu",
        n_jobs=1,
        fold_ids=_FOLDS,
    )
    result = run_nested_cv.FoldResult.from_predictions(
        1,
        pd.DataFrame(
            {
                "pixel_id": pd.Series([1, 2], dtype="int64"),
                "outer_fold": pd.Series([1, 1], dtype="int8"),
                "y_true": pd.Series([0, 1], dtype="int8"),
                "p": pd.Series([0.1, 0.9], dtype="float64"),
            }
        ),
        pd.DataFrame(
            {
                "parcel_id": pd.Series([1, 2], dtype="int64"),
                "outer_fold": pd.Series([1, 1], dtype="int8"),
                "y_true": pd.Series([0, 1], dtype="int8"),
                "p_mean": pd.Series([0.1, 0.9], dtype="float64"),
                "n_pixels": pd.Series([1, 1], dtype="int64"),
            }
        ),
        best_params={"n_channels": 32},
        best_inner_pr_auc=0.8,
        cnn_epochs=8,
    )
    recipe_path = tmp_path / "training_recipe_fold1.json"
    run_nested_cv.write_json(
        run_nested_cv._training_recipe(config, result, patch_size=3), recipe_path
    )
    changed = run_nested_cv.NestedCVConfig(
        feature_set=_FEATURE_SET,
        architecture="cnn_3x3",
        sampling_mode="inverse_prevalence",
        n_per_parcel=20,
        n_hp_trials=2,
        seed=42,
        device="cpu",
        n_jobs=1,
        fold_ids=_FOLDS,
    )

    with pytest.raises(ValueError, match="sampling_mode"):
        run_nested_cv._validate_training_recipe(recipe_path, changed, outer_fold=1, patch_size=3)


def test_training_recipe_validation_rejects_incomplete_provenance(tmp_path: Path) -> None:
    config = run_nested_cv.NestedCVConfig(
        feature_set=_FEATURE_SET,
        architecture="cnn_3x3",
        sampling_mode="none",
        n_per_parcel=20,
        n_hp_trials=2,
        seed=42,
        device="cpu",
        n_jobs=1,
        fold_ids=_FOLDS,
    )
    recipe = {
        "schema_version": run_nested_cv.TRAINING_RECIPE_SCHEMA_VERSION,
        "architecture": config.architecture,
        "feature_set": config.feature_set,
        "feature_names": list(FEATURE_SET_BANDS[_FEATURE_SET]),
        "patch_size": 3,
        "sampling_mode": config.sampling_mode,
        "n_per_parcel": config.n_per_parcel,
        "n_hp_trials": config.n_hp_trials,
        "seed": config.seed,
        "device": config.device,
        "n_jobs": config.n_jobs,
        "fold_ids": list(config.fold_ids),
        "outer_fold": 1,
        "training_folds": [2, 3],
        "selection_metric": run_nested_cv.FINAL_RECIPE_SELECTION_METRIC,
        "best_params": {"n_channels": 32},
        "best_inner_pr_auc": 0.8,
        "cnn_epochs": None,
        "n_train": 10,
    }
    recipe_path = tmp_path / "training_recipe_fold1.json"
    run_nested_cv.write_json(recipe, recipe_path)

    with pytest.raises(ValueError, match="cnn_epochs"):
        run_nested_cv._validate_training_recipe(recipe_path, config, outer_fold=1, patch_size=3)


def test_default_hp_trials_follow_the_published_budgets() -> None:
    # 50 trials for XGBoost, 30 for the CNNs (manuscript Section 2.7); unset on the CLI.
    assert run_nested_cv._default_n_hp_trials("xgboost") == 50
    assert run_nested_cv._default_n_hp_trials("cnn_3x3") == 30
    assert run_nested_cv._parse_args(["--feature-set", _FEATURE_SET]).n_hp_trials is None
    parsed = run_nested_cv._parse_args(["--feature-set", _FEATURE_SET, "--n-hp-trials", "7"])
    assert parsed.n_hp_trials == 7


def test_patch_size_rejected_for_xgboost() -> None:
    with pytest.raises(ValueError, match="only valid for a CNN"):
        run_nested_cv.main(["--feature-set", _FEATURE_SET, "--patch-size", "3"])


def test_patch_size_required_for_cnn() -> None:
    with pytest.raises(ValueError, match="required for"):
        run_nested_cv.main(["--feature-set", _FEATURE_SET, "--architecture", "cnn_3x3"])


def test_patch_size_must_match_architecture() -> None:
    with pytest.raises(ValueError, match="does not match architecture"):
        run_nested_cv.main(
            ["--feature-set", _FEATURE_SET, "--architecture", "cnn_5x5", "--patch-size", "3"]
        )


_TORCH_FREE_CHILD = """
import logging
import sys

import pandas as pd

import scripts.run_nested_cv as m
from utils.cv import NestedCVConfig
from utils.paths import get_project_paths

paths = get_project_paths(sys.argv[1])
index = pd.read_parquet(paths.results / "main_nested_cv" / "pixel_index.parquet")
valid = pd.read_parquet(paths.cache / "pixel_features" / "baseline_valid.parquet")
index = index.assign(valid=valid["valid"].to_numpy())
config = NestedCVConfig(
    feature_set="baseline",
    architecture="xgboost",
    n_per_parcel=20,
    n_hp_trials=1,
    device="cpu",
    n_jobs=1,
    fold_ids=(1, 2, 3),
)
results, feature_path = m._run_one(
    config,
    index,
    paths,
    outer_folds=[1],
    on_trial_end=lambda fold, record: None,
    on_fold_end=lambda fold, result: None,
    fit_guard=None,
    torch_threads=1,
    logger=logging.getLogger("torch_free_probe"),
)
assert results and results[0].outer_fold == 1
assert "torch" not in sys.modules, sorted(k for k in sys.modules if "torch" in k)
print("XGB_TORCH_FREE_OK")
"""


def test_xgboost_run_one_does_not_import_torch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    paths = get_project_paths(tmp_path)
    _build_flat(paths)
    repo_root = Path(run_nested_cv.__file__).resolve().parents[1]
    # Drop pytest-cov's subprocess-coverage trigger: otherwise the child measures
    # run_nested_cv under its absolute path, which the relative "scripts/*" omit
    # does not match, and the omitted driver would leak into the coverage report.
    child_env = {
        key: value
        for key, value in os.environ.items()
        if not key.startswith(("COV_CORE", "COVERAGE"))
    }
    child_env["PYTHONPATH"] = str(repo_root)
    result = subprocess.run(
        [sys.executable, "-c", _TORCH_FREE_CHILD, str(tmp_path)],
        cwd=tmp_path,
        env=child_env,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert "XGB_TORCH_FREE_OK" in result.stdout
