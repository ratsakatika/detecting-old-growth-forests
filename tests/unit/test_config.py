"""Unit tests for the typed run-configuration objects in utils.config."""

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

import utils.config
from utils.config import (
    InferenceConfig,
    RunConfig,
    config_to_dict,
    write_config,
)
from utils.terminology import N_HP_TRIALS, N_PER_PARCEL, SEED


def test_importing_config_creates_nothing(tmp_path: Path) -> None:
    repo_root = Path(utils.config.__file__).resolve().parents[1]
    code = (
        "import pathlib\n"
        "import utils.config\n"
        "print(sorted(p.name for p in pathlib.Path('.').iterdir()))\n"
    )
    result = subprocess.run(
        [sys.executable, "-c", code],
        cwd=tmp_path,
        env={**os.environ, "PYTHONPATH": str(repo_root)},
        capture_output=True,
        text=True,
        check=True,
    )
    assert result.stdout.strip() == "[]"


def test_config_module_imports_no_heavy_libraries(tmp_path: Path) -> None:
    # The configuration layer must not pull in modelling or geospatial stacks.
    repo_root = Path(utils.config.__file__).resolve().parents[1]
    code = (
        "import sys\n"
        "import utils.config\n"
        "banned = {'sklearn', 'torch', 'xgboost', 'rasterio', 'geopandas'}\n"
        "print(sorted(banned & set(sys.modules)))\n"
    )
    result = subprocess.run(
        [sys.executable, "-c", code],
        cwd=tmp_path,
        env={**os.environ, "PYTHONPATH": str(repo_root)},
        capture_output=True,
        text=True,
        check=True,
    )
    assert result.stdout.strip() == "[]"


def test_run_config_accepts_valid_canonical_values() -> None:
    config = RunConfig(architecture="xgboost", feature_set="baseline", fold_id=3)

    assert config.architecture == "xgboost"
    assert config.feature_set == "baseline"
    assert config.fold_id == 3
    # Defaults come from utils.terminology.
    assert config.seed == SEED
    assert config.n_hp_trials == N_HP_TRIALS
    assert config.n_per_parcel == N_PER_PARCEL


def test_run_config_rejects_bad_architecture() -> None:
    with pytest.raises(ValueError, match="architecture"):
        RunConfig(architecture="random_forest", feature_set="baseline", fold_id=1)


def test_run_config_rejects_bad_feature_set() -> None:
    with pytest.raises(ValueError, match="feature set"):
        RunConfig(architecture="xgboost", feature_set="baseline_lidar", fold_id=1)


def test_run_config_rejects_bad_fold_id() -> None:
    for bad_fold in (0, 7, -1):
        with pytest.raises(ValueError, match="fold_id"):
            RunConfig(architecture="xgboost", feature_set="baseline", fold_id=bad_fold)


def test_run_config_rejects_non_positive_counts_and_negative_seed() -> None:
    base = {"architecture": "cnn_3x3", "feature_set": "baseline_conventional_eo", "fold_id": 2}

    with pytest.raises(ValueError, match="seed"):
        RunConfig(**base, seed=-1)
    with pytest.raises(ValueError, match="n_hp_trials"):
        RunConfig(**base, n_hp_trials=0)
    with pytest.raises(ValueError, match="n_per_parcel"):
        RunConfig(**base, n_per_parcel=0)


def test_inference_config_accepts_valid_and_defaults_calibrate_true() -> None:
    config = InferenceConfig(architecture="cnn_5x5", feature_set="baseline_tessera")

    assert config.architecture == "cnn_5x5"
    assert config.feature_set == "baseline_tessera"
    assert config.seed == SEED
    assert config.calibrate is True


def test_inference_config_rejects_bad_names() -> None:
    with pytest.raises(ValueError, match="architecture"):
        InferenceConfig(architecture="nope", feature_set="baseline")
    with pytest.raises(ValueError, match="feature set"):
        InferenceConfig(architecture="xgboost", feature_set="nope")


def test_config_to_dict_returns_plain_dict() -> None:
    config = RunConfig(architecture="xgboost", feature_set="baseline", fold_id=1)

    result = config_to_dict(config)

    assert type(result) is dict
    assert result == {
        "architecture": "xgboost",
        "feature_set": "baseline",
        "fold_id": 1,
        "seed": SEED,
        "n_hp_trials": N_HP_TRIALS,
        "n_per_parcel": N_PER_PARCEL,
    }


def test_config_to_dict_rejects_non_dataclass_objects() -> None:
    with pytest.raises(TypeError):
        config_to_dict({"architecture": "xgboost"})
    with pytest.raises(TypeError):
        config_to_dict("not a dataclass")


def test_inference_config_rejects_negative_seed() -> None:
    with pytest.raises(ValueError, match="seed"):
        InferenceConfig(architecture="xgboost", feature_set="baseline", seed=-1)
    # A dataclass *class* (not an instance) is also rejected.
    with pytest.raises(TypeError):
        config_to_dict(RunConfig)


def test_write_config_writes_deterministic_json(tmp_path: Path) -> None:
    config = RunConfig(architecture="xgboost", feature_set="baseline", fold_id=4)
    target = tmp_path / "run" / "config.json"

    returned = write_config(config, target)

    assert returned == target
    assert target.is_file()
    assert json.loads(target.read_text(encoding="utf-8")) == config_to_dict(config)

    # Re-writing the same config yields a byte-identical file.
    again = write_config(config, tmp_path / "run2" / "config.json")
    assert again.read_text(encoding="utf-8") == target.read_text(encoding="utf-8")
