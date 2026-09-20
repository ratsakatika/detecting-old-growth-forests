"""Unit tests for utils.hp_search."""

import os
import subprocess
import sys
from pathlib import Path

import optuna

from utils import hp_search
from utils.hp_search import (
    CNN_SEARCH_SPACE,
    XGB_SEARCH_SPACE,
    cnn_boundary_diagnostics,
    hp_boundary_diagnostics,
    suggest_cnn_params,
    suggest_xgb_params,
)

optuna.logging.set_verbosity(optuna.logging.WARNING)


def test_import_writes_no_files(tmp_path: Path) -> None:
    repo_root = Path(hp_search.__file__).resolve().parents[1]
    result = subprocess.run(
        [sys.executable, "-c", "import utils.hp_search"],
        cwd=tmp_path,
        env={**os.environ, "PYTHONPATH": str(repo_root)},
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert list(tmp_path.iterdir()) == []


def test_suggest_returns_all_params_in_range_and_typed() -> None:
    study = optuna.create_study(sampler=optuna.samplers.TPESampler(seed=0))
    params = suggest_xgb_params(study.ask())
    assert set(params) == set(XGB_SEARCH_SPACE)
    for name, spec in XGB_SEARCH_SPACE.items():
        value = params[name]
        assert spec.low <= value <= spec.high
        assert isinstance(value, int) if spec.kind == "int" else isinstance(value, float)


def test_boundary_diagnostics_flags_lower_and_upper() -> None:
    names = list(XGB_SEARCH_SPACE)
    best = {}
    for i, (name, spec) in enumerate(XGB_SEARCH_SPACE.items()):
        if i == 0:
            best[name] = spec.low
        elif i == 1:
            best[name] = spec.high
        else:
            best[name] = (spec.low + spec.high) / 2.0
    df = hp_boundary_diagnostics(best)
    assert set(df.columns) == {
        "parameter",
        "value",
        "low",
        "high",
        "at_lower_bound",
        "at_upper_bound",
    }
    assert bool(df.loc[df["parameter"] == names[0], "at_lower_bound"].iloc[0]) is True
    assert bool(df.loc[df["parameter"] == names[1], "at_upper_bound"].iloc[0]) is True


def test_boundary_diagnostics_midpoints_not_flagged() -> None:
    mid = {name: (spec.low + spec.high) / 2.0 for name, spec in XGB_SEARCH_SPACE.items()}
    df = hp_boundary_diagnostics(mid)
    assert not df["at_lower_bound"].any()
    assert not df["at_upper_bound"].any()


def test_boundary_diagnostics_skips_missing_keys() -> None:
    df = hp_boundary_diagnostics({"learning_rate": 0.05})
    assert list(df["parameter"]) == ["learning_rate"]


# --------------------------------------------------------------------------- #
# CNN search space (suggest_cnn_params, cnn_boundary_diagnostics).
# --------------------------------------------------------------------------- #


def test_suggest_cnn_returns_all_params_in_range_and_typed() -> None:
    study = optuna.create_study(sampler=optuna.samplers.TPESampler(seed=0))
    params = suggest_cnn_params(study.ask())
    assert set(params) == set(CNN_SEARCH_SPACE)
    for name, spec in CNN_SEARCH_SPACE.items():
        value = params[name]
        assert spec.low <= value <= spec.high
        assert isinstance(value, int) if spec.kind == "int" else isinstance(value, float)


def test_cnn_boundary_diagnostics_flags_bounds_and_skips_missing() -> None:
    names = list(CNN_SEARCH_SPACE)
    best: dict[str, float | int] = {}
    for i, (name, spec) in enumerate(CNN_SEARCH_SPACE.items()):
        if i == 0:
            best[name] = spec.low
        elif i == 1:
            best[name] = spec.high
        else:
            best[name] = (spec.low + spec.high) / 2.0
    # Drop the final parameter so the missing-key skip branch is exercised.
    dropped = names[-1]
    del best[dropped]

    df = cnn_boundary_diagnostics(best)
    assert set(df.columns) == {
        "parameter",
        "value",
        "low",
        "high",
        "at_lower_bound",
        "at_upper_bound",
    }
    assert dropped not in set(df["parameter"])
    assert bool(df.loc[df["parameter"] == names[0], "at_lower_bound"].iloc[0]) is True
    assert bool(df.loc[df["parameter"] == names[1], "at_upper_bound"].iloc[0]) is True


def test_cnn_boundary_diagnostics_midpoints_not_flagged() -> None:
    mid = {name: (spec.low + spec.high) / 2.0 for name, spec in CNN_SEARCH_SPACE.items()}
    df = cnn_boundary_diagnostics(mid)
    assert not df["at_lower_bound"].any()
    assert not df["at_upper_bound"].any()
