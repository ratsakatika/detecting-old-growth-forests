"""Unit tests for the argparse type validators in utils.cli."""

import argparse
import os
import subprocess
import sys
from pathlib import Path

import pytest

import utils.cli
from utils.cli import (
    existing_dir,
    existing_file,
    fold_id,
    non_negative_int,
    output_path,
    positive_float,
    positive_int,
    unit_interval_float,
)
from utils.terminology import FOLD_IDS


def test_importing_cli_creates_nothing(tmp_path: Path) -> None:
    # Import in a fresh interpreter whose working directory is empty; nothing
    # should be created at import time.
    repo_root = Path(utils.cli.__file__).resolve().parents[1]
    code = (
        "import pathlib\n"
        "import utils.cli\n"
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


def test_positive_int_accepts_and_rejects() -> None:
    assert positive_int("3") == 3
    for bad in ("0", "-1", "1.5", "abc", ""):
        with pytest.raises(argparse.ArgumentTypeError):
            positive_int(bad)


def test_non_negative_int_accepts_and_rejects() -> None:
    assert non_negative_int("0") == 0
    assert non_negative_int("7") == 7
    for bad in ("-1", "2.0", "x"):
        with pytest.raises(argparse.ArgumentTypeError):
            non_negative_int(bad)


def test_positive_float_accepts_and_rejects() -> None:
    assert positive_float("0.5") == pytest.approx(0.5)
    assert positive_float("3") == pytest.approx(3.0)
    for bad in ("0", "0.0", "-2.5", "nan", "abc"):
        with pytest.raises(argparse.ArgumentTypeError):
            positive_float(bad)


def test_unit_interval_float_boundaries() -> None:
    assert unit_interval_float("0") == pytest.approx(0.0)
    assert unit_interval_float("1") == pytest.approx(1.0)
    assert unit_interval_float("0.42") == pytest.approx(0.42)
    for bad in ("-0.01", "1.01", "2", "-1"):
        with pytest.raises(argparse.ArgumentTypeError):
            unit_interval_float(bad)


def test_fold_id_accepts_canonical_and_rejects_others() -> None:
    for valid in FOLD_IDS:
        assert fold_id(str(valid)) == valid
    for bad in ("0", "7", "-1", "3.0", "foo"):
        with pytest.raises(argparse.ArgumentTypeError):
            fold_id(bad)


def test_existing_file_accepts_real_file_and_rejects_others(tmp_path: Path) -> None:
    real = tmp_path / "input.tif"
    real.write_text("data", encoding="utf-8")

    assert existing_file(str(real)) == real

    # Missing path and a directory are both rejected.
    with pytest.raises(argparse.ArgumentTypeError):
        existing_file(str(tmp_path / "missing.tif"))
    with pytest.raises(argparse.ArgumentTypeError):
        existing_file(str(tmp_path))


def test_existing_dir_accepts_real_dir_and_rejects_others(tmp_path: Path) -> None:
    real_file = tmp_path / "input.tif"
    real_file.write_text("data", encoding="utf-8")

    assert existing_dir(str(tmp_path)) == tmp_path

    # Missing path and a file are both rejected.
    with pytest.raises(argparse.ArgumentTypeError):
        existing_dir(str(tmp_path / "missing_dir"))
    with pytest.raises(argparse.ArgumentTypeError):
        existing_dir(str(real_file))


def test_output_path_returns_path_without_creating_anything(tmp_path: Path) -> None:
    target = tmp_path / "nested" / "result.parquet"

    returned = output_path(str(target))

    assert returned == target
    assert isinstance(returned, Path)
    # Nothing is created on disk.
    assert not target.exists()
    assert not target.parent.exists()


def test_validators_are_usable_as_argparse_types(tmp_path: Path) -> None:
    # End-to-end: argparse should surface ArgumentTypeError as a parse failure.
    parser = argparse.ArgumentParser()
    parser.add_argument("--fold", type=fold_id)
    parser.add_argument("--threshold", type=unit_interval_float)

    args = parser.parse_args(["--fold", "2", "--threshold", "0.7"])
    assert args.fold == 2
    assert args.threshold == pytest.approx(0.7)

    with pytest.raises(SystemExit):
        parser.parse_args(["--fold", "0"])
