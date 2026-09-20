"""Tests for scripts/run_coordinate_cnn.py: prerequisites, commands and the launch loop."""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path
from typing import ClassVar

import pytest

from scripts import run_coordinate_cnn as rcc
from utils.paths import get_project_paths


class _FakePopen:
    """Stands in for subprocess.Popen: records the command and returns a fixed exit code."""

    launched: ClassVar[list[list[str]]] = []
    exit_codes: ClassVar[dict[str, int]] = {}

    def __init__(self, command, cwd, stdout, stderr):
        self.command = list(command)
        self.architecture = self.command[self.command.index("--architecture") + 1]
        stdout.write(f"fake search for {self.architecture}\n".encode())
        type(self).launched.append(self.command)

    def wait(self) -> int:
        return type(self).exit_codes.get(self.architecture, 0)


@pytest.fixture
def driver(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """The driver wired to a temporary repository with fake builders and subprocesses."""
    paths = get_project_paths(tmp_path)
    monkeypatch.setattr(rcc, "get_project_paths", lambda: paths)
    calls: list[tuple[str, list[str]]] = []
    for module, name in (
        (rcc.build_coordinate_stack, "stack"),
        (rcc.build_grid_memmap, "memmap"),
        (rcc.build_pixel_table, "table"),
    ):
        monkeypatch.setattr(
            module, "main", lambda argv, name=name: (calls.append((name, list(argv))), 0)[1]
        )
    _FakePopen.launched = []
    _FakePopen.exit_codes = {}
    monkeypatch.setattr(rcc.subprocess, "Popen", _FakePopen)
    monkeypatch.setattr(rcc.time, "sleep", lambda seconds: None)
    return paths, calls


def test_search_command_encodes_the_patch_size_and_budget() -> None:
    command = rcc.search_command("cnn_5x5", n_hp_trials=7, device="cpu", concurrency=3)
    assert command[:3] == [sys.executable, "-m", "scripts.run_nested_cv"]
    pairs = dict(zip(command[3::2], command[4::2], strict=True))
    assert pairs == {
        "--feature-set": "xy_coords",
        "--architecture": "cnn_5x5",
        "--patch-size": "5",
        "--device": "cpu",
        "--n-hp-trials": "7",
        "--concurrency": "3",
    }


def test_dry_run_builds_and_launches_nothing(driver, caplog: pytest.LogCaptureFixture) -> None:
    paths, calls = driver
    caplog.set_level(logging.INFO, logger=rcc.__name__)
    assert rcc.main(["--dry-run", "--n-hp-trials", "2"]) == 0
    assert calls == [] and _FakePopen.launched == []
    assert not (paths.results / rcc._RESULTS_ROLE).exists()
    assert sum("Dry run, cnn_" in record.message for record in caplog.records) == 3


def test_builds_inputs_in_order_then_runs_every_search(driver) -> None:
    paths, calls = driver
    assert rcc.main(["--device", "cpu", "--n-hp-trials", "2"]) == 0
    assert [name for name, _ in calls] == ["stack", "memmap", "table"]
    assert calls[1][1] == ["--feature-sets", "xy_coords"]
    assert calls[2][1] == ["--feature-set", "xy_coords"]
    launched = [command[command.index("--architecture") + 1] for command in _FakePopen.launched]
    assert launched == list(rcc._CNN_ARCHITECTURES)
    assert all("--concurrency" in command and command[-1] == "3" for command in _FakePopen.launched)

    run_dir = next((paths.results / rcc._RESULTS_ROLE).iterdir())
    metadata = json.loads((run_dir / "run_metadata.json").read_text())
    assert metadata["exit_codes"] == dict.fromkeys(rcc._CNN_ARCHITECTURES, 0)
    assert (run_dir / "cnn_3x3.log").read_text() == "fake search for cnn_3x3\n"
    assert (run_dir / "logs" / "run.log").is_file()


def test_architecture_subset_and_failed_search(driver) -> None:
    paths, _calls = driver
    _FakePopen.exit_codes = {"cnn_7x7": 3}
    assert rcc.main(["--architectures", "cnn_3x3", "cnn_7x7", "--device", "cpu"]) == 1
    launched = [command[command.index("--architecture") + 1] for command in _FakePopen.launched]
    assert launched == ["cnn_3x3", "cnn_7x7"]
    run_dir = next((paths.results / rcc._RESULTS_ROLE).iterdir())
    metadata = json.loads((run_dir / "run_metadata.json").read_text())
    assert metadata["exit_codes"] == {"cnn_3x3": 0, "cnn_7x7": 3}


def test_failed_prerequisite_stops_before_launching(driver, monkeypatch) -> None:
    _paths, calls = driver
    monkeypatch.setattr(rcc.build_grid_memmap, "main", lambda argv: 2)
    assert rcc.main(["--device", "cpu"]) == 2
    assert [name for name, _ in calls] == ["stack"]
    assert _FakePopen.launched == []
