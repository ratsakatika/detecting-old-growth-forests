"""Unit tests for the environment-snapshot helpers in utils.env_capture."""

import importlib
import subprocess
import sys
from collections.abc import Sequence
from pathlib import Path

import pytest

import utils.env_capture
from utils.env_capture import (
    capture_environment_snapshot,
    run_command,
    write_environment_snapshot,
)


def _fake_run_factory(stdout: str = "FAKE OUTPUT", returncode: int = 0):
    """Build a stand-in for subprocess.run that echoes the command it received."""

    def fake_run(argv: Sequence[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
        joined = " ".join(argv)
        return subprocess.CompletedProcess(
            args=list(argv), returncode=returncode, stdout=f"{stdout} for: {joined}\n", stderr=""
        )

    return fake_run


def test_importing_runs_no_commands_and_creates_no_files(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)

    def boom(*args: object, **kwargs: object) -> object:
        raise AssertionError("subprocess.run must not be called at import time")

    monkeypatch.setattr(subprocess, "run", boom)

    # Re-executing the module body would invoke boom if it ran any command, and
    # would leave files in the empty working directory if it wrote anything.
    module = importlib.reload(utils.env_capture)

    assert hasattr(module, "capture_environment_snapshot")
    assert list(tmp_path.iterdir()) == []


def test_run_command_returns_stdout_for_successful_command() -> None:
    result = run_command([sys.executable, "-c", "print('hello world')"])
    assert result == "hello world"


def test_run_command_returns_placeholder_for_missing_command() -> None:
    result = run_command(["this-command-definitely-does-not-exist-xyz123"])
    # No exception; a clear placeholder naming the command instead.
    assert "this-command-definitely-does-not-exist-xyz123" in result
    assert result.startswith("<command not found")


def test_run_command_returns_placeholder_on_nonzero_exit() -> None:
    result = run_command([sys.executable, "-c", "import sys; sys.exit(3)"])
    assert "exit 3" in result


def test_capture_snapshot_contains_all_four_headings(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(subprocess, "run", _fake_run_factory())

    snapshot = capture_environment_snapshot()

    for heading in ("pip freeze", "nvidia-smi", "uname -a", "git rev-parse HEAD"):
        assert heading in snapshot
    assert "FAKE OUTPUT" in snapshot


def test_capture_snapshot_survives_missing_nvidia_smi(monkeypatch: pytest.MonkeyPatch) -> None:
    def selective_run(argv: Sequence[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
        if argv[0] == "nvidia-smi":
            raise FileNotFoundError(argv[0])
        return subprocess.CompletedProcess(list(argv), 0, stdout="ok\n", stderr="")

    monkeypatch.setattr(subprocess, "run", selective_run)

    snapshot = capture_environment_snapshot()

    # The missing GPU tool degrades to a placeholder; the rest is still captured.
    assert "nvidia-smi" in snapshot
    assert "<command not found: nvidia-smi>" in snapshot
    assert "ok" in snapshot


def test_write_environment_snapshot_writes_utf8_and_returns_path(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(subprocess, "run", _fake_run_factory())
    target = tmp_path / "run" / "logs" / "env_snapshot.txt"

    returned = write_environment_snapshot(target)

    assert returned == target
    assert target.is_file()
    contents = target.read_text(encoding="utf-8")
    assert contents == capture_environment_snapshot()
    assert "pip freeze" in contents


def test_run_command_returns_placeholder_on_timeout(monkeypatch: pytest.MonkeyPatch) -> None:
    def raise_timeout(argv: Sequence[str], **kwargs: object) -> object:
        raise subprocess.TimeoutExpired(cmd=list(argv), timeout=1)

    monkeypatch.setattr(subprocess, "run", raise_timeout)

    result = run_command(["sleep", "100"], timeout_s=1)

    assert result.startswith("<command timed out after 1s")
    assert "sleep 100" in result


def test_run_command_returns_placeholder_on_os_error(monkeypatch: pytest.MonkeyPatch) -> None:
    def raise_os_error(argv: Sequence[str], **kwargs: object) -> object:
        raise OSError("disk gone")

    monkeypatch.setattr(subprocess, "run", raise_os_error)

    result = run_command(["anything"])

    assert result.startswith("<command unavailable: anything")
    assert "disk gone" in result
