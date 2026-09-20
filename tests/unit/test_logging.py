"""Unit tests for the run-scoped logging helpers in utils.logging."""

import importlib
import logging
from collections.abc import Iterator
from pathlib import Path

import pytest

import utils.logging
from utils.logging import LOG_FORMAT, make_run_logger


@pytest.fixture
def logger_name(tmp_path: Path) -> Iterator[str]:
    """Yield a unique logger name and remove its handlers afterwards.

    Loggers are process-global singletons keyed by name, so each test uses a
    distinct name and tears down its handlers to avoid leaking state between
    tests.
    """
    name = f"test_run_logger.{tmp_path.name}"
    yield name
    leftover = logging.getLogger(name)
    for handler in list(leftover.handlers):
        handler.close()
        leftover.removeHandler(handler)


def test_log_format_matches_agents_md() -> None:
    # Pinned verbatim in AGENTS.md, "Logging".
    assert LOG_FORMAT == "%(asctime)s.%(msecs)03d %(levelname)-7s %(name)s %(message)s"


def test_importing_logging_creates_no_logs_directory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Reloading re-executes the module body. If any logs directory were created
    # at import time it would appear under the current working directory.
    monkeypatch.chdir(tmp_path)
    module = importlib.reload(utils.logging)

    assert hasattr(module, "make_run_logger")
    assert not (tmp_path / "logs").exists()


def test_make_run_logger_creates_run_log_file(tmp_path: Path, logger_name: str) -> None:
    make_run_logger(logger_name, tmp_path)

    assert (tmp_path / "logs" / "run.log").is_file()


def test_logger_info_writes_to_log_file(tmp_path: Path, logger_name: str) -> None:
    logger = make_run_logger(logger_name, tmp_path)
    logger.info("hello from the file handler")

    contents = (tmp_path / "logs" / "run.log").read_text(encoding="utf-8")
    assert "hello from the file handler" in contents
    # The canonical format places the level and logger name on the line.
    assert "INFO" in contents
    assert logger_name in contents


def test_logger_info_writes_to_stdout(
    tmp_path: Path, logger_name: str, capsys: pytest.CaptureFixture[str]
) -> None:
    logger = make_run_logger(logger_name, tmp_path)
    logger.info("hello from stdout")

    captured = capsys.readouterr()
    assert "hello from stdout" in captured.out


def test_repeated_calls_do_not_duplicate_messages(
    tmp_path: Path, logger_name: str, capsys: pytest.CaptureFixture[str]
) -> None:
    first = make_run_logger(logger_name, tmp_path)
    second = make_run_logger(logger_name, tmp_path)

    # Same logger name and run_dir resolve to the same logger with no extra
    # handlers added on the second call.
    assert first is second
    assert len(second.handlers) == 2

    second.info("only once please")

    captured = capsys.readouterr()
    assert captured.out.count("only once please") == 1

    contents = (tmp_path / "logs" / "run.log").read_text(encoding="utf-8")
    assert contents.count("only once please") == 1


def test_reused_name_with_new_run_dir_switches_log_file(tmp_path: Path, logger_name: str) -> None:
    first_dir = tmp_path / "run_a"
    second_dir = tmp_path / "run_b"

    logger = make_run_logger(logger_name, first_dir)
    logger.info("written to the first run")

    # Reusing the same name with a new run_dir must retire the old file handler.
    logger = make_run_logger(logger_name, second_dir)
    logger.info("written to the second run")

    file_handlers = [h for h in logger.handlers if isinstance(h, logging.FileHandler)]
    assert len(file_handlers) == 1
    assert (
        Path(file_handlers[0].baseFilename).resolve() == (second_dir / "logs" / "run.log").resolve()
    )

    first_contents = (first_dir / "logs" / "run.log").read_text(encoding="utf-8")
    second_contents = (second_dir / "logs" / "run.log").read_text(encoding="utf-8")

    # The second message lands only in the second run's log file.
    assert "written to the second run" not in first_contents
    assert "written to the second run" in second_contents
    assert "written to the first run" in first_contents


def test_propagation_is_disabled(tmp_path: Path, logger_name: str) -> None:
    # Disabling propagation prevents duplicate emission via the root logger.
    logger = make_run_logger(logger_name, tmp_path)
    assert logger.propagate is False
