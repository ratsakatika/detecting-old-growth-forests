"""Run-scoped logging helpers for the old-growth-forests publication repository.

``make_run_logger`` is the only sanctioned way to obtain a logger (see AGENTS.md,
"Logging"). It attaches two handlers — one writing to standard output and one
writing to ``<run_dir>/logs/run.log`` — using the canonical format pinned in
AGENTS.md.

Importing this module only binds names: it configures no logging, calls no
``logging.basicConfig`` and touches no filesystem, in keeping with the
repository rule that ``utils`` modules have no side effects on import. The
``logs`` directory is created only when :func:`make_run_logger` is called.
"""

import logging
import sys
from pathlib import Path
from typing import Final

# Canonical log line format (see AGENTS.md, "Logging"). Milliseconds are appended
# explicitly via ``%(msecs)03d`` so the date itself carries no fractional part.
LOG_FORMAT: Final[str] = "%(asctime)s.%(msecs)03d %(levelname)-7s %(name)s %(message)s"

# Date layout for ``%(asctime)s``. Kept free of milliseconds so they are not
# duplicated by the ``.%(msecs)03d`` field in ``LOG_FORMAT``.
_DATE_FORMAT: Final[str] = "%Y-%m-%d %H:%M:%S"


def _has_stdout_handler(logger: logging.Logger) -> bool:
    """Return whether ``logger`` already streams to standard output.

    A :class:`~logging.FileHandler` is a :class:`~logging.StreamHandler`
    subclass, so it is excluded here to avoid mistaking the file handler for the
    stdout handler.

    Args:
        logger: The logger to inspect.

    Returns:
        ``True`` if an existing stdout :class:`~logging.StreamHandler` is
        present.
    """
    return any(
        type(handler) is logging.StreamHandler and handler.stream is sys.stdout
        for handler in logger.handlers
    )


def _reconcile_file_handler(logger: logging.Logger, log_file: Path) -> bool:
    """Drop stale file handlers and report whether ``log_file`` is already wired.

    Any :class:`~logging.FileHandler` targeting a path other than ``log_file`` is
    closed and removed, so a logger reused across run directories never keeps
    writing to a previous ``run.log``.

    Args:
        logger: The logger to inspect and prune.
        log_file: The resolved target log file for this run.

    Returns:
        ``True`` if a :class:`~logging.FileHandler` for ``log_file`` is already
        attached (so none needs adding); ``False`` otherwise.
    """
    target = log_file.resolve()
    has_target = False
    for handler in list(logger.handlers):
        if not isinstance(handler, logging.FileHandler):
            continue
        if Path(handler.baseFilename).resolve() == target:
            has_target = True
        else:
            handler.close()
            logger.removeHandler(handler)
    return has_target


def make_run_logger(
    name: str,
    run_dir: str | Path,
    level: int | str = logging.INFO,
) -> logging.Logger:
    """Create a logger that writes to standard output and the run's log file.

    The logger is configured with the canonical :data:`LOG_FORMAT` and writes to
    both ``sys.stdout`` and ``<run_dir>/logs/run.log``. The ``logs`` directory is
    created (with parents, idempotently) the first time this function runs for a
    given run directory; nothing is written to disk at import time.

    Propagation to the root logger is disabled so messages are not emitted twice.
    Repeated calls with the same ``name`` and ``run_dir`` are idempotent: the
    stdout and file handlers are added at most once, so log lines are never
    duplicated. When a logger name is reused with a different ``run_dir``, any
    file handler pointing at a previous ``run.log`` is closed and removed, so the
    logger is left with exactly one stdout handler and one file handler for the
    current run.

    Args:
        name: Logger name (typically the calling module or run identifier).
        run_dir: Run directory beneath which ``logs/run.log`` is written.
        level: Logging level, as an integer or a level name. Defaults to
            :data:`logging.INFO`.

    Returns:
        The configured logger.
    """
    log_dir = Path(run_dir) / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / "run.log"

    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.propagate = False

    formatter = logging.Formatter(LOG_FORMAT, datefmt=_DATE_FORMAT)

    if not _reconcile_file_handler(logger, log_file):
        file_handler = logging.FileHandler(log_file, encoding="utf-8")
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    if not _has_stdout_handler(logger):
        stream_handler = logging.StreamHandler(sys.stdout)
        stream_handler.setFormatter(formatter)
        logger.addHandler(stream_handler)

    return logger


__all__ = [
    "LOG_FORMAT",
    "make_run_logger",
]
