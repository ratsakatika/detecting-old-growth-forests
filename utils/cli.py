"""Reusable argparse type validators for parameterised scripts.

These small callables are intended as argparse ``type=`` hooks: each parses a
single command-line string, validates it, and either returns the converted
value or raises :class:`argparse.ArgumentTypeError` with a clear message.
Centralising them keeps the numbered scripts in ``scripts/`` consistent about
how they accept counts, probabilities, fold identifiers and paths.

Fold identifiers are validated against :data:`utils.terminology.FOLD_IDS` rather
than hard-coded, so the canonical 1..6 convention has a single source of truth.
Importing this module only binds names: it reads and writes nothing, and the
path validators touch the filesystem only when they are actually called.
"""

import argparse
from pathlib import Path

from utils.terminology import FOLD_IDS


def _parse_int(value: str) -> int:
    """Parse ``value`` as an integer or raise an argparse error.

    Args:
        value: The raw command-line string.

    Returns:
        The parsed integer.

    Raises:
        argparse.ArgumentTypeError: If ``value`` is not an integer.
    """
    try:
        return int(value)
    except (TypeError, ValueError):
        raise argparse.ArgumentTypeError(f"expected an integer, got {value!r}") from None


def _parse_float(value: str) -> float:
    """Parse ``value`` as a float or raise an argparse error.

    Args:
        value: The raw command-line string.

    Returns:
        The parsed float.

    Raises:
        argparse.ArgumentTypeError: If ``value`` is not a float.
    """
    try:
        return float(value)
    except (TypeError, ValueError):
        raise argparse.ArgumentTypeError(f"expected a number, got {value!r}") from None


def positive_int(value: str) -> int:
    """Validate ``value`` as a strictly positive integer.

    Args:
        value: The raw command-line string.

    Returns:
        The parsed integer, guaranteed greater than zero.

    Raises:
        argparse.ArgumentTypeError: If ``value`` is not an integer above zero.
    """
    parsed = _parse_int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError(f"expected a positive integer, got {parsed}")
    return parsed


def non_negative_int(value: str) -> int:
    """Validate ``value`` as an integer of zero or more.

    Args:
        value: The raw command-line string.

    Returns:
        The parsed integer, guaranteed to be zero or positive.

    Raises:
        argparse.ArgumentTypeError: If ``value`` is not a non-negative integer.
    """
    parsed = _parse_int(value)
    if parsed < 0:
        raise argparse.ArgumentTypeError(f"expected a non-negative integer, got {parsed}")
    return parsed


def positive_float(value: str) -> float:
    """Validate ``value`` as a strictly positive number.

    Args:
        value: The raw command-line string.

    Returns:
        The parsed float, guaranteed greater than zero.

    Raises:
        argparse.ArgumentTypeError: If ``value`` is not a number above zero.
    """
    parsed = _parse_float(value)
    if not parsed > 0:
        raise argparse.ArgumentTypeError(f"expected a positive number, got {parsed}")
    return parsed


def unit_interval_float(value: str) -> float:
    """Validate ``value`` as a number in the closed unit interval [0, 1].

    Args:
        value: The raw command-line string.

    Returns:
        The parsed float, guaranteed to lie between 0 and 1 inclusive.

    Raises:
        argparse.ArgumentTypeError: If ``value`` is not a number in [0, 1].
    """
    parsed = _parse_float(value)
    if not 0.0 <= parsed <= 1.0:
        raise argparse.ArgumentTypeError(f"expected a number in the interval [0, 1], got {parsed}")
    return parsed


def fold_id(value: str) -> int:
    """Validate ``value`` as a canonical fold identifier.

    Accepts only the fold IDs pinned in :data:`utils.terminology.FOLD_IDS`
    (1..6); fold 0 and anything above the configured number of folds are
    rejected.

    Args:
        value: The raw command-line string.

    Returns:
        The parsed fold identifier.

    Raises:
        argparse.ArgumentTypeError: If ``value`` is not one of ``FOLD_IDS``.
    """
    parsed = _parse_int(value)
    if parsed not in FOLD_IDS:
        raise argparse.ArgumentTypeError(f"fold id must be one of {list(FOLD_IDS)}, got {parsed}")
    return parsed


def existing_file(value: str) -> Path:
    """Validate ``value`` as the path of an existing file.

    Args:
        value: The raw command-line string.

    Returns:
        The path as a :class:`~pathlib.Path`.

    Raises:
        argparse.ArgumentTypeError: If the path is missing or is not a file.
    """
    path = Path(value)
    if not path.exists():
        raise argparse.ArgumentTypeError(f"file does not exist: {value}")
    if not path.is_file():
        raise argparse.ArgumentTypeError(f"not a file: {value}")
    return path


def existing_dir(value: str) -> Path:
    """Validate ``value`` as the path of an existing directory.

    Args:
        value: The raw command-line string.

    Returns:
        The path as a :class:`~pathlib.Path`.

    Raises:
        argparse.ArgumentTypeError: If the path is missing or is not a directory.
    """
    path = Path(value)
    if not path.exists():
        raise argparse.ArgumentTypeError(f"directory does not exist: {value}")
    if not path.is_dir():
        raise argparse.ArgumentTypeError(f"not a directory: {value}")
    return path


def output_path(value: str) -> Path:
    """Convert ``value`` to a :class:`~pathlib.Path` without touching disk.

    Nothing is created or checked; this only normalises the string to a path so
    a script can decide later where and when to write.

    Args:
        value: The raw command-line string.

    Returns:
        The path as a :class:`~pathlib.Path`.
    """
    return Path(value)


__all__ = [
    "existing_dir",
    "existing_file",
    "fold_id",
    "non_negative_int",
    "output_path",
    "positive_float",
    "positive_int",
    "unit_interval_float",
]
