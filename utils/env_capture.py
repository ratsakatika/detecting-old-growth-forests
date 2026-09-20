"""Environment-snapshot helpers for reproducibility.

Long-running scripts record an ``env_snapshot.txt`` capturing the software and
hardware they ran on: ``pip freeze`` (packages and versions), ``nvidia-smi``
(GPU, driver and CUDA), ``uname -a`` (OS and kernel) and ``git rev-parse HEAD``
(code version) — see AGENTS.md, "Reproducibility floor". The helpers here run
those commands and assemble the snapshot.

Each command is run with ``shell=False`` and, if it is missing, times out or
exits non-zero, a clear placeholder is substituted rather than raising. This
keeps the snapshot useful on CPU-only CI where, for example, ``nvidia-smi`` is
absent. Importing this module runs no commands and writes nothing to disk.
"""

import subprocess
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Final

# Snapshot sections as (heading, command). The heading text is reproduced
# verbatim in the snapshot so each section is self-describing. ``pip freeze`` is
# run via the current interpreter so it reflects this environment exactly.
_SNAPSHOT_COMMANDS: Final[tuple[tuple[str, tuple[str, ...]], ...]] = (
    ("pip freeze", (sys.executable, "-m", "pip", "freeze")),
    ("nvidia-smi", ("nvidia-smi",)),
    ("uname -a", ("uname", "-a")),
    ("git rev-parse HEAD", ("git", "rev-parse", "HEAD")),
)


def run_command(command: Sequence[str], *, timeout_s: int = 30) -> str:
    """Run ``command`` without a shell and return its stripped stdout.

    The command is never interpreted by a shell (``shell=False``). Any failure
    mode — the executable is missing, the command times out, or it exits
    non-zero — yields a clear placeholder string instead of an exception, so a
    snapshot can always be assembled.

    Args:
        command: The command and its arguments as a sequence of strings.
        timeout_s: Seconds to wait before giving up on the command.

    Returns:
        The command's stripped standard output on success, otherwise a
        placeholder describing why no output was captured.
    """
    argv = list(command)
    try:
        completed = subprocess.run(
            argv,
            shell=False,
            capture_output=True,
            text=True,
            timeout=timeout_s,
            check=False,
        )
    except subprocess.TimeoutExpired:
        return f"<command timed out after {timeout_s}s: {' '.join(argv)}>"
    except FileNotFoundError:
        return f"<command not found: {argv[0]}>"
    except OSError as error:
        return f"<command unavailable: {argv[0]} ({error})>"

    if completed.returncode != 0:
        detail = completed.stderr.strip() or completed.stdout.strip()
        suffix = f"\n{detail}" if detail else ""
        return f"<command failed (exit {completed.returncode}): {' '.join(argv)}>{suffix}"

    return completed.stdout.strip()


def capture_environment_snapshot() -> str:
    """Assemble the environment snapshot as a single text block.

    Runs each command in :data:`_SNAPSHOT_COMMANDS` and concatenates the results
    under self-describing headings. Missing tools (such as ``nvidia-smi`` on
    CPU-only machines) contribute a placeholder rather than failing the whole
    snapshot.

    Returns:
        The snapshot text, with one headed section per command.
    """
    sections = [
        f"=== {heading} ===\n{run_command(command)}" for heading, command in _SNAPSHOT_COMMANDS
    ]
    return "\n\n".join(sections) + "\n"


def write_environment_snapshot(path: str | Path) -> Path:
    """Write the environment snapshot to ``path`` as UTF-8 text.

    Parent directories are created when this function is called.

    Args:
        path: Destination text file.

    Returns:
        The path written to.
    """
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(capture_environment_snapshot(), encoding="utf-8")
    return target


__all__ = [
    "capture_environment_snapshot",
    "run_command",
    "write_environment_snapshot",
]
