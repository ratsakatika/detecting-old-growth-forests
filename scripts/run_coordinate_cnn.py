"""Coordinate-only control for the receptive-field CNNs: build the inputs, run the searches.

This is the CNN counterpart of ``scripts/run_coordinate_xgboost.py``. The control runs
the ordinary nested cross-validation with the predictors replaced by the two
pixel-centre coordinates (feature set ``xy_coords``), once per CNN architecture. The
CNN path reads patches from a raster stack, so the control needs the coordinate
stack, its grid memmap and its pixel table on disk. This driver builds them first,
serially, so the searches never race to create them, then runs one
``scripts/run_nested_cv.py`` search per architecture as concurrent subprocesses that
share the GPU. Each search is an ordinary nested-CV run under
``results/main_nested_cv`` (``<run_id>__cnn_<p>x<p>__xy_coords``) and can be resumed
on its own with ``run_nested_cv --resume <run_id>``.

The driver's own record goes to ``results/coordinate_cnn_search/<timestamp>/``: its
log, each search's captured output and ``run_metadata.json`` with the exit codes.
``--dry-run`` logs the commands without building or launching anything.

Usage::

    python -m scripts.run_coordinate_cnn
    python -m scripts.run_coordinate_cnn --architectures cnn_3x3 --n-hp-trials 5 --device cpu
"""

from __future__ import annotations

import argparse
import logging
import subprocess
import sys
import time
from collections.abc import Callable, Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Final

from scripts import build_coordinate_stack, build_grid_memmap, build_pixel_table
from scripts.run_nested_cv import _patch_size_from_architecture
from utils.io import write_json
from utils.logging import LOG_FORMAT, make_run_logger
from utils.paths import ProjectPaths, ensure_writable_dirs, get_project_paths
from utils.terminology import ARCHITECTURES, N_HP_TRIALS

logger = logging.getLogger(__name__)

_FEATURE_SET: Final[str] = "xy_coords"
_RESULTS_ROLE: Final[str] = "coordinate_cnn_search"
_CNN_ARCHITECTURES: Final[tuple[str, ...]] = tuple(
    name for name in ARCHITECTURES if name.startswith("cnn_")
)
# Run ids have second resolution, so searches launched in the same second would collide.
_STAGGER_SECONDS: Final[float] = 5.0


def build_inputs(log: logging.Logger) -> int:
    """Build the coordinate stack, its grid memmap and its pixel table, in that order.

    Each builder is idempotent (an existing artefact is kept), so re-running the driver
    only fills in what is missing.

    Args:
        log: Logger for the progress lines.

    Returns:
        Zero on success, otherwise the exit code of the first builder that failed.
    """
    steps: tuple[tuple[str, Callable[[Sequence[str] | None], int], list[str]], ...] = (
        ("coordinate stack", build_coordinate_stack.main, []),
        ("grid memmap", build_grid_memmap.main, ["--feature-sets", _FEATURE_SET]),
        ("pixel table", build_pixel_table.main, ["--feature-set", _FEATURE_SET]),
    )
    for name, entry, argv in steps:
        log.info("Building the %s.", name)
        code = int(entry(argv))
        if code != 0:
            log.error("Building the %s failed with exit code %d.", name, code)
            return code
    return 0


def search_command(
    architecture: str, *, n_hp_trials: int, device: str, concurrency: int
) -> list[str]:
    """Return the ``run_nested_cv`` command for one architecture's coordinate-only search.

    Args:
        architecture: A ``cnn_<p>x<p>`` architecture name.
        n_hp_trials: Inner Optuna trials per outer fold.
        device: ``"cuda"`` or ``"cpu"``.
        concurrency: Number of searches sharing the machine (sizes the threads per fit).

    Returns:
        The command line, starting with the current interpreter.
    """
    return [
        sys.executable,
        "-m",
        "scripts.run_nested_cv",
        "--feature-set",
        _FEATURE_SET,
        "--architecture",
        architecture,
        "--patch-size",
        str(_patch_size_from_architecture(architecture)),
        "--device",
        device,
        "--n-hp-trials",
        str(n_hp_trials),
        "--concurrency",
        str(concurrency),
    ]


def run_searches(
    commands: Mapping[str, Sequence[str]],
    out_dir: Path,
    *,
    repo_root: Path,
    log: logging.Logger,
    stagger_seconds: float = _STAGGER_SECONDS,
) -> dict[str, int]:
    """Launch every search concurrently, wait for all of them and return their exit codes.

    Each search's standard output and error are captured in ``<out_dir>/<architecture>.log``.

    Args:
        commands: Command line per architecture, in launch order.
        out_dir: Directory receiving the captured outputs.
        repo_root: Working directory for the subprocesses.
        log: Logger for the launch and completion lines.
        stagger_seconds: Pause between launches so run ids never collide.

    Returns:
        Exit code per architecture.
    """
    processes: dict[str, subprocess.Popen[bytes]] = {}
    handles = []
    for index, (architecture, command) in enumerate(commands.items()):
        if index:
            time.sleep(stagger_seconds)
        handle = (out_dir / f"{architecture}.log").open("wb")
        handles.append(handle)
        log.info("Launching %s: %s", architecture, " ".join(command))
        processes[architecture] = subprocess.Popen(
            list(command), cwd=repo_root, stdout=handle, stderr=subprocess.STDOUT
        )
    codes: dict[str, int] = {}
    for architecture, process in processes.items():
        codes[architecture] = int(process.wait())
        log.info("%s finished with exit code %d.", architecture, codes[architecture])
    for handle in handles:
        handle.close()
    return codes


def _parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--architectures",
        nargs="+",
        default=list(_CNN_ARCHITECTURES),
        choices=list(_CNN_ARCHITECTURES),
        help="CNN architectures to search (default: all three).",
    )
    parser.add_argument("--n-hp-trials", type=int, default=N_HP_TRIALS)
    parser.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    parser.add_argument(
        "--dry-run", action="store_true", help="Log the commands; build and launch nothing."
    )
    return parser.parse_args(argv)


def run(args: argparse.Namespace, paths: ProjectPaths) -> int:
    """Execute the driver for parsed arguments; returns a process exit code."""
    architectures = list(args.architectures)
    commands = {
        architecture: search_command(
            architecture,
            n_hp_trials=args.n_hp_trials,
            device=args.device,
            concurrency=len(architectures),
        )
        for architecture in architectures
    }
    if args.dry_run:
        for architecture, command in commands.items():
            logger.info("Dry run, %s: %s", architecture, " ".join(command))
        return 0

    started = datetime.now(UTC)
    out_dir = paths.results / _RESULTS_ROLE / started.strftime("%Y%m%d_%H%M%S")
    out_dir.mkdir(parents=True, exist_ok=True)
    log = make_run_logger(__name__, out_dir)

    code = build_inputs(log)
    if code != 0:
        return code
    codes = run_searches(commands, out_dir, repo_root=paths.repo_root, log=log)
    write_json(
        {
            "created_at": started.isoformat(),
            "script": "scripts/run_coordinate_cnn.py",
            "feature_set": _FEATURE_SET,
            "device": args.device,
            "n_hp_trials": args.n_hp_trials,
            "exit_codes": codes,
        },
        out_dir / "run_metadata.json",
    )
    failed = sorted(name for name, code in codes.items() if code != 0)
    if failed:
        log.error("Searches failed: %s (see the logs in %s).", ", ".join(failed), out_dir)
        return 1
    log.info("All searches finished; runs are under %s.", paths.results / "main_nested_cv")
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    """Command-line entry point."""
    args = _parse_args(argv)
    paths = get_project_paths()
    ensure_writable_dirs(paths)
    return run(args, paths)


if __name__ == "__main__":
    # Console output for the dry run; real runs log through make_run_logger.
    logging.basicConfig(level=logging.INFO, format=LOG_FORMAT, stream=sys.stdout)
    raise SystemExit(main())
