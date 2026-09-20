"""Canonical project paths for the old-growth-forests publication repository.

This module is the single source of truth for where things live on disk. It
locates the repository root by walking upwards for the two markers every clone
carries (``AGENTS.md`` and ``pyproject.toml``) and derives every other path from
that root, so callers never hard-code absolute ``/home/$USER`` paths (hard rule
2).

Importing this module only binds names: it performs no input/output, configures
no logging and inspects no environment, in keeping with the repository rule that
``utils`` modules have no side effects on import. In particular,
:func:`find_repo_root` is never called at import time; path discovery and
validation happen only when a caller asks for them.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Final

# Markers that together identify the repository root. Both must be present so a
# stray ``pyproject.toml`` in a parent directory cannot be mistaken for the root.
_ROOT_MARKERS: Final[tuple[str, ...]] = ("AGENTS.md", "pyproject.toml")

# Reference grid, relative to the repository root (see AGENTS.md, "Spatial
# conventions"): 10 m, EPSG:3035, anchored on the Sentinel-2 / WorldCover grid.
# The file is the stacked (Sentinel-1/2 plus spectral-index bands), tile-merged,
# AOI-clipped WorldCover composite on the 10 m EPSG:3035 grid.
REFERENCE_GRID_RELATIVE: Final[Path] = Path(
    "data/processed/rasters/worldcover_composites_10m/stacked_merged_clipped_to_aoi_3035_10m.tif"
)

# Two-site Natura 2000 area-of-interest boundary, relative to the repository
# root: the canonical AOI polygon in the project CRS (EPSG:3035).
AOI_RELATIVE: Final[Path] = Path("data/processed/vectors/natura_2000/aoi_natura_2000.gpkg")

# Directory attributes of :class:`ProjectPaths` that must exist as directories.
# ``reference_grid`` is a file path, so it does not appear here.
_EXPECTED_DIRECTORIES: Final[tuple[str, ...]] = (
    "repo_root",
    "data",
    "raw",
    "processed",
    "cache",
    "labels",
    "figures",
    "results",
    "tests",
    "docs",
    "scripts",
    "notebooks",
    "utils",
)

# Directories code is permitted to create. Raw inputs are read-only and are never
# created here (see AGENTS.md, "Repository layout").
_WRITABLE_DIRECTORIES: Final[tuple[str, ...]] = (
    "processed",
    "cache",
    "labels",
    "figures",
    "results",
)


@dataclass(frozen=True, slots=True)
class ProjectPaths:
    """Canonical filesystem locations, all derived from :attr:`repo_root`.

    The record is frozen: paths are computed once by :func:`get_project_paths`
    and never mutated. Existence is not implied by construction; use
    :func:`validate_project_paths` to assert the expected layout is present.

    Attributes:
        repo_root: Repository root (contains ``AGENTS.md`` and ``pyproject.toml``).
        data: Root of the data tree.
        raw: Read-only raw inputs (symlinked datasets).
        processed: Writable processed outputs.
        cache: Writable intermediate cache.
        labels: Writable reference-label products.
        figures: Generated figures.
        results: Generated tables, run metadata and registry.
        tests: Unit and integration tests.
        docs: Supporting documentation.
        scripts: Parameterised CLI scripts.
        notebooks: Figure-producing notebooks.
        utils: Importable shared modules.
        reference_grid: Canonical 10 m EPSG:3035 reference raster.
        aoi: Canonical two-site Natura 2000 AOI boundary (project CRS).
    """

    repo_root: Path
    data: Path
    raw: Path
    processed: Path
    cache: Path
    labels: Path
    figures: Path
    results: Path
    tests: Path
    docs: Path
    scripts: Path
    notebooks: Path
    utils: Path
    reference_grid: Path
    aoi: Path


def find_repo_root(start: Path | str | None = None) -> Path:
    """Locate the repository root by walking upwards from ``start``.

    The root is the first directory at or above ``start`` that contains both
    ``AGENTS.md`` and ``pyproject.toml``.

    Args:
        start: Directory or file to start from. Defaults to the current working
            directory when ``None``.

    Returns:
        The resolved repository root.

    Raises:
        FileNotFoundError: If no ancestor directory carries both markers.
    """
    origin = Path.cwd() if start is None else Path(start)
    origin = origin.resolve()

    for candidate in (origin, *origin.parents):
        if all((candidate / marker).is_file() for marker in _ROOT_MARKERS):
            return candidate

    markers = " and ".join(_ROOT_MARKERS)
    raise FileNotFoundError(
        f"Could not find repository root above {origin}: no ancestor contains both {markers}."
    )


def get_project_paths(repo_root: Path | str | None = None) -> ProjectPaths:
    """Build the canonical :class:`ProjectPaths` for a repository root.

    This only composes paths; it does not touch the filesystem beyond
    normalising ``repo_root`` to a :class:`~pathlib.Path`. No path is checked for
    existence here — call :func:`validate_project_paths` for that.

    Args:
        repo_root: Repository root. When ``None``, it is discovered with
            :func:`find_repo_root`.

    Returns:
        The canonical paths derived from ``repo_root``.
    """
    root = find_repo_root() if repo_root is None else Path(repo_root)
    data = root / "data"

    return ProjectPaths(
        repo_root=root,
        data=data,
        raw=data / "raw",
        processed=data / "processed",
        cache=data / "cache",
        labels=data / "processed" / "vectors" / "labels",
        figures=root / "figures",
        results=root / "results",
        tests=root / "tests",
        docs=root / "docs",
        scripts=root / "scripts",
        notebooks=root / "notebooks",
        utils=root / "utils",
        reference_grid=root / REFERENCE_GRID_RELATIVE,
        aoi=root / AOI_RELATIVE,
    )


def validate_project_paths(paths: ProjectPaths) -> None:
    """Assert that ``paths`` describes a correctly laid-out repository.

    Checks that every expected directory exists and that ``reference_grid`` is
    the canonical reference raster path. This is explicit, opt-in validation; it
    is never called at import time.

    Args:
        paths: The paths to validate.

    Raises:
        ValueError: If ``reference_grid`` is not the canonical path.
        FileNotFoundError: If any expected directory is missing.
    """
    expected_grid = paths.repo_root / REFERENCE_GRID_RELATIVE
    if paths.reference_grid != expected_grid:
        raise ValueError(
            "reference_grid is not the canonical path: expected "
            f"{expected_grid}, got {paths.reference_grid}."
        )

    missing: list[str] = []
    for name in _EXPECTED_DIRECTORIES:
        directory = getattr(paths, name)
        if not directory.is_dir():
            missing.append(f"{name}: {directory}")

    if missing:
        joined = "\n  ".join(missing)
        raise FileNotFoundError(
            "Project layout is incomplete; the following paths are missing:\n  " f"{joined}"
        )


def ensure_writable_dirs(paths: ProjectPaths) -> None:
    """Create the writable output directories, leaving read-only inputs alone.

    Creates ``data/processed``, ``data/cache``, ``data/processed/vectors/labels``,
    ``figures`` and ``results`` (with parents, idempotently). Never creates
    ``raw``, which holds read-only inputs.

    Args:
        paths: The paths whose writable directories should be created.
    """
    for name in _WRITABLE_DIRECTORIES:
        getattr(paths, name).mkdir(parents=True, exist_ok=True)


__all__ = [
    "AOI_RELATIVE",
    "REFERENCE_GRID_RELATIVE",
    "ProjectPaths",
    "ensure_writable_dirs",
    "find_repo_root",
    "get_project_paths",
    "validate_project_paths",
]
