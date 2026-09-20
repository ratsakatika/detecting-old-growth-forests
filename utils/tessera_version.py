"""Park and restore everything that depends on the TESSERA embedding version.

Notebook 001 is the only place the embedding version is chosen. Everything
downstream reads version-free names (``data/raw/rasters/tessera/``, the
``baseline_tessera`` feature set, the ``tessera_t*`` bands), so a second version
is swapped in by replacing data rather than code. :data:`VERSION_DEPENDENT`
lists, as repository-relative glob patterns, every artefact whose content
depends on the version: the raw tiles and their download cache, the 10 m mosaic
and the ``baseline_tessera`` stack, the caches built from that stack, the
Carpathian TESSERA layer, every result group that trains, evaluates or maps with
TESSERA, and the figure folders of the notebooks that read those results.

:func:`switch_version` moves the installed version's matches to
``archive/tessera_<version>/<same relative path>`` and brings the requested
version's matches back from its own slot. Moves are renames on one filesystem,
so a switch takes seconds; artefacts the requested version has never produced
are simply absent afterwards and are rebuilt by re-running the pipeline. Nothing
is ever overwritten: a destination that already exists aborts the switch before
anything moves.

Importing this module only binds names.
"""

import json
import logging
import re
import shutil
from pathlib import Path
from types import MappingProxyType
from typing import Final

# Mirror dataset path of each version this repository knows, keyed by the version
# name used in notebook 001, the provenance sidecars and the archive slots.
DATASET_PATHS: Final = MappingProxyType({"v1.1": "v1.1-cam", "v2": "v2-2B-L~beta1"})

# Repository-relative glob patterns of everything whose content depends on the
# version. Directory patterns end in ``/*`` so the directory itself (and any
# ``.gitkeep``) stays in place while its contents move.
VERSION_DEPENDENT: Final[tuple[str, ...]] = (
    "data/raw/rasters/tessera/*",
    "data/cache/tessera/*",
    "data/processed/rasters/tessera_10m/*",
    "data/processed/rasters/stacks_10m/baseline_tessera_3035_10m.tif",
    "data/processed/rasters/carpathians/tessera_100m/*",
    "data/cache/geotessera_carpathians/*",
    "data/cache/pixel_features/baseline_tessera*",
    "data/cache/grid_memmap/baseline_tessera*",
    "data/cache/patch_cache/baseline_tessera_p*",
    "results/main_nested_cv/*__baseline_tessera",
    "results/spatial_buffer/*__baseline_tessera",
    "results/sampling_sensitivity/*__baseline_tessera",
    "results/performance_matrix/*",
    "results/block_size_sensitivity/*",
    "results/seed_sweep/*",
    "results/autocorrelation/*",
    "results/fold_sensitivity/*",
    "results/mapping/*",
    "results/comparison/*",
    "results/final/*",
    "results/area_of_applicability/*",
    "figures/007_autocorrelation/*",
    "figures/009_performance_matrix_and_buffers/*",
    "figures/010_mapping/*",
    "figures/011_existing_product_comparison/*",
    "figures/012_methods_figures/*",
    "figures/013_area_of_applicability/*",
    "figures/014_robustness_checks/*",
    "figures/015_manuscript_tables/*",
)

_ARCHIVE_DIR: Final[Path] = Path("archive")
_TILE_DIR: Final[Path] = Path("data/raw/rasters/tessera")
_LABEL_PATTERN: Final[re.Pattern[str]] = re.compile(r"^TESSERA embeddings \((v[^,]+), (\d{4})\)$")

_LOG = logging.getLogger(__name__)


def dataset_label(version: str, year: int) -> str:
    """The ``dataset`` string the tile provenance sidecars carry for ``version``."""
    if version not in DATASET_PATHS:
        raise ValueError(f"unknown TESSERA version {version!r}; known: {sorted(DATASET_PATHS)}")
    return f"TESSERA embeddings ({version}, {year})"


def archive_slot(repo_root: Path, version: str) -> Path:
    """Where a version's artefacts live while another version is installed."""
    return repo_root / _ARCHIVE_DIR / f"tessera_{version}"


def installed_version(repo_root: Path) -> str | None:
    """The version whose tiles are installed, read from the provenance sidecars.

    Returns ``None`` when no sidecar is present and raises when the sidecars name
    more than one version.
    """
    versions: set[str] = set()
    for sidecar in sorted((repo_root / _TILE_DIR).glob("*.provenance.json")):
        label = json.loads(sidecar.read_text(encoding="utf-8")).get("dataset", "")
        match = _LABEL_PATTERN.match(label)
        if match:
            versions.add(match.group(1))
    if len(versions) > 1:
        raise RuntimeError(f"{repo_root / _TILE_DIR} mixes TESSERA versions {sorted(versions)}")
    return versions.pop() if versions else None


def _moves(source_root: Path, dest_root: Path) -> list[tuple[Path, Path]]:
    """Every version-dependent path under ``source_root`` with its destination."""
    moves: list[tuple[Path, Path]] = []
    for pattern in VERSION_DEPENDENT:
        for match in sorted(source_root.glob(pattern)):
            if match.name.startswith("."):
                continue
            moves.append((match, dest_root / match.relative_to(source_root)))
    return moves


def _prune_empty(root: Path) -> None:
    """Remove empty directories under ``root``, and ``root`` itself if it empties."""
    if not root.is_dir():
        return
    for path in sorted(root.rglob("*"), reverse=True):
        if path.is_dir() and not path.is_symlink() and not any(path.iterdir()):
            path.rmdir()
    if not any(root.iterdir()):
        root.rmdir()


def switch_version(
    repo_root: Path,
    installed: str | None,
    requested: str,
    *,
    logger: logging.Logger | None = None,
) -> dict[str, int]:
    """Park the installed version's artefacts and restore the requested version's.

    Args:
        repo_root: Repository root.
        installed: The version currently in the live tree, or ``None`` when no
            version is installed (nothing is parked then).
        requested: The version to install.
        logger: Logger; defaults to the module logger.

    Returns:
        ``{"parked": n, "restored": m}``, the number of paths moved each way.

    Raises:
        ValueError: If ``requested`` is not a known version.
        FileExistsError: If any destination already exists; nothing is moved.
    """
    log = logger or _LOG
    if requested not in DATASET_PATHS:
        raise ValueError(f"unknown TESSERA version {requested!r}; known: {sorted(DATASET_PATHS)}")
    if installed == requested:
        return {"parked": 0, "restored": 0}

    park = [] if installed is None else _moves(repo_root, archive_slot(repo_root, installed))
    restore = _moves(archive_slot(repo_root, requested), repo_root)
    vacated = {src for src, _ in park}
    conflicts = [dst for _, dst in park if dst.exists()]
    conflicts += [dst for _, dst in restore if dst.exists() and dst not in vacated]
    if conflicts:
        listed = ", ".join(str(p.relative_to(repo_root)) for p in conflicts[:5])
        raise FileExistsError(
            f"{len(conflicts)} destination(s) already exist, e.g. {listed}; nothing was moved"
        )

    for src, dst in park + restore:
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(src), str(dst))
    _prune_empty(archive_slot(repo_root, requested))
    log.info(
        "TESSERA %s -> %s: parked %d path(s), restored %d.",
        installed or "none",
        requested,
        len(park),
        len(restore),
    )
    return {"parked": len(park), "restored": len(restore)}
