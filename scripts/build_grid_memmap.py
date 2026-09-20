"""Build flat, memory-mappable arrays of the feature-set stacks for the CNN.

For each feature set in :data:`utils.terminology.FEATURE_SETS`, this one-time
preprocessing step converts the analysis-ready raster stack into a flat
``float32`` array that the receptive-field CNN can memory-map and slice patches
from at training time. Every band is read exactly once here; nothing is decoded
inside the training loop.

Outputs live under ``data/cache/grid_memmap`` (the parallel of
``build_pixel_table``'s ``data/cache/pixel_features``): per feature set a real
``.npy`` of shape ``(n_bands, H, W)`` in C-order, reopenable with
``numpy.load(mmap_mode="r")``, plus a versioned JSON manifest. The build is
memory-safe -- it copies one spatial block at a time rather than loading the
whole array -- and idempotent: reuse requires matching source bytes,
reference-grid identity, shape and dtype. Legacy or incomplete manifests are
rebuilt unless ``--force`` already requests a rebuild.

Band-order assumption: the memmap preserves the GeoTIFF band order, which must be
the same order ``scripts/build_pixel_table.py`` used for the pixel cache, so the
CNN and XGBoost see identical features under identical names. The band-count
check (against :data:`utils.terminology.FEATURE_SET_BANDS`) is a partial guard;
the manifest's ``band_descriptions`` lets an auditor confirm full parity.

The source stacks are band-interleaved GeoTIFFs, so conversion streams spatial
blocks with every band together. Reading one whole band at a time would repeatedly
decode the other bands in each tile, multiplying TESSERA I/O. The total output is
roughly 38 GB across the four feature sets. The script uses CPU, disk and RAM only
(no GPU), so it can run alongside other jobs. Unlike the long-running modelling
scripts it deliberately writes no reproducibility-floor artefacts (no input
manifest or environment snapshot), matching how the
``pixel_features`` cache is treated. It does not import from or depend on
``archive/``.
"""

import argparse
import json
import logging
import time
from collections.abc import Sequence
from pathlib import Path
from typing import Any, Final

import numpy as np
import rasterio
from rasterio.crs import CRS

from utils import terminology
from utils.cache import (
    CACHE_MANIFEST_VERSION,
    atomic_write_json,
    file_signature,
    is_pre_lineage_manifest,
    npy_matches,
    replace_file,
    temporary_sibling,
)
from utils.logging import make_run_logger
from utils.paths import ProjectPaths, ensure_writable_dirs, get_project_paths
from utils.raster_io import ReferenceGrid, open_reference_grid
from utils.terminology import (
    FEATURE_SETS,
    MODEL_INPUT_BANDS,
    MODEL_INPUTS,
    NODATA,
    REF_RESOLUTION_M,
)

_LOGGER_NAME: Final[str] = "build_grid_memmap"

# Cached grid memmaps live under data/cache (gitignored), parallel to the pixel
# feature cache built by scripts/build_pixel_table.py (data/cache/pixel_features).
_CACHE_SUBDIR: Final[Path] = Path("grid_memmap")

# Stage 1 stack naming, mirroring scripts/build_pixel_table.py (_STACKS_SUBDIR /
# _STACK_SUFFIX) and notebook 004: the analysis-ready stack for a feature set is
# data/processed/rasters/stacks_10m/<feature_set>_3035_10m.tif.
_STACKS_SUBDIR: Final[Path] = Path("rasters") / "stacks_10m"
_STACK_SUFFIX: Final[str] = "_3035_10m.tif"


def _stack_path(paths: ProjectPaths, feature_set: str) -> Path:
    """Resolve a feature set's stack path, exactly as build_pixel_table does."""
    return paths.processed / _STACKS_SUBDIR / f"{feature_set}{_STACK_SUFFIX}"


def _repo_relative(path: Path) -> str:
    """Express ``path`` relative to the repository root, or absolutely if outside.

    Args:
        path: The path to express relative to the repository root.

    Returns:
        The repo-root-relative path as a string when ``path`` lies inside the
        repository, otherwise the resolved absolute path as a string.
    """
    resolved = Path(path).resolve()
    try:
        repo_root = get_project_paths().repo_root
    except FileNotFoundError:
        return str(resolved)
    try:
        return str(resolved.relative_to(repo_root))
    except ValueError:
        return str(resolved)


def _validate_stack(
    src: Any,
    reference_grid: ReferenceGrid,
    *,
    source_tif: Path,
    feature_set: str,
) -> None:
    """Validate a stack's band count and co-registration to the reference grid.

    A pixel ``(row, col)`` in the memmap must index the same ground location as
    ``(row, col)`` in the shared pixel index; that holds only when the source is
    the reference grid (same CRS, width, height and transform), not merely a grid
    with the same CRS and resolution. Misalignment would silently take every
    patch from the wrong place, so any mismatch raises before anything is written.

    Args:
        src: An open rasterio dataset for the source stack.
        reference_grid: The project reference grid the stack must align to.
        source_tif: Path to the source stack, used in error messages.
        feature_set: Canonical feature-set name keying the expected band count.

    Raises:
        ValueError: If the band count does not match
            :data:`utils.terminology.FEATURE_SET_BANDS`, if the CRS is not the
            project CRS, or if the stack is not co-registered to the reference
            grid (width, height or transform differ).
    """
    expected_bands = len(MODEL_INPUT_BANDS[feature_set])
    if src.count != expected_bands:
        raise ValueError(
            f"Stack {source_tif} has {src.count} bands but feature set {feature_set!r} "
            f"declares {expected_bands} in FEATURE_SET_BANDS."
        )

    project_crs = CRS.from_user_input(terminology.CRS)
    src_crs = src.crs
    if src_crs is None or src_crs != project_crs:
        found = src_crs.to_string() if src_crs is not None else "undefined"
        raise ValueError(f"Stack {source_tif} has CRS {found}, expected {terminology.CRS}.")
    if reference_grid.crs != project_crs:
        raise ValueError(
            f"Reference grid CRS {reference_grid.crs.to_string()} is not the project "
            f"CRS {terminology.CRS}."
        )

    if (src.width, src.height) != (reference_grid.width, reference_grid.height):
        raise ValueError(
            f"Stack {source_tif} is {src.width}x{src.height} but the reference grid is "
            f"{reference_grid.width}x{reference_grid.height}; the stack must be "
            "co-registered to the reference grid."
        )
    if not src.transform.almost_equals(reference_grid.transform):
        raise ValueError(
            f"Stack {source_tif} transform {src.transform!r} does not match the reference "
            f"grid transform {reference_grid.transform!r}; pixels must be {REF_RESOLUTION_M} m "
            "and aligned to the reference grid."
        )


def _reference_grid_signature(reference_grid: ReferenceGrid) -> dict[str, object]:
    """Return the grid properties that determine cache pixel identity."""
    transform = reference_grid.transform
    return {
        "crs": reference_grid.crs.to_string(),
        "width": reference_grid.width,
        "height": reference_grid.height,
        "transform": [
            transform.a,
            transform.b,
            transform.c,
            transform.d,
            transform.e,
            transform.f,
        ],
    }


def convert_stack_to_memmap(
    source_tif: Path,
    out_npy: Path,
    out_json: Path,
    *,
    feature_set: str,
    reference_grid: ReferenceGrid,
) -> dict[str, object]:
    """Convert one feature-set stack to an atomic ``.npy`` cache plus manifest.

    The stack is validated (band count and co-registration to ``reference_grid``)
    before any output is written, then copied by spatial block into a C-order
    ``float32`` array of shape ``(n_bands, height, width)`` via
    :func:`numpy.lib.format.open_memmap`. Values are copied verbatim and the
    NoData sentinel is preserved (the patch dataset masks it before
    standardisation later). A companion JSON manifest is written atomically
    after the array validates.

    Args:
        source_tif: Path to the analysis-ready feature-set stack (EPSG:3035, 10 m,
            multi-band ``float32``), co-registered to the reference grid.
        out_npy: Destination ``.npy`` memmap path, atomically replaced only
            after the temporary array validates.
        out_json: Destination JSON manifest path.
        feature_set: Canonical feature-set name (a key of
            :data:`utils.terminology.FEATURE_SETS`).
        reference_grid: The project reference grid the stack must align to.

    Returns:
        The manifest dictionary that was written to ``out_json``.

    Raises:
        ValueError: If validation fails, or if the written array's shape does not
            equal ``(n_bands, height, width)``.
    """
    temp_npy = temporary_sibling(out_npy)
    try:
        with rasterio.open(source_tif) as src:
            _validate_stack(src, reference_grid, source_tif=source_tif, feature_set=feature_set)
            count = int(src.count)
            height = int(src.height)
            width = int(src.width)
            descriptions = tuple(src.descriptions)

            out_npy.parent.mkdir(parents=True, exist_ok=True)
            memmap = np.lib.format.open_memmap(
                temp_npy, mode="w+", dtype=np.float32, shape=(count, height, width)
            )
            try:
                # The stacks are band-interleaved. Read every band for one
                # spatial block so each compressed tile is decoded once.
                for _, window in src.block_windows(1):
                    row_slice, col_slice = window.toslices()
                    memmap[:, row_slice, col_slice] = src.read(window=window, out_dtype=np.float32)
                memmap.flush()
            finally:
                # np.memmap exposes no close(); dropping the reference after
                # flushing releases the mapping for the atomic replacement.
                del memmap

        expected_shape = (count, height, width)
        if not npy_matches(temp_npy, expected_shape, dtype="float32"):
            raise ValueError(f"Wrote {temp_npy} with an invalid array signature.")

        manifest = _grid_manifest(
            feature_set=feature_set,
            shape=expected_shape,
            descriptions=descriptions,
            source_tif=source_tif,
            reference_grid=reference_grid,
        )
        replace_file(temp_npy, out_npy)
        atomic_write_json(manifest, out_json)
        return manifest
    finally:
        temp_npy.unlink(missing_ok=True)


def _grid_manifest(
    *,
    feature_set: str,
    shape: tuple[int, int, int],
    descriptions: tuple[str | None, ...],
    source_tif: Path,
    reference_grid: ReferenceGrid,
) -> dict[str, object]:
    """Build the grid-cache manifest, including the source and reference lineage."""
    band_descriptions = (
        list(descriptions) if any(description is not None for description in descriptions) else None
    )
    return {
        "cache_manifest_version": CACHE_MANIFEST_VERSION,
        "feature_set": feature_set,
        "shape": list(shape),
        "dtype": "float32",
        "n_bands": int(shape[0]),
        "nodata": NODATA,
        "crs": terminology.CRS,
        "resolution_m": REF_RESOLUTION_M,
        "source": _repo_relative(source_tif),
        "source_signature": file_signature(source_tif),
        "reference_grid": _reference_grid_signature(reference_grid),
        "band_descriptions": band_descriptions,
    }


def _adopt_grid_cache(
    out_npy: Path,
    out_json: Path,
    source_tif: Path,
    *,
    feature_set: str,
    reference_grid: ReferenceGrid,
) -> bool:
    """Re-describe a structurally valid pre-lineage grid cache without re-extraction.

    A grid array written before cache lineage existed is byte-identical to a fresh
    build when its source stack is unchanged, so rather than re-reading the
    multi-gigabyte stack it is validated structurally (band count, co-registration
    and array shape/dtype) and stamped with a current-lineage manifest recording
    the present source and reference-grid signatures.

    Args:
        out_npy: Existing grid-cache array.
        out_json: Manifest path to write atomically on adoption.
        source_tif: Source stack the array is assumed to derive from.
        feature_set: Canonical feature-set name.
        reference_grid: Project reference grid.

    Returns:
        ``True`` if the array matched the source shape/dtype and a fresh manifest
        was written; ``False`` if the array must instead be rebuilt.
    """
    with rasterio.open(source_tif) as src:
        _validate_stack(src, reference_grid, source_tif=source_tif, feature_set=feature_set)
        shape = (int(src.count), int(src.height), int(src.width))
        descriptions = tuple(src.descriptions)
    if not npy_matches(out_npy, shape, dtype="float32"):
        return False
    atomic_write_json(
        _grid_manifest(
            feature_set=feature_set,
            shape=shape,
            descriptions=descriptions,
            source_tif=source_tif,
            reference_grid=reference_grid,
        ),
        out_json,
    )
    return True


def _manifest_matches_source(
    out_npy: Path,
    out_json: Path,
    source_tif: Path,
    *,
    feature_set: str,
    reference_grid: ReferenceGrid,
) -> bool:
    """Return whether an existing cache exactly matches its source and grid.

    Args:
        out_json: Path to the existing JSON manifest.
        out_npy: Existing cache array.
        source_tif: Path to the source stack.
        feature_set: Expected canonical feature set.
        reference_grid: Expected project reference grid.

    Returns:
        ``True`` only for a current-version manifest, matching source bytes,
        reference-grid signature and readable array signature.
    """
    try:
        manifest = json.loads(out_json.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return False
    with rasterio.open(source_tif) as src:
        source_shape = (int(src.count), int(src.height), int(src.width))
    return (
        manifest.get("cache_manifest_version") == CACHE_MANIFEST_VERSION
        and manifest.get("feature_set") == feature_set
        and manifest.get("shape") == list(source_shape)
        and manifest.get("dtype") == "float32"
        and manifest.get("source_signature") == file_signature(source_tif)
        and manifest.get("reference_grid") == _reference_grid_signature(reference_grid)
        and npy_matches(out_npy, source_shape, dtype="float32")
    )


def ensure_grid_cache(
    feature_set: str,
    *,
    paths: ProjectPaths,
    logger: logging.Logger,
    force: bool = False,
) -> Path:
    """Build a current grid cache when absent or stale, returning its array path."""
    reference_grid = open_reference_grid()
    source_tif = _stack_path(paths, feature_set)
    out_dir = paths.cache / _CACHE_SUBDIR
    out_npy = out_dir / f"{feature_set}.npy"
    out_json = out_dir / f"{feature_set}.json"
    if (
        not force
        and out_npy.is_file()
        and out_json.is_file()
        and _manifest_matches_source(
            out_npy,
            out_json,
            source_tif,
            feature_set=feature_set,
            reference_grid=reference_grid,
        )
    ):
        logger.info("Feature set %r already cached; skipping.", feature_set)
        return out_npy

    # Adopt a pre-lineage cache: a valid array whose manifest predates lineage is
    # re-described from the current source rather than re-extracted. A mismatching
    # current-version manifest is not pre-lineage, so it still rebuilds below.
    if (
        not force
        and out_npy.is_file()
        and is_pre_lineage_manifest(out_json)
        and _adopt_grid_cache(
            out_npy, out_json, source_tif, feature_set=feature_set, reference_grid=reference_grid
        )
    ):
        logger.info("Adopted existing grid cache for %r without re-extraction.", feature_set)
        return out_npy

    start = time.perf_counter()
    manifest = convert_stack_to_memmap(
        source_tif,
        out_npy,
        out_json,
        feature_set=feature_set,
        reference_grid=reference_grid,
    )
    elapsed = time.perf_counter() - start
    size_gb = out_npy.stat().st_size / 1e9
    logger.info(
        "Feature set %r: shape %s, %.2f GB on disk, %.1f s.",
        feature_set,
        manifest["shape"],
        size_gb,
        elapsed,
    )
    return out_npy


def _parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    """Parse the command-line arguments.

    Args:
        argv: Argument list, or ``None`` to read from ``sys.argv``.

    Returns:
        The parsed arguments.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--feature-sets",
        nargs="+",
        choices=list(MODEL_INPUTS),
        help="Feature set(s) to build; defaults to all four.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Rebuild even when cached outputs already exist.",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    """Build the grid memmaps for the requested feature sets.

    Args:
        argv: Argument list, or ``None`` to read from ``sys.argv``.

    Returns:
        Process exit code (0 on success).
    """
    args = _parse_args(argv)
    paths = get_project_paths()
    ensure_writable_dirs(paths)

    out_dir = paths.cache / _CACHE_SUBDIR
    out_dir.mkdir(parents=True, exist_ok=True)
    logger = make_run_logger(_LOGGER_NAME, out_dir)

    feature_sets = args.feature_sets or list(FEATURE_SETS)
    logger.info("Building grid memmaps for %s (force=%s).", feature_sets, args.force)

    reference_grid = open_reference_grid()
    logger.info(
        "Reference grid: %dx%d, %s.",
        reference_grid.width,
        reference_grid.height,
        terminology.CRS,
    )

    for feature_set in feature_sets:
        ensure_grid_cache(feature_set, paths=paths, logger=logger, force=args.force)

    logger.info("Grid memmap build complete.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
