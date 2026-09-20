"""Build per-pixel patch caches for the receptive-field CNN.

For a ``(feature_set, patch_size)`` pair this one-time preprocessing step
memoises the patch extraction the CNN would otherwise repeat every epoch. Each
training epoch of the nested cross-validation gathers millions of strided
windows from the multi-gigabyte grid memmap; reading them instead from a
contiguous cache turns the loaders from CPU-bound to trivially fast and is the
performance fix that takes a single configuration off a multi-week path.

Outputs live under ``data/cache/patch_cache`` (the parallel of
``build_grid_memmap``'s ``data/cache/grid_memmap``): per ``(feature_set,
patch_size)`` a real ``.npy`` of shape ``(n_pixels, n_bands, P, P)`` in C-order,
reopenable with ``numpy.load(mmap_mode="r")``, plus a JSON manifest. Row ``i`` is
the raw, sentinel-preserving patch (:func:`utils.patches.extract_raw_patch`, no
mean-fill and no standardisation, so the cache is fold-independent) centred on
pixel row ``i`` of ``results/main_nested_cv/pixel_index.parquet``, taken from that
row's ``row``/``col`` columns and cut from ``data/cache/grid_memmap/<fs>.npy``.
The per-fold standardisation is applied later in :class:`utils.patches.PatchDataset`,
so cached patches feed the model unchanged.

The build is memory-safe: it streams the pixel rows in chunks and writes each
patch straight into the on-disk memmap, so peak memory is about one patch rather
than the whole cache (which can be tens of gigabytes for the embedding feature
sets at the larger patch sizes). Chunks are processed in parallel with joblib
when there is more than one, each worker writing a disjoint row range of the
shared ``r+`` memmap. The build is idempotent: a ``(feature_set, patch_size)``
whose ``.npy`` and versioned manifest match the validated grid manifest, exact
pixel-index bytes, feature set, patch size, output shape and dtype is skipped
unless ``--force``. Legacy manifests rebuild. A new array is filled and validated
under a temporary sibling path before atomically replacing the old cache.

Like ``build_grid_memmap`` and the ``pixel_features`` cache, this deliberately
writes no reproducibility-floor artefacts (no input manifest or environment
snapshot). It does not import from or depend on ``archive/``.
"""

import argparse
import json
import logging
import os
import time
from collections.abc import Sequence
from pathlib import Path
from typing import Final

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

from scripts.build_grid_memmap import ensure_grid_cache
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
from utils.patches import extract_raw_patch
from utils.paths import ProjectPaths, ensure_writable_dirs, get_project_paths
from utils.terminology import (
    CNN_PATCH_SIZES,
    FEATURE_SETS,
    MODEL_INPUT_BANDS,
    MODEL_INPUTS,
    NODATA,
)

_LOGGER_NAME: Final[str] = "build_patch_cache"

# The patch cache lives under data/cache (gitignored), parallel to the grid
# memmap (data/cache/grid_memmap) it is built from.
_CACHE_SUBDIR: Final[Path] = Path("patch_cache")
_GRID_CACHE_SUBDIR: Final[Path] = Path("grid_memmap")

# Shared pixel index location (written by scripts/build_pixel_table.py).
_RESULTS_ROLE: Final[str] = "main_nested_cv"
_PIXEL_INDEX_FILENAME: Final[str] = "pixel_index.parquet"

# Pixel rows per build task. Patches are written one at a time into the memmap,
# so this only bounds the joblib work-unit size, not peak memory.
_CHUNK_ROWS: Final[int] = 8192

# Upper bound on the worker count for the parallel chunk fill.
_MAX_N_JOBS: Final[int] = 32


def _cache_paths(paths: ProjectPaths, feature_set: str, patch_size: int) -> tuple[Path, Path]:
    """Return the ``(.npy, .json)`` cache paths for a ``(feature_set, patch_size)``."""
    cache_dir = paths.cache / _CACHE_SUBDIR
    stem = f"{feature_set}_p{patch_size}"
    return cache_dir / f"{stem}.npy", cache_dir / f"{stem}.json"


def _grid_path(paths: ProjectPaths, feature_set: str) -> Path:
    """Return the grid memmap path for a feature set."""
    return paths.cache / _GRID_CACHE_SUBDIR / f"{feature_set}.npy"


def _fill_chunk(
    grid_path: Path,
    out_npy: Path,
    rows: np.ndarray,
    cols: np.ndarray,
    start: int,
    *,
    patch_size: int,
    nodata: float,
) -> None:
    """Write the raw patches for one contiguous range of cache rows.

    Opens the grid and output memmaps read-only and read/write respectively and
    fills ``out[start : start + len(rows)]``. Several tasks run concurrently over
    disjoint ranges, which is safe because each writes only its own rows.

    Args:
        grid_path: Path to the source grid memmap ``(n_bands, height, width)``.
        out_npy: Path to the output cache memmap (already allocated on disk).
        rows: Centre-pixel rows for this range.
        cols: Centre-pixel columns for this range.
        start: First cache row index this range writes.
        patch_size: Odd patch side length.
        nodata: NoData sentinel passed to :func:`utils.patches.extract_raw_patch`.
    """
    grid = np.load(grid_path, mmap_mode="r")
    out = np.load(out_npy, mmap_mode="r+")
    try:
        for offset in range(int(rows.shape[0])):
            out[start + offset] = extract_raw_patch(
                grid, int(rows[offset]), int(cols[offset]), patch_size=patch_size, nodata=nodata
            )
        out.flush()
    finally:
        del out
        del grid


def build_patch_cache(
    grid_path: Path,
    pixel_index: pd.DataFrame,
    out_npy: Path,
    out_json: Path,
    *,
    feature_set: str,
    patch_size: int,
    chunk_rows: int = _CHUNK_ROWS,
    n_jobs: int = 1,
    nodata: float = NODATA,
    grid_manifest_signature: dict[str, str | int],
    pixel_index_signature: dict[str, str | int],
) -> dict[str, object]:
    """Build one ``(feature_set, patch_size)`` patch cache plus its manifest.

    Allocates a C-order ``float32`` memmap of shape ``(n_pixels, n_bands, P, P)``
    then streams the pixel rows in chunks, writing each row's raw patch
    (:func:`utils.patches.extract_raw_patch`) straight to disk so peak memory is
    about one patch. Chunks are filled in parallel when there is more than one and
    ``n_jobs > 1``; otherwise they are filled sequentially in-process.

    Args:
        grid_path: Path to the source grid memmap ``(n_bands, height, width)``.
        pixel_index: Pixel index with ``row`` and ``col`` columns, in cache order.
        out_npy: Destination ``.npy`` memmap path (created or overwritten here).
        out_json: Destination JSON manifest path.
        feature_set: Canonical feature-set name (a key of
            :data:`utils.terminology.FEATURE_SETS`).
        patch_size: Odd patch side length (one of
            :data:`utils.terminology.CNN_PATCH_SIZES`).
        chunk_rows: Pixel rows per build task.
        n_jobs: Worker processes for the parallel chunk fill.
        nodata: NoData sentinel recorded in the manifest and used for out-of-grid
            fill.
        grid_manifest_signature: Content signature of the validated source-grid
            manifest.
        pixel_index_signature: Content signature of the pixel-index Parquet
            whose row order defines this cache.

    Returns:
        The manifest dictionary that was written to ``out_json``.

    Raises:
        ValueError: If the written array's shape does not equal the expected
            ``(n_pixels, n_bands, patch_size, patch_size)``.
    """
    grid_shape = np.load(grid_path, mmap_mode="r").shape
    n_bands, height, width = int(grid_shape[0]), int(grid_shape[1]), int(grid_shape[2])

    rows = pixel_index["row"].to_numpy(dtype=np.int64)
    cols = pixel_index["col"].to_numpy(dtype=np.int64)
    n_pixels = int(rows.shape[0])
    shape = (n_pixels, n_bands, patch_size, patch_size)

    temp_npy = temporary_sibling(out_npy)
    try:
        out_npy.parent.mkdir(parents=True, exist_ok=True)
        # Allocate the full memmap on disk, then release it so the chunk fillers
        # can reopen it read/write and write disjoint row ranges.
        created = np.lib.format.open_memmap(temp_npy, mode="w+", dtype=np.float32, shape=shape)
        del created

        bounds = [
            (start, min(start + chunk_rows, n_pixels)) for start in range(0, n_pixels, chunk_rows)
        ]
        if n_jobs > 1 and len(bounds) > 1:
            Parallel(n_jobs=min(n_jobs, len(bounds)))(
                delayed(_fill_chunk)(
                    grid_path,
                    temp_npy,
                    rows[start:end],
                    cols[start:end],
                    start,
                    patch_size=patch_size,
                    nodata=nodata,
                )
                for start, end in bounds
            )
        else:
            for start, end in bounds:
                _fill_chunk(
                    grid_path,
                    temp_npy,
                    rows[start:end],
                    cols[start:end],
                    start,
                    patch_size=patch_size,
                    nodata=nodata,
                )

        if not npy_matches(temp_npy, shape, dtype="float32"):
            raise ValueError(f"Wrote {temp_npy} with an invalid array signature.")

        manifest = _patch_manifest(
            feature_set=feature_set,
            patch_size=patch_size,
            shape=shape,
            grid_shape=(height, width),
            nodata=nodata,
            grid_manifest_signature=grid_manifest_signature,
            pixel_index_signature=pixel_index_signature,
        )
        replace_file(temp_npy, out_npy)
        atomic_write_json(manifest, out_json)
        return manifest
    finally:
        temp_npy.unlink(missing_ok=True)


def _patch_manifest(
    *,
    feature_set: str,
    patch_size: int,
    shape: tuple[int, int, int, int],
    grid_shape: tuple[int, int],
    nodata: float,
    grid_manifest_signature: dict[str, str | int],
    pixel_index_signature: dict[str, str | int],
) -> dict[str, object]:
    """Build the patch-cache manifest, including grid and pixel-index lineage."""
    return {
        "cache_manifest_version": CACHE_MANIFEST_VERSION,
        "feature_set": feature_set,
        "patch_size": patch_size,
        "shape": [int(dim) for dim in shape],
        "dtype": "float32",
        "nodata": nodata,
        "n_pixels": int(shape[0]),
        "grid_height": int(grid_shape[0]),
        "grid_width": int(grid_shape[1]),
        "grid_manifest_signature": grid_manifest_signature,
        "pixel_index_signature": pixel_index_signature,
    }


def _manifest_matches(
    out_json: Path,
    *,
    feature_set: str,
    patch_size: int,
    shape: tuple[int, int, int, int],
    grid_shape: tuple[int, int],
    grid_manifest_signature: dict[str, str | int],
    pixel_index_signature: dict[str, str | int],
    out_npy: Path,
) -> bool:
    """Return whether an existing manifest matches the requested build.

    Args:
        out_json: Path to the existing JSON manifest.
        feature_set: Expected feature-set name.
        patch_size: Expected patch size.
        shape: Expected output shape ``(n_pixels, n_bands, P, P)``.
        grid_shape: Expected source grid ``(height, width)``.
        grid_manifest_signature: Expected validated grid-manifest signature.
        pixel_index_signature: Expected pixel-index signature.
        out_npy: Existing patch-cache array.

    Returns:
        ``True`` only if lineage, configuration and the actual array signature
        all match.
    """
    try:
        manifest = json.loads(out_json.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return False
    return (
        manifest.get("cache_manifest_version") == CACHE_MANIFEST_VERSION
        and manifest.get("feature_set") == feature_set
        and manifest.get("patch_size") == patch_size
        and manifest.get("shape") == list(shape)
        and manifest.get("dtype") == "float32"
        and manifest.get("grid_height") == grid_shape[0]
        and manifest.get("grid_width") == grid_shape[1]
        and manifest.get("grid_manifest_signature") == grid_manifest_signature
        and manifest.get("pixel_index_signature") == pixel_index_signature
        and npy_matches(out_npy, shape, dtype="float32")
    )


def _ensure(
    feature_set: str,
    patch_size: int,
    *,
    paths: ProjectPaths,
    logger: logging.Logger,
    force: bool,
) -> Path:
    """Build the patch cache when absent or stale and return its ``.npy`` path."""
    grid_path = ensure_grid_cache(feature_set, paths=paths, logger=logger)
    if not grid_path.is_file():
        raise ValueError(
            f"Missing grid memmap {grid_path}. Run scripts/build_grid_memmap.py "
            f"--feature-set {feature_set} first."
        )
    grid_manifest_path = grid_path.with_suffix(".json")
    index_path = paths.results / _RESULTS_ROLE / _PIXEL_INDEX_FILENAME
    if not index_path.is_file():
        raise ValueError(
            f"Missing pixel index {index_path}. Run scripts/build_pixel_table.py first."
        )

    grid_shape = np.load(grid_path, mmap_mode="r").shape
    n_bands, height, width = int(grid_shape[0]), int(grid_shape[1]), int(grid_shape[2])
    expected_bands = len(MODEL_INPUT_BANDS[feature_set])
    if n_bands != expected_bands:
        raise ValueError(
            f"Grid memmap {grid_path} has {n_bands} bands but feature set {feature_set!r} "
            f"declares {expected_bands} in FEATURE_SET_BANDS."
        )

    pixel_index = pd.read_parquet(index_path, columns=["row", "col"])
    n_pixels = len(pixel_index)
    shape = (n_pixels, n_bands, patch_size, patch_size)
    out_npy, out_json = _cache_paths(paths, feature_set, patch_size)
    grid_manifest_signature = file_signature(grid_manifest_path)
    pixel_index_signature = file_signature(index_path)

    if (
        not force
        and out_npy.is_file()
        and out_json.is_file()
        and _manifest_matches(
            out_json,
            feature_set=feature_set,
            patch_size=patch_size,
            shape=shape,
            grid_shape=(height, width),
            grid_manifest_signature=grid_manifest_signature,
            pixel_index_signature=pixel_index_signature,
            out_npy=out_npy,
        )
    ):
        logger.info(
            "Patch cache for %r (patch %dx%d) already built; skipping.",
            feature_set,
            patch_size,
            patch_size,
        )
        return out_npy

    # Adopt a pre-lineage cache: a valid patch array whose manifest predates
    # lineage is re-described from the current grid and pixel index rather than
    # re-extracted. A mismatching current-version manifest still rebuilds below.
    if (
        not force
        and out_npy.is_file()
        and is_pre_lineage_manifest(out_json)
        and npy_matches(out_npy, shape, dtype="float32")
    ):
        atomic_write_json(
            _patch_manifest(
                feature_set=feature_set,
                patch_size=patch_size,
                shape=shape,
                grid_shape=(height, width),
                nodata=NODATA,
                grid_manifest_signature=grid_manifest_signature,
                pixel_index_signature=pixel_index_signature,
            ),
            out_json,
        )
        logger.info(
            "Adopted existing patch cache for %r (patch %dx%d) without re-extraction.",
            feature_set,
            patch_size,
            patch_size,
        )
        return out_npy

    n_jobs = min(_MAX_N_JOBS, os.cpu_count() or 1)
    start = time.perf_counter()
    manifest = build_patch_cache(
        grid_path,
        pixel_index,
        out_npy,
        out_json,
        feature_set=feature_set,
        patch_size=patch_size,
        n_jobs=n_jobs,
        grid_manifest_signature=grid_manifest_signature,
        pixel_index_signature=pixel_index_signature,
    )
    elapsed = time.perf_counter() - start
    size_gb = out_npy.stat().st_size / 1e9
    logger.info(
        "Patch cache for %r (patch %dx%d): shape %s, %.2f GB on disk, %.1f s.",
        feature_set,
        patch_size,
        patch_size,
        manifest["shape"],
        size_gb,
        elapsed,
    )
    return out_npy


def ensure_patch_cache(
    feature_set: str,
    patch_size: int,
    *,
    paths: ProjectPaths,
    logger: logging.Logger,
    force: bool = False,
) -> Path:
    """Build the patch cache if it is absent or stale, returning its ``.npy`` path.

    The on-demand entry point used by ``scripts/run_nested_cv.py``: it builds the
    ``(feature_set, patch_size)`` cache when missing (or when ``force``), and
    otherwise reuses the cached one. The source grid memmap and pixel index must
    already exist.

    Args:
        feature_set: Canonical feature-set name.
        patch_size: Odd patch side length (one of
            :data:`utils.terminology.CNN_PATCH_SIZES`).
        paths: Resolved project paths.
        logger: Run logger.
        force: Rebuild even when a matching cache already exists.

    Returns:
        The path of the patch cache ``.npy``.

    Raises:
        ValueError: If the grid memmap or pixel index is missing, or the grid band
            count disagrees with the feature set.
    """
    return _ensure(feature_set, patch_size, paths=paths, logger=logger, force=force)


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
        "--patch-sizes",
        nargs="+",
        type=int,
        choices=list(CNN_PATCH_SIZES),
        help="Patch size(s) to build; defaults to 3 5 7.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Rebuild even when cached outputs already exist.",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    """Build the patch caches for the requested feature sets and patch sizes.

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
    patch_sizes = args.patch_sizes or list(CNN_PATCH_SIZES)
    logger.info(
        "Building patch caches for %s x patch sizes %s (force=%s).",
        feature_sets,
        patch_sizes,
        args.force,
    )

    for feature_set in feature_sets:
        for patch_size in patch_sizes:
            _ensure(feature_set, patch_size, paths=paths, logger=logger, force=args.force)

    logger.info("Patch cache build complete.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
