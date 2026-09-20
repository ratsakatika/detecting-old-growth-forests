"""Build the pixel index and cached feature matrices for the nested CV.

This is the input/output fix for the nested cross-validation: it turns the
labelled parcels into a single pixel index and, per feature set, an on-disk
``float32`` feature matrix that the cross-validation workers memory-map. Every
GeoTIFF is read exactly once here; nothing is read inside the CV loop.

The build is idempotent and cached. The pixel index is written once to
``results/main_nested_cv/pixel_index.parquet`` and each feature matrix to
``data/cache/pixel_features/<feature_set>.npy`` (a real ``.npy`` memmap) with a
companion validity mask ``<feature_set>_valid.parquet`` and versioned JSON
manifest. Reuse requires matching source-stack and pixel-index bytes plus valid
array/table signatures; legacy or incomplete caches rebuild. New cache files are
validated under temporary sibling paths before atomically replacing old files.

How sampling works (pinned here so the cache contract is explicit): the full
feature matrix is stored for every labelled, fully-covered pixel. Training-time
sampling (up to ``terminology.N_PER_PARCEL`` pixels per parcel) selects *row
indices* into this stored matrix; it never subsets the matrix on disk.
Validation and test pixels are *all* pixels of the held-out parcels.

The "strict full pixel" treatment matches the legacy reference-label pipeline:
each parcel polygon is eroded inward by half a pixel diagonal before
rasterising with ``all_touched=False``, so a pixel is kept only when its whole
extent lies within the parcel (the centre of an eroded polygon is, by
construction, at least half a diagonal inside the original).
"""

import argparse
import json
import logging
import math
import os
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Final

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
from joblib import Parallel, delayed
from numpy.typing import NDArray
from rasterio.crs import CRS
from rasterio.features import rasterize
from tqdm import tqdm

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
from utils.env_capture import write_environment_snapshot
from utils.folds import encode_pixel_id
from utils.forestry import classify_forest_type, composition_metrics
from utils.io import (
    PIXEL_INDEX_SCHEMA,
    PIXEL_VALID_SCHEMA,
    validate_schema,
    write_json,
    write_parquet,
)
from utils.logging import make_run_logger
from utils.manifest import write_input_manifest
from utils.paths import ProjectPaths, ensure_writable_dirs, get_project_paths
from utils.raster_io import ReferenceGrid, open_reference_grid
from utils.terminology import (
    FEATURE_SETS,
    MODEL_INPUT_BANDS,
    MODEL_INPUTS,
    NODATA,
    REF_RESOLUTION_M,
)

_LOGGER_NAME: Final[str] = "build_pixel_table"

# Output locations (see AGENTS.md, "Output paths"). The pixel index sits at the
# top of the headline nested-CV results; the reproducibility floor and logs go in
# a dedicated sub-directory so they never clash with per-run-id directories.
_RESULTS_ROLE: Final[str] = "main_nested_cv"
_FLOOR_SUBDIR: Final[str] = "pixel_table"
_PIXEL_INDEX_FILENAME: Final[str] = "pixel_index.parquet"

# Cached feature matrices and validity masks live under data/cache (gitignored).
_CACHE_SUBDIR: Final[Path] = Path("pixel_features")

# Stage 1 stack naming (matches notebook 004 and run_autocorrelation.py):
# data/processed/rasters/stacks_10m/<feature_set>_3035_10m.tif.
_STACKS_SUBDIR: Final[Path] = Path("rasters") / "stacks_10m"
_STACK_SUFFIX: Final[str] = "_3035_10m.tif"

# Partitioned reference labels (notebook 008): the single-layer parcels file with
# the spatial fold and bootstrap assignments. We only need labelled parcels.
_LABELS_FILENAME: Final[str] = "ogf_reference_labels_partitioned.gpkg"
_OGF_COLUMN: Final[str] = "ogf"
_FOLD_COLUMN: Final[str] = "fold_id"
_PARCEL_ID_COLUMN: Final[str] = "parcel_id"
# Two forest-type stratifiers carried per pixel (stratification only; forest type
# never enters the feature matrix). CORINE land-cover type gives coverage; the
# inventory type gives granularity from the labels' species composition.
_CORINE_COLUMN: Final[str] = "corine_forest_type"
# Inventory class string, e.g. "OGF_Broadleaf"/"Non_OGF_Mixed"; its lower-cased
# suffix is the inventory forest type. It already encodes
# utils.forestry.classify_forest_type at FOREST_TYPE_DOMINANCE_PCT.
_CLASS_COLUMN: Final[str] = "class"
# Fallback source if "class" is absent: the canonical species composition string.
_COMPOSITION_COLUMN: Final[str] = "final_composition"
_UNCLASSIFIED: Final[str] = "unclassified"
# Canonical forest-type vocabulary shared by both stratifiers.
_FOREST_TYPES: Final[frozenset[str]] = frozenset({"broadleaf", "coniferous", "mixed"})

# Inward erosion applied to each parcel before rasterising, in metres: half the
# pixel diagonal. A pixel whose centre lies inside the eroded polygon is wholly
# inside the original polygon (strict full-pixel coverage).
STRICT_FULL_PIXEL_BUFFER_M: Final[float] = REF_RESOLUTION_M * math.sqrt(2.0) / 2.0

# Default worker count for the per-band extract.
_DEFAULT_N_JOBS: Final[int] = min(32, os.cpu_count() or 1)


def _project_epsg() -> int | None:
    """Return the project CRS as an EPSG code."""
    return CRS.from_user_input(terminology.CRS).to_epsg()


def _stack_path(paths: ProjectPaths, feature_set: str) -> Path:
    """Resolve the on-disk path of a feature-set stack."""
    return paths.processed / _STACKS_SUBDIR / f"{feature_set}{_STACK_SUFFIX}"


def _cache_paths(paths: ProjectPaths, feature_set: str) -> tuple[Path, Path, Path]:
    """Return the matrix, validity-table and manifest cache paths."""
    cache_dir = paths.cache / _CACHE_SUBDIR
    return (
        cache_dir / f"{feature_set}.npy",
        cache_dir / f"{feature_set}_valid.parquet",
        cache_dir / f"{feature_set}.json",
    )


def load_labelled_parcels(path: Path, logger: logging.Logger) -> gpd.GeoDataFrame:
    """Load the labelled parcels, reprojected to the project CRS.

    Only parcels carrying an old-growth label are returned (unlabelled parcels
    have no fold and play no part in the cross-validation).

    Args:
        path: Path to the partitioned labels GeoPackage.
        logger: Run logger for the load and any reprojection line.

    Returns:
        The labelled parcels on the project CRS, with a fresh range index.

    Raises:
        ValueError: If the layer has no CRS, or a required column is missing.
    """
    frame = gpd.read_file(path)
    if frame.crs is None:
        raise ValueError(f"Labels layer {path} has no CRS; cannot place parcels on the grid.")
    if frame.crs.to_epsg() != _project_epsg():
        logger.info(
            "Reprojecting labels: source CRS %s -> target %s, input=%s.",
            frame.crs.to_string(),
            terminology.CRS,
            path,
        )
        frame = frame.to_crs(terminology.CRS)

    required = {_OGF_COLUMN, _FOLD_COLUMN, _PARCEL_ID_COLUMN}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"Labels layer {path} is missing required column(s): {missing}.")

    labelled = frame[frame[_OGF_COLUMN].notna()].reset_index(drop=True)
    logger.info(
        "Loaded %d parcels; %d carry an old-growth label (%d old-growth).",
        len(frame),
        len(labelled),
        int(labelled[_OGF_COLUMN].sum()),
    )
    return labelled


def rasterise_parcel_index(
    parcels: gpd.GeoDataFrame, ref: ReferenceGrid, *, buffer_m: float
) -> NDArray[np.int32]:
    """Burn a one-based parcel position onto the reference grid.

    Each parcel polygon is eroded inward by ``buffer_m`` and rasterised with
    ``all_touched=False`` (a pixel is burned only when its centre lies in the
    eroded polygon), carrying the parcel's one-based position in ``parcels``.
    Background pixels are ``0``.

    Args:
        parcels: Labelled parcels on ``ref.crs`` with a range index.
        ref: Reference grid to burn onto.
        buffer_m: Inward erosion distance in metres; ``0`` disables erosion.

    Returns:
        A 2-D ``int32`` array of shape ``ref.shape``; value ``p`` marks pixels
        wholly inside the parcel at zero-based position ``p - 1``.
    """
    geometries = parcels.geometry
    if buffer_m > 0:
        geometries = geometries.buffer(-buffer_m)
    shapes = [
        (geometry, position)
        for position, geometry in enumerate(geometries, start=1)
        if geometry is not None and not geometry.is_empty
    ]
    return rasterize(
        shapes,
        out_shape=ref.shape,
        transform=ref.transform,
        fill=0,
        all_touched=False,
        dtype="int32",
    ).astype(np.int32)


def _corine_forest_type(parcels: gpd.GeoDataFrame) -> NDArray[np.object_]:
    """Per-parcel CORINE forest type, with missing values as ``"unclassified"``."""
    if _CORINE_COLUMN in parcels.columns:
        return parcels[_CORINE_COLUMN].fillna(_UNCLASSIFIED).astype(object).to_numpy()
    return np.full(len(parcels), _UNCLASSIFIED, dtype=object)


def _inventory_forest_type(parcels: gpd.GeoDataFrame) -> NDArray[np.object_]:
    """Per-parcel inventory forest type, derived from the labels' composition.

    Uses the lower-cased forest-type suffix of the inventory ``class`` column
    (for example ``"OGF_Broadleaf"`` -> ``"broadleaf"``), which already encodes
    :func:`utils.forestry.classify_forest_type` at the project dominance
    threshold. If ``class`` is absent, classifies the species ``final_composition``
    directly. Anything outside the canonical vocabulary maps to ``"unclassified"``.

    Args:
        parcels: Labelled parcels.

    Returns:
        An object array of forest-type strings, one per parcel.
    """
    if _CLASS_COLUMN in parcels.columns:
        suffix = parcels[_CLASS_COLUMN].astype("string").str.rsplit("_", n=1).str[-1].str.lower()
        valid = suffix.where(suffix.isin(_FOREST_TYPES), other=_UNCLASSIFIED).fillna(_UNCLASSIFIED)
        return valid.astype(object).to_numpy()
    if _COMPOSITION_COLUMN in parcels.columns:
        derived = [
            classify_forest_type(metrics.pct_broadleaf, metrics.pct_coniferous) or _UNCLASSIFIED
            for metrics in map(composition_metrics, parcels[_COMPOSITION_COLUMN])
        ]
        return np.asarray(derived, dtype=object)
    return np.full(len(parcels), _UNCLASSIFIED, dtype=object)


def build_pixel_index(
    parcels: gpd.GeoDataFrame,
    ref: ReferenceGrid,
    *,
    buffer_m: float = STRICT_FULL_PIXEL_BUFFER_M,
    logger: logging.Logger | None = None,
) -> pd.DataFrame:
    """Build the pixel index over every labelled, fully-covered pixel.

    Args:
        parcels: Labelled parcels on ``ref.crs`` (as returned by
            :func:`load_labelled_parcels`).
        ref: Reference grid defining the pixel geometry.
        buffer_m: Inward erosion for the strict full-pixel treatment.
        logger: Optional run logger for the per-parcel coverage summary.

    Returns:
        A frame conforming to :data:`utils.io.PIXEL_INDEX_SCHEMA`, sorted
        ascending by ``pixel_id``.
    """
    index_raster = rasterise_parcel_index(parcels, ref, buffer_m=buffer_m)
    rows, cols = np.nonzero(index_raster)
    position = index_raster[rows, cols] - 1  # zero-based row in ``parcels``

    transform = ref.transform
    col_centre = cols.astype(np.float64) + 0.5
    row_centre = rows.astype(np.float64) + 0.5
    x = transform.a * col_centre + transform.b * row_centre + transform.c
    y = transform.d * col_centre + transform.e * row_centre + transform.f

    parcel_id = parcels[_PARCEL_ID_COLUMN].astype("int64").to_numpy()
    fold_id = parcels[_FOLD_COLUMN].astype("int8").to_numpy()
    ogf = parcels[_OGF_COLUMN].astype("int8").to_numpy()
    forest_type_corine = _corine_forest_type(parcels)
    forest_type_inventory = _inventory_forest_type(parcels)

    frame = pd.DataFrame(
        {
            "pixel_id": np.asarray(encode_pixel_id(rows, cols, ref.width), dtype=np.int64),
            "row": rows.astype(np.int32),
            "col": cols.astype(np.int32),
            "x": x,
            "y": y,
            "parcel_id": parcel_id[position],
            "fold_id": fold_id[position],
            "ogf": ogf[position],
            "forest_type_corine": forest_type_corine[position],
            "forest_type_inventory": forest_type_inventory[position],
        }
    ).sort_values("pixel_id", ignore_index=True)

    if logger is not None:
        contributing = np.unique(position).size
        logger.info(
            "Pixel index: %d labelled pixels over %d of %d parcels "
            "(%d parcels too small for a full pixel after erosion).",
            len(frame),
            contributing,
            len(parcels),
            len(parcels) - contributing,
        )
    return frame


def _read_band_column(
    stack_path: Path, band_index: int, rows: NDArray[np.int32], cols: NDArray[np.int32]
) -> tuple[int, NDArray[np.float32]]:
    """Read one band fully and return its values at the indexed pixels.

    Args:
        stack_path: Path to the feature-set stack.
        band_index: One-based band index.
        rows: Pixel rows to sample.
        cols: Pixel columns to sample.

    Returns:
        A ``(zero_based_band, values)`` pair; ``values`` is ``float32`` of length
        ``len(rows)``.
    """
    with rasterio.open(stack_path) as src:
        band = src.read(band_index)
    return band_index - 1, band[rows, cols].astype(np.float32)


def extract_features_to_memmap(
    stack_path: Path,
    rows: NDArray[np.int32],
    cols: NDArray[np.int32],
    *,
    n_bands: int,
    memmap_path: Path,
    nodata: float = NODATA,
    n_jobs: int = 1,
    show_progress: bool = False,
) -> NDArray[np.bool_]:
    """Extract a pixel-by-band feature matrix to an on-disk ``.npy`` memmap.

    Each band is read once (optionally across processes) and written as a column
    of the memmap. A pixel is marked valid only when every band is finite and not
    the NoData sentinel.

    Args:
        stack_path: Path to the feature-set stack.
        rows: Pixel rows in pixel-index order.
        cols: Pixel columns in pixel-index order.
        n_bands: Number of bands (matrix columns).
        memmap_path: Destination ``.npy`` file (created/overwritten here).
        nodata: NoData sentinel; a pixel equal to it in any band is invalid.
        n_jobs: Worker processes for the per-band read.
        show_progress: Whether to show a tqdm bar over the bands.

    Returns:
        A boolean validity mask of length ``len(rows)``, in pixel-index order.
    """
    n_pixels = int(rows.shape[0])
    memmap_path.parent.mkdir(parents=True, exist_ok=True)
    matrix = np.lib.format.open_memmap(
        memmap_path, mode="w+", dtype=np.float32, shape=(n_pixels, n_bands)
    )
    valid = np.ones(n_pixels, dtype=bool)

    generator = Parallel(n_jobs=n_jobs, return_as="generator")(
        delayed(_read_band_column)(stack_path, band + 1, rows, cols) for band in range(n_bands)
    )
    if show_progress:
        generator = tqdm(generator, total=n_bands, desc=stack_path.stem)
    for band, column in generator:
        matrix[:, band] = column
        valid &= np.isfinite(column) & (column != nodata)

    matrix.flush()
    del matrix
    return valid


def _check_stack(stack_path: Path, feature_set: str, logger: logging.Logger) -> None:
    """Confirm a stack is on the project CRS with the expected band count.

    Args:
        stack_path: Path to the stack.
        feature_set: Canonical feature-set name.
        logger: Run logger for the one INFO line.

    Raises:
        ValueError: If the CRS is wrong or the band count does not match
            :data:`utils.terminology.FEATURE_SET_BANDS`.
    """
    expected = len(MODEL_INPUT_BANDS[feature_set])
    with rasterio.open(stack_path) as src:
        crs = src.crs
        count = src.count
    if crs is None or crs.to_epsg() != _project_epsg():
        found = crs.to_string() if crs is not None else "undefined"
        raise ValueError(f"Stack {stack_path} has CRS {found}, expected {terminology.CRS}.")
    if count != expected:
        raise ValueError(
            f"Stack {stack_path} has {count} bands but feature set {feature_set!r} "
            f"declares {expected} in FEATURE_SET_BANDS."
        )
    logger.info(
        "Stack %s: CRS %s, %d bands (as expected).", stack_path.name, terminology.CRS, count
    )


def _pixel_cache_structurally_valid(
    memmap_path: Path,
    valid_path: Path,
    *,
    pixel_ids: NDArray[np.int64],
    shape: tuple[int, int],
) -> bool:
    """Whether the matrix and validity table are well-formed and pixel-aligned."""
    try:
        valid = pd.read_parquet(valid_path)
        validate_schema(valid, PIXEL_VALID_SCHEMA)
    except (OSError, ValueError):
        return False
    return (
        npy_matches(memmap_path, shape, dtype="float32")
        and np.array_equal(valid["pixel_id"].to_numpy(dtype=np.int64), pixel_ids)
        and str(valid["valid"].dtype) == "bool"
    )


def _pixel_manifest(
    *,
    feature_set: str,
    shape: tuple[int, int],
    valid_len: int,
    source_signature: dict[str, str | int],
    pixel_index_signature: dict[str, str | int],
) -> dict[str, object]:
    """Build the pixel-cache manifest, including source and pixel-index lineage."""
    return {
        "cache_manifest_version": CACHE_MANIFEST_VERSION,
        "feature_set": feature_set,
        "shape": list(shape),
        "dtype": "float32",
        "valid_shape": [int(valid_len)],
        "valid_dtype": "bool",
        "source_signature": source_signature,
        "pixel_index_signature": pixel_index_signature,
    }


def _pixel_cache_matches(
    memmap_path: Path,
    valid_path: Path,
    manifest_path: Path,
    *,
    feature_set: str,
    source_signature: dict[str, str | int],
    pixel_index_signature: dict[str, str | int],
    pixel_ids: NDArray[np.int64],
    n_bands: int,
) -> bool:
    """Return whether a pixel cache and its lineage exactly match the inputs."""
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return False
    shape = (int(pixel_ids.shape[0]), int(n_bands))
    return (
        manifest.get("cache_manifest_version") == CACHE_MANIFEST_VERSION
        and manifest.get("feature_set") == feature_set
        and manifest.get("shape") == list(shape)
        and manifest.get("dtype") == "float32"
        and manifest.get("source_signature") == source_signature
        and manifest.get("pixel_index_signature") == pixel_index_signature
        and _pixel_cache_structurally_valid(
            memmap_path, valid_path, pixel_ids=pixel_ids, shape=shape
        )
    )


def ensure_pixel_cache_current(
    paths: ProjectPaths,
    feature_set: str,
    *,
    pixel_index_path: Path,
    pixel_ids: NDArray[np.int64],
    logger: logging.Logger,
) -> None:
    """Ensure a feature set's pixel cache matches the current inputs, adopting legacy.

    A cache whose lineage manifest already matches is accepted untouched. A
    pre-lineage cache (built before fingerprinting) whose matrix, validity table
    and pixel-id ordering are structurally valid is adopted in place: a current
    manifest is written so the existing matrix is never re-extracted. Anything
    else raises, directing the caller to (re)build the cache.

    Args:
        paths: Resolved project paths.
        feature_set: Canonical feature-set name.
        pixel_index_path: Path to the shared pixel index defining the row order.
        pixel_ids: Expected ``pixel_id`` ordering for the cache.
        logger: Run logger (records an adoption).

    Raises:
        ValueError: If the cache is absent, structurally invalid, or a
            current-version manifest disagrees with the present source bytes.
    """
    memmap_path, valid_path, manifest_path = _cache_paths(paths, feature_set)
    stack_path = _stack_path(paths, feature_set)
    n_bands = len(MODEL_INPUT_BANDS[feature_set])
    instruction = f"Run scripts/build_pixel_table.py --feature-set {feature_set} first."
    if not all(path.is_file() for path in (memmap_path, valid_path, stack_path, pixel_index_path)):
        raise ValueError(f"Pixel cache for {feature_set!r} is absent. {instruction}")

    source_signature = file_signature(stack_path)
    pixel_index_signature = file_signature(pixel_index_path)
    if manifest_path.is_file() and _pixel_cache_matches(
        memmap_path,
        valid_path,
        manifest_path,
        feature_set=feature_set,
        source_signature=source_signature,
        pixel_index_signature=pixel_index_signature,
        pixel_ids=pixel_ids,
        n_bands=n_bands,
    ):
        return

    shape = (int(pixel_ids.shape[0]), int(n_bands))
    if is_pre_lineage_manifest(manifest_path) and _pixel_cache_structurally_valid(
        memmap_path, valid_path, pixel_ids=pixel_ids, shape=shape
    ):
        valid_len = len(pd.read_parquet(valid_path))
        atomic_write_json(
            _pixel_manifest(
                feature_set=feature_set,
                shape=shape,
                valid_len=valid_len,
                source_signature=source_signature,
                pixel_index_signature=pixel_index_signature,
            ),
            manifest_path,
        )
        logger.info("Adopted existing pixel cache for %r without re-extraction.", feature_set)
        return

    raise ValueError(f"Pixel cache for {feature_set!r} is stale or invalid. {instruction}")


def _library_versions() -> dict[str, str]:
    """Return versions of the key libraries for the run metadata."""
    versions: dict[str, str] = {}
    for library in ("numpy", "pandas", "rasterio", "geopandas", "joblib"):
        try:
            versions[library] = version(library)
        except PackageNotFoundError:
            versions[library] = "unknown"
    return versions


@dataclass(frozen=True, slots=True)
class _FeatureSetResult:
    """Summary of one feature set's cached matrix, for the run metadata."""

    feature_set: str
    n_pixels: int
    n_bands: int
    gigabytes: float
    valid_count: int
    skipped: bool


def _parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    """Parse the command-line arguments.

    Args:
        argv: Argument list, or ``None`` to read from ``sys.argv``.

    Returns:
        The parsed arguments.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--feature-set",
        action="append",
        choices=list(MODEL_INPUTS),
        help="Feature set to build; repeatable. Defaults to all feature sets.",
    )
    parser.add_argument(
        "--n-jobs",
        type=int,
        default=_DEFAULT_N_JOBS,
        help=f"Worker processes for the per-band extract (default {_DEFAULT_N_JOBS}).",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Rebuild even when cached outputs already exist.",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    """Build the pixel index and cached feature matrices.

    Args:
        argv: Argument list, or ``None`` to read from ``sys.argv``.

    Returns:
        Process exit code (0 on success).
    """
    args = _parse_args(argv)
    paths = get_project_paths()
    ensure_writable_dirs(paths)

    results_dir = paths.results / _RESULTS_ROLE
    floor_dir = results_dir / _FLOOR_SUBDIR
    floor_dir.mkdir(parents=True, exist_ok=True)
    logger = make_run_logger(_LOGGER_NAME, floor_dir)

    feature_sets = args.feature_set or list(FEATURE_SETS)
    logger.info(
        "Building pixel table for %s (n_jobs=%d, force=%s, strict-full-pixel buffer=%.3f m).",
        feature_sets,
        args.n_jobs,
        args.force,
        STRICT_FULL_PIXEL_BUFFER_M,
    )

    ref = open_reference_grid()
    logger.info("Reference grid: %dx%d, %s.", ref.width, ref.height, terminology.CRS)

    labels_path = paths.labels / _LABELS_FILENAME
    parcels = load_labelled_parcels(labels_path, logger)

    # The pixel index is rebuilt every run (cheap, deterministic) and rewritten,
    # so column changes always take effect. Its pixel_id ordering is identical to
    # the cached feature memmaps, which are reused untouched below.
    pixel_index_path = results_dir / _PIXEL_INDEX_FILENAME
    pixel_index = build_pixel_index(parcels, ref, logger=logger)
    temp_pixel_index = temporary_sibling(pixel_index_path)
    try:
        write_parquet(pixel_index, temp_pixel_index, PIXEL_INDEX_SCHEMA)
        replace_file(temp_pixel_index, pixel_index_path)
    finally:
        temp_pixel_index.unlink(missing_ok=True)
    logger.info("Wrote pixel index (%d pixels) to %s.", len(pixel_index), pixel_index_path)
    pixel_index_signature = file_signature(pixel_index_path)

    rows = pixel_index["row"].to_numpy(dtype=np.int32)
    cols = pixel_index["col"].to_numpy(dtype=np.int32)
    pixel_ids = pixel_index["pixel_id"].to_numpy(dtype=np.int64)

    inputs: list[Path] = [labels_path, paths.reference_grid]
    fs_results: list[_FeatureSetResult] = []
    for feature_set in feature_sets:
        stack_path = _stack_path(paths, feature_set)
        memmap_path, valid_path, manifest_path = _cache_paths(paths, feature_set)
        n_bands = len(MODEL_INPUT_BANDS[feature_set])
        gigabytes = len(pixel_index) * n_bands * 4 / 1e9
        inputs.append(stack_path)
        _check_stack(stack_path, feature_set, logger)
        source_signature = file_signature(stack_path)

        if (
            not args.force
            and memmap_path.is_file()
            and valid_path.is_file()
            and manifest_path.is_file()
            and _pixel_cache_matches(
                memmap_path,
                valid_path,
                manifest_path,
                feature_set=feature_set,
                source_signature=source_signature,
                pixel_index_signature=pixel_index_signature,
                pixel_ids=pixel_ids,
                n_bands=n_bands,
            )
        ):
            logger.info("Feature set %r already cached; skipping.", feature_set)
            fs_results.append(
                _FeatureSetResult(feature_set, len(pixel_index), n_bands, gigabytes, -1, True)
            )
            continue

        logger.info(
            "Extracting %r: %d pixels x %d bands (~%.2f GB) with n_jobs=%d.",
            feature_set,
            len(pixel_index),
            n_bands,
            gigabytes,
            args.n_jobs,
        )
        temp_memmap = temporary_sibling(memmap_path)
        temp_valid = temporary_sibling(valid_path)
        try:
            valid = extract_features_to_memmap(
                stack_path,
                rows,
                cols,
                n_bands=n_bands,
                memmap_path=temp_memmap,
                n_jobs=args.n_jobs,
                show_progress=True,
            )
            valid_frame = pd.DataFrame({"pixel_id": pixel_ids, "valid": valid})
            write_parquet(valid_frame, temp_valid, PIXEL_VALID_SCHEMA)
            shape = (len(pixel_index), n_bands)
            if not npy_matches(temp_memmap, shape, dtype="float32"):
                raise ValueError(f"Wrote {temp_memmap} with an invalid array signature.")
            written_valid = pd.read_parquet(temp_valid)
            validate_schema(written_valid, PIXEL_VALID_SCHEMA)
            if not np.array_equal(written_valid["pixel_id"].to_numpy(dtype=np.int64), pixel_ids):
                raise ValueError(
                    f"Wrote {temp_valid} with pixel ids that do not match the pixel index."
                )
            manifest = _pixel_manifest(
                feature_set=feature_set,
                shape=shape,
                valid_len=len(valid_frame),
                source_signature=source_signature,
                pixel_index_signature=pixel_index_signature,
            )
            replace_file(temp_memmap, memmap_path)
            replace_file(temp_valid, valid_path)
            atomic_write_json(manifest, manifest_path)
        finally:
            temp_memmap.unlink(missing_ok=True)
            temp_valid.unlink(missing_ok=True)
        valid_count = int(valid.sum())
        logger.info(
            "Feature set %r: matrix %s (%.2f GB), %d/%d valid pixels (%.1f%%).",
            feature_set,
            memmap_path.name,
            gigabytes,
            valid_count,
            len(valid),
            100.0 * valid_count / len(valid),
        )
        fs_results.append(
            _FeatureSetResult(feature_set, len(pixel_index), n_bands, gigabytes, valid_count, False)
        )

    # Reproducibility floor.
    write_input_manifest(inputs, floor_dir / "input_manifest.json", repo_root=paths.repo_root)
    write_environment_snapshot(floor_dir / "env_snapshot.txt")
    run_metadata = {
        "run_id": datetime.now(UTC).strftime("%Y%m%d_%H%M%S"),
        "created_at": datetime.now(UTC).isoformat(),
        "script": "scripts/build_pixel_table.py",
        "seed": terminology.SEED,
        "n_jobs": args.n_jobs,
        "force": args.force,
        "strict_full_pixel_buffer_m": STRICT_FULL_PIXEL_BUFFER_M,
        "reference_grid": {"width": ref.width, "height": ref.height, "crs": terminology.CRS},
        "n_pixels": len(pixel_index),
        "feature_sets": [
            {
                "feature_set": result.feature_set,
                "n_bands": result.n_bands,
                "gigabytes": result.gigabytes,
                "valid_count": result.valid_count,
                "skipped": result.skipped,
            }
            for result in fs_results
        ],
        "library_versions": _library_versions(),
        "outputs": {
            "pixel_index": str(pixel_index_path.relative_to(paths.repo_root)),
            "feature_matrices": {
                result.feature_set: str(
                    _cache_paths(paths, result.feature_set)[0].relative_to(paths.repo_root)
                )
                for result in fs_results
            },
        },
    }
    write_json(run_metadata, floor_dir / "run_metadata.json")
    logger.info("Pixel-table build complete.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
