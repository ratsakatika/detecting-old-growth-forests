r"""Download TESSERA 2020 embeddings for the whole Carpathian massif.

Fetches every 0.1-degree embedding tile intersecting the Carpathian massif
polygon (extracted by notebook 013) straight from the Source Cooperative mirror
through :mod:`scripts.download_tessera_tiles`, dequantises each tile, warps it
from its native UTM grid onto an EPSG:3035 Carpathian grid at the requested
resolution, masks pixels outside the massif to the project NoData sentinel and
writes one aligned 128-band GeoTIFF per tile. A final virtual mosaic
(``tessera_2020_3035_{res}m.vrt``) with canonical band descriptions is built
over the tiles.

The embedding version is the one notebook 001 installed for the study area
(read from the tile provenance sidecars), so the map and the training features
come from the same product; this script takes no version option.

The run is resumable: existing valid tile GeoTIFFs are skipped, tiles found to
lie outside the massif are remembered in a checkpoint file, and cached raw
downloads are deleted after each tile unless ``--keep-cache`` is given. Run it
inside ``screen``, for example::

    screen -S tessera_carpathians
    .venv/bin/python -m scripts.download_carpathians_tessera --grid-res 100

Use ``--dry-run`` first: it reports tile counts, mirror coverage gaps,
estimated download and output sizes, and free disk space, without downloading.
"""

import argparse
import json
import logging
import shutil
import subprocess
import tempfile
import time
import xml.etree.ElementTree as ET
from collections import deque
from collections.abc import Sequence
from concurrent.futures import Future, ThreadPoolExecutor
from pathlib import Path
from typing import Final

import numpy as np
import rasterio
from rasterio.enums import Resampling
from rasterio.errors import RasterioIOError
from rasterio.transform import array_bounds
from rasterio.warp import reproject, transform_bounds
from rasterio.windows import Window
from tqdm.auto import tqdm

from scripts.download_tessera_tiles import (
    fetch_embedding,
    make_session,
    remote_size,
    tile_name,
    tile_urls,
)
from utils import raster_io, terminology
from utils.carpathians import (
    carpathian_grid,
    carpathians_massif_path,
    degree_tiles_intersecting,
    estimated_valid_pixels,
    load_carpathian_massif,
    window_grid,
)
from utils.cli import positive_int
from utils.env_capture import write_environment_snapshot
from utils.logging import make_run_logger
from utils.manifest import write_input_manifest, write_json
from utils.paths import ProjectPaths, ensure_writable_dirs, get_project_paths
from utils.tessera_version import DATASET_PATHS, installed_version

logger = logging.getLogger(__name__)

_DEFAULT_YEAR: Final[int] = 2020
_TILE_STEP_DEG: Final[float] = 0.1
_N_BANDS: Final[int] = 128
_FETCH_ATTEMPTS: Final[int] = 3
_FETCH_RETRY_SLEEP_S: Final[tuple[float, ...]] = (5.0, 25.0)
# Observed deflate compression of the Făgăraș TESSERA stack (float32, masked):
# 20.3 GB raw -> 8.2 GB on disk. Used only for the dry-run disk forecast.
_DEFLATE_RATIO: Final[float] = 2.5
_OUTSIDE_CHECKPOINT: Final[str] = "tiles_outside_massif.json"


def _tessera_band_names() -> list[str]:
    """Canonical TESSERA band names, derived from the terminology feature sets."""
    baseline = terminology.FEATURE_SET_BANDS["baseline"]
    combined = terminology.FEATURE_SET_BANDS["baseline_tessera"]
    return list(combined[len(baseline) :])


def _grid_name(lon: float, lat: float) -> str:
    """Tile stem used on the mirror, e.g. ``grid_24.05_45.35`` (the cell centre)."""
    return tile_name(lon, lat)


def _tile_output_path(tiles_dir: Path, lon: float, lat: float, year: int, res: int) -> Path:
    return tiles_dir / f"{_grid_name(lon, lat)}_{year}_3035_{res}m.tif"


def _tile_is_valid(path: Path) -> bool:
    """Return True when an existing tile GeoTIFF opens cleanly with 128 bands."""
    if not path.exists():
        return False
    try:
        with rasterio.open(path) as src:
            return src.count == _N_BANDS
    except RasterioIOError:
        return False


def _snapped_window(
    ref: raster_io.ReferenceGrid, bounds_3035: tuple[float, float, float, float]
) -> Window | None:
    """Snap projected bounds outward to the grid lattice, clipped to the grid.

    Returns ``None`` when the bounds do not overlap the grid.
    """
    res_x, _ = ref.resolution
    left, top = ref.bounds.left, ref.bounds.top
    minx, miny, maxx, maxy = bounds_3035
    col0 = int(np.floor((minx - left) / res_x))
    col1 = int(np.ceil((maxx - left) / res_x))
    row0 = int(np.floor((top - maxy) / res_x))
    row1 = int(np.ceil((top - miny) / res_x))
    col0, col1 = max(col0, 0), min(col1, ref.width)
    row0, row1 = max(row0, 0), min(row1, ref.height)
    if col1 <= col0 or row1 <= row0:
        return None
    return Window(col0, row0, col1 - col0, row1 - row0)


def _purge_tile_cache(cache_dir: Path, lon: float, lat: float, log: logging.Logger) -> None:
    """Delete a tile's cached raw artefacts (quantised npy, scales, landmask)."""
    removed = 0
    for path in cache_dir.rglob(f"*{_grid_name(lon, lat)}*"):
        if path.is_file():
            path.unlink()
            removed += 1
    if removed:
        log.debug("Purged %d cached file(s) for %s", removed, _grid_name(lon, lat))


def _write_vrt(
    tiles: Sequence[Path],
    vrt_path: Path,
    ref: raster_io.ReferenceGrid,
    band_names: Sequence[str],
    log: logging.Logger,
) -> None:
    """Build a full-grid VRT over the aligned tiles and set band descriptions."""
    with tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False) as handle:
        handle.write("\n".join(str(t) for t in tiles))
        list_path = Path(handle.name)
    try:
        res_x, res_y = ref.resolution
        bounds = ref.bounds
        subprocess.run(
            [
                "gdalbuildvrt",
                "-q",
                "-te",
                str(bounds.left),
                str(bounds.bottom),
                str(bounds.right),
                str(bounds.top),
                "-tr",
                str(res_x),
                str(res_y),
                "-input_file_list",
                str(list_path),
                str(vrt_path),
            ],
            check=True,
        )
    finally:
        list_path.unlink(missing_ok=True)

    # gdalbuildvrt carries no band descriptions over; insert them directly
    # into the VRT XML in canonical order.
    tree = ET.parse(vrt_path)
    bands = tree.getroot().findall("VRTRasterBand")
    if len(bands) != len(band_names):
        raise RuntimeError(
            f"VRT has {len(bands)} bands but {len(band_names)} descriptions were expected."
        )
    for element, name in zip(bands, band_names, strict=True):
        description = ET.Element("Description")
        description.text = name
        element.insert(0, description)
    tree.write(vrt_path)

    with rasterio.open(vrt_path) as src:
        if src.count != _N_BANDS or (src.width, src.height) != (ref.width, ref.height):
            raise RuntimeError(
                f"VRT validation failed: {src.count} bands, {src.width}x{src.height} px; "
                f"expected {_N_BANDS} bands, {ref.width}x{ref.height} px."
            )
    log.info("Wrote VRT %s over %d tile(s)", vrt_path, len(tiles))

    # Consolidate to a single multiband GeoTIFF with overviews: windowed and
    # decimated reads through the ~2000-file VRT decode thousands of tiny
    # blocks and are far too slow for analysis or quicklooks. The VRT is kept
    # as the tile index; downstream readers prefer the .tif.
    tif_path = vrt_path.with_suffix(".tif")
    if not tif_path.exists():
        log.info("Consolidating VRT into a single GeoTIFF (one-off)")
        partial = tif_path.with_name(f".{tif_path.name}.part.tif")
        subprocess.run(
            [
                "gdal_translate",
                "-q",
                "-co",
                "TILED=YES",
                "-co",
                "COMPRESS=DEFLATE",
                "-co",
                "NUM_THREADS=8",
                "-co",
                "BIGTIFF=IF_SAFER",
                str(vrt_path),
                str(partial),
            ],
            check=True,
        )
        partial.rename(tif_path)
    if not tif_path.with_suffix(".tif.ovr").exists():
        log.info("Building external overviews on the consolidated GeoTIFF")
        subprocess.run(
            ["gdaladdo", "-q", "-ro", "-r", "average", str(tif_path), "4", "16"], check=True
        )


def _parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--grid-res",
        type=positive_int,
        required=True,
        help=(
            "Output grid resolution in metres; a whole multiple of the 10 m reference "
            "resolution. 10 reproduces the study pipeline exactly (roughly 300 GB on "
            "disk for the Carpathians); 100 samples the same feature space at about "
            "3 GB. Run --dry-run first."
        ),
    )
    parser.add_argument(
        "--year", type=positive_int, default=_DEFAULT_YEAR, help="Embedding year (default 2020)."
    )
    parser.add_argument(
        "--massif",
        type=Path,
        default=None,
        help="Carpathian massif GeoPackage (default: the processed stage-020 location).",
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=None,
        help="Raw tile cache (default data/cache/geotessera_carpathians).",
    )
    parser.add_argument(
        "--num-threads",
        type=positive_int,
        default=8,
        help="Warp threads per tile (default 8; keep bounded on shared workstations).",
    )
    parser.add_argument(
        "--workers",
        type=positive_int,
        default=4,
        help="Tiles fetched concurrently ahead of the warp (the mirror caps each stream).",
    )
    parser.add_argument(
        "--max-tiles",
        type=positive_int,
        default=None,
        help="Process at most this many tiles (smoke-testing aid).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Report tile counts, coverage gaps and size estimates; download nothing.",
    )
    parser.add_argument(
        "--force", action="store_true", help="Rebuild tiles even when a valid output exists."
    )
    parser.add_argument(
        "--keep-cache",
        action="store_true",
        help="Keep cached raw downloads instead of deleting them after each tile.",
    )
    return parser.parse_args(argv)


def run(args: argparse.Namespace, paths: ProjectPaths) -> int:
    """Execute the download; returns a process exit code."""
    res = int(args.grid_res)
    year = int(args.year)
    out_dir = paths.processed / "rasters" / "carpathians" / f"tessera_{res}m"
    tiles_dir = out_dir / "tiles"
    cache_dir = (
        paths.cache / "geotessera_carpathians" if args.cache_dir is None else Path(args.cache_dir)
    )
    global logger
    logger = make_run_logger(__name__, out_dir)

    version = installed_version(paths.repo_root)
    if version is None:
        raise RuntimeError(
            "No TESSERA tiles are installed under data/raw/rasters/tessera; run the TESSERA "
            "cell of notebook 001 first so the Carpathian layer uses the same version."
        )
    dataset_path = DATASET_PATHS[version]

    massif = load_carpathian_massif(args.massif, logger=logger)
    grid = carpathian_grid(massif, float(res), logger=logger)
    massif_4326 = massif.to_crs("EPSG:4326")
    massif_geometry = massif.geometry

    # Every 0.1-degree cell intersecting the massif; tile names are cell centres.
    requested = sorted(
        (round(lon + 0.05, 2), round(lat + 0.05, 2))
        for lon, lat in degree_tiles_intersecting(massif_4326, step_deg=_TILE_STEP_DEG)
    )
    logger.info(
        "Tiles: %d intersect the massif; fetching TESSERA %s (%s) for %d from the mirror.",
        len(requested),
        version,
        dataset_path,
        year,
    )

    valid_px = estimated_valid_pixels(massif, float(res))
    est_output_gb = valid_px * _N_BANDS * 4 / _DEFLATE_RATIO / 1e9
    session = make_session()
    if args.dry_run:
        # Probe every tile's objects on the mirror for coverage and size.
        sizes: dict[tuple[float, float], int | None] = {}

        def probe(centre: tuple[float, float]) -> tuple[tuple[float, float], int | None]:
            urls = tile_urls(dataset_path, year, _grid_name(*centre))
            found = [remote_size(session, url) for url in urls.values()]
            return centre, None if any(size is None for size in found) else sum(found)  # type: ignore[arg-type]

        with ThreadPoolExecutor(max_workers=8) as pool:
            for centre, size in pool.map(probe, requested):
                sizes[centre] = size
        missing = sorted(_grid_name(*centre) for centre, size in sizes.items() if size is None)
        est_download_gb = sum(size for size in sizes.values() if size is not None) / 1e9
        logger.info(
            "Mirror coverage: %d of %d tiles present, %d missing%s",
            len(requested) - len(missing),
            len(requested),
            len(missing),
            f" (first 20: {missing[:20]})" if missing else "",
        )
    else:
        est_download_gb = 0.12 * len(requested)  # typical quantised tile + scales
    free_gb = shutil.disk_usage(out_dir).free / 1e9
    logger.info(
        "Estimates: download %.0f GB (quantised), output %.1f GB (float32 deflate, "
        "%s valid px at %d m), free disk %.0f GB",
        est_download_gb,
        est_output_gb,
        f"{valid_px:,}",
        res,
        free_gb,
    )

    if args.dry_run:
        logger.info("Dry run complete; nothing downloaded.")
        return 0

    if est_output_gb > free_gb * 0.8:
        raise RuntimeError(
            f"Estimated output ({est_output_gb:.0f} GB) exceeds 80% of free disk "
            f"({free_gb:.0f} GB). Choose a coarser --grid-res or free space first."
        )

    tiles_dir.mkdir(parents=True, exist_ok=True)
    outside_path = out_dir / _OUTSIDE_CHECKPOINT
    outside: set[str] = (
        set(json.loads(outside_path.read_text())) if outside_path.exists() else set()
    )

    band_names = _tessera_band_names()
    ordered = sorted(requested, key=lambda t: (t[1], t[0]))
    if args.max_tiles is not None:
        ordered = ordered[: args.max_tiles]

    # Tiles still to build; the rest are valid on disk or known to miss the massif.
    todo = [
        (lon, lat)
        for lon, lat in ordered
        if _grid_name(lon, lat) not in outside
        and (args.force or not _tile_is_valid(_tile_output_path(tiles_dir, lon, lat, year, res)))
    ]
    written, skipped = 0, len(ordered) - len(todo)
    failed: list[tuple[str, str]] = []
    missing = []

    def fetch(name: str) -> tuple[np.ndarray, object, object] | None:
        """Download and dequantise one tile, retrying transient mirror errors."""
        for attempt in range(_FETCH_ATTEMPTS):
            try:
                return fetch_embedding(
                    name, session=session, cache_dir=cache_dir, dataset_path=dataset_path, year=year
                )
            except Exception:
                if attempt == _FETCH_ATTEMPTS - 1:
                    raise
                time.sleep(_FETCH_RETRY_SLEEP_S[attempt])
        return None

    # The mirror caps each connection, so the next tiles download while this one is
    # warped: a pool of --workers fetches runs ahead of the sequential warp-and-write.
    pending: deque[tuple[tuple[float, float], Future]] = deque()
    upcoming = iter(todo)
    pool = ThreadPoolExecutor(max_workers=int(args.workers))

    def prefetch() -> None:
        centre = next(upcoming, None)
        if centre is not None:
            pending.append((centre, pool.submit(fetch, _grid_name(*centre))))

    for _ in range(int(args.workers)):
        prefetch()
    progress = tqdm(total=len(todo), desc=f"TESSERA {year} -> {res} m", unit="tile")
    while pending:
        (lon, lat), future = pending.popleft()
        prefetch()
        name = _grid_name(lon, lat)
        out_tile = _tile_output_path(tiles_dir, lon, lat, year, res)
        progress.update(1)
        try:
            fetched = future.result()
        except Exception as exc:
            logger.error("Tile %s failed after %d attempts: %s", name, _FETCH_ATTEMPTS, exc)
            failed.append((name, str(exc)))
            continue
        if fetched is None:
            missing.append(name)  # not on the mirror
            continue
        embedding, src_crs, src_transform = fetched

        source = np.moveaxis(embedding, 2, 0).astype("float32", copy=False)
        height, width = source.shape[1], source.shape[2]
        src_bounds = array_bounds(height, width, src_transform)
        bounds_3035 = transform_bounds(src_crs, grid.crs, *src_bounds)
        window = _snapped_window(grid, bounds_3035)
        if window is None:
            outside.add(name)
            outside_path.write_text(json.dumps(sorted(outside)))
            skipped += 1
            continue

        sub_grid = window_grid(grid, window)
        destination = np.full(
            (_N_BANDS, sub_grid.height, sub_grid.width), terminology.NODATA, "float32"
        )
        reproject(
            source=source,
            destination=destination,
            src_transform=src_transform,
            src_crs=src_crs,
            src_nodata=None,
            dst_transform=sub_grid.transform,
            dst_crs=sub_grid.crs,
            dst_nodata=terminology.NODATA,
            resampling=Resampling.bilinear,
            num_threads=int(args.num_threads),
        )
        # The source declares no nodata; NaN cells survive the warp.
        destination[np.isnan(destination)] = terminology.NODATA

        mask = raster_io.rasterize_mask(massif_geometry, sub_grid, all_touched=True)
        if not mask.any():
            outside.add(name)
            outside_path.write_text(json.dumps(sorted(outside)))
            skipped += 1
            continue
        destination[:, mask == 0] = terminology.NODATA

        raster_io.write_geotiff(
            out_tile,
            destination,
            sub_grid,
            dtype="float32",
            nodata=terminology.NODATA,
            band_descriptions=band_names,
            logger=logger,
        )
        written += 1
        if not args.keep_cache:
            _purge_tile_cache(cache_dir, lon, lat, logger)

    progress.close()
    pool.shutdown()

    tile_paths = sorted(tiles_dir.glob(f"grid_*_{year}_3035_{res}m.tif"))
    if tile_paths:
        vrt_path = out_dir / f"tessera_{year}_3035_{res}m.vrt"
        _write_vrt(tile_paths, vrt_path, grid, band_names, logger)

    massif_path = Path(args.massif) if args.massif else carpathians_massif_path(paths.repo_root)
    write_input_manifest([massif_path], out_dir / "input_manifest.json", repo_root=paths.repo_root)
    write_environment_snapshot(out_dir / "env_snapshot.txt")
    write_json(
        {
            "dataset_version": version,
            "dataset_path": dataset_path,
            "year": year,
            "grid_res_m": res,
            "grid": {
                "crs": terminology.CRS,
                "width": grid.width,
                "height": grid.height,
                "bounds": list(grid.bounds),
            },
            "tiles_requested": len(ordered),
            "tiles_written": written,
            "tiles_skipped": skipped,
            "tiles_failed": [name for name, _ in failed],
            "mirror_missing_tiles": missing,
            "band_names": band_names,
        },
        out_dir / "run_metadata.json",
    )

    logger.info(
        "Done: %d written, %d skipped, %d failed, %d not on the mirror",
        written,
        skipped,
        len(failed),
        len(missing),
    )
    if failed:
        logger.error("Failed tiles (rerun to resume): %s", [name for name, _ in failed])
        return 1
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    """Command-line entry point."""
    args = _parse_args(argv)
    paths = get_project_paths()
    ensure_writable_dirs(paths)
    return run(args, paths)


if __name__ == "__main__":
    raise SystemExit(main())
