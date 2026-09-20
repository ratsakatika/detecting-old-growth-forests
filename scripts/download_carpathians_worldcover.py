"""Download ESA WorldCover 2020 tree cover for the Carpathian AOA forest mask.

Downloads the WorldCover 2020 v100 classification-map tiles (3-degree, EPSG:4326,
10 m) intersecting the Carpathian massif from the public AWS Open Data bucket and
aggregates the tree-cover class (10) to a tree-cover *fraction* on the EPSG:3035
Carpathian grid at the requested resolution (average resampling of the binary
class). Pixels outside the massif are NoData; "forest" at analysis time is
fraction >= 0.5. This is the gap-free forest mask for the AOA analysis — CORINE
CLC2018 does not cover Ukraine — and reproduces the construction of the archive
prototype (archive/scripts/034_carpathian_tessera_zarr_aoa.py, WorldCover
tree-fraction mask).

Outputs
    data/processed/rasters/carpathians/worldcover_{res}m/worldcover_tree_fraction_3035_{res}m.tif
    raw tiles in data/raw/rasters/worldcover_map/ (with provenance sidecars)

Example (about 15 tiles, roughly 1.5 GB of raw downloads)::

    .venv/bin/python -m scripts.download_carpathians_worldcover --grid-res 100
"""

import argparse
import logging
import shutil
from collections.abc import Sequence
from pathlib import Path
from typing import Final

import numpy as np
import rasterio
import requests
from rasterio.enums import Resampling
from rasterio.errors import RasterioIOError
from rasterio.warp import reproject, transform_bounds

from utils import raster_io, terminology
from utils.carpathians import (
    carpathian_grid,
    carpathians_massif_path,
    degree_tiles_intersecting,
    load_carpathian_massif,
    snapped_window,
    window_grid,
)
from utils.cli import positive_int
from utils.env_capture import write_environment_snapshot
from utils.logging import make_run_logger
from utils.manifest import write_input_manifest, write_json
from utils.paths import ProjectPaths, ensure_writable_dirs, get_project_paths
from utils.provenance import provenance_path_for, write_provenance

logger = logging.getLogger(__name__)

_YEAR: Final[int] = 2020
_VERSION: Final[str] = "v100"
_TREE_CLASS: Final[int] = 10
_TILE_STEP_DEG: Final[float] = 3.0
_URL: Final[str] = (
    "https://esa-worldcover.s3.eu-central-1.amazonaws.com/"
    f"{_VERSION}/{_YEAR}/map/ESA_WorldCover_10m_{_YEAR}_{_VERSION}_{{tile}}_Map.tif"
)
_DOWNLOAD_CHUNK: Final[int] = 1 << 20


def _tile_id(lon: float, lat: float) -> str:
    """WorldCover 3-degree tile identifier from its south-west corner."""
    if lat < 0 or lon < 0:
        raise ValueError(f"Southern/western hemisphere tiles are not supported: {(lon, lat)}.")
    return f"N{int(lat):02d}E{int(lon):03d}"


def _stream_download(url: str, out_file: Path, log: logging.Logger) -> None:
    """Stream a URL to disk via a ``.part`` temp file with atomic rename."""
    tmp = out_file.with_suffix(out_file.suffix + ".part")
    log.info("Downloading %s", url)
    with requests.get(url, stream=True, timeout=600) as response:
        response.raise_for_status()
        with open(tmp, "wb") as handle:
            for chunk in response.iter_content(chunk_size=_DOWNLOAD_CHUNK):
                handle.write(chunk)
    tmp.replace(out_file)
    log.info("Saved %s (%.1f MB)", out_file.name, out_file.stat().st_size / 1e6)


def _tile_is_valid(path: Path) -> bool:
    if not path.exists():
        return False
    try:
        with rasterio.open(path) as src:
            return src.count == 1
    except RasterioIOError:
        return False


def _parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--grid-res",
        type=positive_int,
        required=True,
        help="Output grid resolution in metres; must match the other Carpathian scripts.",
    )
    parser.add_argument(
        "--massif",
        type=Path,
        default=None,
        help="Carpathian massif GeoPackage (default: the processed stage-020 location).",
    )
    parser.add_argument(
        "--num-threads",
        type=positive_int,
        default=8,
        help="Warp threads per tile (default 8).",
    )
    parser.add_argument(
        "--dry-run", action="store_true", help="Report the tile list and estimates only."
    )
    parser.add_argument(
        "--force", action="store_true", help="Rebuild the fraction raster even if present."
    )
    parser.add_argument(
        "--keep-raw", action="store_true", help="Keep raw map tiles (default: keep, always)."
    )
    return parser.parse_args(argv)


def run(args: argparse.Namespace, paths: ProjectPaths) -> int:
    """Execute the download and aggregation; returns a process exit code."""
    res = int(args.grid_res)
    out_dir = paths.processed / "rasters" / "carpathians" / f"worldcover_{res}m"
    out_path = out_dir / f"worldcover_tree_fraction_3035_{res}m.tif"
    raw_dir = paths.raw / "rasters" / "worldcover_map"
    global logger
    logger = make_run_logger(__name__, out_dir)

    massif = load_carpathian_massif(args.massif, logger=logger)
    grid = carpathian_grid(massif, float(res), logger=logger)
    corners = degree_tiles_intersecting(
        massif.to_crs("EPSG:4326"), step_deg=_TILE_STEP_DEG, logger=logger
    )
    tiles = [_tile_id(lon, lat) for lon, lat in corners]
    have = [
        t
        for t in tiles
        if _tile_is_valid(raw_dir / f"ESA_WorldCover_10m_{_YEAR}_{_VERSION}_{t}_Map.tif")
    ]
    logger.info(
        "WorldCover tiles wanted: %d (%s); already present: %d", len(tiles), tiles, len(have)
    )

    if args.dry_run:
        free_gb = shutil.disk_usage(paths.processed).free / 1e9
        logger.info(
            "Dry run: ~%.1f GB of raw tiles to fetch, output %dx%d px fraction raster, "
            "free disk %.0f GB",
            0.1 * (len(tiles) - len(have)),
            grid.width,
            grid.height,
            free_gb,
        )
        return 0

    raw_dir.mkdir(parents=True, exist_ok=True)
    failed: list[str] = []
    for tile in tiles:
        out_file = raw_dir / f"ESA_WorldCover_10m_{_YEAR}_{_VERSION}_{tile}_Map.tif"
        if _tile_is_valid(out_file):
            continue
        url = _URL.format(tile=tile)
        try:
            _stream_download(url, out_file, logger)
            if not provenance_path_for(out_file).exists():
                write_provenance(
                    out_file,
                    source=url,
                    extra={"dataset": f"ESA WorldCover 10 m {_YEAR} {_VERSION} classification map"},
                )
        except Exception as exc:
            logger.error("Tile %s failed: %s", tile, exc)
            failed.append(tile)

    if out_path.exists() and not args.force:
        logger.info("Fraction raster already present: %s", out_path.name)
    else:
        nodata = terminology.NODATA
        fraction = np.full(grid.shape, nodata, dtype="float32")
        for tile in tiles:
            tile_path = raw_dir / f"ESA_WorldCover_10m_{_YEAR}_{_VERSION}_{tile}_Map.tif"
            if not tile_path.exists():
                continue
            with rasterio.open(tile_path) as src:
                bounds_3035 = transform_bounds(src.crs, grid.crs, *src.bounds)
                window = snapped_window(grid, bounds_3035)
                if window is None:
                    continue
                sub = window_grid(grid, window)
                # Binary tree-class layer at native 10 m, averaged into the target
                # cells -> per-cell tree-cover fraction.
                binary = (src.read(1) == _TREE_CLASS).astype("float32")
                destination = np.full((sub.height, sub.width), nodata, dtype="float32")
                reproject(
                    source=binary,
                    destination=destination,
                    src_transform=src.transform,
                    src_crs=src.crs,
                    src_nodata=None,
                    dst_transform=sub.transform,
                    dst_crs=sub.crs,
                    dst_nodata=nodata,
                    resampling=Resampling.average,
                    num_threads=int(args.num_threads),
                )
                del binary
            rows = slice(int(window.row_off), int(window.row_off + window.height))
            cols = slice(int(window.col_off), int(window.col_off + window.width))
            block = fraction[rows, cols]
            fraction[rows, cols] = np.where(destination != nodata, destination, block)
            logger.info("Aggregated %s into the fraction grid", tile)

        mask = raster_io.rasterize_mask(massif.geometry, grid, all_touched=True)
        fraction[mask == 0] = nodata
        raster_io.write_geotiff(
            out_path,
            fraction,
            grid,
            dtype="float32",
            nodata=nodata,
            band_descriptions=["worldcover_tree_fraction"],
            logger=logger,
        )

    write_input_manifest(
        [Path(args.massif) if args.massif else carpathians_massif_path(paths.repo_root)],
        out_dir / "input_manifest.json",
        repo_root=paths.repo_root,
    )
    write_environment_snapshot(out_dir / "env_snapshot.txt")
    write_json(
        {
            "year": _YEAR,
            "version": _VERSION,
            "tree_class": _TREE_CLASS,
            "grid_res_m": res,
            "tiles": tiles,
            "tiles_failed": failed,
            "output": out_path.name,
        },
        out_dir / "run_metadata.json",
    )
    logger.info("Done: %d tiles, %d failed", len(tiles), len(failed))
    return 1 if failed else 0


def main(argv: Sequence[str] | None = None) -> int:
    """Command-line entry point."""
    args = _parse_args(argv)
    paths = get_project_paths()
    ensure_writable_dirs(paths)
    return run(args, paths)


if __name__ == "__main__":
    raise SystemExit(main())
