r"""Download and derive the baseline feature set for the Carpathian massif.

Builds the six baseline bands (``utils.terminology.FEATURE_SET_BANDS["baseline"]``:
elevation, slope, heat-load index and the three distance-to-roads bands) over
the Carpathian massif polygon (extracted by notebook 013) on an EPSG:3035 grid
at the requested resolution, reproducing the notebook 001/002 recipes at
Carpathian scale:

- FABDEM V1-2 1-degree tiles are downloaded from the University of Bristol
  archives, mosaicked, warped to EPSG:3035 at FABDEM's native ground sampling,
  and slope / heat-load index are derived there before resampling to the target
  grid (port of notebook 002, FABDEM cell).
- OpenStreetMap roads for the seven Carpathian countries come from Geofabrik
  GeoPackage extracts, are classified into paved / unpaved / footpath and
  rasterised to distance bands from a network buffered 10 km beyond the massif
  (port of notebook 002, roads cell).

Outputs land in ``data/processed/rasters/carpathians/baseline_{res}m/``; raw
downloads join the existing ``data/raw`` trees with provenance sidecars.

Steps run in order and are individually resumable and selectable::

    --steps fabdem,terrain,osm,roads   (default: all)

Run inside ``screen``, for example::

    screen -S baseline_carpathians
    .venv/bin/python -m scripts.download_carpathians_baseline --grid-res 100

Use ``--dry-run`` first for tile lists and disk/memory estimates.
"""

import argparse
import logging
import math
import shutil
import subprocess
import tempfile
import zipfile
from collections.abc import Sequence
from pathlib import Path
from typing import Final

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
import requests
from rasterio.enums import Resampling
from rasterio.transform import from_origin
from rasterio.warp import calculate_default_transform, reproject

from utils import predictors, raster_io, terminology
from utils.carpathians import (
    carpathian_grid,
    carpathians_massif_path,
    degree_tiles_intersecting,
    load_carpathian_massif,
)
from utils.cli import positive_int
from utils.env_capture import write_environment_snapshot
from utils.logging import make_run_logger
from utils.manifest import write_input_manifest, write_json
from utils.paths import ProjectPaths, ensure_writable_dirs, get_project_paths
from utils.provenance import provenance_path_for, write_provenance
from utils.vector_io import repair_geometries

logger = logging.getLogger(__name__)

_STEPS: Final[tuple[str, ...]] = ("fabdem", "terrain", "osm", "roads")

_FABDEM_URL: Final[str] = (
    "https://data.bris.ac.uk/datasets/s5hqmjcdj8yo2ibzi9b4ew3sn/{block}_FABDEM_V1-2.zip"
)
_GEOFABRIK_URL: Final[str] = "https://download.geofabrik.de/europe/{country}-latest-free.gpkg.zip"
# Countries whose territory the Carpathian massif polygon touches.
_OSM_COUNTRIES: Final[tuple[str, ...]] = (
    "romania",
    "ukraine",
    "slovakia",
    "poland",
    "hungary",
    "czech-republic",
    "serbia",
)
_OSM_ROADS_LAYER: Final[str] = "gis_osm_roads_free"

# Road classification, copied verbatim from notebooks/002_raster_pre_processing.ipynb
# (roads cell), where it is defined inline; kept identical so Carpathian distance
# bands share the training features' semantics.
_OSM_ROAD_GROUP_MAP: Final[dict[str, str]] = {
    "motorway": "paved_road",
    "trunk": "paved_road",
    "primary": "paved_road",
    "secondary": "paved_road",
    "tertiary": "paved_road",
    "unclassified": "paved_road",
    "residential": "paved_road",
    "living_street": "paved_road",
    "motorway_link": "paved_road",
    "trunk_link": "paved_road",
    "primary_link": "paved_road",
    "secondary_link": "paved_road",
    "tertiary_link": "paved_road",
    "road": "paved_road",
    "service": "unpaved_road",
    "track": "unpaved_road",
    "track_grade1": "unpaved_road",
    "track_grade2": "unpaved_road",
    "track_grade3": "unpaved_road",
    "track_grade4": "unpaved_road",
    "track_grade5": "unpaved_road",
    "path": "footpath",
    "footway": "footpath",
    "steps": "footpath",
    "cycleway": "footpath",
    "bridleway": "footpath",
    "pedestrian": "footpath",
    "trail": "footpath",
}
_OSM_ROAD_GROUP_ORDER: Final[tuple[str, ...]] = ("paved_road", "unpaved_road", "footpath")
_LINE_TYPES: Final[tuple[str, ...]] = ("LineString", "MultiLineString")

# Terrain output file stems, matching the fabdem_10m/ file naming of notebook 002;
# the band descriptions inside the files carry the canonical band names.
_TERRAIN_FILE_STEMS: Final[tuple[str, ...]] = ("elevation", "slope_deg", "heat_load_index")

_DOWNLOAD_CHUNK: Final[int] = 1 << 20
# Bytes per padded-grid pixel held at the peak of a scipy Euclidean distance
# transform: uint8 input + float64 distances + int32 feature transform (2 axes).
_EDT_BYTES_PER_PX: Final[int] = 17


def _fabdem_tile_name(lon: float, lat: float) -> str:
    """FABDEM 1-degree tile filename from its south-west corner."""
    if lat < 0 or lon < 0:
        raise ValueError(f"Southern/western hemisphere tiles are not supported: {(lon, lat)}.")
    return f"N{int(lat):02d}E{int(lon):03d}_FABDEM_V1-2.tif"


def _fabdem_block_name(lon: float, lat: float) -> str:
    """FABDEM 10-degree archive stem containing a 1-degree tile."""
    lat0 = int(math.floor(lat / 10.0) * 10)
    lon0 = int(math.floor(lon / 10.0) * 10)
    return f"N{lat0}E{lon0:03d}-N{lat0 + 10}E{lon0 + 10:03d}"


def _stream_download(url: str, out_file: Path, log: logging.Logger, timeout: int = 600) -> None:
    """Stream a URL to disk via a ``.part`` temp file with atomic rename."""
    tmp = out_file.with_suffix(out_file.suffix + ".part")
    log.info("Downloading %s", url)
    with requests.get(url, stream=True, timeout=timeout) as response:
        response.raise_for_status()
        with open(tmp, "wb") as handle:
            for chunk in response.iter_content(chunk_size=_DOWNLOAD_CHUNK):
                handle.write(chunk)
    tmp.replace(out_file)
    log.info("Saved %s (%.1f MB)", out_file.name, out_file.stat().st_size / 1e6)


def _available_ram_bytes() -> int:
    """Available system memory from ``/proc/meminfo`` (Linux)."""
    for line in Path("/proc/meminfo").read_text().splitlines():
        if line.startswith("MemAvailable:"):
            return int(line.split()[1]) * 1024
    raise RuntimeError("MemAvailable not found in /proc/meminfo.")


def _matches_grid(path: Path, grid: raster_io.ReferenceGrid) -> bool:
    """Return True when an existing raster matches the expected grid exactly.

    Guards the resume logic: an output produced for a different extent or
    resolution (e.g. a smoke run with a substitute massif) is rebuilt rather
    than skipped.
    """
    if not path.exists():
        return False
    try:
        with rasterio.open(path) as src:
            return (
                src.crs == grid.crs
                and src.transform == grid.transform
                and (src.width, src.height) == (grid.width, grid.height)
            )
    except rasterio.errors.RasterioIOError:
        return False


def _terrain_band_names() -> list[str]:
    return list(terminology.FEATURE_SET_BANDS["baseline"][:3])


def _roads_band_names() -> list[str]:
    names = list(terminology.FEATURE_SET_BANDS["baseline"][3:])
    derived = [f"dist_{group}_m" for group in _OSM_ROAD_GROUP_ORDER]
    if names != derived:
        raise RuntimeError(
            f"Road group order {derived} no longer matches the canonical baseline "
            f"band order {names}; reconcile before writing rasters."
        )
    return names


def _wanted_fabdem_tiles(
    massif: gpd.GeoDataFrame, log: logging.Logger
) -> list[tuple[float, float]]:
    """South-west corners of 1-degree FABDEM tiles needed for the massif.

    A 1 km metric buffer keeps neighbouring tiles whose edge pixels feed the
    3x3 terrain kernels at the massif boundary.
    """
    buffered = massif.geometry.buffer(1_000.0)
    return degree_tiles_intersecting(buffered.to_crs("EPSG:4326"), step_deg=1.0, logger=log)


def step_fabdem(
    massif: gpd.GeoDataFrame, raw_dir: Path, *, keep_archives: bool, log: logging.Logger
) -> list[Path]:
    """Download the FABDEM tiles intersecting the massif; returns tile paths."""
    raw_dir.mkdir(parents=True, exist_ok=True)
    wanted = _wanted_fabdem_tiles(massif, log)
    by_block: dict[str, list[tuple[float, float]]] = {}
    for lon, lat in wanted:
        by_block.setdefault(_fabdem_block_name(lon, lat), []).append((lon, lat))

    tile_paths = [raw_dir / _fabdem_tile_name(lon, lat) for lon, lat in wanted]
    for block, corners in sorted(by_block.items()):
        missing = [
            (lon, lat)
            for lon, lat in corners
            if not (raw_dir / _fabdem_tile_name(lon, lat)).exists()
        ]
        if not missing:
            log.info("Block %s: all %d tile(s) already present", block, len(corners))
            continue

        url = _FABDEM_URL.format(block=block)
        zip_path = raw_dir / f"{block}_FABDEM_V1-2.zip"
        if not zip_path.exists():
            _stream_download(url, zip_path, log)
        with zipfile.ZipFile(zip_path) as archive:
            members = archive.namelist()
            for lon, lat in missing:
                name = _fabdem_tile_name(lon, lat)
                member = next((m for m in members if m.endswith(name)), None)
                if member is None:
                    log.warning("%s absent from %s (likely all-water cell); skipping", name, block)
                    continue
                out_file = raw_dir / name
                with archive.open(member) as src, open(out_file, "wb") as dst:
                    for chunk in iter(lambda: src.read(_DOWNLOAD_CHUNK), b""):
                        dst.write(chunk)
                if not provenance_path_for(out_file).exists():
                    write_provenance(out_file, source=url, extra={"dataset": "FABDEM V1-2"})
                log.info("Extracted %s", name)
        if not keep_archives:
            zip_path.unlink()

    present = [p for p in tile_paths if p.exists()]
    log.info("FABDEM tiles present: %d of %d wanted", len(present), len(tile_paths))
    return present


def step_terrain(
    massif: gpd.GeoDataFrame,
    grid: raster_io.ReferenceGrid,
    tiles: Sequence[Path],
    out_dir: Path,
    *,
    num_threads: int,
    force: bool,
    log: logging.Logger,
) -> list[Path]:
    """Derive elevation, slope and heat-load index on the Carpathian grid.

    Port of notebook 002 (FABDEM cell): derivatives are computed at FABDEM's
    native ground sampling in EPSG:3035, then resampled; raw aspect is never
    resampled.
    """
    res = int(grid.resolution[0])
    out_paths = [out_dir / f"{stem}_3035_{res}m.tif" for stem in _TERRAIN_FILE_STEMS]
    if all(_matches_grid(p, grid) for p in out_paths) and not force:
        log.info("Terrain rasters already present on the expected grid; skipping.")
        return out_paths
    if not tiles:
        raise RuntimeError("No FABDEM tiles available; run the fabdem step first.")

    nodata = terminology.NODATA
    with tempfile.TemporaryDirectory() as tmp:
        vrt_path = Path(tmp) / "fabdem_4326.vrt"
        subprocess.run(["gdalbuildvrt", "-q", str(vrt_path), *map(str, tiles)], check=True)

        with rasterio.open(vrt_path) as src:
            src_crs = src.crs
            src_nodata = src.nodata
            src_transform = src.transform
            native_transform, native_w, native_h = calculate_default_transform(
                src_crs, grid.crs, src.width, src.height, *src.bounds
            )
            log.info(
                "FABDEM mosaic %dx%d px (%s) -> native EPSG:3035 %dx%d px at %.2fx%.2f m",
                src.width,
                src.height,
                src_crs,
                native_w,
                native_h,
                abs(native_transform.a),
                abs(native_transform.e),
            )
            source = src.read(1)

    elevation_native = np.full((native_h, native_w), np.nan, dtype="float64")
    reproject(
        source=source,
        destination=elevation_native,
        src_transform=src_transform,
        src_crs=src_crs,
        src_nodata=src_nodata,
        dst_transform=native_transform,
        dst_crs=grid.crs,
        dst_nodata=np.nan,
        resampling=Resampling.bilinear,
        num_threads=num_threads,
    )
    del source

    elevation_native = np.where(np.isnan(elevation_native), nodata, elevation_native)
    native_res_x = abs(native_transform.a)
    native_res_y = abs(native_transform.e)
    slope_native = predictors.horn_slope(
        elevation_native, x_res=native_res_x, y_res=native_res_y, nodata=nodata
    )
    aspect_native = predictors.horn_aspect(
        elevation_native, x_res=native_res_x, y_res=native_res_y, nodata=nodata
    )
    latitude_native = predictors.latitude_grid(native_transform, grid.crs, native_w, native_h)
    hli_native = predictors.heat_load_index(
        slope_native, aspect_native, latitude_native, nodata=nodata
    )
    del aspect_native, latitude_native
    log.info("Derived slope and heat-load index at %.2f m native sampling", native_res_x)

    massif_mask = raster_io.rasterize_mask(massif.geometry, grid, all_touched=True)

    def to_target(native: np.ndarray, path: Path, band_name: str) -> None:
        destination = np.full(grid.shape, nodata, dtype="float32")
        reproject(
            source=native.astype("float32", copy=False),
            destination=destination,
            src_transform=native_transform,
            src_crs=grid.crs,
            src_nodata=nodata,
            dst_transform=grid.transform,
            dst_crs=grid.crs,
            dst_nodata=nodata,
            resampling=Resampling.bilinear,
            num_threads=num_threads,
        )
        destination[massif_mask == 0] = nodata
        raster_io.write_geotiff(
            path,
            destination,
            grid,
            dtype="float32",
            nodata=nodata,
            band_descriptions=[band_name],
            logger=log,
        )

    for native, path, name in (
        (elevation_native, out_paths[0], _terrain_band_names()[0]),
        (slope_native, out_paths[1], _terrain_band_names()[1]),
        (hli_native, out_paths[2], _terrain_band_names()[2]),
    ):
        to_target(native, path, name)
    return out_paths


def step_osm(raw_dir: Path, *, log: logging.Logger) -> tuple[list[Path], list[str]]:
    """Download Geofabrik GeoPackage extracts for the Carpathian countries."""
    raw_dir.mkdir(parents=True, exist_ok=True)
    present: list[Path] = []
    failed: list[str] = []
    for country in _OSM_COUNTRIES:
        out_file = raw_dir / f"{country}.gpkg"
        if out_file.exists():
            log.info("%s already present", out_file.name)
            present.append(out_file)
            continue
        url = _GEOFABRIK_URL.format(country=country)
        zip_path = raw_dir / f"{country}-latest-free.gpkg.zip"
        try:
            _stream_download(url, zip_path, log)
            with zipfile.ZipFile(zip_path) as archive:
                member = next((m for m in archive.namelist() if m.lower().endswith(".gpkg")), None)
                if member is None:
                    raise RuntimeError(f"No .gpkg inside {zip_path.name}.")
                with archive.open(member) as src, open(out_file, "wb") as dst:
                    for chunk in iter(lambda: src.read(_DOWNLOAD_CHUNK), b""):
                        dst.write(chunk)
            zip_path.unlink()
            write_provenance(
                out_file,
                source=url,
                extra={"dataset": f"OpenStreetMap {country} (Geofabrik GeoPackage)"},
            )
            present.append(out_file)
        except Exception as exc:
            log.error("OSM download failed for %s: %s", country, exc)
            zip_path.unlink(missing_ok=True)
            failed.append(country)
    return present, failed


def step_roads(
    massif: gpd.GeoDataFrame,
    grid: raster_io.ReferenceGrid,
    osm_files: Sequence[Path],
    vector_out: Path,
    raster_out: Path,
    *,
    buffer_km: float,
    force: bool,
    log: logging.Logger,
) -> Path:
    """Classify OSM roads and build the three distance bands on the grid.

    Port of notebook 002 (roads cell): distances are computed on a grid padded
    by the buffer so roads beyond the massif still set boundary distances, then
    cropped and masked to the massif.
    """
    if _matches_grid(raster_out, grid) and not force:
        log.info("Roads distance raster already present on the expected grid; skipping.")
        return raster_out
    missing = [c for c in _OSM_COUNTRIES if not any(f.stem == c for f in osm_files)]
    if missing:
        raise RuntimeError(
            f"OSM extracts missing for {missing}; run the osm step first. If Geofabrik's "
            "free .gpkg extract is unavailable (404 or stub archive), derive the layer "
            "from the country's raw .osm.pbf with the GDAL OSM driver instead: lines "
            "with a highway tag, column fclass=highway, layer name "
            f"{_OSM_ROADS_LAYER!r}, saved as data/raw/vectors/open_street_map/<country>.gpkg."
        )

    nodata = terminology.NODATA
    buffer_m = buffer_km * 1_000.0
    massif_buffer = massif.geometry.buffer(buffer_m)  # GeoSeries, EPSG:3035
    bbox_4326 = tuple(massif_buffer.to_crs("EPSG:4326").total_bounds)

    frames = []
    for path in osm_files:
        frame = gpd.read_file(path, layer=_OSM_ROADS_LAYER, bbox=bbox_4326)
        log.info("%s: %d road features in massif bbox", path.name, len(frame))
        frames.append(frame)
    # Normalise CRS variants before concatenation: Geofabrik extracts declare
    # EPSG:4326 while PBF-derived fallback layers declare OGC:CRS84 (the same
    # datum); pandas refuses to concatenate differing CRS objects.
    common_crs = frames[0].crs
    frames = [f if f.crs == common_crs else f.to_crs(common_crs) for f in frames]
    roads = gpd.GeoDataFrame(pd.concat(frames, ignore_index=True), crs=common_crs)
    del frames

    log.info("Clipping %d road features to the buffered massif ...", len(roads))
    roads = gpd.clip(roads, massif_buffer.to_crs(roads.crs))
    log.info("Clip complete: %d features retained", len(roads))
    roads = roads.to_crs(terminology.CRS)
    roads = repair_geometries(roads, logger=log)
    roads = roads[roads.geom_type.isin(_LINE_TYPES)].copy()
    roads["fclass"] = roads["fclass"].astype("string").str.lower().str.strip()
    roads["road_group"] = roads["fclass"].map(_OSM_ROAD_GROUP_MAP).fillna("other")
    roads = roads.loc[
        roads["road_group"].isin(_OSM_ROAD_GROUP_ORDER),
        ["osm_id", "fclass", "road_group", "geometry"],
    ].copy()
    log.info(
        "Classified network: %s",
        roads.groupby("road_group").size().to_dict(),
    )

    vector_out.parent.mkdir(parents=True, exist_ok=True)
    if not vector_out.exists() or force:
        roads.to_file(vector_out, driver="GPKG")
        log.info("Wrote %s (%d features)", vector_out.name, len(roads))

    res = grid.resolution[0]
    pad_px = int(round(buffer_m / res))
    padded = raster_io.ReferenceGrid(
        crs=grid.crs,
        transform=from_origin(
            grid.bounds.left - pad_px * res, grid.bounds.top + pad_px * res, res, res
        ),
        width=grid.width + 2 * pad_px,
        height=grid.height + 2 * pad_px,
    )
    estimate = padded.width * padded.height * _EDT_BYTES_PER_PX
    available = _available_ram_bytes()
    log.info(
        "Distance transform domain %dx%d px; peak estimate %.0f GB of %.0f GB available",
        padded.width,
        padded.height,
        estimate / 1e9,
        available / 1e9,
    )
    if estimate > 0.6 * available:
        raise RuntimeError(
            f"Distance transform needs about {estimate / 1e9:.0f} GB but only "
            f"{available / 1e9:.0f} GB is available; choose a coarser --grid-res."
        )

    massif_mask = raster_io.rasterize_mask(massif.geometry, grid, all_touched=True)
    band_names = _roads_band_names()
    profile = grid.profile(count=3, dtype="float32", nodata=nodata)
    profile["bigtiff"] = "IF_SAFER"
    raster_out.parent.mkdir(parents=True, exist_ok=True)
    # Bands are written one at a time (write_geotiff would require all three
    # distance surfaces in memory together).
    with rasterio.open(raster_out, "w", **profile) as dst:
        for index, group in enumerate(_OSM_ROAD_GROUP_ORDER, start=1):
            lines = roads.loc[roads["road_group"] == group, "geometry"]
            if lines.empty:
                raise ValueError(f"No {group} features in the buffered massif.")
            padded_distance = raster_io.distance_to_lines_on_ref(lines, padded, logger=log)
            band = padded_distance[
                pad_px : pad_px + grid.height, pad_px : pad_px + grid.width
            ].copy()
            del padded_distance
            band[massif_mask == 0] = nodata
            dst.write(band, index)
            dst.set_band_description(index, band_names[index - 1])
            inside = band[band != nodata]
            log.info("%s: max %.0f m over %s valid px", group, inside.max(), f"{inside.size:,}")
            del band, inside
    log.info("Wrote %s", raster_out)
    return raster_out


def _parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--grid-res",
        type=positive_int,
        required=True,
        help=(
            "Output grid resolution in metres; a whole multiple of the 10 m reference "
            "resolution. Must match the TESSERA download so the two sources stay "
            "co-registered."
        ),
    )
    parser.add_argument(
        "--steps",
        default=",".join(_STEPS),
        help=f"Comma-separated subset of {_STEPS} to run (default: all, in order).",
    )
    parser.add_argument(
        "--massif",
        type=Path,
        default=None,
        help="Carpathian massif GeoPackage (default: the processed stage-020 location).",
    )
    parser.add_argument(
        "--buffer-km",
        type=positive_int,
        default=10,
        help="Road-network buffer beyond the massif, in km (default 10, as the AOI).",
    )
    parser.add_argument(
        "--num-threads",
        type=positive_int,
        default=8,
        help="Warp threads (default 8; keep bounded on shared workstations).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Report tile lists and size estimates; download and compute nothing.",
    )
    parser.add_argument(
        "--force", action="store_true", help="Rebuild outputs even when they exist."
    )
    parser.add_argument(
        "--keep-archives",
        action="store_true",
        help="Keep the 10-degree FABDEM zip archives after extraction.",
    )
    return parser.parse_args(argv)


def run(args: argparse.Namespace, paths: ProjectPaths) -> int:
    """Execute the selected steps; returns a process exit code."""
    steps = tuple(s.strip() for s in args.steps.split(",") if s.strip())
    unknown = [s for s in steps if s not in _STEPS]
    if unknown:
        raise ValueError(f"Unknown steps {unknown}; choose from {_STEPS}.")

    res = int(args.grid_res)
    out_dir = paths.processed / "rasters" / "carpathians" / f"baseline_{res}m"
    global logger
    logger = make_run_logger(__name__, out_dir)

    massif = load_carpathian_massif(args.massif, logger=logger)
    grid = carpathian_grid(massif, float(res), logger=logger)

    fabdem_raw = paths.raw / "rasters" / "fabdem"
    osm_raw = paths.raw / "vectors" / "open_street_map"
    roads_vector = (
        paths.processed
        / "vectors"
        / "open_street_map"
        / f"osm_roads_carpathians_buffer_{int(args.buffer_km)}km.gpkg"
    )
    roads_raster = out_dir / f"distance_to_roads_3035_{res}m.tif"

    if args.dry_run:
        wanted = _wanted_fabdem_tiles(massif, logger)
        blocks = sorted({_fabdem_block_name(lon, lat) for lon, lat in wanted})
        have = sum((fabdem_raw / _fabdem_tile_name(lon, lat)).exists() for lon, lat in wanted)
        osm_have = [c for c in _OSM_COUNTRIES if (osm_raw / f"{c}.gpkg").exists()]
        est_terrain_gb = 3 * grid.width * grid.height * 4 / 2.5 / 1e9
        free_gb = shutil.disk_usage(paths.processed).free / 1e9
        logger.info(
            "Dry run: %d FABDEM tiles wanted (%d present) from archives %s; "
            "OSM extracts present: %s, to download: %s; grid %dx%d px at %d m; "
            "terrain output estimate %.1f GB; free disk %.0f GB",
            len(wanted),
            have,
            blocks,
            osm_have,
            [c for c in _OSM_COUNTRIES if c not in osm_have],
            grid.width,
            grid.height,
            res,
            est_terrain_gb,
            free_gb,
        )
        return 0

    osm_failed: list[str] = []
    if "fabdem" in steps:
        step_fabdem(massif, fabdem_raw, keep_archives=args.keep_archives, log=logger)
    if "terrain" in steps:
        wanted_paths = [
            fabdem_raw / _fabdem_tile_name(lon, lat)
            for lon, lat in _wanted_fabdem_tiles(massif, logger)
        ]
        step_terrain(
            massif,
            grid,
            [p for p in wanted_paths if p.exists()],
            out_dir,
            num_threads=int(args.num_threads),
            force=args.force,
            log=logger,
        )
    if "osm" in steps:
        _, osm_failed = step_osm(osm_raw, log=logger)
    if "roads" in steps:
        osm_files = [
            osm_raw / f"{c}.gpkg" for c in _OSM_COUNTRIES if (osm_raw / f"{c}.gpkg").exists()
        ]
        step_roads(
            massif,
            grid,
            osm_files,
            roads_vector,
            roads_raster,
            buffer_km=float(args.buffer_km),
            force=args.force,
            log=logger,
        )

    band_files = {
        name: f"{stem}_3035_{res}m.tif"
        for name, stem in zip(_terrain_band_names(), _TERRAIN_FILE_STEMS, strict=True)
    }
    for index, name in enumerate(_roads_band_names(), start=1):
        band_files[name] = f"{roads_raster.name}#band={index}"
    massif_path = Path(args.massif) if args.massif else carpathians_massif_path(paths.repo_root)
    write_input_manifest([massif_path], out_dir / "input_manifest.json", repo_root=paths.repo_root)
    write_environment_snapshot(out_dir / "env_snapshot.txt")
    write_json(
        {
            "grid_res_m": res,
            "steps": list(steps),
            "grid": {
                "crs": terminology.CRS,
                "width": grid.width,
                "height": grid.height,
                "bounds": list(grid.bounds),
            },
            "band_files": band_files,
            "band_order": list(terminology.FEATURE_SET_BANDS["baseline"]),
            "osm_countries": list(_OSM_COUNTRIES),
            "osm_failed": osm_failed,
            "roads_buffer_km": float(args.buffer_km),
        },
        out_dir / "run_metadata.json",
    )
    logger.info("Baseline steps complete: %s", ", ".join(steps))
    return 1 if osm_failed else 0


def main(argv: Sequence[str] | None = None) -> int:
    """Command-line entry point."""
    args = _parse_args(argv)
    paths = get_project_paths()
    ensure_writable_dirs(paths)
    return run(args, paths)


if __name__ == "__main__":
    raise SystemExit(main())
