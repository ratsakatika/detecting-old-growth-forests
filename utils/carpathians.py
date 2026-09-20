"""Carpathian-extension helpers: massif polygon, country boundaries and grids.

Stage-020 support module for the area-of-applicability (AOA) analysis, which
extends prediction-side data acquisition from the Făgăraș AOI to the entire
Carpathian mountain range. The massif polygon is the largest
``m_massive == "carp"`` feature of the EEA European mountain massifs layer
(``data/raw/vectors/european_mountain_areas/m_massifs_v1.shp``); the smaller
``carp`` features are sub-ranges nested inside it. Notebook 013 extracts and
saves the polygon; the download scripts and later AOA computation load it from
its processed location via :func:`load_carpathian_massif`.

Grids built here share the project CRS (EPSG:3035, from ``utils.terminology``)
and snap their bounds outward to a lattice of the requested resolution. Any
resolution that is a whole multiple of the reference resolution keeps the grid
co-registered with the Făgăraș 10 m reference lattice, because the snapped
origins are themselves multiples of the reference resolution.

Importing this module only binds names and a module logger; it performs no
input/output (see AGENTS.md).
"""

import logging
import math
import subprocess
import tempfile
from pathlib import Path
from typing import Final

import geopandas as gpd
import numpy as np
import pandas as pd
import pyogrio
import requests
from rasterio.crs import CRS
from rasterio.transform import from_origin
from rasterio.windows import Window
from rasterio.windows import transform as window_transform
from shapely.geometry import box

from utils import terminology
from utils.io import write_json
from utils.manifest import write_input_manifest
from utils.paths import ProjectPaths, get_project_paths
from utils.raster_io import ReferenceGrid
from utils.vector_io import audit_vector, repair_geometries

# Module logger. No handlers are configured here; callers own logging setup.
logger = logging.getLogger(__name__)

# Processed location of the single Carpathian massif polygon, relative to the
# repository root. Written by notebook 013; read by the Carpathian download
# scripts and the AOA computation.
CARPATHIANS_MASSIF_RELATIVE: Final[Path] = Path(
    "data/processed/vectors/european_mountain_areas/carpathians_3035.gpkg"
)

# Attribute field and value identifying Carpathian features in the EEA
# European mountain massifs layer.
MASSIF_FIELD: Final[str] = "m_massive"
MASSIF_VALUE: Final[str] = "carp"

# Tolerance for accepting a requested grid resolution as a whole multiple of
# the reference resolution, in metres.
_RESOLUTION_TOLERANCE_M: Final[float] = 1e-6

# National boundaries of the seven countries the massif touches: Geofabrik extract
# name -> country code. Ukraine lies outside NUTS, so its ISO 3166-1 alpha-2 code
# stands in for the NUTS-0 code.
CARPATHIAN_COUNTRIES: Final[dict[str, str]] = {
    "romania": "RO",
    "ukraine": "UA",
    "slovakia": "SK",
    "poland": "PL",
    "hungary": "HU",
    "czech-republic": "CZ",
    "serbia": "RS",
}

# Processed location of the dissolved national polygons (one feature per
# country), relative to the repository root. Built on demand by
# :func:`build_carpathian_countries` for the per-country breakdown of the
# area-of-applicability notebook.
CARPATHIAN_COUNTRIES_RELATIVE: Final[Path] = Path(
    "data/processed/vectors/open_street_map/carpathian_countries_3035.gpkg"
)

# Geofabrik admin-areas layer carrying the national polygons, the raw PBF used
# where a free extract lacks that layer, and the margin added around the massif
# bounding box before clipping the national polygons (metres).
_ADMIN_LAYER: Final[str] = "gis_osm_adminareas_a_free"
_PBF_URL: Final[str] = "https://download.geofabrik.de/europe/{country}-latest.osm.pbf"
_COUNTRY_CLIP_MARGIN_M: Final[float] = 20_000.0


def carpathians_massif_path(repo_root: Path | None = None) -> Path:
    """Resolve the processed Carpathian massif GeoPackage path.

    Args:
        repo_root: Repository root. Defaults to the discovered project root.

    Returns:
        Absolute path to ``carpathians_3035.gpkg`` (which may not exist yet).
    """
    root = get_project_paths().repo_root if repo_root is None else Path(repo_root)
    return root / CARPATHIANS_MASSIF_RELATIVE


def extract_carpathian_massif(
    source: Path,
    *,
    massif_field: str = MASSIF_FIELD,
    massif_value: str = MASSIF_VALUE,
    logger: logging.Logger | None = None,
) -> gpd.GeoDataFrame:
    """Extract the single largest Carpathian polygon from the massifs layer.

    Filters the EEA massifs layer to ``massif_field == massif_value``, keeps
    the row with the largest ``area_km2`` (the range proper; the remaining
    rows are nested sub-ranges), reprojects to the project CRS and repairs
    the geometry.

    Args:
        source: Path to the massifs source layer (``m_massifs_v1.shp``).
        massif_field: Attribute field naming the massif group.
        massif_value: Field value selecting the Carpathians.
        logger: Logger for progress lines; defaults to the module logger.

    Returns:
        A single-row GeoDataFrame in the project CRS.

    Raises:
        ValueError: If no feature matches, or the ``area_km2`` field is absent.
    """
    log = logger if logger is not None else logging.getLogger(__name__)

    massifs = gpd.read_file(source)
    if massif_field not in massifs.columns or "area_km2" not in massifs.columns:
        raise ValueError(
            f"{source} lacks required fields {massif_field!r} and/or 'area_km2'; "
            f"found columns {list(massifs.columns)}."
        )

    matches = massifs[massifs[massif_field] == massif_value]
    if matches.empty:
        raise ValueError(f"No features with {massif_field} == {massif_value!r} in {source}.")

    largest = matches.sort_values("area_km2", ascending=False).head(1).copy()
    log.info(
        "Selected largest of %d %r features: area_km2=%s (next largest %s)",
        len(matches),
        massif_value,
        f"{float(largest['area_km2'].iloc[0]):,.0f}",
        f"{float(matches['area_km2'].sort_values().iloc[-2]):,.0f}" if len(matches) > 1 else "n/a",
    )

    source_crs = largest.crs
    largest = largest.to_crs(terminology.CRS)
    log.info("Reprojected massif %s -> %s", source_crs, terminology.CRS)
    return repair_geometries(largest, logger=log).reset_index(drop=True)


def load_carpathian_massif(
    path: Path | None = None,
    *,
    logger: logging.Logger | None = None,
) -> gpd.GeoDataFrame:
    """Load the processed Carpathian massif polygon.

    Args:
        path: GeoPackage path. Defaults to :func:`carpathians_massif_path`.
        logger: Logger for the one INFO line; defaults to the module logger.

    Returns:
        A single-row GeoDataFrame in the project CRS.

    Raises:
        FileNotFoundError: If the massif GeoPackage does not exist yet.
        ValueError: If the layer does not hold exactly one feature, or its CRS
            does not match the project CRS.
    """
    log = logger if logger is not None else logging.getLogger(__name__)

    massif_path = carpathians_massif_path() if path is None else Path(path)
    if not massif_path.exists():
        raise FileNotFoundError(
            f"Carpathian massif polygon not found at {massif_path}. Run "
            "scripts/build_carpathian_massif.py first."
        )

    massif = gpd.read_file(massif_path)
    if len(massif) != 1:
        raise ValueError(f"{massif_path} holds {len(massif)} features; expected exactly one.")

    project_crs = CRS.from_user_input(terminology.CRS)
    if massif.crs is None or CRS.from_user_input(massif.crs) != project_crs:
        raise ValueError(
            f"{massif_path} has CRS {massif.crs}; expected the project CRS {terminology.CRS}."
        )

    log.info("Loaded Carpathian massif from %s (%.0f km2)", massif_path, massif.area.iloc[0] / 1e6)
    return massif


def carpathian_grid(
    massif: gpd.GeoDataFrame | gpd.GeoSeries,
    resolution_m: float,
    *,
    margin_m: float = 0.0,
    logger: logging.Logger | None = None,
) -> ReferenceGrid:
    """Build a Carpathian reference grid at the requested resolution.

    Bounds are the massif's total bounds, optionally padded by ``margin_m``,
    snapped outward to a lattice of ``resolution_m``. Because ``resolution_m``
    must be a whole multiple of the reference resolution, the snapped origins
    are multiples of the reference resolution and the grid stays co-registered
    with the Făgăraș 10 m reference lattice.

    Args:
        massif: Massif polygon (single row) in the project CRS.
        resolution_m: Grid resolution in metres; a positive whole multiple of
            ``terminology.REF_RESOLUTION_M``.
        margin_m: Extra margin added on every side before snapping.
        logger: Logger for the one INFO line; defaults to the module logger.

    Returns:
        The snapped :class:`ReferenceGrid` in the project CRS.

    Raises:
        ValueError: If ``resolution_m`` is not a positive whole multiple of the
            reference resolution, or the massif CRS differs from the project CRS.
    """
    log = logger if logger is not None else logging.getLogger(__name__)

    ratio = resolution_m / terminology.REF_RESOLUTION_M
    if resolution_m <= 0 or abs(ratio - round(ratio)) > _RESOLUTION_TOLERANCE_M or ratio < 1:
        raise ValueError(
            f"resolution_m must be a positive whole multiple of the reference "
            f"resolution ({terminology.REF_RESOLUTION_M:g} m); got {resolution_m!r}."
        )

    project_crs = CRS.from_user_input(terminology.CRS)
    if massif.crs is None or CRS.from_user_input(massif.crs) != project_crs:
        raise ValueError(
            f"Massif CRS {massif.crs} does not match the project CRS {terminology.CRS}."
        )

    minx, miny, maxx, maxy = massif.total_bounds
    minx = math.floor((minx - margin_m) / resolution_m) * resolution_m
    miny = math.floor((miny - margin_m) / resolution_m) * resolution_m
    maxx = math.ceil((maxx + margin_m) / resolution_m) * resolution_m
    maxy = math.ceil((maxy + margin_m) / resolution_m) * resolution_m
    width = int(round((maxx - minx) / resolution_m))
    height = int(round((maxy - miny) / resolution_m))

    grid = ReferenceGrid(
        crs=project_crs,
        transform=from_origin(minx, maxy, resolution_m, resolution_m),
        width=width,
        height=height,
    )
    log.info(
        "Carpathian grid: %dx%d px at %g m, bbox=(%.0f, %.0f, %.0f, %.0f)",
        width,
        height,
        resolution_m,
        minx,
        miny,
        maxx,
        maxy,
    )
    return grid


def snapped_window(ref: ReferenceGrid, bounds: tuple[float, float, float, float]) -> Window | None:
    """Snap projected bounds outward to the grid lattice, clipped to the grid.

    Args:
        ref: Reference grid (project CRS).
        bounds: ``(minx, miny, maxx, maxy)`` in the grid CRS.

    Returns:
        The covering :class:`rasterio.windows.Window`, or ``None`` when the
        bounds do not overlap the grid.
    """
    res_x, _ = ref.resolution
    left, top = ref.bounds.left, ref.bounds.top
    minx, miny, maxx, maxy = bounds
    col0 = max(int(math.floor((minx - left) / res_x)), 0)
    col1 = min(int(math.ceil((maxx - left) / res_x)), ref.width)
    row0 = max(int(math.floor((top - maxy) / res_x)), 0)
    row1 = min(int(math.ceil((top - miny) / res_x)), ref.height)
    if col1 <= col0 or row1 <= row0:
        return None
    return Window(col0, row0, col1 - col0, row1 - row0)


def window_grid(ref: ReferenceGrid, window: Window) -> ReferenceGrid:
    """Build the sub-grid covered by a window of a reference grid.

    Args:
        ref: Parent reference grid.
        window: Rasterio window into ``ref`` (integer offsets and sizes).

    Returns:
        A :class:`ReferenceGrid` describing exactly the window's extent, for
        windowed writes of tile-sized arrays.
    """
    return ReferenceGrid(
        crs=ref.crs,
        transform=window_transform(window, ref.transform),
        width=int(window.width),
        height=int(window.height),
    )


def degree_tiles_intersecting(
    geometry_4326: gpd.GeoDataFrame | gpd.GeoSeries,
    *,
    step_deg: float,
    logger: logging.Logger | None = None,
) -> list[tuple[float, float]]:
    """List south-west corners of step-degree cells intersecting a geometry.

    Used to enumerate FABDEM 1-degree tiles (``step_deg=1``) and the expected
    TESSERA 0.1-degree tile lattice (``step_deg=0.1``) covering the massif.
    The lattice is anchored at integer multiples of ``step_deg`` from the
    origin, matching both products' tiling.

    Args:
        geometry_4326: Geometry in EPSG:4326 (GeoDataFrame or GeoSeries).
        step_deg: Cell size in degrees.
        logger: Logger for the one INFO line; defaults to the module logger.

    Returns:
        Sorted ``(lon, lat)`` south-west corners of intersecting cells.
    """
    log = logger if logger is not None else logging.getLogger(__name__)

    union = geometry_4326.union_all()
    minx, miny, maxx, maxy = union.bounds
    lon_start = math.floor(minx / step_deg)
    lon_stop = math.ceil(maxx / step_deg)
    lat_start = math.floor(miny / step_deg)
    lat_stop = math.ceil(maxy / step_deg)

    corners: list[tuple[float, float]] = []
    for lon_i in range(lon_start, lon_stop):
        for lat_i in range(lat_start, lat_stop):
            lon = lon_i * step_deg
            lat = lat_i * step_deg
            if union.intersects(box(lon, lat, lon + step_deg, lat + step_deg)):
                # Round away float noise from the multiplication so corner
                # values are exact lattice coordinates (e.g. 24.1, not
                # 24.100000000000001).
                corners.append((round(lon, 6), round(lat, 6)))

    log.info(
        "%d of %d candidate %g-degree cells intersect the geometry",
        len(corners),
        (lon_stop - lon_start) * (lat_stop - lat_start),
        step_deg,
    )
    return sorted(corners)


def estimated_valid_pixels(massif: gpd.GeoDataFrame | gpd.GeoSeries, resolution_m: float) -> int:
    """Estimate the count of grid pixels inside the massif polygon.

    Area-based estimate used for disk-space forecasts before any download.

    Args:
        massif: Massif polygon in the project (equal-area) CRS.
        resolution_m: Grid resolution in metres.

    Returns:
        Approximate number of pixels whose centres fall inside the polygon.
    """
    area_m2 = float(np.sum(massif.area))
    return int(area_m2 / (resolution_m * resolution_m))


def carpathian_countries_path(repo_root: Path | None = None) -> Path:
    """Return the processed Carpathian country-boundary GeoPackage path.

    Args:
        repo_root: Repository root. When ``None`` it is discovered through
            :func:`utils.paths.get_project_paths`.

    Returns:
        ``<repo_root>/`` :data:`CARPATHIAN_COUNTRIES_RELATIVE`.
    """
    root = get_project_paths().repo_root if repo_root is None else Path(repo_root)
    return root / CARPATHIAN_COUNTRIES_RELATIVE


def _national_boundary_from_gpkg(path: Path) -> gpd.GeoDataFrame:
    """Return the ``fclass = 'national'`` polygons of a Geofabrik GeoPackage extract."""
    return gpd.read_file(path, layer=_ADMIN_LAYER, where="fclass = 'national'")


def _national_boundary_from_pbf(
    country: str, cache_dir: Path, log: logging.Logger
) -> gpd.GeoDataFrame:
    """Return a country's admin-level-2 boundary from its raw Geofabrik PBF.

    Used for the countries whose free GeoPackage extract lacks the admin-areas
    layer (Czechia and Poland). The PBF is downloaded from Geofabrik into
    ``cache_dir`` if absent and converted with the GDAL OSM driver (``ogr2ogr``).

    Args:
        country: Geofabrik extract name (a key of :data:`CARPATHIAN_COUNTRIES`).
        cache_dir: Directory holding (or receiving) ``<country>-latest.osm.pbf``.
        log: Logger for the download line.

    Returns:
        The administrative boundary polygons, in the PBF's CRS (EPSG:4326).
    """
    pbf = cache_dir / f"{country}-latest.osm.pbf"
    if not pbf.exists():
        alt = cache_dir / f"{country}.osm.pbf"
        pbf = alt if alt.exists() else pbf
    if not pbf.exists():
        url = _PBF_URL.format(country=country)
        log.info("Downloading %s", url)
        tmp = pbf.with_suffix(".part")
        with requests.get(url, stream=True, timeout=1200) as response:
            response.raise_for_status()
            with open(tmp, "wb") as handle:
                for chunk in response.iter_content(1 << 20):
                    handle.write(chunk)
        tmp.replace(pbf)
    with tempfile.TemporaryDirectory() as tmp_dir:
        out = Path(tmp_dir) / f"{country}_admin.gpkg"
        subprocess.run(
            [
                "ogr2ogr",
                "-f",
                "GPKG",
                str(out),
                str(pbf),
                "multipolygons",
                "-where",
                "boundary = 'administrative' AND admin_level = '2'",
            ],
            check=True,
        )
        return gpd.read_file(out)


def build_carpathian_countries(
    paths: ProjectPaths,
    *,
    massif_path: Path | None = None,
    force: bool = False,
    logger: logging.Logger | None = None,
) -> Path:
    """Assemble the national polygons of the seven Carpathian countries.

    Each country's national polygon is read from the Geofabrik OpenStreetMap
    extract in ``data/raw/vectors/open_street_map`` (downloaded by
    ``scripts/download_carpathians_baseline.py``) or, for extracts without the
    admin-areas layer, from the raw ``.osm.pbf`` (see
    :func:`_national_boundary_from_pbf`). Polygons are reprojected to the project
    CRS, clipped to the massif bounding box plus
    :data:`_COUNTRY_CLIP_MARGIN_M`, dissolved to one feature per country and
    written to :data:`CARPATHIAN_COUNTRIES_RELATIVE` with an input manifest and
    run-metadata sidecar. The build is idempotent unless ``force``.

    Args:
        paths: Project paths.
        massif_path: Massif GeoPackage; defaults to the processed location.
        force: Rebuild even when the output already exists.
        logger: Logger for progress lines; defaults to the module logger.

    Returns:
        The output GeoPackage path.

    Raises:
        RuntimeError: If a country yields no national polygon.
    """
    log = logger if logger is not None else logging.getLogger(__name__)
    out_path = paths.repo_root / CARPATHIAN_COUNTRIES_RELATIVE
    if out_path.exists() and not force:
        log.info("Carpathian countries present: %s", audit_vector(out_path))
        return out_path

    massif = load_carpathian_massif(massif_path, logger=log)
    minx, miny, maxx, maxy = massif.total_bounds
    margin = _COUNTRY_CLIP_MARGIN_M
    clip_box = box(minx - margin, miny - margin, maxx + margin, maxy + margin)
    raw_dir = paths.raw / "vectors" / "open_street_map"
    cache_dir = paths.cache / "osm_pbf"
    cache_dir.mkdir(parents=True, exist_ok=True)

    parts = []
    inputs = []
    for country, code in CARPATHIAN_COUNTRIES.items():
        gpkg = raw_dir / f"{country}.gpkg"
        layers = {name for name, _ in pyogrio.list_layers(gpkg)}
        if _ADMIN_LAYER in layers:
            frame = _national_boundary_from_gpkg(gpkg)
            inputs.append(gpkg)
        else:
            frame = _national_boundary_from_pbf(country, cache_dir, log)
            inputs.append(cache_dir / f"{country}-latest.osm.pbf")
        if frame.empty:
            raise RuntimeError(f"No national polygon found for {country}.")
        clipped = gpd.clip(frame.to_crs(terminology.CRS), clip_box)
        parts.append({"country": code, "geometry": clipped.geometry.union_all()})
        log.info("%s (%s): national polygon clipped to the massif box", country, code)

    countries = gpd.GeoDataFrame(pd.DataFrame(parts), geometry="geometry", crs=terminology.CRS)
    countries = repair_geometries(countries, logger=log)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    countries.to_file(out_path, driver="GPKG")
    log.info("Wrote %s", audit_vector(out_path))

    write_input_manifest(
        [path for path in inputs if path.exists()],
        out_path.parent / "carpathian_countries_input_manifest.json",
        repo_root=paths.repo_root,
    )
    write_json(
        {"countries": list(CARPATHIAN_COUNTRIES.values()), "output": out_path.name},
        out_path.parent / "carpathian_countries_run_metadata.json",
    )
    return out_path


__all__ = [
    "CARPATHIANS_MASSIF_RELATIVE",
    "CARPATHIAN_COUNTRIES",
    "CARPATHIAN_COUNTRIES_RELATIVE",
    "MASSIF_FIELD",
    "MASSIF_VALUE",
    "build_carpathian_countries",
    "carpathian_countries_path",
    "carpathian_grid",
    "carpathians_massif_path",
    "degree_tiles_intersecting",
    "estimated_valid_pixels",
    "extract_carpathian_massif",
    "load_carpathian_massif",
    "snapped_window",
    "window_grid",
]
