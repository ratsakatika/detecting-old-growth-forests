"""Generic, reusable vector helpers anchored to the project CRS.

These are the shared building blocks for the spatial stages: an area-of-interest
loader that checks and logs CRS, geometry repair, an equal-area unique-area
measure and a structured :class:`VectorAudit`. Source-specific raw-to-processed
steps (clipping the OpenStreetMap extract and pulling out road classes, clipping
and forest-class filtering CORINE, and so on) are *not* here; they are done cell
by cell in ``notebooks/002`` using these helpers.

Spatial constants are never hard-coded here: the project CRS is read from
:mod:`utils.terminology` and compared by :class:`pyproj.CRS` equality (not string
comparison), and the default AOI path is resolved through :mod:`utils.paths`. The
project CRS is EPSG:3035 (ETRS89 / LAEA Europe), an equal-area metric CRS, so
areas measured on it are in square metres.

Importing this module only binds names and a module logger: it performs no
input/output, configures no logging handlers and inspects no environment, in
keeping with the repository rule that ``utils`` modules have no side effects on
import (see AGENTS.md). Unlike :mod:`utils.raster_io`, this module imports
geopandas and shapely at module top, since it returns and operates on
``GeoDataFrame`` objects.
"""

import logging
from dataclasses import dataclass
from pathlib import Path

import geopandas
import shapely
from pyproj import CRS

from utils import terminology
from utils.paths import get_project_paths

# Module logger. No handlers are configured here; callers own logging setup.
logger = logging.getLogger(__name__)


def _crs_label(crs: CRS | None) -> str:
    """Return a short label for a CRS: its EPSG code when resolvable, else its name."""
    if crs is None:
        return "undefined CRS"
    epsg = crs.to_epsg()
    return f"EPSG:{epsg}" if epsg is not None else crs.name


@dataclass(frozen=True, slots=True)
class VectorAudit:
    """Structured metadata for a vector dataset.

    The audit summarises a single vector layer: its CRS, the distinct geometry
    types present, the feature count, the non-geometry attribute columns, the
    spatial extent and the number of invalid and empty/missing geometries. It
    performs no logging itself; notebooks print ``str(audit)`` on load and
    long-running scripts pass the same data to ``logger.info``.

    Attributes:
        path: Path to the audited vector.
        crs: Coordinate reference system, or ``None`` if undefined.
        geometry_types: Distinct geometry type names present (e.g. ``"Polygon"``).
        feature_count: Number of features (rows).
        columns: Non-geometry attribute column names.
        bounds: Spatial extent ``(minx, miny, maxx, maxy)`` in the layer CRS.
        invalid_count: Number of present geometries failing shapely ``is_valid``.
        empty_count: Number of empty or missing geometries.
    """

    path: Path
    crs: CRS | None
    geometry_types: tuple[str, ...]
    feature_count: int
    columns: tuple[str, ...]
    bounds: tuple[float, float, float, float]
    invalid_count: int
    empty_count: int

    def __str__(self) -> str:
        """Return a single readable summary line for the vector."""
        minx, miny, maxx, maxy = self.bounds
        geom = ", ".join(self.geometry_types) if self.geometry_types else "none"
        return (
            f"{self.path.name}: {_crs_label(self.crs)} {self.feature_count} feature(s) "
            f"[{geom}] bounds=({minx:g}, {miny:g}, {maxx:g}, {maxy:g})"
        )


def audit_vector(path: Path, *, layer: str | None = None) -> VectorAudit:
    """Read a vector and return its :class:`VectorAudit`.

    Invalid geometries (present, but failing shapely ``is_valid``) and
    empty/missing geometries are counted separately. This reads every feature, so
    it is intended for AOI-scale and processed vectors, not multi-gigabyte raw
    country extracts. This function performs no logging; callers print
    ``str(audit)`` in notebooks or pass it to ``logger.info`` in scripts.

    Args:
        path: Path to the vector dataset.
        layer: Layer to read for multi-layer sources (e.g. a GeoPackage). When
            ``None`` and the source has a single layer, that layer is read.

    Returns:
        The :class:`VectorAudit`.

    Raises:
        ValueError: If ``layer`` is ``None`` and the source has multiple layers;
            the message names the available layers.
    """
    path = Path(path)
    if layer is None:
        available = [str(name) for name in geopandas.list_layers(path)["name"]]
        if len(available) > 1:
            joined = ", ".join(available)
            raise ValueError(
                f"{path} contains multiple layers ({joined}); pass layer= to choose one."
            )

    gdf = geopandas.read_file(path, layer=layer)
    geometries = gdf.geometry

    missing = geometries.isna()
    empty_or_missing = geometries.is_empty | missing
    geometry_types = tuple(sorted(geometries.geom_type.dropna().unique()))

    return VectorAudit(
        path=path,
        crs=gdf.crs,
        geometry_types=geometry_types,
        feature_count=int(len(gdf)),
        columns=tuple(c for c in gdf.columns if c != geometries.name),
        bounds=tuple(float(v) for v in gdf.total_bounds),  # type: ignore[arg-type]
        invalid_count=int(((~geometries.is_valid) & (~missing)).sum()),
        empty_count=int(empty_or_missing.sum()),
    )


def repair_geometries(
    gdf: geopandas.GeoDataFrame,
    *,
    logger: logging.Logger | None = None,
) -> geopandas.GeoDataFrame:
    """Return a copy with geometries flattened to 2D, made valid, and empties dropped.

    Any Z dimension is dropped with :func:`shapely.force_2d` (areas and overlaps
    are planar in the project CRS and the processed-vector writers expect 2D),
    invalid geometries are repaired with :func:`shapely.make_valid`, and empty or
    missing geometries are dropped. The input is never mutated. One INFO line
    reports how many geometries were flattened, repaired, and dropped.

    Args:
        gdf: Input GeoDataFrame.
        logger: Logger for the one INFO line; defaults to the module logger.

    Returns:
        A repaired copy of ``gdf``.
    """
    log = logger if logger is not None else logging.getLogger(__name__)

    geometries = gdf.geometry
    geometry_name = geometries.name
    missing = geometries.isna()
    drop_mask = geometries.is_empty | missing
    keep_mask = ~drop_mask
    repaired_count = int(((~geometries.is_valid) & keep_mask).sum())
    dropped_count = int(drop_mask.sum())

    kept = gdf.loc[keep_mask].copy()
    flattened_count = int(kept.geometry.has_z.sum())
    if flattened_count:
        kept[geometry_name] = shapely.force_2d(kept.geometry.to_numpy())
    if repaired_count:
        kept[geometry_name] = shapely.make_valid(kept.geometry.to_numpy())

    log.info(
        "Repaired geometries: %d flattened to 2D, %d invalid fixed, "
        "%d empty or missing dropped (%d features in, %d out)",
        flattened_count,
        repaired_count,
        dropped_count,
        len(gdf),
        len(kept),
    )
    return kept


def load_aoi(
    path: Path | None = None,
    *,
    dissolve: bool = False,
    logger: logging.Logger | None = None,
) -> geopandas.GeoDataFrame:
    """Load the area-of-interest boundary in the project CRS.

    The file is read, its CRS checked against the project CRS by
    :class:`pyproj.CRS` equality (not string comparison) and reprojected if it
    differs, and its geometries repaired with :func:`repair_geometries`. The
    canonical AOI on disk is the two-site Natura 2000 polygon already in
    EPSG:3035, but the loader still checks and logs, per AGENTS.md. Exactly one
    INFO line is emitted for the load, naming the CRS and path and, when a
    reprojection happens, recording it (source CRS -> EPSG:3035); vectors need no
    resampling, so no resampling method is reported.

    Args:
        path: AOI path. Defaults to ``get_project_paths().aoi``.
        dissolve: When ``True``, return a single-feature GeoDataFrame holding the
            union of all input features; otherwise return the features as loaded.
        logger: Logger for the one INFO line; defaults to the module logger.

    Returns:
        The AOI as a GeoDataFrame in the project CRS.

    Raises:
        ValueError: If the loaded AOI has no CRS; an undefined CRS cannot be
            reprojected safely, so the correct CRS must be assigned at source.
    """
    log = logger if logger is not None else logging.getLogger(__name__)

    aoi_path = get_project_paths().aoi if path is None else Path(path)
    gdf = geopandas.read_file(aoi_path)

    source_crs = gdf.crs
    if source_crs is None:
        raise ValueError("AOI has no CRS; assign the correct CRS at source first.")

    project_crs = CRS.from_user_input(terminology.CRS)
    if source_crs == project_crs:
        log.info("Loaded AOI in %s from %s", _crs_label(source_crs), aoi_path)
    else:
        gdf = gdf.to_crs(project_crs)
        log.info(
            "Loaded AOI from %s and reprojected from %s to EPSG:3035",
            aoi_path,
            _crs_label(source_crs),
        )

    gdf = repair_geometries(gdf, logger=log)

    if dissolve:
        union = gdf.geometry.union_all()
        gdf = geopandas.GeoDataFrame(geometry=[union], crs=project_crs)

    return gdf


def unique_area_ha(
    geometry: geopandas.GeoDataFrame | geopandas.GeoSeries,
    *,
    logger: logging.Logger | None = None,
) -> float:
    """Return the area in hectares of the geometry's spatial union.

    Overlapping features are unioned first, so shared area is not double-counted.
    A CRS is required; areas are only meaningful in the equal-area project CRS
    (EPSG:3035), so a geometry in any other CRS is reprojected to it first, with
    one INFO line recording the reprojection. The union area is computed in square
    metres and divided by 10,000.

    Args:
        geometry: A GeoDataFrame or GeoSeries with a defined CRS.
        logger: Logger for the reprojection INFO line; defaults to the module
            logger. No line is emitted when the input is already in the project
            CRS.

    Returns:
        The unique area in hectares.

    Raises:
        ValueError: If ``geometry`` has no CRS.
    """
    log = logger if logger is not None else logging.getLogger(__name__)

    crs = geometry.crs
    if crs is None:
        raise ValueError(
            "geometry has no CRS; cannot measure area. Assign the correct CRS at source first."
        )

    project_crs = CRS.from_user_input(terminology.CRS)
    if crs != project_crs:
        geometry = geometry.to_crs(project_crs)
        log.info(
            "Reprojected geometry from %s to EPSG:3035 before measuring area "
            "(area is only meaningful in the equal-area project CRS)",
            _crs_label(crs),
        )

    geoms = geometry.geometry if isinstance(geometry, geopandas.GeoDataFrame) else geometry
    return float(geoms.union_all().area) / 10_000.0


__all__ = [
    "VectorAudit",
    "audit_vector",
    "load_aoi",
    "repair_geometries",
    "unique_area_ha",
]
