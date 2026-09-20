"""Geometry utilities for vector pre-processing and label construction.

Side-effect-free helpers used across the forestry-vector pre-processing
and the reference-label construction. The module
holds no project constants; callers pass geometries and read constants from
:mod:`utils.terminology`.

"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import geopandas as gpd
import numpy as np
import rasterio
import shapely
from rasterio.crs import CRS
from rasterio.features import geometry_mask, shapes
from shapely.geometry import shape

__all__ = [
    "DeoverlapStats",
    "RoadDisturbanceCut",
    "clip_to_aoi_preserve_rows",
    "deoverlap_polygons",
    "remove_roads_and_disturbance",
]

_POLYGONAL = ("Polygon", "MultiPolygon")


def clip_to_aoi_preserve_rows(
    gdf: gpd.GeoDataFrame,
    aoi: gpd.GeoDataFrame,
    *,
    keep_geom_type: bool = True,
) -> gpd.GeoDataFrame:
    """Clip ``gdf`` to the AOI, keeping one output row per intersecting input row.

    Every feature is clipped to the AOI boundary; features lying entirely outside
    are removed. Clipping never splits a feature into several rows, so the index
    and attributes of each surviving feature are preserved unchanged. Empty or
    null geometries after clipping are dropped. With ``keep_geom_type`` true, only
    the parts matching the input's geometry dimension are kept, so a boundary
    sliver of lower dimension (a shared edge clipped to a line) is discarded
    rather than left as an empty or mixed geometry.

    Args:
        gdf: Features to clip. Must have a defined CRS.
        aoi: Area-of-interest polygon(s). Must share ``gdf``'s CRS. Multiple rows
            are treated as their union.
        keep_geom_type: Keep only parts matching the input geometry dimension.

    Returns:
        The clipped features, a row-preserving subset of ``gdf``.

    Raises:
        ValueError: If either input lacks a CRS, or their CRSs differ.
    """
    if gdf.crs is None or aoi.crs is None:
        raise ValueError("Both 'gdf' and 'aoi' must have a defined CRS.")
    if gdf.crs != aoi.crs:
        raise ValueError(f"CRS mismatch: gdf is {gdf.crs}, aoi is {aoi.crs}.")

    clipped = gpd.clip(gdf, aoi, keep_geom_type=keep_geom_type)
    keep = clipped.geometry.notna() & ~clipped.geometry.is_empty
    if keep_geom_type:
        # keep_geom_type extracts matching parts from mixed collections but can leave a
        # pure lower-dimension sliver (a shared edge clipped to a line); drop those by
        # restricting to the input's own geometry family.
        families = {
            "Polygon": {"Polygon", "MultiPolygon"},
            "MultiPolygon": {"Polygon", "MultiPolygon"},
            "LineString": {"LineString", "MultiLineString"},
            "MultiLineString": {"LineString", "MultiLineString"},
            "Point": {"Point", "MultiPoint"},
            "MultiPoint": {"Point", "MultiPoint"},
        }
        input_types = set(gdf.geom_type.dropna())
        allowed = (
            set().union(*(families.get(t, {t}) for t in input_types)) if input_types else set()
        )
        if allowed:
            keep &= clipped.geom_type.isin(allowed)
    return clipped.loc[keep].copy()


def _keep_polygonal(geom):
    """Return the polygonal component of a geometry, or None if there is none."""
    if geom is None or geom.is_empty:
        return None
    kind = geom.geom_type
    if kind in _POLYGONAL:
        return geom
    if kind == "GeometryCollection":
        parts = [g for g in geom.geoms if g.geom_type in _POLYGONAL and not g.is_empty]
        if not parts:
            return None
        return shapely.union_all(parts) if len(parts) > 1 else parts[0]
    return None  # a pure line or point fragment from a boundary-only difference


@dataclass(frozen=True, slots=True)
class DeoverlapStats:
    """Counts from a de-overlap pass: how the input rows were trimmed or dropped."""

    n_input: int
    n_output: int
    n_trimmed: int
    n_absorbed: int
    n_below_threshold: int

    @property
    def n_dropped(self) -> int:
        """Rows removed: absorbed by higher-priority neighbours or below a residual filter."""
        return self.n_absorbed + self.n_below_threshold

    @property
    def n_unchanged(self) -> int:
        """Surviving rows whose geometry was not trimmed."""
        return self.n_output - self.n_trimmed

    def __str__(self) -> str:
        """Return a one-line summary of the counts."""
        return (
            f"{self.n_input} in -> {self.n_output} out "
            f"({self.n_dropped} dropped: {self.n_absorbed} absorbed, "
            f"{self.n_below_threshold} below threshold; "
            f"{self.n_trimmed} trimmed, {self.n_unchanged} unchanged)"
        )


def deoverlap_polygons(
    gdf: gpd.GeoDataFrame,
    *,
    priority: Literal["smaller_area", "larger_area"] = "smaller_area",
    min_area: float = 0.0,
    min_residual_fraction: float = 0.0,
    return_stats: bool = False,
    logger: logging.Logger | None = None,
) -> gpd.GeoDataFrame | tuple[gpd.GeoDataFrame, DeoverlapStats]:
    """Resolve overlaps within a polygon layer so the output geometries are disjoint.

    Overlapping area is assigned by a deterministic priority: with
    ``priority="smaller_area"`` (default) the smaller polygon keeps the shared area
    and the larger one cedes it, so nested subdivisions stay intact and the
    enclosing polygon gains a hole; ``"larger_area"`` reverses this. Ties are broken
    by row position, so the result is fully determined by the input.

    Each input row is preserved with its geometry trimmed to its exclusive area.
    Two optional filters then drop residual polygons: ``min_area`` removes any whose
    trimmed area falls below it (an absolute sliver cut), and
    ``min_residual_fraction`` removes any retaining less than that fraction of its
    original area, that is, one mostly absorbed by higher-priority neighbours. Both
    default to 0.0, which keeps every non-empty fragment so the output partitions
    the union of the inputs. A positive ``min_residual_fraction`` deliberately
    discards absorbed polygons and the unique area they held.

    Args:
        gdf: Polygon or multipolygon features with a defined, ideally equal-area CRS.
        priority: Which polygon keeps overlapping area.
        min_area: Drop polygons trimmed below this area (CRS units).
        min_residual_fraction: Drop polygons retaining less than this fraction of
            their original area (0.0 to 1.0).
        return_stats: If ``True``, return ``(gdf, stats)`` where ``stats`` is a
            :class:`DeoverlapStats`; otherwise return only the GeoDataFrame.
        logger: Logger for the one INFO line; defaults to the module logger.

    Returns:
        A row-preserving subset of ``gdf`` with mutually disjoint geometries, or that
        subset paired with a :class:`DeoverlapStats` when ``return_stats`` is true.

    Raises:
        ValueError: If ``gdf`` has no CRS, or contains non-polygonal geometries.
    """
    log = logger if logger is not None else logging.getLogger(__name__)
    if gdf.crs is None:
        raise ValueError("'gdf' must have a defined CRS.")
    present = set(gdf.geometry.geom_type.dropna().unique())
    if not present <= set(_POLYGONAL):
        raise ValueError(f"deoverlap_polygons requires polygonal input; found {sorted(present)}.")

    geoms = gdf.geometry.to_numpy()
    n = len(geoms)
    if n == 0:
        stats = DeoverlapStats(0, 0, 0, 0, 0)
        return (gdf.copy(), stats) if return_stats else gdf.copy()

    areas = gdf.geometry.area.to_numpy()
    positions = np.arange(n)
    sort_area = areas if priority == "smaller_area" else -areas
    order = np.lexsort((positions, sort_area))  # by area, ties by position
    rank = np.empty(n, dtype=np.int64)
    rank[order] = np.arange(n)

    tree = shapely.STRtree(geoms)
    out = np.empty(n, dtype=object)
    for i, geom in enumerate(geoms):
        if geom is None or geom.is_empty:
            out[i] = None
            continue
        candidates = tree.query(geom, predicate="intersects")
        higher = candidates[rank[candidates] < rank[i]]
        if higher.size:
            out[i] = _keep_polygonal(geom.difference(shapely.union_all(geoms[higher])))
        else:
            out[i] = geom

    result = gdf.copy().set_geometry(gpd.GeoSeries(out, index=gdf.index, crs=gdf.crs))
    empty = result.geometry.isna().to_numpy() | result.geometry.is_empty.to_numpy()
    surviving = np.where(empty, 0.0, result.geometry.area.to_numpy())
    keep = (surviving > 0.0) & (surviving >= min_area) & (surviving > min_residual_fraction * areas)

    stats = DeoverlapStats(
        n_input=n,
        n_output=int(keep.sum()),
        n_trimmed=int((keep & (surviving + 1e-6 < areas)).sum()),
        n_absorbed=int(empty.sum()),
        n_below_threshold=int((~empty & ~keep).sum()),
    )
    log.info("De-overlapped polygons (%s wins): %s", priority, stats)
    kept = result[keep].copy()
    return (kept, stats) if return_stats else kept


@dataclass(frozen=True, slots=True)
class RoadDisturbanceCut:
    """Counts and areas from cutting roads and disturbance out of parcel geometries."""

    n_selected: int
    n_patches: int
    n_removed: int
    area_before_ha: float
    area_after_ha: float

    @property
    def area_removed_ha(self) -> float:
        """Area cut out of the selected parcels, in hectares."""
        return self.area_before_ha - self.area_after_ha

    def __str__(self) -> str:
        """Return a one-line summary of the cut."""
        return (
            f"{self.n_selected} parcels, {self.n_patches} disturbed patches: "
            f"{self.area_before_ha:,.0f} -> {self.area_after_ha:,.0f} ha "
            f"({self.area_removed_ha:,.0f} ha removed; {self.n_removed} parcels removed entirely)"
        )


def remove_roads_and_disturbance(
    parcels: gpd.GeoDataFrame,
    selected: Sequence[bool] | np.ndarray,
    *,
    roads: gpd.GeoDataFrame,
    disturbance_path: str | Path,
    road_buffer_m: float,
) -> tuple[gpd.GeoDataFrame, RoadDisturbanceCut]:
    """Cut buffered roads and disturbed pixels out of the selected parcels.

    This is the geometry step of the reference-label construction. Within the
    union of the selected (old-growth) parcels, every pixel of the disturbance
    raster with a value above zero is polygonised; all roads are buffered by
    ``road_buffer_m`` each side; the union of both is subtracted from the
    selected geometries. Parcels left with no geometry are dropped; unselected
    parcels are returned unchanged. The input is never mutated.

    Args:
        parcels: Parcel polygons with a defined CRS.
        selected: Boolean mask, one entry per row of ``parcels``, marking the
            parcels to trim.
        roads: Road and trail lines with a defined CRS (reprojected to the
            parcels' CRS).
        disturbance_path: Single-band raster in the parcels' CRS whose values
            above zero mark disturbed pixels.
        road_buffer_m: Buffer applied to the roads, in metres each side.

    Returns:
        The trimmed parcels and a :class:`RoadDisturbanceCut` summary.

    Raises:
        ValueError: If ``selected`` does not have one entry per parcel, if
            ``parcels`` or ``roads`` lack a CRS, or if the raster CRS differs
            from the parcels' CRS.
    """
    if parcels.crs is None or roads.crs is None:
        raise ValueError("Both 'parcels' and 'roads' must have a defined CRS.")
    mask = np.asarray(selected, dtype=bool)
    if mask.shape != (len(parcels),):
        raise ValueError(
            f"'selected' must have one entry per parcel ({len(parcels)}); got shape {mask.shape}."
        )

    parcels = parcels.copy()
    geometry_name = parcels.geometry.name
    chosen = parcels.loc[mask].copy()
    area_before = float(chosen.geometry.area.sum()) / 1e4
    if chosen.empty:
        return parcels, RoadDisturbanceCut(0, 0, 0, 0.0, 0.0)

    chosen_union = chosen.geometry.union_all()
    parcel_crs = CRS.from_user_input(parcels.crs)
    with rasterio.open(disturbance_path) as src:
        if src.crs is None or src.crs != parcel_crs:
            raise ValueError(
                f"Disturbance raster CRS {src.crs} differs from the parcels' CRS {parcels.crs}."
            )
        band = src.read(1, masked=True)
        inside = geometry_mask(
            [chosen_union], out_shape=band.shape, transform=src.transform, invert=True
        )
        disturbed = np.ma.filled(band > 0, False) & inside
        patches = [
            shape(geom)
            for geom, value in shapes(
                disturbed.astype(np.uint8), mask=disturbed, transform=src.transform
            )
            if value == 1
        ]
    disturbance_union = shapely.union_all(patches) if patches else None

    road_union = roads.to_crs(parcels.crs).geometry.buffer(road_buffer_m).union_all()
    cutter = shapely.union_all([g for g in (disturbance_union, road_union) if g is not None])

    chosen = chosen.set_geometry(chosen.geometry.difference(cutter))
    removed = chosen.index[chosen.geometry.is_empty | chosen.geometry.isna()]
    chosen = chosen.drop(index=removed)
    parcels.loc[chosen.index, geometry_name] = chosen.geometry
    parcels = parcels.drop(index=removed)
    area_after = float(chosen.geometry.area.sum()) / 1e4
    stats = RoadDisturbanceCut(
        n_selected=int(mask.sum()),
        n_patches=len(patches),
        n_removed=len(removed),
        area_before_ha=area_before,
        area_after_ha=area_after,
    )
    return parcels, stats
