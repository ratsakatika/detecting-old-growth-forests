from __future__ import annotations

from collections.abc import Sequence

import geopandas as gpd
import pandas as pd
import shapely

__all__ = ["overlap_pairs"]


def overlap_pairs(
    target: gpd.GeoDataFrame,
    source: gpd.GeoDataFrame,
    *,
    source_columns: Sequence[str] = (),
    target_id: str = "parcel_id",
) -> pd.DataFrame:
    """Pair each target polygon with the source polygons it intersects.

    For every intersecting (target, source) pair the result records the shared
    area and two overlap ratios: ``overlap_target`` is the shared area as a
    fraction of the target polygon, ``overlap_source`` as a fraction of the source
    polygon. Requested ``source_columns`` are carried through so callers can select
    the dominant or best-overlapping source per target. Non-intersecting targets
    contribute no rows.

    Args:
        target: Polygons to enrich, with a unique ``target_id`` column and a CRS.
        source: Polygons to overlap against, in the same CRS as ``target``.
        source_columns: Source columns to copy onto each pair.
        target_id: Identifier column on ``target`` to key the output by.

    Returns:
        A long DataFrame with columns ``target_id``, ``intersection_area``,
        ``overlap_target``, ``overlap_source`` and any ``source_columns``; empty
        when nothing intersects.

    Raises:
        ValueError: If either layer has no CRS, or the two CRSs differ.
    """
    if target.crs is None or source.crs is None:
        raise ValueError("both layers must have a defined CRS")
    if target.crs != source.crs:
        raise ValueError(f"CRS mismatch: {target.crs} vs {source.crs}")
    columns = [target_id, "intersection_area", "overlap_target", "overlap_source", *source_columns]
    target_geoms = target.geometry.to_numpy()
    source_geoms = source.geometry.to_numpy()
    if len(target_geoms) == 0 or len(source_geoms) == 0:
        return pd.DataFrame(columns=columns)
    target_area = target.geometry.area.to_numpy()
    source_area = source.geometry.area.to_numpy()
    ti, si = shapely.STRtree(source_geoms).query(target_geoms, predicate="intersects")
    if len(ti) == 0:
        return pd.DataFrame(columns=columns)
    inter_area = shapely.area(shapely.intersection(target_geoms[ti], source_geoms[si]))
    keep = inter_area > 0.0
    ti, si, inter_area = ti[keep], si[keep], inter_area[keep]
    data = {
        target_id: target[target_id].to_numpy()[ti],
        "intersection_area": inter_area,
        "overlap_target": inter_area / target_area[ti],
        "overlap_source": inter_area / source_area[si],
    }
    for col in source_columns:
        data[col] = source[col].to_numpy()[si]
    return pd.DataFrame(data)
