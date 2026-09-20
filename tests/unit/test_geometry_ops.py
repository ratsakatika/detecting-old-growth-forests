import geopandas as gpd
import numpy as np
import pytest
import shapely
from shapely.geometry import GeometryCollection, LineString, Point, Polygon, box

from utils.geometry_ops import (
    DeoverlapStats,
    _keep_polygonal,
    clip_to_aoi_preserve_rows,
    deoverlap_polygons,
    remove_roads_and_disturbance,
)

CRS = "EPSG:3035"


def _aoi():
    return gpd.GeoDataFrame({"site": ["aoi"]}, geometry=[box(0, 0, 10, 10)], crs=CRS)


def _poly_gdf(geoms, **cols):
    data = cols or {"pid": list(range(len(geoms)))}
    return gpd.GeoDataFrame(data, geometry=geoms, crs=CRS)


def test_keeps_inside_drops_outside_clips_straddling():
    gdf = gpd.GeoDataFrame(
        {"pid": ["inside", "outside", "straddle"]},
        geometry=[box(2, 2, 4, 4), box(20, 20, 22, 22), box(8, 8, 12, 12)],
        crs=CRS,
    )
    out = clip_to_aoi_preserve_rows(gdf, _aoi())
    assert set(out["pid"]) == {"inside", "straddle"}
    assert out.loc[out["pid"] == "inside", "geometry"].area.iloc[0] == pytest.approx(4.0)
    assert out.loc[out["pid"] == "straddle", "geometry"].area.iloc[0] == pytest.approx(4.0)


def test_preserves_index_and_attributes():
    gdf = gpd.GeoDataFrame(
        {"pid": [101, 102]}, geometry=[box(1, 1, 2, 2), box(3, 3, 4, 4)], index=[7, 9], crs=CRS
    )
    out = clip_to_aoi_preserve_rows(gdf, _aoi())
    assert list(out.index) == [7, 9]
    assert list(out["pid"]) == [101, 102]


def test_edge_only_intersection_is_dropped():
    gdf = gpd.GeoDataFrame({"pid": ["edge"]}, geometry=[box(10, 0, 12, 2)], crs=CRS)
    out = clip_to_aoi_preserve_rows(gdf, _aoi())
    assert len(out) == 0


def test_crs_mismatch_raises():
    gdf = gpd.GeoDataFrame({"pid": [1]}, geometry=[box(1, 1, 2, 2)], crs="EPSG:4326")
    with pytest.raises(ValueError, match="CRS mismatch"):
        clip_to_aoi_preserve_rows(gdf, _aoi())


def test_missing_crs_raises():
    gdf = gpd.GeoDataFrame({"pid": [1]}, geometry=[box(1, 1, 2, 2)])
    with pytest.raises(ValueError, match="must have a defined CRS"):
        clip_to_aoi_preserve_rows(gdf, _aoi())


def test_deoverlap_partial_smaller_keeps_shared_area():
    g = _poly_gdf([box(0, 0, 10, 10), box(8, 8, 12, 12)], pid=["big", "small"])
    out = deoverlap_polygons(g)
    a = out.set_index("pid").geometry.area
    assert a["small"] == pytest.approx(16.0)
    assert a["big"] == pytest.approx(96.0)
    assert out.union_all().area == pytest.approx(112.0)


def test_deoverlap_larger_priority_reverses():
    out = deoverlap_polygons(
        _poly_gdf([box(0, 0, 10, 10), box(8, 8, 12, 12)], pid=["big", "small"]),
        priority="larger_area",
    )
    a = out.set_index("pid").geometry.area
    assert a["big"] == pytest.approx(100.0)
    assert a["small"] == pytest.approx(12.0)


def test_deoverlap_containment_punches_hole():
    g = _poly_gdf([box(0, 0, 10, 10), box(3, 3, 6, 6)], pid=["outer", "inner"])
    out = deoverlap_polygons(g)
    a = out.set_index("pid").geometry.area
    assert a["inner"] == pytest.approx(9.0)
    assert a["outer"] == pytest.approx(91.0)


def test_deoverlap_full_duplicate_dropped():
    out = deoverlap_polygons(_poly_gdf([box(0, 0, 4, 4), box(0, 0, 4, 4)], pid=["first", "second"]))
    assert list(out["pid"]) == ["first"]
    assert out.union_all().area == pytest.approx(16.0)


def test_deoverlap_partitions_and_is_disjoint():
    rng = np.random.default_rng(42)
    geoms = [
        box(*(p := rng.uniform(0, 20, 2)), p[0] + w, p[1] + h)
        for w, h in rng.uniform(1, 6, (40, 2))
    ]
    out = deoverlap_polygons(_poly_gdf(geoms))
    assert out.geometry.area.sum() == pytest.approx(shapely.union_all(geoms).area)


def test_deoverlap_preserves_index_and_attributes():
    gdf = gpd.GeoDataFrame(
        {"name": ["x", "y"]}, geometry=[box(0, 0, 2, 2), box(5, 5, 7, 7)], index=[11, 22], crs=CRS
    )
    out = deoverlap_polygons(gdf)
    assert list(out.index) == [11, 22] and list(out["name"]) == ["x", "y"]


def test_deoverlap_missing_crs_raises():
    with pytest.raises(ValueError, match="defined CRS"):
        deoverlap_polygons(gpd.GeoDataFrame({"pid": [1]}, geometry=[box(0, 0, 1, 1)]))


def test_deoverlap_non_polygonal_raises():
    gdf = gpd.GeoDataFrame({"pid": [1]}, geometry=[LineString([(0, 0), (1, 1)])], crs=CRS)
    with pytest.raises(ValueError, match="polygonal"):
        deoverlap_polygons(gdf)


def test_min_residual_fraction_drops_absorbed_container():
    # A container 70% covered by a smaller polygon is dropped; its unique area is lost.
    g = gpd.GeoDataFrame(geometry=[box(0, 0, 10, 10), box(0, 0, 7, 10)], crs=CRS)
    kept = deoverlap_polygons(g, min_residual_fraction=0.40)
    assert len(kept) == 1
    assert kept.geometry.area.sum() == pytest.approx(70.0)


def test_default_is_a_partition():
    g = gpd.GeoDataFrame(geometry=[box(0, 0, 10, 10), box(0, 0, 7, 10)], crs=CRS)
    kept = deoverlap_polygons(g)
    assert kept.geometry.area.sum() == pytest.approx(g.union_all().area)


def test_min_area_drops_sliver():
    g = gpd.GeoDataFrame(geometry=[box(0, 0, 10, 10), box(0, 0, 10, 10.05)], crs=CRS)
    kept = deoverlap_polygons(g, min_area=1.0)
    assert len(kept) == 1


def test_deoverlap_return_stats():
    # box 0 trimmed to nothing useful: 1 below-threshold container, 1 absorbed duplicate, 1 kept.
    g = gpd.GeoDataFrame(geometry=[box(0, 0, 10, 10), box(0, 0, 7, 10), box(0, 0, 10, 10)], crs=CRS)
    kept, stats = deoverlap_polygons(g, min_residual_fraction=0.40, return_stats=True)
    assert isinstance(stats, DeoverlapStats)
    assert (stats.n_input, stats.n_output) == (3, 1)
    assert (stats.n_absorbed, stats.n_below_threshold, stats.n_trimmed) == (1, 1, 0)
    assert stats.n_dropped == 2
    assert len(kept) == stats.n_output


def test_keep_polygonal_extracts_polygons_from_geometrycollection() -> None:
    poly_a = box(0, 0, 2, 2)
    poly_b = box(3, 0, 4, 1)
    line = LineString([(0, 0), (5, 5)])
    # One polygonal part comes back unchanged.
    assert _keep_polygonal(GeometryCollection([poly_a, line])).geom_type == "Polygon"
    # Several polygonal parts are merged into one.
    assert _keep_polygonal(GeometryCollection([poly_a, poly_b])).geom_type == "MultiPolygon"
    # A collection with no polygonal part yields nothing.
    assert _keep_polygonal(GeometryCollection([line])) is None


def test_keep_polygonal_drops_pure_lower_dimension() -> None:
    assert _keep_polygonal(LineString([(0, 0), (1, 1)])) is None
    assert _keep_polygonal(Point(0, 0)) is None


def test_deoverlap_stats_str_and_unchanged_property() -> None:
    stats = DeoverlapStats(n_input=5, n_output=4, n_trimmed=1, n_absorbed=1, n_below_threshold=0)
    assert stats.n_unchanged == 3
    assert str(stats) == (
        "5 in -> 4 out (1 dropped: 1 absorbed, 0 below threshold; 1 trimmed, 3 unchanged)"
    )


def test_deoverlap_empty_input_returns_empty() -> None:
    empty = gpd.GeoDataFrame({"pid": []}, geometry=gpd.GeoSeries([], crs=CRS))
    assert len(deoverlap_polygons(empty)) == 0
    out, stats = deoverlap_polygons(empty, return_stats=True)
    assert len(out) == 0
    assert (stats.n_input, stats.n_output) == (0, 0)


def test_deoverlap_skips_empty_geometry_rows() -> None:
    gdf = gpd.GeoDataFrame(
        {"pid": ["solid", "empty"]}, geometry=[box(0, 0, 2, 2), Polygon()], crs=CRS
    )
    kept, stats = deoverlap_polygons(gdf, return_stats=True)
    assert list(kept["pid"]) == ["solid"]
    assert stats.n_absorbed == 1


# --------------------------------------------------------------------------- #
# remove_roads_and_disturbance
# --------------------------------------------------------------------------- #


def _disturbance_raster(tmp_path, *, crs: str = CRS):
    """A 20 x 20 cell, 10 m raster over (0, 0)-(200, 200) with a disturbed 20 x 20 m block."""
    import rasterio
    from rasterio.transform import from_origin

    data = np.zeros((20, 20), dtype="uint8")
    data[18:20, 0:2] = 1  # the block (0, 0)-(20, 20): bottom-left corner
    path = tmp_path / "disturbance.tif"
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        height=20,
        width=20,
        count=1,
        dtype="uint8",
        crs=crs,
        transform=from_origin(0, 200, 10, 10),
        nodata=255,
    ) as dst:
        dst.write(data, 1)
    return path


def _two_parcels():
    return gpd.GeoDataFrame(
        {"pid": ["a", "b"]}, geometry=[box(0, 0, 100, 100), box(100, 100, 200, 200)], crs=CRS
    )


def _road():
    return gpd.GeoDataFrame(geometry=[LineString([(50, -10), (50, 110)])], crs=CRS)


def test_remove_roads_and_disturbance_trims_selected_parcel_only(tmp_path):
    parcels = _two_parcels()
    out, stats = remove_roads_and_disturbance(
        parcels,
        [True, False],
        roads=_road(),
        disturbance_path=_disturbance_raster(tmp_path),
        road_buffer_m=5.0,
    )
    # Parcel a loses the 20 x 20 m disturbed block and the 10 m wide road corridor.
    area_a = out.loc[out["pid"] == "a"].geometry.area.iloc[0]
    assert area_a == pytest.approx(10_000 - 400 - 1_000, rel=1e-6)
    assert out.loc[out["pid"] == "b"].geometry.area.iloc[0] == pytest.approx(10_000)
    assert parcels.geometry.area.tolist() == [10_000, 10_000]  # input untouched
    assert (stats.n_selected, stats.n_patches, stats.n_removed) == (1, 1, 0)
    assert stats.area_before_ha == pytest.approx(1.0)
    assert stats.area_after_ha == pytest.approx(0.86)
    assert stats.area_removed_ha == pytest.approx(0.14)
    assert "0 parcels removed entirely" in str(stats)


def test_remove_roads_and_disturbance_drops_fully_covered_parcel(tmp_path):
    parcels = _two_parcels()
    out, stats = remove_roads_and_disturbance(
        parcels,
        [True, True],
        roads=_road(),
        disturbance_path=_disturbance_raster(tmp_path),
        road_buffer_m=60.0,  # the corridor (x from -10 to 110) swallows parcel a entirely
    )
    assert out["pid"].tolist() == ["b"]
    assert stats.n_removed == 1
    assert out.geometry.area.iloc[0] < 10_000  # b is clipped by the wide corridor too


def test_remove_roads_and_disturbance_nothing_selected(tmp_path):
    parcels = _two_parcels()
    out, stats = remove_roads_and_disturbance(
        parcels,
        [False, False],
        roads=_road(),
        disturbance_path=_disturbance_raster(tmp_path),
        road_buffer_m=5.0,
    )
    assert out.geometry.area.tolist() == [10_000, 10_000]
    assert (stats.n_selected, stats.n_patches, stats.n_removed) == (0, 0, 0)
    assert (stats.area_before_ha, stats.area_after_ha) == (0.0, 0.0)


def test_remove_roads_and_disturbance_reprojects_roads(tmp_path):
    parcels = _two_parcels()
    out, _stats = remove_roads_and_disturbance(
        parcels,
        [True, False],
        roads=_road().to_crs("EPSG:4326"),
        disturbance_path=_disturbance_raster(tmp_path),
        road_buffer_m=5.0,
    )
    assert out.loc[out["pid"] == "a"].geometry.area.iloc[0] == pytest.approx(8_600, rel=1e-3)


def test_remove_roads_and_disturbance_validation(tmp_path):
    parcels = _two_parcels()
    raster = _disturbance_raster(tmp_path)
    with pytest.raises(ValueError, match="one entry per parcel"):
        remove_roads_and_disturbance(
            parcels, [True], roads=_road(), disturbance_path=raster, road_buffer_m=5.0
        )
    with pytest.raises(ValueError, match="defined CRS"):
        remove_roads_and_disturbance(
            parcels.set_crs(None, allow_override=True),
            [True, False],
            roads=_road(),
            disturbance_path=raster,
            road_buffer_m=5.0,
        )
    with pytest.raises(ValueError, match="defined CRS"):
        remove_roads_and_disturbance(
            parcels,
            [True, False],
            roads=_road().set_crs(None, allow_override=True),
            disturbance_path=raster,
            road_buffer_m=5.0,
        )
    wgs84_dir = tmp_path / "wgs84"
    wgs84_dir.mkdir()
    with pytest.raises(ValueError, match="differs from the parcels' CRS"):
        remove_roads_and_disturbance(
            parcels,
            [True, False],
            roads=_road(),
            disturbance_path=_disturbance_raster(wgs84_dir, crs="EPSG:4326"),
            road_buffer_m=5.0,
        )
