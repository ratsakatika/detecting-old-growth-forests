import geopandas as gpd
import pytest
from shapely.geometry import box

from utils.overlay import overlap_pairs

CRS = "EPSG:3035"
OX, OY = 4_000_000.0, 2_800_000.0


def test_overlap_pairs_areas_and_ratios():
    target = gpd.GeoDataFrame(
        {"parcel_id": ["p0", "p1"]},
        geometry=[box(OX, OY, OX + 10, OY + 10), box(OX + 20, OY, OX + 30, OY + 10)],
        crs=CRS,
    )
    source = gpd.GeoDataFrame(
        {"kind": ["A", "B"]},
        geometry=[box(OX + 5, OY, OX + 15, OY + 10), box(OX + 25, OY, OX + 35, OY + 10)],
        crs=CRS,
    )
    pairs = overlap_pairs(target, source, source_columns=["kind"]).set_index("parcel_id")
    assert set(pairs.index) == {"p0", "p1"}
    assert pairs.loc["p0", "intersection_area"] == 50.0
    assert pairs.loc["p0", "overlap_target"] == 0.5
    assert pairs.loc["p0", "overlap_source"] == 0.5
    assert pairs.loc["p0", "kind"] == "A"
    assert pairs.loc["p1", "kind"] == "B"


def test_overlap_pairs_no_intersection_returns_empty():
    target = gpd.GeoDataFrame(
        {"parcel_id": ["p0"]}, geometry=[box(OX, OY, OX + 10, OY + 10)], crs=CRS
    )
    source = gpd.GeoDataFrame(
        {"kind": ["C"]}, geometry=[box(OX + 500, OY, OX + 510, OY + 10)], crs=CRS
    )
    pairs = overlap_pairs(target, source, source_columns=["kind"])
    assert pairs.empty
    assert list(pairs.columns) == [
        "parcel_id",
        "intersection_area",
        "overlap_target",
        "overlap_source",
        "kind",
    ]


def test_overlap_pairs_crs_mismatch_raises():
    target = gpd.GeoDataFrame(
        {"parcel_id": ["p0"]}, geometry=[box(OX, OY, OX + 10, OY + 10)], crs=CRS
    )
    source = target.to_crs("EPSG:4326")
    with pytest.raises(ValueError, match="CRS mismatch"):
        overlap_pairs(target, source)


def test_overlap_pairs_missing_crs_raises():
    target = gpd.GeoDataFrame({"parcel_id": ["p0"]}, geometry=[box(OX, OY, OX + 1, OY + 1)])
    source = gpd.GeoDataFrame({"kind": ["A"]}, geometry=[box(OX, OY, OX + 1, OY + 1)], crs=CRS)
    with pytest.raises(ValueError, match="defined CRS"):
        overlap_pairs(target, source)


def test_overlap_pairs_empty_layer_returns_empty():
    target = gpd.GeoDataFrame(
        {"parcel_id": ["p0"]}, geometry=[box(OX, OY, OX + 1, OY + 1)], crs=CRS
    )
    source = gpd.GeoDataFrame({"kind": []}, geometry=gpd.GeoSeries([], crs=CRS))
    pairs = overlap_pairs(target, source, source_columns=["kind"])
    assert pairs.empty
    assert list(pairs.columns) == [
        "parcel_id",
        "intersection_area",
        "overlap_target",
        "overlap_source",
        "kind",
    ]
