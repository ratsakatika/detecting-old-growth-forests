"""Unit tests for utils.vector_io.

All tests are offline: they build small synthetic GeoPackages in ``tmp_path``
(or in-memory GeoDataFrames) with geopandas. None of them depend on the real AOI
file existing.
"""

import logging
from pathlib import Path

import geopandas
import pytest
from pyproj import CRS
from shapely.geometry import Polygon, box

from utils.terminology import CRS as PROJECT_CRS
from utils.vector_io import (
    VectorAudit,
    audit_vector,
    load_aoi,
    repair_geometries,
    unique_area_ha,
)

# A benign EPSG:3035 origin well inside the projection's valid European extent.
_OX = 4_000_000.0
_OY = 2_800_000.0
_LOGGER_NAME = "utils.vector_io"
_PROJECT_CRS = CRS.from_user_input(PROJECT_CRS)


def _bowtie() -> Polygon:
    """A self-intersecting (bowtie) polygon: present but not valid."""
    return Polygon([(_OX, _OY), (_OX + 1, _OY + 1), (_OX + 1, _OY), (_OX, _OY + 1), (_OX, _OY)])


def _info_records(caplog: pytest.LogCaptureFixture) -> list[logging.LogRecord]:
    """INFO records emitted on the vector_io module logger."""
    return [r for r in caplog.records if r.name == _LOGGER_NAME and r.levelno == logging.INFO]


# --------------------------------------------------------------------------- #
# audit_vector
# --------------------------------------------------------------------------- #


def test_audit_vector_reports_metadata(tmp_path: Path) -> None:
    poly1 = box(_OX, _OY, _OX + 10, _OY + 10)
    poly2 = box(_OX + 20, _OY + 20, _OX + 30, _OY + 30)
    path = tmp_path / "good.gpkg"
    geopandas.GeoDataFrame(
        {"a": [1, 2], "b": ["x", "y"]}, geometry=[poly1, poly2], crs=PROJECT_CRS
    ).to_file(path, driver="GPKG")

    audit = audit_vector(path)

    assert isinstance(audit, VectorAudit)
    assert audit.feature_count == 2
    assert audit.geometry_types == ("Polygon",)
    assert set(audit.columns) == {"a", "b"}
    assert audit.crs.to_epsg() == 3035
    assert audit.bounds == (_OX, _OY, _OX + 30, _OY + 30)
    assert audit.invalid_count == 0
    assert audit.empty_count == 0


def test_audit_vector_counts_invalid(tmp_path: Path) -> None:
    path = tmp_path / "bowtie.gpkg"
    geopandas.GeoDataFrame(geometry=[_bowtie()], crs=PROJECT_CRS).to_file(path, driver="GPKG")

    audit = audit_vector(path)

    assert audit.invalid_count == 1
    assert audit.empty_count == 0


def test_audit_vector_counts_empty(tmp_path: Path) -> None:
    good = box(_OX, _OY, _OX + 10, _OY + 10)
    path = tmp_path / "empty.gpkg"
    # One valid polygon, one empty geometry and one missing (NULL) geometry.
    geopandas.GeoDataFrame(geometry=[good, Polygon(), None], crs=PROJECT_CRS).to_file(
        path, driver="GPKG"
    )

    audit = audit_vector(path)

    assert audit.feature_count == 3
    assert audit.empty_count == 2
    assert audit.invalid_count == 0


def test_audit_vector_multilayer_requires_layer(tmp_path: Path) -> None:
    path = tmp_path / "multi.gpkg"
    geopandas.GeoDataFrame(geometry=[box(_OX, _OY, _OX + 1, _OY + 1)], crs=PROJECT_CRS).to_file(
        path, layer="roads", driver="GPKG"
    )
    geopandas.GeoDataFrame(geometry=[box(_OX, _OY, _OX + 2, _OY + 2)], crs=PROJECT_CRS).to_file(
        path, layer="parcels", driver="GPKG"
    )

    with pytest.raises(ValueError, match="multiple layers") as excinfo:
        audit_vector(path)
    assert "roads" in str(excinfo.value)
    assert "parcels" in str(excinfo.value)

    # Selecting a layer explicitly succeeds.
    audit = audit_vector(path, layer="parcels")
    assert audit.feature_count == 1


def test_audit_vector_str_is_single_line(tmp_path: Path) -> None:
    path = tmp_path / "line.gpkg"
    geopandas.GeoDataFrame(geometry=[box(_OX, _OY, _OX + 5, _OY + 5)], crs=PROJECT_CRS).to_file(
        path, driver="GPKG"
    )

    text = str(audit_vector(path))

    assert "\n" not in text
    assert "line.gpkg" in text
    assert "EPSG:3035" in text


def test_audit_vector_emits_no_log(tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
    path = tmp_path / "silent.gpkg"
    geopandas.GeoDataFrame(geometry=[box(_OX, _OY, _OX + 1, _OY + 1)], crs=PROJECT_CRS).to_file(
        path, driver="GPKG"
    )

    with caplog.at_level(logging.INFO, logger=_LOGGER_NAME):
        audit_vector(path)

    assert [r for r in caplog.records if r.name == _LOGGER_NAME] == []


# --------------------------------------------------------------------------- #
# repair_geometries
# --------------------------------------------------------------------------- #


def test_repair_geometries_flattens_3d(caplog: pytest.LogCaptureFixture) -> None:
    flat = box(_OX, _OY, _OX + 10, _OY + 10)
    raised = Polygon([(_OX, _OY, 5), (_OX + 10, _OY, 5), (_OX + 10, _OY + 10, 5), (_OX, _OY, 5)])
    gdf = geopandas.GeoDataFrame({"id": [1, 2]}, geometry=[flat, raised], crs=PROJECT_CRS)
    assert bool(gdf.geometry.has_z.iloc[1])

    with caplog.at_level(logging.INFO, logger=_LOGGER_NAME):
        repaired = repair_geometries(gdf)

    assert len(repaired) == 2
    assert not bool(repaired.geometry.has_z.any())
    assert "1 flattened" in _info_records(caplog)[0].getMessage()


def test_repair_geometries_fixes_and_drops(caplog: pytest.LogCaptureFixture) -> None:
    good = box(_OX, _OY, _OX + 10, _OY + 10)
    gdf = geopandas.GeoDataFrame(
        {"id": [1, 2, 3, 4]}, geometry=[good, _bowtie(), Polygon(), None], crs=PROJECT_CRS
    )

    with caplog.at_level(logging.INFO, logger=_LOGGER_NAME):
        repaired = repair_geometries(gdf)

    # The bowtie is fixed and the empty/missing geometries dropped.
    assert len(repaired) == 2
    assert bool(repaired.geometry.is_valid.all())
    assert not bool(repaired.geometry.is_empty.any())

    # Exactly one INFO line reporting the repaired (1) and dropped (2) counts.
    info = _info_records(caplog)
    assert len(info) == 1
    message = info[0].getMessage()
    assert "1" in message
    assert "2" in message


def test_repair_geometries_does_not_mutate_input() -> None:
    good = box(_OX, _OY, _OX + 10, _OY + 10)
    gdf = geopandas.GeoDataFrame(geometry=[good, _bowtie(), None], crs=PROJECT_CRS)

    repair_geometries(gdf)

    # The original frame is untouched: same length, still invalid, still missing.
    assert len(gdf) == 3
    assert not bool(gdf.geometry.iloc[1].is_valid)
    assert bool(gdf.geometry.isna().iloc[2])


# --------------------------------------------------------------------------- #
# load_aoi
# --------------------------------------------------------------------------- #


def test_load_aoi_already_in_project_crs(tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
    path = tmp_path / "aoi_3035.gpkg"
    feats = [box(_OX, _OY, _OX + 100, _OY + 100), box(_OX + 200, _OY, _OX + 300, _OY + 100)]
    geopandas.GeoDataFrame(geometry=feats, crs=PROJECT_CRS).to_file(path, driver="GPKG")

    with caplog.at_level(logging.INFO, logger=_LOGGER_NAME):
        gdf = load_aoi(path)

    assert len(gdf) == 2
    assert gdf.crs == _PROJECT_CRS

    # Exactly one INFO line names the path; it states the CRS and not a reprojection.
    load_lines = [r for r in _info_records(caplog) if str(path) in r.getMessage()]
    assert len(load_lines) == 1
    message = load_lines[0].getMessage()
    assert "EPSG:3035" in message
    assert "reproject" not in message.lower()


def test_load_aoi_reprojects_when_crs_differs(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    path = tmp_path / "aoi_4326.gpkg"
    # Two small lon/lat boxes near the Făgăraş Mountains.
    feats = [box(24.5, 45.5, 24.6, 45.6), box(24.7, 45.5, 24.8, 45.6)]
    geopandas.GeoDataFrame(geometry=feats, crs="EPSG:4326").to_file(path, driver="GPKG")

    with caplog.at_level(logging.INFO, logger=_LOGGER_NAME):
        gdf = load_aoi(path)

    assert gdf.crs == _PROJECT_CRS

    load_lines = [r for r in _info_records(caplog) if str(path) in r.getMessage()]
    assert len(load_lines) == 1
    message = load_lines[0].getMessage()
    assert "reproject" in message.lower()
    assert "EPSG:4326" in message
    assert "EPSG:3035" in message


def test_load_aoi_raises_when_crs_is_none(tmp_path: Path) -> None:
    # A CRS-less AOI cannot be reprojected safely; load_aoi must refuse it.
    path = tmp_path / "aoi_no_crs.gpkg"
    geopandas.GeoDataFrame(geometry=[box(_OX, _OY, _OX + 100, _OY + 100)], crs=None).to_file(
        path, driver="GPKG"
    )

    with pytest.raises(ValueError, match="AOI has no CRS"):
        load_aoi(path)


def test_load_aoi_dissolve_returns_single_feature(tmp_path: Path) -> None:
    path = tmp_path / "aoi_two.gpkg"
    feats = [box(_OX, _OY, _OX + 100, _OY + 100), box(_OX + 50, _OY + 50, _OX + 150, _OY + 150)]
    geopandas.GeoDataFrame(geometry=feats, crs=PROJECT_CRS).to_file(path, driver="GPKG")

    dissolved = load_aoi(path, dissolve=True)

    assert len(dissolved) == 1
    assert dissolved.crs == _PROJECT_CRS


# --------------------------------------------------------------------------- #
# unique_area_ha
# --------------------------------------------------------------------------- #


def test_unique_area_ha_single_polygon(caplog: pytest.LogCaptureFixture) -> None:
    # 100 m x 200 m = 20,000 m^2 = 2.0 ha.
    gdf = geopandas.GeoDataFrame(geometry=[box(_OX, _OY, _OX + 100, _OY + 200)], crs=PROJECT_CRS)

    with caplog.at_level(logging.INFO, logger=_LOGGER_NAME):
        area = unique_area_ha(gdf)

    assert area == pytest.approx(2.0)
    # Already in the project CRS: no reprojection line.
    assert _info_records(caplog) == []


def test_unique_area_ha_overlap_not_double_counted() -> None:
    # Two 100 m squares overlapping on a 50 m x 50 m corner.
    a = box(_OX, _OY, _OX + 100, _OY + 100)
    b = box(_OX + 50, _OY + 50, _OX + 150, _OY + 150)
    gdf = geopandas.GeoDataFrame(geometry=[a, b], crs=PROJECT_CRS)

    area = unique_area_ha(gdf)
    naive = (a.area + b.area) / 10_000.0

    # Union is 17,500 m^2 = 1.75 ha, strictly less than the naive 2.0 ha sum.
    assert area == pytest.approx(1.75)
    assert area < naive


def test_unique_area_ha_reprojects_geographic(caplog: pytest.LogCaptureFixture) -> None:
    series = geopandas.GeoSeries([box(24.5, 45.5, 24.6, 45.6)], crs="EPSG:4326")

    with caplog.at_level(logging.INFO, logger=_LOGGER_NAME):
        area = unique_area_ha(series)

    assert area > 0.0

    info = _info_records(caplog)
    assert len(info) == 1
    message = info[0].getMessage()
    assert "EPSG:4326" in message
    assert "EPSG:3035" in message


def test_unique_area_ha_requires_crs() -> None:
    gdf = geopandas.GeoDataFrame(geometry=[box(_OX, _OY, _OX + 1, _OY + 1)])

    with pytest.raises(ValueError, match="no CRS"):
        unique_area_ha(gdf)


def test_audit_vector_str_reports_undefined_crs(tmp_path: Path) -> None:
    path = tmp_path / "no_crs.gpkg"
    geopandas.GeoDataFrame(geometry=[box(_OX, _OY, _OX + 1, _OY + 1)], crs=None).to_file(
        path, driver="GPKG"
    )

    audit = audit_vector(path)

    assert audit.crs is None
    assert "undefined CRS" in str(audit)
