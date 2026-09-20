"""Unit tests for utils.published: rebuilding pipeline layers from the published dataset."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import geopandas as gpd
import numpy as np
import pytest
import rasterio
from rasterio.transform import from_origin
from shapely.geometry import LineString, box

from utils import published
from utils.paths import get_project_paths
from utils.published import (
    LABEL_COLUMNS,
    parcel_map_from_published,
    published_labels_path,
    read_published,
    rebuild_from_published,
    reference_labels_from_published,
)

CRS = "EPSG:3035"


def test_import_writes_no_files(tmp_path: Path) -> None:
    repo_root = Path(published.__file__).resolve().parents[1]
    result = subprocess.run(
        [sys.executable, "-c", "import utils.published"],
        cwd=tmp_path,
        env={**os.environ, "PYTHONPATH": str(repo_root)},
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert list(tmp_path.iterdir()) == []


def _published_frame() -> gpd.GeoDataFrame:
    return gpd.GeoDataFrame(
        {
            "parcel_id": np.array([1, 2, 3], dtype=np.int64),
            "ogf_reference_label": ["ogf", "non_ogf", None],
            "CORINE_forest_type": ["coniferous", "mixed", None],
            "OGF_probability": [0.9, 0.1, 0.5],
            "OGF_binary": [True, False, True],
        },
        geometry=[box(0, 0, 100, 100), box(100, 0, 200, 100), box(0, 100, 100, 200)],
        crs=CRS,
    )


def _write_inputs(paths, *, layer: str | None = "ogf_labels_predictions") -> Path:
    """The published file plus the roads and disturbance inputs the rebuild needs."""
    paths.labels.mkdir(parents=True, exist_ok=True)
    source = published_labels_path(paths)
    _published_frame().to_file(source, layer=layer, driver="GPKG")

    roads_path = paths.processed / published.ROADS_RELATIVE
    roads_path.parent.mkdir(parents=True, exist_ok=True)
    gpd.GeoDataFrame(geometry=[LineString([(50, -10), (50, 110)])], crs=CRS).to_file(
        roads_path, driver="GPKG"
    )

    raster = paths.processed / published.DISTURBANCE_RELATIVE
    raster.parent.mkdir(parents=True, exist_ok=True)
    data = np.zeros((20, 20), dtype="uint8")
    data[18:20, 0:2] = 1  # a disturbed 20 x 20 m block in the corner of parcel 1
    with rasterio.open(
        raster,
        "w",
        driver="GTiff",
        height=20,
        width=20,
        count=1,
        dtype="uint8",
        crs=CRS,
        transform=from_origin(0, 200, 10, 10),
        nodata=255,
    ) as dst:
        dst.write(data, 1)
    return source


def test_rebuild_writes_both_layers(tmp_path: Path) -> None:
    paths = get_project_paths(tmp_path)
    _write_inputs(paths)
    parcel_map_path, labels_path = rebuild_from_published(paths)

    parcel_map = gpd.read_file(parcel_map_path, layer="parcel_map")
    assert list(parcel_map.columns) == [
        "parcel_id",
        "species_composition",
        "average_stand_age",
        "management_plan_year",
        "padure_virgina_pv",
        "geometry",
    ]
    assert parcel_map["parcel_id"].tolist() == ["00001", "00002", "00003"]
    assert parcel_map["management_plan_year"].tolist() == [2018, 2018, 2018]
    assert parcel_map["average_stand_age"].isna().all()
    assert parcel_map["species_composition"].isna().all()
    assert parcel_map.geometry.area.tolist() == pytest.approx([10_000, 10_000, 10_000])

    labels = gpd.read_file(labels_path, layer="parcels")
    assert list(labels.columns) == list(LABEL_COLUMNS)
    assert labels["parcel_id"].tolist() == ["00001", "00002", "00003"]
    assert labels["ogf"].tolist()[:2] == [1.0, 0.0] and np.isnan(labels["ogf"].iloc[2])
    assert labels["class"].tolist() == ["OGF_Unclassified", "Non_OGF_Unclassified", "Unlabelled"]
    assert labels["corine_forest_type"].tolist()[:2] == ["coniferous", "mixed"]
    assert labels["corine_forest_type"].isna().iloc[2]
    assert labels["final_age"].isna().all() and labels["ownership_type"].isna().all()
    # The old-growth parcel loses the disturbed block (400 m2) and the 20 m wide road corridor
    # (OGF_ROAD_BUFFER_M = 10 m each side over 100 m: 2,000 m2).
    assert labels.geometry.area.tolist() == pytest.approx([7_600, 10_000, 10_000], rel=1e-6)
    assert labels.crs.to_epsg() == 3035


def test_rebuild_skips_existing_layers_unless_forced(tmp_path: Path) -> None:
    paths = get_project_paths(tmp_path)
    _write_inputs(paths)
    parcel_map_path, labels_path = rebuild_from_published(paths)
    first = (parcel_map_path.stat().st_mtime_ns, labels_path.stat().st_mtime_ns)
    assert rebuild_from_published(paths) == (parcel_map_path, labels_path)
    assert (parcel_map_path.stat().st_mtime_ns, labels_path.stat().st_mtime_ns) == first
    rebuild_from_published(paths, force=True)
    assert labels_path.stat().st_mtime_ns >= first[1]


def test_rebuild_requires_the_published_file(tmp_path: Path) -> None:
    paths = get_project_paths(tmp_path)
    with pytest.raises(FileNotFoundError, match="zenodo"):
        rebuild_from_published(paths)


def test_rebuild_requires_the_preprocessed_inputs(tmp_path: Path) -> None:
    paths = get_project_paths(tmp_path)
    source = _write_inputs(paths)
    (paths.processed / published.ROADS_RELATIVE).unlink()
    with pytest.raises(FileNotFoundError, match="raster pre-processing notebook"):
        rebuild_from_published(paths, published_path=source)


def test_read_published_validates_columns_and_labels(tmp_path: Path) -> None:
    bad = _published_frame().drop(columns=["CORINE_forest_type"])
    path = tmp_path / "bad.gpkg"
    bad.to_file(path, driver="GPKG")
    with pytest.raises(ValueError, match="required columns"):
        read_published(path)

    wrong = _published_frame()
    wrong.loc[0, "ogf_reference_label"] = "old"
    path = tmp_path / "wrong.gpkg"
    wrong.to_file(path, driver="GPKG")
    with pytest.raises(ValueError, match="Unknown reference-label values"):
        read_published(path)


def test_read_published_accepts_default_layer_and_reprojects(tmp_path: Path) -> None:
    path = tmp_path / "wgs84.gpkg"
    _published_frame().to_crs("EPSG:4326").to_file(path, driver="GPKG")  # layer "wgs84"
    frame = read_published(path)
    assert frame.crs.to_epsg() == 3035
    assert frame.geometry.area.tolist() == pytest.approx([10_000] * 3, rel=1e-3)


def test_reference_labels_from_published_uses_buffer_argument(tmp_path: Path) -> None:
    paths = get_project_paths(tmp_path)
    _write_inputs(paths)
    frame = read_published(published_labels_path(paths))
    labels = reference_labels_from_published(
        frame,
        roads=gpd.read_file(paths.processed / published.ROADS_RELATIVE),
        disturbance_path=paths.processed / published.DISTURBANCE_RELATIVE,
        road_buffer_m=0.5,
    )
    assert labels.geometry.area.iloc[0] == pytest.approx(10_000 - 400 - 100, rel=1e-6)
    assert parcel_map_from_published(frame).geometry.area.iloc[0] == pytest.approx(10_000)
