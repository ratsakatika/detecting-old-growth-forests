"""Smoke test for scripts/build_final_outputs.py (public output packaging)."""

import sqlite3
import xml.etree.ElementTree as ET
from pathlib import Path

import geopandas as gpd
import numpy as np
import pyogrio
import rasterio
from affine import Affine
from rasterio.crs import CRS
from shapely.geometry import box

from scripts import build_final_outputs as bfo
from utils.terminology import COMPARISON_STUDIES


def _comparison(n: int = 4) -> gpd.GeoDataFrame:
    data: dict[str, object] = {
        "parcel_id": np.arange(n),
        "reference_label": [1.0, 0.0, np.nan, 1.0],
        "ogf_ratsakatika_probability": [0.9, 0.1, 0.5, 0.8],
        "ogf_ratsakatika": [True, False, True, True],
        "corine_forest_type": ["broadleaf"] * n,
        "dominant_english": ["European beech"] * n,
        "total_ogf": [3, 0, 2, 4],
        "altitude_band": ["1000-1400 m"] * n,
    }
    for study in COMPARISON_STUDIES:
        data[f"ogf_{study}_frac"] = [0.5] * n
        data[f"ogf_{study}"] = [True, False, True, False]
    return gpd.GeoDataFrame(data, geometry=[box(i, 0, i + 1, 1) for i in range(n)], crs="EPSG:3035")


def test_public_frame_schema() -> None:
    public = bfo.public_frame(_comparison())
    assert list(public.columns)[:5] == [
        "parcel_id",
        "ogf_reference_label",
        "CORINE_forest_type",
        "OGF_probability",
        "OGF_binary",
    ]
    # Everything that is not this study's own output is dropped.
    for dropped in ("altitude_band", "dominant_english", "dominant_species", "total_ogf"):
        assert dropped not in public.columns
    for study in COMPARISON_STUDIES:
        assert not {c for c in public.columns if study in c}
    assert list(public["ogf_reference_label"]) == ["ogf", "non_ogf", None, "ogf"]


def test_write_vector_embeds_style_and_metadata(tmp_path: Path) -> None:
    comparison = tmp_path / "comparison.gpkg"
    _comparison().to_file(comparison, driver="GPKG")
    out = tmp_path / "ogf_labels_predictions.gpkg"
    bfo.write_vector(comparison, out, Path("assets/qgis_styles"), threshold=0.3164)
    assert out.is_file()
    # Style and metadata travel both as sidecars and embedded in the gpkg.
    assert out.with_suffix(".qml").is_file() and out.with_suffix(".qmd").is_file()
    with sqlite3.connect(out) as con:
        styles = con.execute("SELECT useAsDefault FROM layer_styles").fetchall()
        meta = con.execute(
            "SELECT metadata FROM gpkg_metadata WHERE md_standard_uri LIKE '%qgis%'"
        ).fetchall()
        contents = con.execute("SELECT description FROM gpkg_contents").fetchall()
    assert styles and styles[0][0] == 1  # embedded default style
    assert meta and bfo._AUTHORS in meta[0][0]  # embedded QGIS metadata names the authors
    assert "TBC" not in meta[0][0] and bfo._DOI in meta[0][0]
    assert contents and "Forest parcels (n = 4)" in contents[0][0]
    tags = pyogrio.read_info(out, layer="ogf_labels_predictions")["layer_metadata"] or {}
    assert tags["OGF_THRESHOLD"] == "0.3164"
    assert (
        "OGF_binary is 1 where OGF_probability is at or above 0.3164" in tags["OGF_THRESHOLD_NOTE"]
    )
    # The QGIS-format metadata (what the QGIS metadata tab shows) carries the threshold too.
    assert "at or above 0.3164" in meta[0][0]
    assert "at or above 0.3164" in out.with_suffix(".qmd").read_text()


def test_write_raster_masks_and_tags(tmp_path: Path) -> None:
    ref = dict(
        driver="GTiff",
        height=2,
        width=3,
        count=1,
        dtype="uint8",
        nodata=255,
        crs=CRS.from_epsg(3035),
        transform=Affine(10, 0, 0, 0, -10, 20),
    )
    src = tmp_path / "binary.tif"
    with rasterio.open(src, "w", **ref) as dst:
        dst.write(np.array([[1, 0, 1], [0, 1, 0]], dtype="uint8"), 1)
    domain = np.array([[True, True, False], [True, False, False]])  # mask out the rest
    out = tmp_path / "out.tif"
    bfo.write_raster(
        src,
        out,
        band_name="OGF binary",
        title="t",
        description="d",
        style_qml=Path("assets/qgis_styles/ogf_binary.qml"),
        predictor="2",
        resampling="NEAREST",
        threshold=0.3719,
        output="binary",
        domain_mask=domain,
    )
    with rasterio.open(out) as r:
        data = r.read(1)
        assert r.descriptions[0] == "OGF binary"
        assert r.tags()["AUTHOR"] == bfo._AUTHORS and r.tags()["LICENSE"] == bfo._LICENCE
        assert r.tags()["TIFFTAG_COPYRIGHT"] == bfo._COPYRIGHT
        assert r.tags()["OGF_THRESHOLD"] == "0.3719"
        assert "at or above 0.3719" in r.tags()["OGF_THRESHOLD_NOTE"]
        assert r.tags(1)["STATISTICS_MAXIMUM"] == "1"  # exact statistics live in the file
        assert data[0, 2] == 255 and data[1, 1] == 255  # outside the domain -> NoData
        assert data[0, 0] == 1  # inside the domain kept
        assert r.files == [str(out)]  # a COG needs no .aux.xml alongside it
    assert not out.with_name(f"{out.name}.aux.xml").exists()
    assert out.with_suffix(".qml").is_file()
    qmd = ET.parse(out.with_suffix(".qmd")).getroot()
    assert qmd.find("extent/spatial").get("crs") == "EPSG:3035"
    assert "at or above 0.3719" in qmd.findtext("abstract")
    assert qmd.findtext("license") == bfo._LICENCE_NAME
