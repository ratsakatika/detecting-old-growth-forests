"""Synthetic smoke test for scripts/build_existing_product_masks.py.

A tiny reference grid and four miniature product sources (two rasters, two
vectors) exercise the upsampling, rasterisation and area accounting without
touching the real continental products.
"""

import logging
import zipfile
from pathlib import Path

import geopandas as gpd
import numpy as np
import pytest
import rasterio
from affine import Affine
from rasterio.crs import CRS
from shapely.geometry import box

from scripts import build_existing_product_masks as bpm
from utils.io import validate_schema
from utils.raster_io import ReferenceGrid

# A 6 x 6, 10 m EPSG:3035 grid at a realistic AOI origin.
_ORIGIN_X, _ORIGIN_Y = 5_427_190.0, 2_634_920.0
_TRANSFORM = Affine(10.0, 0.0, _ORIGIN_X, 0.0, -10.0, _ORIGIN_Y)
_W = _H = 6


def _ref() -> ReferenceGrid:
    return ReferenceGrid(crs=CRS.from_epsg(3035), transform=_TRANSFORM, width=_W, height=_H)


def _write_raster(path: Path, data: np.ndarray, *, res: float) -> None:
    transform = Affine(res, 0.0, _ORIGIN_X, 0.0, -res, _ORIGIN_Y)
    profile = {
        "driver": "GTiff",
        "height": data.shape[0],
        "width": data.shape[1],
        "count": 1,
        "dtype": "float32",
        "crs": CRS.from_epsg(3035),
        "transform": transform,
        "nodata": -3.4e38,
    }
    with rasterio.open(path, "w", **profile) as dst:
        dst.write(data.astype(np.float32), 1)


def _cover(frac: float) -> box:
    """A 3035 box covering the left ``frac`` of the grid."""
    return box(_ORIGIN_X, _ORIGIN_Y - _H * 10.0, _ORIGIN_X + _W * 10.0 * frac, _ORIGIN_Y)


def _write_kmz(path: Path) -> None:
    """A minimal KMZ (zipped KML) polygon, reprojected from a 3035 box to WGS84."""
    geom84 = gpd.GeoSeries([_cover(0.5)], crs="EPSG:3035").to_crs("EPSG:4326").iloc[0]
    ring = " ".join(f"{x},{y},0" for x, y in geom84.exterior.coords)
    kml = (
        '<?xml version="1.0" encoding="UTF-8"?>'
        '<kml xmlns="http://www.opengis.net/kml/2.2"><Document><Placemark><Polygon>'
        f"<outerBoundaryIs><LinearRing><coordinates>{ring}</coordinates></LinearRing>"
        "</outerBoundaryIs></Polygon></Placemark></Document></kml>"
    )
    with zipfile.ZipFile(path, "w") as zf:
        zf.writestr("doc.kml", kml)


def _build_sources(raw_root: Path, template_path: Path) -> None:
    (raw_root / "sabatini").mkdir(parents=True)
    (raw_root / "munteanu").mkdir(parents=True)
    (raw_root / "kathmann").mkdir(parents=True)
    (raw_root / "schickhofer").mkdir(parents=True)
    # Sabatini: a 30 m 0/1 composite (left half old-growth).
    sab = np.zeros((2, 2), dtype=np.float32)
    sab[:, 0] = 1.0
    _write_raster(raw_root / "sabatini" / "x_primary_composite_2019.tif", sab, res=30.0)
    # Munteanu: a 20 m HCVF four-class raster; classes 12 and 22 are old-growth.
    mun = np.array([[11, 12, 21], [22, 11, 12], [21, 22, 11]], dtype=np.float32)
    _write_raster(raw_root / "munteanu" / "Romania_HCVF_4classes.tif", mun, res=20.0)
    # Kathmann: a shapefile polygon over the left 40% of the grid.
    gpd.GeoDataFrame(geometry=[_cover(0.4)], crs="EPSG:3035").to_file(
        raw_root / "kathmann" / "Potential_Primary_Forests.shp"
    )
    # Schickhofer: a KMZ over the left half.
    _write_kmz(raw_root / "schickhofer" / "primofaro.kmz")
    # Parcel-map domain: the whole grid is inside a parcel.
    template_path.parent.mkdir(parents=True, exist_ok=True)
    gpd.GeoDataFrame(geometry=[_cover(1.0)], crs="EPSG:3035").to_file(template_path)


def test_build_existing_product_masks(tmp_path: Path) -> None:
    raw_root = tmp_path / "raw"
    out_dir = tmp_path / "out"
    out_dir.mkdir()
    template_path = tmp_path / "parcel_template.gpkg"
    _build_sources(raw_root, template_path)

    areas, inputs = bpm.build_existing_product_masks(
        raw_root,
        out_dir,
        template_path,
        _ref(),
        studies=list(bpm.COMPARISON_STUDIES),
        run_logger=logging.getLogger("test"),
    )

    validate_schema(areas, bpm.PRODUCT_AREAS_SCHEMA)
    assert list(areas["study"]) == list(bpm.COMPARISON_STUDIES)
    for study in bpm.COMPARISON_STUDIES:
        mask_path = out_dir / f"{study}_ogf_3035_10m.tif"
        assert mask_path.is_file()
        with rasterio.open(mask_path) as src:
            assert src.shape == (_H, _W)
            assert set(np.unique(src.read(1))).issubset({0, 1})
    # The whole grid is inside a parcel, so every product's old-growth area lies within the domain.
    assert (areas["ogf_area_domain_ha"] <= areas["ogf_area_ha"] + 1e-9).all()
    assert (areas["n_domain_pixels"] == _W * _H).all()
    # Sabatini's left-column composite burns the left third of the 6-wide grid.
    sabatini = areas[areas["study"] == "sabatini"].iloc[0]
    assert sabatini["n_ogf_pixels"] > 0
    assert len(inputs) == 1 + len(bpm.COMPARISON_STUDIES)  # template + one source per study


def test_native_ogf_area_ha(tmp_path: Path) -> None:
    raw_root = tmp_path / "raw"
    template_path = tmp_path / "parcel_template.gpkg"
    _build_sources(raw_root, template_path)

    # Native old-growth area of each synthetic source within the whole 60 x 60 m grid,
    # measured at each source's own resolution/geometry (not the 10 m reference grid).
    boundary = gpd.GeoDataFrame(geometry=[_cover(1.0)], crs="EPSG:3035")
    expected_ha = {
        "sabatini": 0.18,  # 30 m raster, left column (2 px) old-growth
        "munteanu": 0.16,  # 20 m raster, four class-12/22 pixels
        "kathmann": 0.144,  # shapefile box over the left 40%
        "schickhofer": 0.18,  # KMZ box over the left half
    }
    for study, area_ha in expected_ha.items():
        got = bpm.native_ogf_area_ha(study, boundary, raw_root)
        assert got == pytest.approx(area_ha, rel=0.02), f"{study}: {got} vs {area_ha}"

    # Clipping to a smaller boundary (left 20%) trims every product's area.
    left = gpd.GeoDataFrame(geometry=[_cover(0.2)], crs="EPSG:3035")
    for study in expected_ha:
        assert bpm.native_ogf_area_ha(study, left, raw_root) < expected_ha[study] + 1e-9
