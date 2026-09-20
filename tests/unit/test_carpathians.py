"""Unit tests for utils.carpathians.

All tests are offline: they build small synthetic massif polygons on EPSG:3035
(or EPSG:4326 for the degree-lattice helper) so grid snapping, extraction and
validation outcomes are exact and predictable.
"""

import logging
from pathlib import Path

import geopandas as gpd
import pytest
from rasterio.windows import Window
from shapely.geometry import box

from utils import carpathians, terminology
from utils.carpathians import (
    build_carpathian_countries,
    carpathian_countries_path,
    carpathian_grid,
    degree_tiles_intersecting,
    estimated_valid_pixels,
    extract_carpathian_massif,
    load_carpathian_massif,
    window_grid,
)

CRS = terminology.CRS


def _massif_gdf(minx: float, miny: float, maxx: float, maxy: float) -> gpd.GeoDataFrame:
    return gpd.GeoDataFrame(
        {"m_massive": ["carp"], "area_km2": [1.0]},
        geometry=[box(minx, miny, maxx, maxy)],
        crs=CRS,
    )


class TestExtractCarpathianMassif:
    def _source(self, tmp_path: Path) -> Path:
        frame = gpd.GeoDataFrame(
            {
                "m_massive": ["carp", "carp", "alps"],
                "name_mm": ["Carpathians", "Carpathians", "Alps"],
                "area_km2": [100, 900, 5000],
            },
            geometry=[
                box(0, 0, 10_000, 10_000),
                box(20_000, 0, 50_000, 30_000),
                box(-90_000, 0, -20_000, 70_000),
            ],
            crs=CRS,
        )
        path = tmp_path / "massifs.gpkg"
        frame.to_file(path, driver="GPKG")
        return path

    def test_picks_largest_carp_row(self, tmp_path: Path) -> None:
        largest = extract_carpathian_massif(self._source(tmp_path))
        assert len(largest) == 1
        assert float(largest["area_km2"].iloc[0]) == 900

    def test_output_is_project_crs(self, tmp_path: Path) -> None:
        largest = extract_carpathian_massif(self._source(tmp_path))
        assert largest.crs is not None
        assert largest.crs.to_epsg() == 3035

    def test_missing_value_raises(self, tmp_path: Path) -> None:
        with pytest.raises(ValueError, match="No features"):
            extract_carpathian_massif(self._source(tmp_path), massif_value="pyrenees")

    def test_missing_field_raises(self, tmp_path: Path) -> None:
        frame = gpd.GeoDataFrame({"other": ["carp"]}, geometry=[box(0, 0, 1, 1)], crs=CRS)
        path = tmp_path / "bad.gpkg"
        frame.to_file(path, driver="GPKG")
        with pytest.raises(ValueError, match="required fields"):
            extract_carpathian_massif(path)


class TestLoadCarpathianMassif:
    def test_roundtrip(self, tmp_path: Path) -> None:
        path = tmp_path / "carpathians_3035.gpkg"
        _massif_gdf(0, 0, 1000, 1000).to_file(path, driver="GPKG")
        massif = load_carpathian_massif(path)
        assert len(massif) == 1
        assert massif.crs.to_epsg() == 3035

    def test_missing_file_raises(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError, match="build_carpathian_massif"):
            load_carpathian_massif(tmp_path / "absent.gpkg")

    def test_multi_feature_raises(self, tmp_path: Path) -> None:
        path = tmp_path / "two.gpkg"
        frame = gpd.GeoDataFrame(
            {"m_massive": ["carp", "carp"]},
            geometry=[box(0, 0, 1, 1), box(2, 2, 3, 3)],
            crs=CRS,
        )
        frame.to_file(path, driver="GPKG")
        with pytest.raises(ValueError, match="expected exactly one"):
            load_carpathian_massif(path)

    def test_wrong_crs_raises(self, tmp_path: Path) -> None:
        path = tmp_path / "wgs84.gpkg"
        gpd.GeoDataFrame(
            {"m_massive": ["carp"]}, geometry=[box(24.0, 45.0, 25.0, 46.0)], crs="EPSG:4326"
        ).to_file(path, driver="GPKG")
        with pytest.raises(ValueError, match="project CRS"):
            load_carpathian_massif(path)


class TestCarpathianGrid:
    def test_snaps_outward_to_resolution(self) -> None:
        massif = _massif_gdf(1_000_003.0, 2_000_007.0, 1_050_001.0, 2_040_002.0)
        grid = carpathian_grid(massif, 100.0)
        assert grid.bounds.left == 1_000_000.0
        assert grid.bounds.bottom == 2_000_000.0
        assert grid.bounds.right == 1_050_100.0
        assert grid.bounds.top == 2_040_100.0
        assert grid.resolution == (100.0, 100.0)
        assert grid.width == 501
        assert grid.height == 401

    def test_bounds_cover_massif(self) -> None:
        massif = _massif_gdf(4_852_557.9, 2_464_194.6, 5_641_630.4, 3_032_655.3)
        grid = carpathian_grid(massif, 10.0)
        minx, miny, maxx, maxy = massif.total_bounds
        assert grid.bounds.left <= minx and grid.bounds.bottom <= miny
        assert grid.bounds.right >= maxx and grid.bounds.top >= maxy
        # Snapped origins stay on the reference 10 m lattice.
        assert grid.bounds.left % terminology.REF_RESOLUTION_M == 0
        assert grid.bounds.top % terminology.REF_RESOLUTION_M == 0

    def test_margin_expands_bounds(self) -> None:
        massif = _massif_gdf(1_000_000.0, 2_000_000.0, 1_001_000.0, 2_001_000.0)
        no_margin = carpathian_grid(massif, 100.0)
        with_margin = carpathian_grid(massif, 100.0, margin_m=500.0)
        assert with_margin.bounds.left == no_margin.bounds.left - 500.0
        assert with_margin.bounds.top == no_margin.bounds.top + 500.0

    @pytest.mark.parametrize("resolution", [0.0, -10.0, 15.0, 5.0])
    def test_invalid_resolution_raises(self, resolution: float) -> None:
        massif = _massif_gdf(0, 0, 1000, 1000)
        with pytest.raises(ValueError, match="whole multiple"):
            carpathian_grid(massif, resolution)

    def test_wrong_crs_raises(self) -> None:
        massif = gpd.GeoDataFrame(
            {"m_massive": ["carp"]}, geometry=[box(24, 45, 25, 46)], crs="EPSG:4326"
        )
        with pytest.raises(ValueError, match="project CRS"):
            carpathian_grid(massif, 10.0)


class TestWindowGrid:
    def test_window_transform_and_shape(self) -> None:
        massif = _massif_gdf(1_000_000.0, 2_000_000.0, 1_010_000.0, 2_010_000.0)
        grid = carpathian_grid(massif, 100.0)
        sub = window_grid(grid, Window(10, 20, 30, 40))
        assert (sub.width, sub.height) == (30, 40)
        assert sub.transform.c == grid.bounds.left + 10 * 100.0
        assert sub.transform.f == grid.bounds.top - 20 * 100.0
        assert sub.crs == grid.crs


class TestDegreeTilesIntersecting:
    def test_one_degree_cells(self) -> None:
        geometry = gpd.GeoSeries([box(24.2, 45.3, 25.9, 46.1)], crs="EPSG:4326")
        corners = degree_tiles_intersecting(geometry, step_deg=1.0)
        assert corners == [(24.0, 45.0), (24.0, 46.0), (25.0, 45.0), (25.0, 46.0)]

    def test_tenth_degree_cells_are_exact_lattice(self) -> None:
        geometry = gpd.GeoSeries([box(24.01, 45.01, 24.19, 45.09)], crs="EPSG:4326")
        corners = degree_tiles_intersecting(geometry, step_deg=0.1)
        assert corners == [(24.0, 45.0), (24.1, 45.0)]

    def test_excludes_disjoint_cells(self) -> None:
        # An L-shaped geometry leaves the far corner cell untouched.
        geometry = gpd.GeoSeries(
            [box(24.0, 45.0, 26.0, 45.4).union(box(24.0, 45.0, 24.4, 47.0))],
            crs="EPSG:4326",
        )
        corners = degree_tiles_intersecting(geometry, step_deg=1.0)
        assert (25.0, 46.0) not in corners
        assert (24.0, 45.0) in corners and (25.0, 45.0) in corners and (24.0, 46.0) in corners


class TestDownloadScriptHelpers:
    """Pure helpers of the Carpathian download scripts (no network, no I/O)."""

    def test_fabdem_tile_and_block_names(self) -> None:
        from scripts.download_carpathians_baseline import _fabdem_block_name, _fabdem_tile_name

        assert _fabdem_tile_name(24.0, 45.0) == "N45E024_FABDEM_V1-2.tif"
        assert _fabdem_block_name(24.0, 45.0) == "N40E020-N50E030"
        assert _fabdem_block_name(17.0, 49.0) == "N40E010-N50E020"
        with pytest.raises(ValueError, match="not supported"):
            _fabdem_tile_name(-1.0, 45.0)

    def test_roads_band_names_match_terminology(self) -> None:
        from scripts.download_carpathians_baseline import _roads_band_names

        assert tuple(_roads_band_names()) == terminology.FEATURE_SET_BANDS["baseline"][3:]

    def test_tessera_band_names(self) -> None:
        from scripts.download_carpathians_tessera import _tessera_band_names

        names = _tessera_band_names()
        assert len(names) == 128
        assert names[0] == "tessera_t000" and names[-1] == "tessera_t127"

    def test_snapped_window(self) -> None:
        from scripts.download_carpathians_tessera import _snapped_window

        massif = _massif_gdf(1_000_000.0, 2_000_000.0, 1_010_000.0, 2_010_000.0)
        grid = carpathian_grid(massif, 100.0)
        window = _snapped_window(grid, (1_000_150.0, 2_000_150.0, 1_000_950.0, 2_000_950.0))
        assert (window.col_off, window.row_off) == (1, 90)
        assert (window.width, window.height) == (9, 9)
        assert _snapped_window(grid, (0.0, 0.0, 100.0, 100.0)) is None


class TestEstimatedValidPixels:
    def test_area_over_pixel_area(self) -> None:
        massif = _massif_gdf(0.0, 0.0, 10_000.0, 5_000.0)
        assert estimated_valid_pixels(massif, 100.0) == 5_000
        assert estimated_valid_pixels(massif, 10.0) == 500_000


class TestBuildCarpathianCountries:
    """Offline tests: the boundary readers are replaced by synthetic polygons."""

    @staticmethod
    def _setup(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, *, admin_layer: bool = True):
        from utils.paths import get_project_paths

        paths = get_project_paths(tmp_path)
        massif_path = tmp_path / "carpathians_3035.gpkg"
        _massif_gdf(4_800_000, 2_700_000, 4_900_000, 2_800_000).to_file(massif_path, driver="GPKG")
        raw = paths.raw / "vectors" / "open_street_map"
        raw.mkdir(parents=True)
        for country in carpathians.CARPATHIAN_COUNTRIES:
            (raw / f"{country}.gpkg").touch()
        layers = [("gis_osm_adminareas_a_free", "Polygon")] if admin_layer else []
        monkeypatch.setattr(carpathians.pyogrio, "list_layers", lambda path: layers)
        national = gpd.GeoDataFrame(
            {"fclass": ["national"]},
            geometry=[box(4_700_000, 2_600_000, 5_000_000, 2_900_000)],
            crs=CRS,
        ).to_crs("EPSG:4326")
        return paths, massif_path, national

    def test_builds_one_feature_per_country(self, tmp_path: Path, monkeypatch) -> None:
        paths, massif_path, national = self._setup(tmp_path, monkeypatch)
        calls: list[Path] = []

        def fake_gpkg(path: Path) -> gpd.GeoDataFrame:
            calls.append(path)
            return national.copy()

        monkeypatch.setattr(carpathians, "_national_boundary_from_gpkg", fake_gpkg)
        out = build_carpathian_countries(paths, massif_path=massif_path)
        assert out == carpathian_countries_path(tmp_path)
        assert len(calls) == len(carpathians.CARPATHIAN_COUNTRIES)
        countries = gpd.read_file(out)
        assert list(countries["country"]) == list(carpathians.CARPATHIAN_COUNTRIES.values())
        assert countries.crs.to_epsg() == 3035
        # Clipped to the massif box plus the 20 km margin on every side (140 x 140 km).
        assert countries.geometry.area.max() == pytest.approx(140_000.0**2, rel=1e-6)
        assert (out.parent / "carpathian_countries_input_manifest.json").is_file()
        assert (out.parent / "carpathian_countries_run_metadata.json").is_file()

        # Idempotent: a second call neither rebuilds nor touches the readers.
        monkeypatch.setattr(carpathians, "_national_boundary_from_gpkg", _raise_if_called)
        assert build_carpathian_countries(paths, massif_path=massif_path) == out
        # ... unless forced.
        monkeypatch.setattr(carpathians, "_national_boundary_from_gpkg", fake_gpkg)
        build_carpathian_countries(paths, massif_path=massif_path, force=True)
        assert len(calls) == 2 * len(carpathians.CARPATHIAN_COUNTRIES)

    def test_falls_back_to_pbf_without_admin_layer(self, tmp_path: Path, monkeypatch) -> None:
        paths, massif_path, national = self._setup(tmp_path, monkeypatch, admin_layer=False)
        seen: list[str] = []

        def fake_pbf(country: str, cache_dir: Path, log) -> gpd.GeoDataFrame:
            seen.append(country)
            assert cache_dir == paths.cache / "osm_pbf"
            return national.copy()

        monkeypatch.setattr(carpathians, "_national_boundary_from_gpkg", _raise_if_called)
        monkeypatch.setattr(carpathians, "_national_boundary_from_pbf", fake_pbf)
        out = build_carpathian_countries(paths, massif_path=massif_path)
        assert seen == list(carpathians.CARPATHIAN_COUNTRIES)
        assert len(gpd.read_file(out)) == len(carpathians.CARPATHIAN_COUNTRIES)

    def test_empty_boundary_raises(self, tmp_path: Path, monkeypatch) -> None:
        paths, massif_path, national = self._setup(tmp_path, monkeypatch)
        monkeypatch.setattr(
            carpathians, "_national_boundary_from_gpkg", lambda path: national.iloc[0:0].copy()
        )
        with pytest.raises(RuntimeError, match="No national polygon"):
            build_carpathian_countries(paths, massif_path=massif_path)


def _raise_if_called(*args, **kwargs):
    raise AssertionError("boundary reader should not have been called")


class TestNationalBoundaryFromPbf:
    @staticmethod
    def _fake_ogr2ogr(national: gpd.GeoDataFrame):
        def run(command, check):
            out = Path(command[3])
            national.to_file(out, driver="GPKG")
            return None

        return run

    def test_uses_cached_pbf_without_download(self, tmp_path: Path, monkeypatch) -> None:
        cache = tmp_path / "osm_pbf"
        cache.mkdir()
        (cache / "poland.osm.pbf").write_bytes(b"pbf")  # the alternative cached name
        national = gpd.GeoDataFrame({"a": [1]}, geometry=[box(0, 0, 1, 1)], crs="EPSG:4326")
        monkeypatch.setattr(carpathians.requests, "get", _raise_if_called)
        monkeypatch.setattr(carpathians.subprocess, "run", self._fake_ogr2ogr(national))
        frame = carpathians._national_boundary_from_pbf("poland", cache, logging.getLogger("t"))
        assert len(frame) == 1

    def test_downloads_missing_pbf(self, tmp_path: Path, monkeypatch) -> None:
        cache = tmp_path / "osm_pbf"
        cache.mkdir()
        national = gpd.GeoDataFrame({"a": [1]}, geometry=[box(0, 0, 1, 1)], crs="EPSG:4326")

        class _Response:
            def __enter__(self):
                return self

            def __exit__(self, *exc):
                return False

            @staticmethod
            def raise_for_status() -> None:
                return None

            @staticmethod
            def iter_content(size):
                yield b"pbf-bytes"

        urls: list[str] = []

        def fake_get(url, stream, timeout):
            urls.append(url)
            return _Response()

        monkeypatch.setattr(carpathians.requests, "get", fake_get)
        monkeypatch.setattr(carpathians.subprocess, "run", self._fake_ogr2ogr(national))
        frame = carpathians._national_boundary_from_pbf(
            "czech-republic", cache, logging.getLogger("t")
        )
        assert urls == ["https://download.geofabrik.de/europe/czech-republic-latest.osm.pbf"]
        assert (cache / "czech-republic-latest.osm.pbf").read_bytes() == b"pbf-bytes"
        assert len(frame) == 1
