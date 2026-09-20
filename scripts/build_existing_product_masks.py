"""Align the four existing old-growth products onto the 10 m reference grid.

Each published product is reduced to a binary old-growth mask on the project
reference grid (EPSG:3035, 10 m), using the authors' own predictions with no
tuning (Stage 9). The raster products are nearest-neighbour upsampled (so the
binary verdict is preserved); the vector products are reprojected and rasterised:

  - Sabatini (2020): the ``primary_composite`` 0/1 raster (250 m).
  - Munteanu et al. (2022): the HCVF four-class raster (30 m), old-growth = the
    high-conservation classes {12, 22}.
  - Kathmann et al. (2017), Greenpeace: the potential-primary-forest polygons (a shapefile).
  - Schickhofer and Schwarz (2019), PRIMOFARO: the primary-forest polygons (a KMZ).

Old-growth areas are reported on the shared domain (the parcel map rasterised onto
the reference grid, the same area this study predicts) so the totals are comparable
with this study's map. The
masks and the per-product areas feed both the mapping notebook (the area
comparison) and the cross-study comparison (notebook 011). Outputs go to
``data/processed/rasters/existing_products_10m/``.
"""

import argparse
import logging
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Final

import geopandas as gpd
import numpy as np
import numpy.typing as npt
import pandas as pd
import rasterio
from rasterio.enums import Resampling
from rasterio.mask import mask as rio_mask

from utils.env_capture import write_environment_snapshot
from utils.inference import PIXEL_AREA_HA
from utils.io import ColumnSpec, Schema, write_csv, write_json
from utils.logging import make_run_logger
from utils.manifest import write_input_manifest
from utils.paths import ProjectPaths, ensure_writable_dirs, get_project_paths
from utils.raster_io import (
    ReferenceGrid,
    open_reference_grid,
    rasterize_mask,
    reproject_raster_to_ref,
    write_geotiff,
)
from utils.terminology import COMPARISON_STUDIES, CRS

logger = logging.getLogger(__name__)

_LOGGER_NAME: Final[str] = "build_existing_product_masks"
_RAW_SUBDIR: Final[Path] = Path("data/raw/existing_products")
_OUT_SUBDIR: Final[Path] = Path("data/processed/rasters/existing_products_10m")
_PARCEL_TEMPLATE: Final[Path] = Path("data/processed/vectors/forest_records/parcel_map_clean.gpkg")

# Source-file globs (request-only files are matched leniently by pattern).
_SOURCES: Final[dict[str, tuple[str, str]]] = {
    "sabatini": ("sabatini", "*primary_composite*.tif"),
    "munteanu": ("munteanu", "Romania_HCVF_4classes.tif"),
    "kathmann": ("kathmann", "Potential_Primary_Forests.shp"),
    "schickhofer": ("schickhofer", "*.kmz"),
}
_MUNTEANU_OGF_CLASSES: Final[tuple[int, ...]] = (12, 22)

PRODUCT_AREAS_SCHEMA: Final[Schema] = {
    "study": ColumnSpec("object"),
    "display": ColumnSpec("object"),
    "n_ogf_pixels": ColumnSpec("int64"),
    "n_domain_pixels": ColumnSpec("int64"),
    "ogf_area_ha": ColumnSpec("float64"),
    "ogf_area_domain_ha": ColumnSpec("float64"),
}


def _resolve_source(raw_root: Path, study: str) -> Path:
    """Return the single source file for a product, matched by its glob."""
    subdir, pattern = _SOURCES[study]
    matches = sorted((raw_root / subdir).glob(pattern))
    if not matches:
        raise FileNotFoundError(f"no source file for {study!r} matching {subdir}/{pattern}.")
    return matches[0]


def _raster_ogf_mask(
    src_path: Path, ref: ReferenceGrid, *, classes: Sequence[int] | None, run_logger: logging.Logger
) -> npt.NDArray[np.uint8]:
    """Nearest-neighbour upsample a raster product to a binary old-growth mask."""
    reprojected = reproject_raster_to_ref(
        src_path,
        ref,
        resampling=Resampling.nearest,
        dst_nodata=0.0,
        dtype="float32",
        logger=run_logger,
    )
    band = reprojected[0]
    if classes is None:  # Sabatini: a 0/1 composite, old-growth where the value rounds to 1
        mask = np.isclose(band, 1.0)
    else:  # Munteanu: old-growth where the class is one of the high-conservation classes.
        # Nearest resampling preserves exact integer class values, so compare as floats and
        # avoid an int cast that would overflow on the source NoData sentinel.
        mask = np.isin(band, np.asarray(classes, dtype=np.float64))
    return mask.astype(np.uint8)


def _vector_ogf_mask(src_path: Path, ref: ReferenceGrid) -> npt.NDArray[np.uint8]:
    """Reproject and rasterise a vector product to a binary old-growth mask."""
    frame = gpd.read_file(src_path)
    if frame.crs is None:
        raise ValueError(f"vector product {src_path} has no CRS.")
    geoms = frame.to_crs(CRS).geometry
    return rasterize_mask(geoms, ref, all_touched=False).astype(np.uint8)


def _product_mask(
    study: str, src_path: Path, ref: ReferenceGrid, run_logger: logging.Logger
) -> npt.NDArray[np.uint8]:
    """Dispatch a product to its raster or vector mask builder."""
    if study == "sabatini":
        return _raster_ogf_mask(src_path, ref, classes=None, run_logger=run_logger)
    if study == "munteanu":
        return _raster_ogf_mask(src_path, ref, classes=_MUNTEANU_OGF_CLASSES, run_logger=run_logger)
    return _vector_ogf_mask(src_path, ref)


def native_ogf_area_ha(study: str, boundary: gpd.GeoDataFrame, raw_root: Path) -> float:
    """Old-growth area (ha) of a product in its native form, clipped to ``boundary``.

    Unlike :func:`build_existing_product_masks` -- which reprojects every product onto the
    shared 10 m grid and reports area within the parcel-map domain -- this measures each
    product on its own source (native raster resolution, native vector geometry), with only
    a clip to ``boundary`` and no domain restriction. It answers "how much old-growth does
    the product itself claim inside this boundary, before any processing". The old-growth
    rules match the mask builder (Sabatini value == 1; Munteanu classes 12/22; the vector
    products are wholly old-growth). Vector polygons are made valid and dissolved so
    overlapping polygons are not double-counted.

    Args:
        study: One of :data:`COMPARISON_STUDIES`.
        boundary: Clip polygon(s); reprojected to each source's CRS as needed.
        raw_root: Root directory of the raw product sources.

    Returns:
        Old-growth area within ``boundary``, in hectares.
    """
    src_path = _resolve_source(raw_root, study)
    raster_classes: dict[str, tuple[int, ...] | None] = {
        "sabatini": None,
        "munteanu": _MUNTEANU_OGF_CLASSES,
    }
    if study in raster_classes:
        classes = raster_classes[study]
        with rasterio.open(src_path) as src:
            shapes = boundary.to_crs(src.crs).geometry.values
            band = rio_mask(src, shapes, crop=True, filled=True, nodata=0.0)[0][0]
            ogf = (
                np.isclose(band, 1.0)
                if classes is None
                else np.isin(band, np.asarray(classes, dtype=np.float64))
            )
            return float(ogf.sum()) * abs(src.res[0] * src.res[1]) / 1e4
    frame = gpd.read_file(src_path).to_crs(CRS)
    frame["geometry"] = frame.geometry.make_valid()
    clipped = gpd.clip(frame, boundary.to_crs(CRS))
    return float(clipped.union_all().area / 1e4)


def _parcel_domain(template_path: Path, ref: ReferenceGrid) -> npt.NDArray[np.bool_]:
    """Rasterise the parcel map to the shared domain mask (the area this study predicts)."""
    template = gpd.read_file(template_path).to_crs(CRS)
    return rasterize_mask(template.geometry, ref, all_touched=False).astype(bool)


def build_existing_product_masks(
    raw_root: Path,
    out_dir: Path,
    template_path: Path,
    ref: ReferenceGrid,
    *,
    studies: Sequence[str],
    run_logger: logging.Logger,
) -> tuple[pd.DataFrame, list[Path]]:
    """Build and write each product's old-growth mask and the per-product areas."""
    domain = _parcel_domain(template_path, ref)
    n_domain = int(domain.sum())
    rows: list[dict[str, object]] = []
    inputs: list[Path] = [template_path]
    for study in studies:
        src_path = _resolve_source(raw_root, study)
        inputs.append(src_path)
        mask = _product_mask(study, src_path, ref, run_logger)
        write_geotiff(
            out_dir / f"{study}_ogf_3035_10m.tif",
            mask,
            ref,
            dtype="uint8",
            nodata=255,
            band_descriptions=[f"{study}_ogf"],
            logger=run_logger,
        )
        n_ogf = int(mask.sum())
        n_ogf_domain = int(np.logical_and(mask.astype(bool), domain).sum())
        rows.append(
            {
                "study": study,
                "display": COMPARISON_STUDIES[study].display,
                "n_ogf_pixels": n_ogf,
                "n_domain_pixels": n_domain,
                "ogf_area_ha": n_ogf * PIXEL_AREA_HA,
                "ogf_area_domain_ha": n_ogf_domain * PIXEL_AREA_HA,
            }
        )
        run_logger.info(
            "%s: %d old-growth pixels (%.0f ha; %.0f ha within the parcel map).",
            study,
            n_ogf,
            n_ogf * PIXEL_AREA_HA,
            n_ogf_domain * PIXEL_AREA_HA,
        )
    areas = pd.DataFrame(rows)[list(PRODUCT_AREAS_SCHEMA)]
    return areas, inputs


def _parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    """Parse the command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--studies", nargs="+", default=list(COMPARISON_STUDIES), choices=list(COMPARISON_STUDIES)
    )
    return parser.parse_args(argv)


def _run(args: argparse.Namespace, paths: ProjectPaths, out_dir: Path) -> dict:
    """Build the product masks and write the areas table and metadata."""
    run_logger = make_run_logger(_LOGGER_NAME, out_dir)
    ref = open_reference_grid()
    raw_root = paths.repo_root / _RAW_SUBDIR
    template_path = paths.repo_root / _PARCEL_TEMPLATE
    run_logger.info(
        "Reference grid %dx%d %s; studies %s.", ref.width, ref.height, ref.crs, args.studies
    )

    areas, inputs = build_existing_product_masks(
        raw_root, out_dir, template_path, ref, studies=args.studies, run_logger=run_logger
    )
    areas_path = write_csv(areas, out_dir / "product_areas.csv", PRODUCT_AREAS_SCHEMA)
    write_input_manifest(inputs, out_dir / "input_manifest.json", repo_root=paths.repo_root)
    write_environment_snapshot(out_dir / "env_snapshot.txt")
    return {
        "created_at": datetime.now(UTC).isoformat(),
        "script": "scripts/build_existing_product_masks.py",
        "studies": list(args.studies),
        "reference_grid": {"width": ref.width, "height": ref.height, "crs": str(ref.crs)},
        "outputs": {"product_areas": str(areas_path.relative_to(paths.repo_root))},
    }


def main(argv: Sequence[str] | None = None) -> int:
    """Build the aligned existing-product masks and per-product areas.

    Args:
        argv: Argument list, or ``None`` to read from ``sys.argv``.

    Returns:
        Process exit code (0 on success).
    """
    args = _parse_args(argv)
    paths = get_project_paths()
    ensure_writable_dirs(paths)
    out_dir = paths.repo_root / _OUT_SUBDIR
    out_dir.mkdir(parents=True, exist_ok=True)
    summary = _run(args, paths, out_dir)
    write_json(summary, out_dir / "run_metadata.json")
    logger.info("Existing-product masks complete: %s.", out_dir.name)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
