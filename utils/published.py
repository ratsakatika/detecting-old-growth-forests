"""Rebuild the parcel and reference-label layers from the published dataset.

The published GeoPackage ``ogf_labels_predictions.gpkg`` (Zenodo,
https://doi.org/10.5281/zenodo.22693148) carries the cleaned forest parcels with
their reference label, CORINE forest type and this study's predictions. When the
Fundația Conservation Carpathia forest records are not available, this module
turns it back into the two layers the pipeline consumes:

- the cleaned parcel map that the vector pre-processing notebook writes
  (:data:`PARCEL_MAP_RELATIVE`), with the forest-record attributes left empty;
- the reference-label layer that the label-construction notebook writes
  (:data:`LABELS_FILENAME`), with the old-growth geometries trimmed of roads and
  1985-2020 disturbance by the same helper that notebook uses
  (:func:`utils.geometry_ops.remove_roads_and_disturbance`), from the public
  inputs of the raster pre-processing notebook.

The published geometry is the cleaned parcel map, so parcel ids and geometries
are exact, as are the labels and the CORINE forest types. Species composition,
stand age, ownership and the virgin-forest overlap ratio come only from the
forest records and are written as missing values; the ``class`` column therefore
carries ``OGF_Unclassified``, ``Non_OGF_Unclassified`` or ``Unlabelled``, and the
statistics that need those attributes cannot be reproduced from the published
file.

Importing this module only binds names; it performs no input/output.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Final

import geopandas as gpd
import numpy as np
import pandas as pd
import pyogrio
from pyproj import CRS

from utils.geometry_ops import remove_roads_and_disturbance
from utils.paths import ProjectPaths
from utils.terminology import CRS as PROJECT_CRS
from utils.terminology import OGF_ROAD_BUFFER_M
from utils.vector_io import repair_geometries

# The published dataset, as placed by the user under ``paths.labels``.
PUBLISHED_LABELS_FILENAME: Final[str] = "ogf_labels_predictions.gpkg"
PUBLISHED_LAYER: Final[str] = "ogf_labels_predictions"

# The two rebuilt layers (paths relative to ``paths.processed`` and ``paths.labels``).
PARCEL_MAP_RELATIVE: Final[Path] = Path("vectors/forest_records/parcel_map_clean.gpkg")
PARCEL_MAP_LAYER: Final[str] = "parcel_map"
LABELS_FILENAME: Final[str] = "ogf_reference_labels.gpkg"
LABELS_LAYER: Final[str] = "parcels"

# Public inputs of the raster pre-processing notebook used to trim old-growth
# parcels (paths relative to ``paths.processed``).
ROADS_RELATIVE: Final[Path] = Path("vectors/open_street_map/osm_roads_aoi_buffer_10km.gpkg")
DISTURBANCE_RELATIVE: Final[Path] = Path(
    "rasters/european_forest_disturbance_atlas_10m/disturbance_1985_2020_3035_10m.tif"
)

# Published label vocabulary -> the pipeline's binary ``ogf`` column.
_LABEL_TO_OGF: Final[dict[str, float]] = {"ogf": 1.0, "non_ogf": 0.0}
_REQUIRED_COLUMNS: Final[tuple[str, ...]] = (
    "parcel_id",
    "ogf_reference_label",
    "CORINE_forest_type",
)
# Cadastral baseline year the parcel map carries (as in the vector pre-processing notebook).
_PLAN_YEAR: Final[int] = 2018

# Column order of the reference-label layer, as written by the label notebook.
LABEL_COLUMNS: Final[tuple[str, ...]] = (
    "parcel_id",
    "ogf",
    "class",
    "final_composition",
    "final_age",
    "pct_coniferous",
    "pct_broadleaf",
    "dominant_code",
    "dominant_english",
    "dominant_latin",
    "dominant_romanian",
    "ogf_overlap_ratio",
    "ownership_type",
    "corine_forest_type",
    "geometry",
)


def published_labels_path(paths: ProjectPaths) -> Path:
    """Return where the published GeoPackage is expected: ``paths.labels`` / its filename."""
    return paths.labels / PUBLISHED_LABELS_FILENAME


def read_published(path: str | Path, *, logger: logging.Logger | None = None) -> gpd.GeoDataFrame:
    """Read and validate the published dataset, reprojecting to the project CRS if needed.

    Args:
        path: The published GeoPackage.
        logger: Logger for the load and reprojection lines; defaults to the module logger.

    Returns:
        The published parcels in the project CRS.

    Raises:
        FileNotFoundError: If ``path`` does not exist.
        ValueError: If a required column is missing or the label vocabulary is unknown.
    """
    log = logger if logger is not None else logging.getLogger(__name__)
    published_path = Path(path)
    if not published_path.exists():
        raise FileNotFoundError(
            f"Published dataset not found at {published_path}. Download "
            "ogf_labels_predictions.gpkg from https://doi.org/10.5281/zenodo.22693148 "
            "and place it there."
        )
    layers = {name for name, _ in pyogrio.list_layers(published_path)}
    layer = PUBLISHED_LAYER if PUBLISHED_LAYER in layers else None
    published = gpd.read_file(published_path, layer=layer)
    missing = [column for column in _REQUIRED_COLUMNS if column not in published.columns]
    if missing:
        raise ValueError(f"{published_path.name} lacks the required columns {missing}.")
    unknown = sorted(set(published["ogf_reference_label"].dropna().unique()) - set(_LABEL_TO_OGF))
    if unknown:
        raise ValueError(
            f"Unknown reference-label values {unknown}; expected {list(_LABEL_TO_OGF)}."
        )
    log.info(
        "Loaded %d published parcels from %s (%s)", len(published), published_path, published.crs
    )
    if published.crs is None or not CRS.from_user_input(published.crs).equals(
        CRS.from_user_input(PROJECT_CRS)
    ):
        log.info("Reprojecting published parcels %s -> %s", published.crs, PROJECT_CRS)
        published = published.to_crs(PROJECT_CRS)
    return published.reset_index(drop=True)


def _parcel_ids(published: gpd.GeoDataFrame) -> list[str]:
    """Zero-padded parcel ids, as the parcel map and label layers store them."""
    return [f"{int(value):05d}" for value in published["parcel_id"].to_numpy()]


def parcel_map_from_published(published: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """Rebuild the cleaned parcel map: published geometry, forest-record attributes empty.

    Args:
        published: The published parcels, from :func:`read_published`.

    Returns:
        The parcel map with the columns the vector pre-processing notebook writes.
    """
    n = len(published)
    return gpd.GeoDataFrame(
        {
            "parcel_id": _parcel_ids(published),
            "species_composition": pd.Series([None] * n, dtype=object),
            "average_stand_age": np.full(n, np.nan),
            "management_plan_year": np.full(n, _PLAN_YEAR, dtype=np.int64),
            "padure_virgina_pv": pd.Series([None] * n, dtype=object),
        },
        geometry=published.geometry.to_numpy(),
        crs=published.crs,
    )


def reference_labels_from_published(
    published: gpd.GeoDataFrame,
    *,
    roads: gpd.GeoDataFrame,
    disturbance_path: str | Path,
    road_buffer_m: float = OGF_ROAD_BUFFER_M,
    logger: logging.Logger | None = None,
) -> gpd.GeoDataFrame:
    """Rebuild the reference-label layer from the published labels.

    Old-growth parcels have buffered roads and disturbed pixels removed exactly as
    the label notebook does; every forest-record attribute is left missing.

    Args:
        published: The published parcels, from :func:`read_published`.
        roads: Road and trail lines (the raster pre-processing notebook's output).
        disturbance_path: The 1985-2020 disturbance mask on the reference grid.
        road_buffer_m: Road corridor half-width in metres.
        logger: Logger for the trimming summary; defaults to the module logger.

    Returns:
        The label layer with the columns of :data:`LABEL_COLUMNS`.
    """
    log = logger if logger is not None else logging.getLogger(__name__)
    n = len(published)
    ogf = published["ogf_reference_label"].map(_LABEL_TO_OGF).astype("float64").to_numpy()
    class_values = np.where(
        ogf == 1.0, "OGF_Unclassified", np.where(ogf == 0.0, "Non_OGF_Unclassified", "Unlabelled")
    )
    empty_text = pd.Series([None] * n, dtype=object)
    frame = gpd.GeoDataFrame(
        {
            "parcel_id": _parcel_ids(published),
            "ogf": ogf,
            "class": class_values.astype(object),
            "final_composition": empty_text.copy(),
            "final_age": np.full(n, np.nan),
            "pct_coniferous": np.full(n, np.nan),
            "pct_broadleaf": np.full(n, np.nan),
            "dominant_code": empty_text.copy(),
            "dominant_english": empty_text.copy(),
            "dominant_latin": empty_text.copy(),
            "dominant_romanian": empty_text.copy(),
            "ogf_overlap_ratio": np.full(n, np.nan),
            "ownership_type": empty_text.copy(),
            "corine_forest_type": published["CORINE_forest_type"].astype(object).to_numpy(),
        },
        geometry=published.geometry.to_numpy(),
        crs=published.crs,
    )
    frame, cut = remove_roads_and_disturbance(
        frame,
        ogf == 1.0,
        roads=roads,
        disturbance_path=disturbance_path,
        road_buffer_m=road_buffer_m,
    )
    log.info("Trimmed old-growth parcels: %s", cut)
    frame = repair_geometries(frame, logger=log)
    return frame[list(LABEL_COLUMNS)]


def rebuild_from_published(
    paths: ProjectPaths,
    *,
    published_path: str | Path | None = None,
    force: bool = False,
    logger: logging.Logger | None = None,
) -> tuple[Path, Path]:
    """Write the parcel map and the reference labels from the published dataset.

    Existing outputs are kept unless ``force``; each layer is rebuilt only when
    absent (or forced), so a parcel map cleaned from the forest records is never
    overwritten by accident.

    Args:
        paths: Project paths.
        published_path: The published GeoPackage; defaults to
            :func:`published_labels_path`.
        force: Rewrite both layers even when present.
        logger: Logger for progress lines; defaults to the module logger.

    Returns:
        The parcel-map path and the reference-label path.

    Raises:
        FileNotFoundError: If the published dataset, the roads layer or the
            disturbance mask is missing.
    """
    log = logger if logger is not None else logging.getLogger(__name__)
    source = published_labels_path(paths) if published_path is None else Path(published_path)
    parcel_map_out = paths.processed / PARCEL_MAP_RELATIVE
    labels_out = paths.labels / LABELS_FILENAME
    need_parcels = force or not parcel_map_out.exists()
    need_labels = force or not labels_out.exists()
    if not (need_parcels or need_labels):
        log.info("Parcel map and reference labels already present; nothing rebuilt.")
        return parcel_map_out, labels_out

    published = read_published(source, logger=log)
    if need_parcels:
        parcel_map = parcel_map_from_published(published)
        parcel_map_out.parent.mkdir(parents=True, exist_ok=True)
        parcel_map.to_file(parcel_map_out, layer=PARCEL_MAP_LAYER, driver="GPKG")
        log.info("Wrote %d parcels -> %s", len(parcel_map), parcel_map_out)
    if need_labels:
        roads_path = paths.processed / ROADS_RELATIVE
        disturbance_path = paths.processed / DISTURBANCE_RELATIVE
        for required in (roads_path, disturbance_path):
            if not required.exists():
                raise FileNotFoundError(
                    f"{required} is missing; run the raster pre-processing notebook first."
                )
        labels = reference_labels_from_published(
            published,
            roads=gpd.read_file(roads_path),
            disturbance_path=disturbance_path,
            logger=log,
        )
        labels_out.parent.mkdir(parents=True, exist_ok=True)
        labels.to_file(labels_out, layer=LABELS_LAYER, driver="GPKG")
        log.info("Wrote %d labelled parcels -> %s", len(labels), labels_out)
    return parcel_map_out, labels_out


__all__ = [
    "DISTURBANCE_RELATIVE",
    "LABELS_FILENAME",
    "LABELS_LAYER",
    "LABEL_COLUMNS",
    "PARCEL_MAP_LAYER",
    "PARCEL_MAP_RELATIVE",
    "PUBLISHED_LABELS_FILENAME",
    "PUBLISHED_LAYER",
    "ROADS_RELATIVE",
    "parcel_map_from_published",
    "published_labels_path",
    "read_published",
    "rebuild_from_published",
    "reference_labels_from_published",
]
