"""Assemble the public-facing final outputs into ``results/final/``.

Packages the three public products from the latest comparison GeoPackage and the latest
mapping run's GeoTIFFs, with tidy attribute/band names, full citation and rights metadata
and embedded QGIS styles. The analysis is not re-run; this is a deterministic packaging step.

Outputs (``results/final/``):
  - ``ogf_labels_predictions.gpkg`` - the parcel table (reference label, CORINE forest type
    and this study's probability and verdict).
  - ``ogf_probability_3035_10m.tif`` - the wall-to-wall old-growth probability, as a COG.
  - ``ogf_binary_3035_10m.tif`` - the binary old-growth map, masked to the predicted
    domain (within the AOI), NoData elsewhere, as a COG.

The rasters carry their exact band statistics inside the file (so no ``.aux.xml`` sidecar is
needed). Each output gets a ``.qml`` style and ``.qmd`` metadata sidecar (QGIS auto-loads
them); the GeoPackage also embeds its style and metadata internally so the file stays
self-describing. The citation, licence and rights are declared once, below, and written to
every output: GDAL ``-mo`` tags on the TIFFs, ``gpkg_metadata``/``gpkg_contents`` on the
GeoPackage, and matching fields in all three ``.qmd`` sidecars.
"""

import argparse
import json
import logging
import sqlite3
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Final
from xml.sax.saxutils import escape

import geopandas as gpd
import numpy as np
import pandas as pd
import pyogrio
import rasterio
import rasterio.shutil

from utils.env_capture import write_environment_snapshot
from utils.io import write_json
from utils.logging import make_run_logger
from utils.manifest import write_input_manifest
from utils.paths import ProjectPaths, ensure_writable_dirs, get_project_paths
from utils.terminology import NODATA

logger = logging.getLogger(__name__)

_LOGGER_NAME: Final[str] = "build_final_outputs"

# --- Citation and rights, shared by every output (GDAL -mo tags, gpkg_metadata and .qmd) ---
_AUTHORS: Final[str] = "Thomas Ratsakatika, Mihai Zotta, Srinivasan Keshav, Emily R. Lines"
_CONTACT: Final[str] = "trr26@cam.ac.uk"
_LICENCE: Final[str] = "CC-BY-NC-SA-4.0, https://creativecommons.org/licenses/by-nc-sa/4.0/"
_LICENCE_NAME: Final[str] = (
    "Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International"
)
_SOURCES: Final[str] = (
    "Predictors: TESSERA v2 embeddings, derived from modified Copernicus Sentinel data (2020); "
    "FABDEM (CC BY-NC-SA 4.0) and © OpenStreetMap contributors (ODbL). "
    "CLMS CORINE 2018 © European Union supplies the forest-type attribute only"
)
_DOI: Final[str] = "10.5281/zenodo.22693148"
_DOI_URL: Final[str] = f"https://doi.org/{_DOI}"
_CITATION: Final[str] = (
    "Ratsakatika, T., Zotta, M., Keshav, S., Lines, E.R., 2026. Old-growth forest reference "
    "labels and model predictions for the Făgăraș Mountains, Romania "
    f"(2020, 10 m) (v1.0.0) [dataset]. Zenodo. {_DOI_URL}"
)
_VERSION: Final[str] = "1.0.0"
_REFERENCE_YEAR: Final[str] = "2020"
_URL: Final[str] = "https://github.com/ratsakatika/detecting-old-growth-forests"
_COPYRIGHT: Final[str] = (
    "© 2026 T. Ratsakatika, M. Zotta, S. Keshav and E. R. Lines. CC BY-NC-SA 4.0."
)
_COPYRIGHT_VECTOR: Final[str] = (
    f"{_COPYRIGHT} Reference labels and geometries derived from data compiled by "
    "Fundația Conservation Carpathia."
)
# The parcel geometries are third-party; CC BY-NC-SA 4.0 covers only what the authors added.
_RIGHTS_NOTE: Final[str] = (
    "CC BY-NC-SA 4.0 applies to the authors' contributions (reference labels, predictions and "
    "compilation). Parcel geometries derive from public forest management plans and are "
    "redistributed as compiled by Fundația Conservation Carpathia; rights in the "
    "underlying plans remain with their holders."
)
_ACCESS_CONSTRAINT: Final[str] = "none"
_USE_CONSTRAINT: Final[str] = "attribution, non-commercial use, share alike (CC BY-NC-SA 4.0)"
_KEYWORDS: Final[tuple[str, ...]] = (
    "old-growth forest",
    "Carpathians",
    "Făgăraș",
    "Romania",
    "TESSERA",
)
_CATEGORIES: Final[tuple[str, ...]] = ("Biota", "Environment")
_TEMPORAL_START: Final[str] = f"{_REFERENCE_YEAR}-01-01T00:00:00Z"
_TEMPORAL_END: Final[str] = f"{_REFERENCE_YEAR}-12-31T23:59:59Z"
_EPSG: Final[int] = 3035
_CRS_AUTHID: Final[str] = f"EPSG:{_EPSG}"

_MOUNTAINS: Final[str] = "Făgăraș Mountains"
_TITLE_VECTOR: Final[str] = f"Old-growth forest labels and predictions, {_MOUNTAINS}"
_TITLE_PROB: Final[str] = f"Old-growth forest probability, {_MOUNTAINS}"
_TITLE_BINARY: Final[str] = f"Old-growth forest extent (binary), {_MOUNTAINS}"
# The descriptions double as the GDAL/gpkg_contents DESCRIPTION and the .qmd abstract.
_DESC_VECTOR: Final[str] = (
    "Forest parcels (n = {n:,}), " + _MOUNTAINS + ", Romania, EPSG:3035, with old-growth "
    "reference labels (ogf_reference_label) and parcel-level predictions (OGF_probability, "
    "OGF_binary); CORINE_forest_type from CLMS CORINE 2018."
)
_DESC_PROB: Final[str] = (
    f"Calibrated old-growth forest probability, {_MOUNTAINS}, Romania, 10 m, EPSG:3035, 2020. "
    "0–1; -9999 = no data."  # noqa: RUF001 - en dash is deliberate in the published range
)
_DESC_BINARY: Final[str] = (
    f"Binary old-growth forest map, {_MOUNTAINS}, Romania, 10 m, EPSG:3035, 2020. "
    "1 = old-growth, 0 = non-old-growth, 255 = no data."
)
_BAND_PROB: Final[str] = "OGF probability (calibrated)"
_BAND_BINARY: Final[str] = "OGF binary"

_STYLE_DIR: Final[Path] = Path("assets/qgis_styles")
_COMPARISON_GPKG: Final[str] = "product_parcel_comparison.gpkg"
_PROB_TIF: Final[str] = "ogf_probability_3035_10m.tif"
_BINARY_TIF: Final[str] = "ogf_binary_3035_10m.tif"
_VECTOR_OUT: Final[str] = "ogf_labels_predictions.gpkg"
_LAYER: Final[str] = "ogf_labels_predictions"

_LAYER_STYLES_DDL: Final[str] = """
CREATE TABLE IF NOT EXISTS layer_styles (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  f_table_catalog TEXT, f_table_schema TEXT, f_table_name TEXT,
  f_geometry_column TEXT, styleName TEXT, styleQML TEXT, styleSLD TEXT,
  useAsDefault BOOLEAN, description TEXT, owner TEXT, ui TEXT,
  update_time DATETIME DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ','now'))
);
"""


def _reference_label(value: object) -> str | None:
    """Map the boolean reference to 'ogf' / 'non_ogf' / NULL (unlabelled)."""
    if pd.isna(value):
        return None
    return "ogf" if bool(value) else "non_ogf"


def public_frame(comparison: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """Reorder and rename the comparison table into the public schema.

    The public table carries this study's own outputs only: the comparison products, the
    cross-study agreement, the dominant species and the altitude band are all dropped.
    """
    frame = comparison.copy()
    frame["parcel_id"] = frame["parcel_id"].astype("int64")
    frame["ogf_reference_label"] = frame["reference_label"].map(_reference_label)
    rename = {
        "corine_forest_type": "CORINE_forest_type",
        "ogf_ratsakatika_probability": "OGF_probability",
        "ogf_ratsakatika": "OGF_binary",
    }
    frame = frame.rename(columns=rename)
    # A parcel without a pixel centre has no probability, so its verdict is unknown, not 0:
    # upstream ``probability >= threshold`` compares NaN as False. Null it in the public file.
    frame["OGF_binary"] = (
        frame["OGF_binary"].astype("boolean").mask(frame["OGF_probability"].isna())
    )
    order = [
        "parcel_id",
        "ogf_reference_label",
        "CORINE_forest_type",
        "OGF_probability",
        "OGF_binary",
        frame.geometry.name,
    ]
    return frame[order]


def common_tags() -> dict[str, str]:
    """The citation/rights tags every output carries (GDAL ``-mo`` items, gpkg metadata)."""
    return {
        "AUTHOR": _AUTHORS,
        "CONTACT": _CONTACT,
        "LICENSE": _LICENCE,
        "SOURCES": _SOURCES,
        "DOI": _DOI,
        "CITATION": _CITATION,
        "VERSION": _VERSION,
        "REFERENCE_YEAR": _REFERENCE_YEAR,
        "URL": _URL,
    }


def threshold_note(threshold: float, *, output: str) -> str:
    """What the operating threshold means for one output, in plain terms, with its value.

    Args:
        threshold: The calibrated probability at or above which a pixel or parcel is old-growth.
        output: ``"probability"``, ``"binary"`` or ``"vector"``; selects the wording.
    """
    value = repr(float(threshold))
    chosen = (
        "The threshold gave the best balance of precision and recall (maximum F1) on held-out "
        "test parcels."
    )
    adjust = " Use a higher value for a stricter map, a lower one for a more inclusive map."
    notes = {
        "probability": (
            f"Pixels with a probability at or above {value} are old-growth in the binary map "
            "(ogf_binary_3035_10m.tif); below it, non-old-growth. " + chosen + adjust
        ),
        "binary": (
            "1 where the calibrated probability (ogf_probability_3035_10m.tif) is at or above "
            f"{value}, 0 below. " + chosen
        ),
        "vector": (
            f"OGF_binary is 1 where OGF_probability is at or above {value}, 0 below, and null "
            "where the parcel has no prediction (no pixel centre). " + chosen + adjust
        ),
    }
    return notes[output]


def threshold_tags(threshold: float, *, output: str) -> dict[str, str]:
    """The operating-threshold tags for one output: the value and :func:`threshold_note`."""
    return {
        "OGF_THRESHOLD": repr(float(threshold)),
        "OGF_THRESHOLD_NOTE": threshold_note(threshold, output=output),
    }


def _crs_block() -> str:
    """The ``<crs>`` element, so QGIS reads the extent in the layer's own projection."""
    crs = rasterio.crs.CRS.from_epsg(_EPSG)
    return (
        "  <crs>\n"
        '    <spatialrefsys nativeFormat="Wkt">\n'
        f"      <wkt>{escape(crs.to_wkt())}</wkt>\n"
        f"      <proj4>{escape(crs.to_proj4())}</proj4>\n"
        f"      <srid>{_EPSG}</srid>\n"
        f"      <authid>{_CRS_AUTHID}</authid>\n"
        "      <description>ETRS89-extended / LAEA Europe</description>\n"
        "      <projectionacronym>laea</projectionacronym>\n"
        "      <ellipsoidacronym>EPSG:7019</ellipsoidacronym>\n"
        "      <geographicflag>false</geographicflag>\n"
        "    </spatialrefsys>\n"
        "  </crs>\n"
    )


def qgis_metadata_xml(
    identifier: str,
    title: str,
    abstract: str,
    *,
    bounds: tuple[float, float, float, float],
    rights: str,
    use_constraint: str = _USE_CONSTRAINT,
) -> str:
    """A QGIS-native layer-metadata document (.qmd / gpkg_metadata) for one output.

    Args:
        identifier: Layer identifier (the file stem, or the GeoPackage layer name).
        title: Human-readable dataset title.
        abstract: Dataset description; the same text as the GDAL ``DESCRIPTION`` tag.
        bounds: The layer's own ``(minx, miny, maxx, maxy)`` extent, in EPSG:3035.
        rights: The copyright statement for this output.
        use_constraint: The use constraint; the vector adds the third-party rights note.

    Returns:
        The ``.qmd`` document as XML text.
    """
    minx, miny, maxx, maxy = bounds
    categories = "".join(f"    <keyword>{k}</keyword>\n" for k in _CATEGORIES)
    keywords = "".join(f"    <keyword>{k}</keyword>\n" for k in _KEYWORDS)
    return (
        "<!DOCTYPE qgis PUBLIC 'http://mrcc.com/qgis.dtd' 'SYSTEM'>\n"
        '<qgis version="3.44.10-Solothurn">\n'
        f"  <identifier>{escape(identifier)}</identifier>\n"
        f"  <parentidentifier>{_DOI_URL}</parentidentifier>\n"
        "  <language>ENG</language>\n"
        "  <type>dataset</type>\n"
        f"  <title>{escape(title)}</title>\n"
        f"  <abstract>{escape(abstract)}</abstract>\n"
        f'  <keywords vocabulary="gmd:topicCategory">\n{categories}  </keywords>\n'
        f'  <keywords vocabulary="keywords">\n{keywords}  </keywords>\n'
        "  <contact>\n"
        f"    <name>{escape(_AUTHORS)}</name>\n"
        "    <organization></organization>\n"
        "    <position></position>\n"
        "    <voice></voice>\n"
        "    <fax></fax>\n"
        f"    <email>{_CONTACT}</email>\n"
        "    <role>Author</role>\n"
        "  </contact>\n"
        "  <links>\n"
        f'    <link name="Source code" type="WWW:LINK" url="{_URL}" '
        'description="Analysis code" format="" mimeType="text/html" size=""/>\n'
        f'    <link name="DOI" type="WWW:LINK" url="{_DOI_URL}" '
        f'description="Zenodo record {_DOI}" format="" mimeType="text/html" size=""/>\n'
        "  </links>\n"
        f"  <history>Version {_VERSION}; reference year {_REFERENCE_YEAR}.</history>\n"
        f"  <history>Sources: {escape(_SOURCES)}</history>\n"
        f"  <history>Cite as: {escape(_CITATION)}</history>\n"
        "  <fees></fees>\n"
        f'  <constraints type="access">{_ACCESS_CONSTRAINT}</constraints>\n'
        f'  <constraints type="use">{escape(use_constraint)}</constraints>\n'
        f"  <rights>{escape(rights)}</rights>\n"
        f"  <license>{_LICENCE_NAME}</license>\n"
        "  <encoding></encoding>\n"
        f"{_crs_block()}"
        "  <extent>\n"
        f'    <spatial dimensions="2" minx="{minx:.4f}" miny="{miny:.4f}" maxx="{maxx:.4f}" '
        f'maxy="{maxy:.4f}" minz="0" maxz="0" crs="{_CRS_AUTHID}"/>\n'
        "    <temporal>\n"
        "      <period>\n"
        f"        <start>{_TEMPORAL_START}</start>\n"
        f"        <end>{_TEMPORAL_END}</end>\n"
        "      </period>\n"
        "    </temporal>\n"
        "  </extent>\n"
        "</qgis>\n"
    )


def _embed_gpkg_metadata(gpkg: Path, layer: str, qmd_xml: str) -> None:
    """Embed a QGIS-format metadata record in the GeoPackage metadata extension tables."""
    with sqlite3.connect(gpkg) as con:
        cursor = con.execute(
            "INSERT INTO gpkg_metadata (md_scope, md_standard_uri, mime_type, metadata) "
            "VALUES ('dataset', 'http://mrcc.com/qgis.dtd', 'text/xml', ?)",
            (qmd_xml,),
        )
        con.execute(
            "INSERT INTO gpkg_metadata_reference (reference_scope, table_name, md_file_id) "
            "VALUES ('table', ?, ?)",
            (layer, cursor.lastrowid),
        )


def _embed_gpkg_style(gpkg: Path, layer: str, qml_text: str) -> None:
    """Embed a QML style in the GeoPackage ``layer_styles`` table (QGIS loads it by default)."""
    with sqlite3.connect(gpkg) as con:
        con.executescript(_LAYER_STYLES_DDL)
        geom_col = con.execute(
            "SELECT column_name FROM gpkg_geometry_columns WHERE table_name = ?", (layer,)
        ).fetchone()
        con.execute(
            "INSERT INTO layer_styles (f_table_name, f_geometry_column, styleName, styleQML, "
            "useAsDefault, description, owner) VALUES (?, ?, ?, ?, 1, ?, ?)",
            (layer, geom_col[0] if geom_col else "geom", layer, qml_text, _TITLE_VECTOR, _AUTHORS),
        )


def _set_gpkg_contents_description(gpkg: Path, layer: str, description: str) -> None:
    """Set the layer's ``gpkg_contents`` description (what a GeoPackage reader shows first)."""
    with sqlite3.connect(gpkg) as con:
        con.execute(
            "UPDATE gpkg_contents SET description = ? WHERE table_name = ?", (description, layer)
        )


def write_vector(
    comparison_gpkg: Path, out_gpkg: Path, style_dir: Path, *, threshold: float
) -> None:
    """Write the public parcel GeoPackage with the style and metadata embedded and as sidecars.

    QGIS does not reliably auto-apply a GeoPackage's embedded ``layer_styles`` / ``gpkg_metadata``,
    so the ``.qml`` (style) and ``.qmd`` (metadata) sidecars are written alongside it; the style and
    metadata are also embedded so the file itself stays self-describing. ``threshold`` is the
    calibrated parcel probability at or above which ``OGF_binary`` is 1, recorded in the metadata.
    """
    public = public_frame(gpd.read_file(comparison_gpkg))
    description = _DESC_VECTOR.format(n=len(public))
    out_gpkg.unlink(missing_ok=True)
    pyogrio.write_dataframe(
        public,
        out_gpkg,
        layer=_LAYER,
        metadata={
            "TITLE": _TITLE_VECTOR,
            "DESCRIPTION": description,
            "COPYRIGHT": _COPYRIGHT_VECTOR,
            "RIGHTS_NOTE": _RIGHTS_NOTE,
            **common_tags(),
            **threshold_tags(threshold, output="vector"),
        },
    )
    _set_gpkg_contents_description(out_gpkg, _LAYER, description)
    qmd = qgis_metadata_xml(
        _LAYER,
        _TITLE_VECTOR,
        f"{description} {threshold_note(threshold, output='vector')}",
        bounds=tuple(public.total_bounds),
        rights=_COPYRIGHT_VECTOR,
        use_constraint=f"{_USE_CONSTRAINT}. {_RIGHTS_NOTE}",
    )
    _embed_gpkg_metadata(out_gpkg, _LAYER, qmd)
    out_gpkg.with_suffix(".qmd").write_text(qmd, "utf-8")
    qml = (style_dir / "ogf_labels_predictions.qml").read_text("utf-8")
    _embed_gpkg_style(out_gpkg, _LAYER, qml)
    out_gpkg.with_suffix(".qml").write_text(qml, "utf-8")


def write_raster(
    src_tif: Path,
    out_tif: Path,
    *,
    band_name: str,
    title: str,
    description: str,
    style_qml: Path,
    predictor: str,
    resampling: str,
    threshold: float,
    output: str,
    domain_mask: np.ndarray | None = None,
) -> None:
    """Write a GeoTIFF as a COG with band name, tags and embedded exact statistics.

    The COG driver only supports ``CreateCopy``, so the tagged band is written to a plain
    GeoTIFF first and then copied into place; ``STATISTICS=YES`` bakes the exact band
    statistics into the file itself, which is what keeps a ``.aux.xml`` from appearing.

    Args:
        src_tif: The mapping run's GeoTIFF to package.
        out_tif: Destination path in ``results/final/``.
        band_name: Band description.
        title: Dataset title (``TIFFTAG_DOCUMENTNAME`` / ``TITLE``).
        description: Dataset description; the ``.qmd`` abstract is this plus the
            threshold note.
        style_qml: QGIS style to copy alongside the raster.
        predictor: DEFLATE predictor - ``"2"`` for integer, ``"3"`` for floating point.
        resampling: Overview resampling - ``"NEAREST"`` for categorical, ``"AVERAGE"`` else.
        threshold: The calibrated pixel probability at or above which a pixel is old-growth,
            recorded in the tags with a plain-language note (see :func:`threshold_tags`).
        output: ``"probability"`` or ``"binary"``; selects the note's wording.
        domain_mask: Optional mask; pixels outside it are set to NoData.
    """
    with rasterio.open(src_tif) as src:
        data = src.read(1)
        profile = src.profile.copy()
        bounds = tuple(src.bounds)
    if domain_mask is not None:  # binary: keep data only where the model predicted
        data = np.where(domain_mask, data, profile["nodata"]).astype(profile["dtype"])
    tags = {
        "TIFFTAG_DOCUMENTNAME": title,
        "TIFFTAG_IMAGEDESCRIPTION": description,
        "TIFFTAG_ARTIST": _AUTHORS,
        "TIFFTAG_COPYRIGHT": _COPYRIGHT,
        "TITLE": title,
        "DESCRIPTION": description,
        **common_tags(),
        **threshold_tags(threshold, output=output),
    }
    staged = out_tif.with_suffix(".staging.tif")
    try:
        profile.update(compress="deflate")
        with rasterio.open(staged, "w", **profile) as dst:
            dst.write(data, 1)
            dst.set_band_description(1, band_name)
            dst.update_tags(**tags)
        rasterio.shutil.copy(
            staged,
            out_tif,
            driver="COG",
            COMPRESS="DEFLATE",
            PREDICTOR=predictor,
            RESAMPLING=resampling,
            STATISTICS="YES",
            BIGTIFF="IF_SAFER",
            NUM_THREADS="ALL_CPUS",
        )
    finally:
        staged.unlink(missing_ok=True)
    out_tif.with_name(f"{out_tif.name}.aux.xml").unlink(missing_ok=True)  # stats live in the COG
    out_tif.with_suffix(".qml").write_text(style_qml.read_text("utf-8"), "utf-8")
    out_tif.with_suffix(".qmd").write_text(
        qgis_metadata_xml(
            out_tif.stem,
            title,
            f"{description} {threshold_note(threshold, output=output)}",
            bounds=bounds,
            rights=_COPYRIGHT,
        ),
        "utf-8",
    )


def _latest_dir(root: Path, marker: str) -> Path:
    """Newest sub-directory of ``root`` containing ``marker``."""
    candidates = sorted(
        (p for p in root.iterdir() if p.is_dir() and (p / marker).is_file()), reverse=True
    )
    if not candidates:
        raise FileNotFoundError(f"no run under {root} with {marker}.")
    return candidates[0]


def _run(args: argparse.Namespace, paths: ProjectPaths, out_dir: Path) -> dict:
    run_logger = make_run_logger(_LOGGER_NAME, out_dir)
    style_dir = paths.repo_root / _STYLE_DIR
    comparison_run = _latest_dir(paths.results / "comparison", "overall_performance.csv")
    mapping_run = _latest_dir(paths.results / "mapping", "run_metadata.json")
    run_logger.info(
        "Packaging from comparison %s and mapping %s.", comparison_run.name, mapping_run.name
    )

    # The calibrated operating thresholds: the pixel raster binary is
    # ``ogf_probability_3035_10m >= pixel_threshold_calibrated`` and the parcel verdict
    # (``OGF_binary``) is ``OGF_probability >= parcel_threshold_calibrated``. Both go into
    # every output's metadata and into the run metadata below.
    mapping_meta = json.loads((mapping_run / "run_metadata.json").read_text("utf-8"))
    pixel_threshold = float(mapping_meta["pixel_threshold_calibrated"])
    parcel_threshold = float(mapping_meta["parcel_threshold_calibrated"])

    write_vector(
        comparison_run / _COMPARISON_GPKG,
        out_dir / _VECTOR_OUT,
        style_dir,
        threshold=parcel_threshold,
    )
    write_raster(
        mapping_run / _PROB_TIF,
        out_dir / _PROB_TIF,
        band_name=_BAND_PROB,
        title=_TITLE_PROB,
        description=_DESC_PROB,
        style_qml=style_dir / "ogf_probability.qml",
        predictor="3",  # floating-point predictor for the float32 probability
        resampling="AVERAGE",
        threshold=pixel_threshold,
        output="probability",
    )
    with rasterio.open(mapping_run / _PROB_TIF) as src:
        domain_mask = src.read(1) != NODATA  # the predicted parcel-map domain (within the AOI)
    write_raster(
        mapping_run / _BINARY_TIF,
        out_dir / _BINARY_TIF,
        band_name=_BAND_BINARY,
        title=_TITLE_BINARY,
        description=_DESC_BINARY,
        style_qml=style_dir / "ogf_binary.qml",
        predictor="2",  # standard predictor for the uint8 class map
        resampling="NEAREST",  # categorical: never average class codes
        threshold=pixel_threshold,
        output="binary",
        domain_mask=domain_mask,
    )
    run_logger.info("Wrote final outputs to %s.", out_dir)

    write_input_manifest(
        [comparison_run / _COMPARISON_GPKG, mapping_run / _PROB_TIF, mapping_run / _BINARY_TIF],
        out_dir / "final_outputs_manifest.json",
        repo_root=paths.repo_root,
    )
    write_environment_snapshot(out_dir / "final_outputs_env.txt")

    thresholds = {
        "pixel_threshold_calibrated": pixel_threshold,
        "parcel_threshold_calibrated": parcel_threshold,
    }
    return {
        "created_at": datetime.now(UTC).isoformat(),
        "script": "scripts/build_final_outputs.py",
        "comparison_run": comparison_run.name,
        "mapping_run": mapping_run.name,
        "author": _AUTHORS,
        "licence": _LICENCE,
        "citation": _CITATION,
        "doi": _DOI,
        "version": _VERSION,
        "outputs": [_VECTOR_OUT, _PROB_TIF, _BINARY_TIF],
        "calibrated_thresholds": thresholds,
    }


def _parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    """Build the public final outputs.

    Args:
        argv: Argument list, or ``None`` to read from ``sys.argv``.

    Returns:
        Process exit code (0 on success).
    """
    args = _parse_args(argv)
    paths = get_project_paths()
    ensure_writable_dirs(paths)
    out_dir = paths.results / "final"
    out_dir.mkdir(parents=True, exist_ok=True)
    summary = _run(args, paths, out_dir)
    write_json(summary, out_dir / "final_outputs_metadata.json")
    logger.info("Final outputs ready in %s.", out_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
