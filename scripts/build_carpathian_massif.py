"""Extract and save the Carpathian massif polygon for the stage-020 AOA analysis.

Selects the largest ``m_massive == "carp"`` feature of the EEA European mountain
massifs layer (downloaded by notebook 001; the smaller ``carp`` features are
sub-ranges nested inside it), reprojects it to the project CRS and saves it to
the processed location read by every Carpathian download script and the AOA
computation (``utils.carpathians.CARPATHIANS_MASSIF_RELATIVE``).

Idempotent: an existing output is validated and kept unless ``--force``. Run::

    .venv/bin/python -m scripts.build_carpathian_massif
"""

import argparse
import logging
from collections.abc import Sequence
from pathlib import Path

from utils.carpathians import (
    carpathians_massif_path,
    extract_carpathian_massif,
    load_carpathian_massif,
)
from utils.env_capture import write_environment_snapshot
from utils.logging import make_run_logger
from utils.manifest import write_input_manifest, write_json
from utils.paths import ProjectPaths, ensure_writable_dirs, get_project_paths
from utils.vector_io import audit_vector

logger = logging.getLogger(__name__)

# Relative geometry-area tolerance against the source ``area_km2`` attribute.
_AREA_TOLERANCE = 0.02


def _parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source",
        type=Path,
        default=None,
        help="Massifs source layer (default: data/raw/vectors/.../m_massifs_v1.shp).",
    )
    parser.add_argument(
        "--force", action="store_true", help="Re-extract even when the output exists."
    )
    return parser.parse_args(argv)


def run(args: argparse.Namespace, paths: ProjectPaths) -> int:
    """Execute the extraction; returns a process exit code."""
    source = (
        Path(args.source)
        if args.source
        else paths.raw / "vectors" / "european_mountain_areas" / "m_massifs_v1.shp"
    )
    out_path = carpathians_massif_path(paths.repo_root)
    global logger
    logger = make_run_logger(__name__, out_path.parent)

    if out_path.exists() and not args.force:
        massif = load_carpathian_massif(out_path, logger=logger)
        logger.info("Present: %s", audit_vector(out_path))
    else:
        massif = extract_carpathian_massif(source, logger=logger)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        massif.to_file(out_path, driver="GPKG")
        logger.info("Wrote: %s", audit_vector(out_path))

    area_attr = float(massif["area_km2"].iloc[0])
    area_geom = float(massif.geometry.area.iloc[0]) / 1e6
    logger.info("Area: attribute %.0f km2, geometry %.0f km2", area_attr, area_geom)
    if abs(area_attr - area_geom) / area_attr > _AREA_TOLERANCE:
        raise ValueError(
            f"Geometry area ({area_geom:.0f} km2) deviates more than "
            f"{_AREA_TOLERANCE:.0%} from the source attribute ({area_attr:.0f} km2)."
        )

    write_input_manifest(
        [source], out_path.parent / "input_manifest.json", repo_root=paths.repo_root
    )
    write_environment_snapshot(out_path.parent / "env_snapshot.txt")
    write_json(
        {
            "output": out_path.name,
            "source": str(source.relative_to(paths.repo_root)),
            "area_km2_attribute": area_attr,
            "area_km2_geometry": round(area_geom, 1),
        },
        out_path.parent / "run_metadata.json",
    )
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    """Command-line entry point."""
    args = _parse_args(argv)
    paths = get_project_paths()
    ensure_writable_dirs(paths)
    return run(args, paths)


if __name__ == "__main__":
    raise SystemExit(main())
