"""Aggregate the continuous existing-product surfaces to parcels by mean pixel value.

Some existing products publish a continuous 0-1 surface, not just a binary mask: Sabatini's
boosted-regression-tree old-growth probability and Munteanu's structural-complexity index. This
script takes each surface's mean pixel value within every parcel (the products' own scores, not
the thresholded masks), for the continuous-surface sensitivity check in notebook 011.

Aggregation matches the binary product masks: the denominator is every pixel inside the parcel
and NoData counts as 0 (non-old-growth), so a value exists for every parcel and partly-covered
parcels are not inflated by dropping their empty pixels.

Writes ``continuous_product_parcels.csv`` (parcel_id + one mean column per surface) into the
latest cross-study comparison run, alongside ``product_parcel_comparison.gpkg``.
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
from rasterio.enums import Resampling
from rasterio.features import rasterize

from utils.env_capture import write_environment_snapshot
from utils.io import write_json
from utils.logging import make_run_logger
from utils.manifest import write_input_manifest
from utils.metrics import parcel_aggregations
from utils.paths import ProjectPaths, get_project_paths
from utils.raster_io import ReferenceGrid, open_reference_grid, reproject_raster_to_ref
from utils.terminology import NODATA

logger = logging.getLogger(__name__)

_LOGGER_NAME: Final[str] = "build_continuous_product_parcels"
_PARCEL_TEMPLATE: Final[Path] = Path("data/processed/vectors/forest_records/parcel_map_clean.gpkg")
_SURFACES: Final[dict[str, Path]] = {
    "sabatini_probability": Path(
        "data/raw/existing_products/sabatini/1841_22_1841_2_pred_ogbrt_ssbtc5_lr02_March20.tif"
    ),
    "munteanu_complexity": Path(
        "data/raw/existing_products/munteanu/Romania_structural_complexity_index.tif"
    ),
}
_OUTPUT: Final[str] = "continuous_product_parcels.csv"


def parcel_id_grid(template: gpd.GeoDataFrame, ref: ReferenceGrid) -> npt.NDArray[np.int64]:
    """Rasterise the parcel template to a per-pixel parcel-id grid (-1 outside parcels)."""
    shapes = list(zip(template.geometry, template["parcel_id"].astype(np.int64), strict=True))
    burned = rasterize(shapes, out_shape=ref.shape, transform=ref.transform, fill=-1, dtype="int64")
    return burned.astype(np.int64)


def surface_parcel_means(
    surface_path: Path, ref: ReferenceGrid, parcel_grid: npt.NDArray[np.int64]
) -> pd.Series:
    """Mean of a continuous surface per parcel, with NoData counted as 0 (non-old-growth).

    The aggregation matches the binary product masks exactly: the denominator is every pixel
    inside the parcel, and out-of-coverage / NoData pixels count as 0 rather than being dropped.
    So a parcel only partly covered by the surface is not inflated by ignoring its empty pixels,
    and every parcel with at least one pixel gets a value (nearest resample preserves native
    values).
    """
    band = reproject_raster_to_ref(surface_path, ref, resampling=Resampling.nearest)[0]
    values = np.asarray(band, dtype=np.float64).reshape(-1)
    parcels = parcel_grid.reshape(-1)
    covered = np.isfinite(values) & (values != NODATA)
    values[~covered] = 0.0  # NA = 0 (non-old-growth), as in the binary masks; kept in denominator
    inside = parcels >= 0
    return parcel_aggregations(parcels[inside], values[inside]).set_index("parcel_id")["p_mean"]


def _latest_comparison(paths: ProjectPaths) -> Path:
    root = paths.results / "comparison"
    runs = sorted(
        (
            p
            for p in root.iterdir()
            if p.is_dir() and (p / "product_parcel_comparison.gpkg").is_file()
        ),
        reverse=True,
    )
    if not runs:
        raise FileNotFoundError(f"no comparison run with a gpkg under {root}.")
    return runs[0]


def _run(args: argparse.Namespace, paths: ProjectPaths, out_dir: Path) -> dict:
    run_logger = make_run_logger(_LOGGER_NAME, out_dir)
    ref = open_reference_grid()
    template = gpd.read_file(paths.repo_root / _PARCEL_TEMPLATE)[["parcel_id", "geometry"]]
    template["parcel_id"] = template["parcel_id"].astype("int64")
    parcel_grid = parcel_id_grid(template, ref)

    frame = template[["parcel_id"]].copy()
    for name, rel in _SURFACES.items():
        means = surface_parcel_means(paths.repo_root / rel, ref, parcel_grid)
        # Parcels too small to capture a pixel centre get no row above; count them as 0, so a
        # value exists for every parcel (identical to the binary masks' NA = 0 handling).
        frame[name] = frame["parcel_id"].map(means).fillna(0.0)
        run_logger.info(
            "Aggregated %s with NoData as 0: all %d parcels valued, %d non-zero.",
            name,
            len(frame),
            int((frame[name] > 0).sum()),
        )
    frame.to_csv(out_dir / _OUTPUT, index=False)
    run_logger.info("Wrote %s to %s.", _OUTPUT, out_dir)

    write_input_manifest(
        [paths.repo_root / _PARCEL_TEMPLATE, *(paths.repo_root / p for p in _SURFACES.values())],
        out_dir / "continuous_product_parcels_manifest.json",
        repo_root=paths.repo_root,
    )
    write_environment_snapshot(out_dir / "continuous_product_parcels_env.txt")
    return {
        "created_at": datetime.now(UTC).isoformat(),
        "script": "scripts/build_continuous_product_parcels.py",
        "comparison_run": out_dir.name,
        "surfaces": {name: str(rel) for name, rel in _SURFACES.items()},
        "n_parcels": int(len(frame)),
    }


def _parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    """Aggregate the continuous product surfaces to parcels.

    Args:
        argv: Argument list, or ``None`` to read from ``sys.argv``.

    Returns:
        Process exit code (0 on success).
    """
    args = _parse_args(argv)
    paths = get_project_paths()
    out_dir = _latest_comparison(paths)
    summary = _run(args, paths, out_dir)
    write_json(summary, out_dir / "continuous_product_parcels_metadata.json")
    logger.info("Continuous product parcels ready in %s.", out_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
