"""Build the coordinate-only control's raster stack (easting, northing).

The coordinate-only control is a spatial null model: an ordinary run of the pipeline in
which the predictors are replaced by the two pixel-centre coordinates. The XGBoost version
(``scripts/run_coordinate_xgboost.py``) synthesises those two columns in memory, but the CNN
path reads patches from a raster stack, so the control needs a real stack on disk like any
other feature set.

This writes ``data/processed/rasters/stacks_10m/xy_coords_3035_10m.tif``: a two-band float32
GeoTIFF, co-registered to the project reference grid, holding the easting and northing of
each cell centre in metres. The values reproduce
:func:`scripts.run_coordinate_xgboost.coordinate_features` exactly -- same affine evaluation at
the cell centre and the same per-axis shift -- so the CNN and the XGBoost control see
identical coordinates and their results stay comparable.

The shift subtracts the minimum over the *modelled pixels* (the pixel index), not over the
whole grid, because that is what the XGBoost control does; cells outside the pixel index can
therefore go slightly negative, which is immaterial (a per-axis shift is monotonic, and the
CNN standardises its inputs).

Once written, the ordinary chain works unchanged::

    python -m scripts.build_grid_memmap --feature-sets xy_coords
    python -m scripts.build_pixel_table --feature-set xy_coords
    python -m scripts.run_nested_cv --feature-set xy_coords --architecture cnn_3x3 ...
"""

import argparse
import logging
from collections.abc import Sequence
from pathlib import Path
from typing import Final

import numpy as np
import pandas as pd
import rasterio

from utils.logging import make_run_logger
from utils.paths import ProjectPaths, ensure_writable_dirs, get_project_paths
from utils.raster_io import open_reference_grid
from utils.terminology import MODEL_INPUT_BANDS, NODATA

logger = logging.getLogger(__name__)

_FEATURE_SET: Final[str] = "xy_coords"
_STACK_RELPATH: Final[Path] = Path("rasters") / "stacks_10m" / f"{_FEATURE_SET}_3035_10m.tif"
_PIXEL_INDEX: Final[Path] = Path("main_nested_cv") / "pixel_index.parquet"
_BLOCK_ROWS: Final[int] = 512  # write in row blocks so the full grid is never materialised
# The control must model the same pixels as the headline stack (run_coordinate_xgboost
# uses this set's validity mask too), so its NoData is stamped into the coordinates.
_DEFAULT_VALID_FROM: Final[str] = "baseline_tessera"


def _pixel_index_offsets(paths: ProjectPaths, width: int) -> tuple[np.ndarray, np.ndarray]:
    """Row and column indices of the modelled pixels, decoded from their pixel ids."""
    index_path = paths.results / _PIXEL_INDEX
    if not index_path.is_file():
        raise FileNotFoundError(
            f"Missing {index_path}; the pixel index defines which cells the control models."
        )
    pixel_ids = pd.read_parquet(index_path, columns=["pixel_id"])["pixel_id"].to_numpy(np.int64)
    return pixel_ids // width, pixel_ids % width


def _axis_origins(transform, rows: np.ndarray, cols: np.ndarray) -> tuple[float, float]:
    """The easting/northing minima over the modelled pixels (the shift applied to each axis).

    Mirrors :func:`scripts.run_coordinate_xgboost.coordinate_features`, which shifts each axis
    by its minimum across the pixel index.
    """
    centre_col = cols + 0.5
    centre_row = rows + 0.5
    easting = transform.a * centre_col + transform.b * centre_row + transform.c
    northing = transform.d * centre_col + transform.e * centre_row + transform.f
    return float(easting.min()), float(northing.min())


def _invalid_cells(
    paths: ProjectPaths, valid_from: str, rows: np.ndarray, cols: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Row/col of the modelled pixels that ``valid_from`` marks invalid.

    Coordinates are defined at every cell, so an unmasked stack would make the control
    valid on 100% of the pixel index while the modelling stacks are not. The control must
    see exactly the pixels the models see, so the reference set's mask is stamped into the
    stack as NoData; both the pixel table and the CNN patch cache then inherit it.
    """
    valid_path = paths.cache / "pixel_features" / f"{valid_from}_valid.parquet"
    if not valid_path.is_file():
        raise FileNotFoundError(
            f"Missing {valid_path}; build the {valid_from!r} pixel table first "
            "(it defines which pixels the control is allowed to model)."
        )
    valid = pd.read_parquet(valid_path)["valid"].to_numpy()
    if len(valid) != len(rows):
        raise ValueError(
            f"{valid_from} validity has {len(valid)} rows but the pixel index has {len(rows)}."
        )
    return rows[~valid], cols[~valid]


def build_coordinate_stack(
    paths: ProjectPaths, *, valid_from: str = _DEFAULT_VALID_FROM, force: bool = False
) -> Path:
    """Write the two-band coordinate stack and return its path."""
    grid = open_reference_grid()
    height, width = int(grid.height), int(grid.width)
    transform = grid.transform
    out_path = paths.processed / _STACK_RELPATH
    if out_path.is_file() and not force:
        logger.info("Coordinate stack already present: %s (use --force to rebuild).", out_path)
        return out_path
    out_path.parent.mkdir(parents=True, exist_ok=True)

    rows, cols = _pixel_index_offsets(paths, width)
    east0, north0 = _axis_origins(transform, rows, cols)
    mask_rows, mask_cols = _invalid_cells(paths, valid_from, rows, cols)
    logger.info(
        "Masking %d cell(s) that %s marks invalid, so the control models the same pixels.",
        len(mask_rows),
        valid_from,
    )
    logger.info(
        "Grid %dx%d; shifting easting by %.1f m and northing by %.1f m (pixel-index minima).",
        height,
        width,
        east0,
        north0,
    )

    profile = {
        "driver": "GTiff",
        "height": height,
        "width": width,
        "count": len(MODEL_INPUT_BANDS[_FEATURE_SET]),
        "dtype": "float32",
        "crs": grid.crs,
        "transform": transform,
        "nodata": NODATA,
        "tiled": True,
        "blockxsize": 256,
        "blockysize": 256,
        "compress": "deflate",
        "predictor": 2,
    }
    all_cols = np.arange(width, dtype=np.float64) + 0.5
    with rasterio.open(out_path, "w", **profile) as dst:
        for name, i in zip(MODEL_INPUT_BANDS[_FEATURE_SET], (1, 2), strict=True):
            dst.set_band_description(i, name)
        for row0 in range(0, height, _BLOCK_ROWS):
            block_rows = np.arange(row0, min(row0 + _BLOCK_ROWS, height), dtype=np.float64) + 0.5
            # Outer sums: every (row, col) centre in this block, via the affine transform.
            easting = (
                transform.a * all_cols[None, :] + transform.b * block_rows[:, None] + transform.c
            ) - east0
            northing = (
                transform.d * all_cols[None, :] + transform.e * block_rows[:, None] + transform.f
            ) - north0
            # Stamp NoData at the reference set's invalid cells falling in this row block.
            in_block = (mask_rows >= row0) & (mask_rows < row0 + len(block_rows))
            if in_block.any():
                local = mask_rows[in_block] - row0
                easting[local, mask_cols[in_block]] = NODATA
                northing[local, mask_cols[in_block]] = NODATA
            window = rasterio.windows.Window(0, row0, width, len(block_rows))
            dst.write(easting.astype(np.float32), 1, window=window)
            dst.write(northing.astype(np.float32), 2, window=window)
    logger.info("Wrote coordinate stack %s.", out_path)
    return out_path


def _parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--valid-from",
        default=_DEFAULT_VALID_FROM,
        help="Feature set whose validity mask the control inherits (its NoData cells "
        "are stamped into the coordinates so both model the same pixels).",
    )
    parser.add_argument("--force", action="store_true", help="Rebuild even if the stack exists.")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    """Build the coordinate stack.

    Args:
        argv: Argument list, or ``None`` to read from ``sys.argv``.

    Returns:
        Process exit code (0 on success).
    """
    args = _parse_args(argv)
    paths = get_project_paths()
    ensure_writable_dirs(paths)
    global logger
    logger = make_run_logger(__name__, paths.processed / _STACK_RELPATH.parent)
    build_coordinate_stack(paths, valid_from=args.valid_from, force=args.force)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
