"""Reusable predictor transforms carrying a science decision or correctness rule.

This module holds only the small set of predictor transforms that encode a
modelling decision or a correctness constraint: Horn slope and aspect, the
McCune & Keon Heat Load Index, a per-pixel latitude grid for that index, and
float32 stack assembly. Per-source raw-to-processed orchestration (mosaicking,
reprojecting and clipping each input, downscaling climate, aligning the learned
embeddings and building the four feature-set stacks) is done cell by cell in
``notebooks/002`` using these functions together with :mod:`utils.raster_io`.
Distance-to-roads is :func:`utils.raster_io.distance_to_lines_on_ref`, not here.
Roughness and TPI are intentionally excluded to avoid neighbourhood spatial
leakage; any useful neighbourhood signal is left to the receptive-field CNN.

Terrain workflow (important). Terrain derivatives are computed at the DEM's native
resolution in a metric CRS -- the project CRS, EPSG:3035 (see
:data:`utils.terminology.CRS`), at roughly 30 m for FABDEM -- and only then are
the continuous products resampled to the 10 m reference grid. The reasons are
twofold. First, resampling elevation to 10 m before differencing manufactures
detail the 30 m source does not contain and biases slope; differencing must
happen at the native posting. Second, raw aspect must never be resampled, because
its 0/360 degree wraparound corrupts any interpolation. The functions here
therefore operate on a single native-resolution array; resampling of their
outputs onto the reference grid happens afterwards in ``notebooks/002`` via
:mod:`utils.raster_io`.

All raster arrays are 2-D float. Missing data is marked with
:data:`utils.terminology.NODATA` on input and output; internally the functions use
NaN and convert at the boundaries. Importing this module only binds names and a
module logger: it performs no input/output, configures no logging handlers and
inspects no environment, in keeping with the repository rule that ``utils``
modules have no side effects on import (see AGENTS.md).
"""

import logging
from collections.abc import Sequence
from pathlib import Path
from typing import Final

import numpy as np
from affine import Affine
from pyproj import Transformer

from utils import raster_io, terminology

# Module logger. No handlers are configured here; callers own logging setup.
logger = logging.getLogger(__name__)

# WGS84 geographic CRS: the natural CRS for latitude in degrees. This is not the
# project (metric) CRS; it is inherent to expressing a latitude.
_GEOGRAPHIC_CRS: Final[str] = "EPSG:4326"


def _to_nan(array: np.ndarray, nodata: float) -> np.ndarray:
    """Return a float64 copy of ``array`` with ``nodata`` cells set to NaN.

    The input is never mutated; existing NaNs are preserved.
    """
    out = np.array(array, dtype=np.float64)
    out[out == nodata] = np.nan
    return out


def _horn_gradients(
    elevation: np.ndarray, x_res: float, y_res: float
) -> tuple[np.ndarray, np.ndarray]:
    """Compute Horn (1981) 8-neighbour gradients, NaN where they are undefined.

    ``elevation`` is expected to carry NaN for missing cells. The returned
    ``dz/dx`` and ``dz/dy`` arrays match the input shape and are NaN on the array
    border (no full 3x3 window) and wherever any cell of the 3x3 window -- the
    centre included -- is missing.
    """
    height, width = elevation.shape
    dzdx = np.full((height, width), np.nan, dtype=np.float64)
    dzdy = np.full((height, width), np.nan, dtype=np.float64)
    if height < 3 or width < 3:
        return dzdx, dzdy

    # 3x3 Horn window labels:  a b c / d e f / g h i  (e is the centre cell).
    a = elevation[0:-2, 0:-2]
    b = elevation[0:-2, 1:-1]
    c = elevation[0:-2, 2:]
    d = elevation[1:-1, 0:-2]
    e = elevation[1:-1, 1:-1]
    f = elevation[1:-1, 2:]
    g = elevation[2:, 0:-2]
    h = elevation[2:, 1:-1]
    i = elevation[2:, 2:]

    dzdx_int = ((c + 2.0 * f + i) - (a + 2.0 * d + g)) / (8.0 * x_res)
    dzdy_int = ((g + 2.0 * h + i) - (a + 2.0 * b + c)) / (8.0 * y_res)

    # Arithmetic above propagates NaN from the eight neighbours; the centre cell
    # is not in either formula, so propagate its missingness explicitly.
    centre_missing = ~np.isfinite(e)
    dzdx_int = np.where(centre_missing, np.nan, dzdx_int)
    dzdy_int = np.where(centre_missing, np.nan, dzdy_int)

    dzdx[1:-1, 1:-1] = dzdx_int
    dzdy[1:-1, 1:-1] = dzdy_int
    return dzdx, dzdy


def horn_slope(
    elevation: np.ndarray,
    *,
    x_res: float,
    y_res: float,
    nodata: float = terminology.NODATA,
) -> np.ndarray:
    """Compute slope in degrees by Horn's (1981) 8-neighbour method.

    The gradients over the 3x3 window ``a b c / d e f / g h i`` are
    ``dz/dx = ((c + 2f + i) - (a + 2d + g)) / (8 * x_res)`` and
    ``dz/dy = ((g + 2h + i) - (a + 2b + c)) / (8 * y_res)``, and the slope is
    ``degrees(atan(hypot(dz/dx, dz/dy)))``. Pixel sizes are in the same metric
    units as the elevation values (metres), so the slope is correct only when the
    array is at its native resolution in a metric CRS.

    Cells on the array border (no full 3x3 window) and cells whose 3x3 window
    touches a missing cell are set to ``nodata``. Input ``nodata`` cells are
    treated as missing.

    Args:
        elevation: 2-D elevation array at native resolution.
        x_res: West-east pixel size in metres (positive).
        y_res: North-south pixel size in metres (positive).
        nodata: Sentinel marking missing input and output cells.

    Returns:
        A 2-D float64 slope array in degrees, with ``nodata`` where undefined.
    """
    elev = _to_nan(elevation, nodata)
    dzdx, dzdy = _horn_gradients(elev, x_res, y_res)

    slope = np.degrees(np.arctan(np.hypot(dzdx, dzdy)))
    undefined = ~np.isfinite(dzdx) | ~np.isfinite(dzdy)
    return np.where(undefined, nodata, slope)


def horn_aspect(
    elevation: np.ndarray,
    *,
    x_res: float,
    y_res: float,
    nodata: float = terminology.NODATA,
    flat_value: float = -1.0,
) -> np.ndarray:
    """Compute aspect in degrees (geographic convention) from Horn gradients.

    Aspect is the downslope facing direction measured clockwise from north
    (0 = North, 90 = East, 180 = South, 270 = West), in ``[0, 360)``, derived
    from the same Horn ``dz/dx`` and ``dz/dy`` as :func:`horn_slope` via
    ``degrees(atan2(-dz/dx, dz/dy))``. Flat cells (zero gradient) are set to
    ``flat_value``. The exact ``flat_value`` is immaterial to
    :func:`heat_load_index`, whose aspect terms all carry ``sin(slope)`` and so
    vanish where the slope is zero.

    Cells on the array border and cells whose 3x3 window touches a missing cell
    are set to ``nodata``; input ``nodata`` cells are treated as missing (the same
    propagation as :func:`horn_slope`).

    Args:
        elevation: 2-D elevation array at native resolution.
        x_res: West-east pixel size in metres (positive).
        y_res: North-south pixel size in metres (positive).
        nodata: Sentinel marking missing input and output cells.
        flat_value: Value assigned to flat (zero-gradient) cells.

    Returns:
        A 2-D float64 aspect array in degrees, with ``flat_value`` on flat cells
        and ``nodata`` where undefined.
    """
    elev = _to_nan(elevation, nodata)
    dzdx, dzdy = _horn_gradients(elev, x_res, y_res)

    magnitude = np.hypot(dzdx, dzdy)
    aspect = np.degrees(np.arctan2(-dzdx, dzdy))
    aspect = np.where(aspect < 0.0, aspect + 360.0, aspect)
    aspect = np.where(magnitude == 0.0, flat_value, aspect)
    undefined = ~np.isfinite(dzdx) | ~np.isfinite(dzdy)
    return np.where(undefined, nodata, aspect)


def heat_load_index(
    slope_deg: np.ndarray,
    aspect_deg: np.ndarray,
    latitude_deg: np.ndarray,
    *,
    nodata: float = terminology.NODATA,
) -> np.ndarray:
    """Compute the per-pixel heat load index (McCune & Keon 2002).

    Reference: McCune, B. & Keon, D. (2002), "Equations for potential annual
    direct incident radiation and heat load", Journal of Vegetation Science 13,
    603-606, doi:10.1111/j.1654-1103.2002.tb02087.x.

    This implements their equation 1 (Table 2, p. 605): four parameters,
    dependent variable ln(Rad), valid for latitudes 0-60 degrees N and slopes
    0-90 degrees, adjusted R-squared 0.958.

        HLI = exp( -1.467 + 1.582*cos(L)*cos(S) - 1.500*cos(F)*sin(S)*sin(L)
                - 0.262*sin(L)*sin(S) + 0.607*sin(F)*sin(S) )

    Output is ``nodata`` wherever slope, aspect or latitude is ``nodata`` or NaN.

    Args:
        slope_deg: 2-D slope array in degrees.
        aspect_deg: 2-D aspect array in degrees (geographic convention).
        latitude_deg: 2-D per-pixel latitude array in degrees north (positive).
        nodata: Sentinel marking missing input and output cells.

    Returns:
        A 2-D float64 heat load index array (unitless), with ``nodata`` where
        undefined.
    """
    slope = _to_nan(slope_deg, nodata)
    aspect = _to_nan(aspect_deg, nodata)
    latitude = _to_nan(latitude_deg, nodata)

    reduced = ((aspect - 225.0 + 180.0) % 360.0) - 180.0
    folded = 180.0 - np.abs(reduced)

    big_s = np.radians(slope)
    big_f = np.radians(folded)
    big_l = np.radians(latitude)

    exponent = (
        -1.467
        + 1.582 * np.cos(big_l) * np.cos(big_s)
        - 1.500 * np.cos(big_f) * np.sin(big_s) * np.sin(big_l)
        - 0.262 * np.sin(big_l) * np.sin(big_s)
        + 0.607 * np.sin(big_f) * np.sin(big_s)
    )
    hli = np.exp(exponent)

    undefined = ~np.isfinite(slope) | ~np.isfinite(aspect) | ~np.isfinite(latitude)
    return np.where(undefined, nodata, hli)


def latitude_grid(transform: Affine, crs: object, width: int, height: int) -> np.ndarray:
    """Build a per-pixel latitude grid in degrees for the Heat Load Index.

    Each cell centre is taken in ``crs`` (the native-resolution metric grid),
    transformed to WGS84 geographic coordinates with a
    :class:`pyproj.Transformer` (``always_xy=True``), and its latitude returned.
    Per-pixel latitude is used deliberately rather than a single AOI-mean
    constant, so the index varies correctly across the extent.

    Args:
        transform: Affine transform of the native-resolution grid.
        crs: CRS of that grid -- anything :class:`pyproj.Transformer` accepts
            (a pyproj or rasterio CRS, an EPSG code or an authority string).
        width: Grid width in pixels.
        height: Grid height in pixels.

    Returns:
        A ``(height, width)`` float64 array of latitudes in degrees.
    """
    cols, rows = np.meshgrid(np.arange(width) + 0.5, np.arange(height) + 0.5)
    xs = transform.a * cols + transform.b * rows + transform.c
    ys = transform.d * cols + transform.e * rows + transform.f

    transformer = Transformer.from_crs(crs, _GEOGRAPHIC_CRS, always_xy=True)
    _, latitudes = transformer.transform(xs, ys)
    return np.asarray(latitudes, dtype=np.float64)


def assemble_stack(
    path: Path,
    bands: Sequence[np.ndarray],
    ref: raster_io.ReferenceGrid,
    *,
    band_names: Sequence[str],
    nodata: float = terminology.NODATA,
    logger: logging.Logger | None = None,
) -> Path:
    """Stack aligned single-band arrays into one float32 GeoTIFF on the grid.

    Every array must already be aligned to ``ref`` (same shape); this function
    does not resample. Arrays are cast to float32, NaN cells are written as the
    ``nodata`` sentinel, and the write is delegated to
    :func:`utils.raster_io.write_geotiff` with ``band_names`` recorded as the band
    descriptions so the band order is stored in the file. Band names are supplied
    by the caller (sourced from terminology in the notebook); this function does
    not invent names or a band-order constant.

    Args:
        path: Output GeoTIFF path.
        bands: Single-band 2-D arrays, each of shape ``ref.shape``.
        ref: Reference grid the stack is written on.
        band_names: One name per band, in band order.
        nodata: NoData sentinel written for missing and NaN cells.
        logger: Logger passed through to the write; defaults to the module logger.

    Returns:
        The output path.

    Raises:
        ValueError: If ``len(bands) != len(band_names)`` or any array's shape does
            not match ``ref.shape``.
    """
    log = logger if logger is not None else logging.getLogger(__name__)

    if len(bands) != len(band_names):
        raise ValueError(
            f"bands has {len(bands)} array(s) but band_names has {len(band_names)} "
            "name(s); they must correspond one-to-one."
        )

    stack = np.empty((len(bands), ref.shape[0], ref.shape[1]), dtype=np.float32)
    for index, band in enumerate(bands):
        array = np.asarray(band, dtype=np.float32)
        if array.shape != ref.shape:
            raise ValueError(
                f"Band {index} ('{band_names[index]}') has shape {array.shape}, which "
                f"does not match the reference grid shape {ref.shape}."
            )
        stack[index] = np.where(np.isnan(array), np.float32(nodata), array)

    return raster_io.write_geotiff(
        path,
        stack,
        ref,
        dtype="float32",
        nodata=nodata,
        band_descriptions=list(band_names),
        logger=log,
    )


__all__ = [
    "assemble_stack",
    "heat_load_index",
    "horn_aspect",
    "horn_slope",
    "latitude_grid",
]
