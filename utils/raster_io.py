"""Raster input/output helpers anchored to the project reference grid.

This module is the dependency root of 002 notebook pre-processing: the reference-grid
build and every later raster loader call into it. It provides a lightweight
:class:`ReferenceGrid` geometry template, reprojection and resampling onto that
grid, GeoTIFF writing, mask and distance rasterisation, block iteration and a
structured :class:`RasterAudit`.

Spatial constants are never hard-coded here: the project CRS, reference-grid
resolution and the NoData sentinel are read from :mod:`utils.terminology`, and
the default reference-grid path is resolved through :mod:`utils.paths`. The
reference grid is EPSG:3035 (ETRS89 / LAEA Europe), an equal-area metric CRS, so
distances measured on it are in metres.

Importing this module only binds names and a module logger: it performs no
input/output, configures no logging handlers and inspects no environment, in
keeping with the repository rule that ``utils`` modules have no side effects on
import (see AGENTS.md).
"""

import logging
from collections.abc import Iterable, Iterator, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Final

import numpy as np
import rasterio
from affine import Affine
from rasterio.coords import BoundingBox
from rasterio.crs import CRS
from rasterio.enums import Resampling
from rasterio.features import rasterize
from rasterio.transform import array_bounds
from rasterio.warp import reproject
from rasterio.windows import Window
from scipy import ndimage

from utils import terminology
from utils.paths import get_project_paths

# Module logger. No handlers are configured here; callers own logging setup.
logger = logging.getLogger(__name__)

# Tolerance, in metres, for accepting a grid's pixel size as the reference
# resolution and as near-square. A reprojected 10 m grid is exact to far better
# than this, while a genuinely wrong grid (e.g. 30 m) is rejected.
_PIXEL_TOLERANCE_M: Final[float] = 1e-3


@dataclass(frozen=True, slots=True)
class ReferenceGrid:
    """Geometry template for the project reference raster grid.

    A :class:`ReferenceGrid` carries only georeferencing: it has no value
    semantics and no NoData field. Loaders use it to guarantee that every
    processed raster shares an identical CRS, transform, width and height so that
    a pixel index maps to the same ground location across every layer.

    Attributes:
        crs: Coordinate reference system (the project CRS, EPSG:3035).
        transform: Affine transform mapping pixel (col, row) to map coordinates.
        width: Grid width in pixels (columns).
        height: Grid height in pixels (rows).
    """

    crs: CRS
    transform: Affine
    width: int
    height: int

    @property
    def resolution(self) -> tuple[float, float]:
        """Absolute pixel size as ``(x_resolution, y_resolution)`` in metres."""
        return (abs(self.transform.a), abs(self.transform.e))

    @property
    def shape(self) -> tuple[int, int]:
        """Array shape as ``(height, width)``, matching ``numpy`` convention."""
        return (self.height, self.width)

    @property
    def bounds(self) -> BoundingBox:
        """Spatial extent as ``(left, bottom, right, top)`` in the grid CRS."""
        left, bottom, right, top = array_bounds(self.height, self.width, self.transform)
        return BoundingBox(left, bottom, right, top)

    def profile(
        self,
        count: int,
        dtype: str,
        nodata: float = terminology.NODATA,
        compress: str = "deflate",
    ) -> dict[str, object]:
        """Build a rasterio dataset profile aligned to this grid.

        Args:
            count: Number of bands.
            dtype: NumPy dtype name for the dataset (e.g. ``"float32"``).
            nodata: NoData sentinel; defaults to the project float sentinel.
            compress: GeoTIFF compression; ``"deflate"`` by default.

        Returns:
            A profile dictionary with driver ``GTiff``, this grid's
            CRS/transform/width/height and tiled storage, ready to pass to
            :func:`rasterio.open` in write mode.
        """
        return {
            "driver": "GTiff",
            "crs": self.crs,
            "transform": self.transform,
            "width": self.width,
            "height": self.height,
            "count": count,
            "dtype": dtype,
            "nodata": nodata,
            "compress": compress,
            "tiled": True,
        }


@dataclass(frozen=True, slots=True)
class BandStats:
    """Summary statistics for a single raster band over its valid pixels.

    Attributes:
        index: One-based band index.
        minimum: Smallest valid value.
        maximum: Largest valid value.
        mean: Mean of the valid values.
        std: Population standard deviation of the valid values.
        valid_count: Number of unmasked (valid) pixels.
        nodata_count: Number of pixels masked as NoData or NaN.
    """

    index: int
    minimum: float
    maximum: float
    mean: float
    std: float
    valid_count: int
    nodata_count: int


@dataclass(frozen=True, slots=True)
class RasterAudit:
    """Structured metadata (and optional per-band statistics) for a raster.

    The audit holds metadata always; per-band statistics are populated only when
    requested. It performs no logging itself: notebooks print ``str(audit)`` on
    load and long-running scripts pass the same data to ``logger.info``.

    Attributes:
        path: Path to the audited raster.
        crs: Coordinate reference system.
        transform: Affine transform.
        resolution: Absolute pixel size ``(x, y)`` in CRS units.
        width: Width in pixels.
        height: Height in pixels.
        count: Number of bands.
        dtype: NumPy dtype name of the dataset.
        nodata: Per-band NoData values from the dataset (``None`` where unset).
        bounds: Spatial extent ``(left, bottom, right, top)``.
        band_stats: Per-band statistics, or ``None`` when not computed.
    """

    path: Path
    crs: CRS
    transform: Affine
    resolution: tuple[float, float]
    width: int
    height: int
    count: int
    dtype: str
    nodata: tuple[float | None, ...]
    bounds: BoundingBox
    band_stats: tuple[BandStats, ...] | None

    def __str__(self) -> str:
        """Return a single readable summary line for the raster."""
        epsg = self.crs.to_epsg() if self.crs is not None else None
        crs_label = f"EPSG:{epsg}" if epsg is not None else "undefined CRS"
        res_x, res_y = self.resolution
        summary = (
            f"{self.path.name}: {crs_label} {self.width}x{self.height} "
            f"{self.count} band(s) {self.dtype} "
            f"res={res_x:g}x{res_y:g}m nodata={self.nodata}"
        )
        if self.band_stats is None:
            return summary
        stats = "; ".join(
            f"b{s.index}[min={s.minimum:g} max={s.maximum:g} mean={s.mean:g} "
            f"std={s.std:g} valid={s.valid_count} nodata={s.nodata_count}]"
            for s in self.band_stats
        )
        return f"{summary} | {stats}"


def open_reference_grid(path: Path | None = None) -> ReferenceGrid:
    """Open the reference raster and build its :class:`ReferenceGrid`.

    Only georeferencing is read; band data is never loaded. The grid is
    validated against the project conventions: its CRS must equal the project
    CRS and its pixels must be square and at the reference resolution.

    Args:
        path: Reference raster path. Defaults to
            ``get_project_paths().reference_grid``.

    Returns:
        The validated :class:`ReferenceGrid`.

    Raises:
        ValueError: If the CRS does not equal the project CRS, or the pixel size
            is not the reference resolution within tolerance, or pixels are not
            near-square.
    """
    grid_path = get_project_paths().reference_grid if path is None else Path(path)

    with rasterio.open(grid_path) as src:
        crs = src.crs
        transform = src.transform
        width = src.width
        height = src.height

    project_crs = CRS.from_user_input(terminology.CRS)
    if crs is None or crs != project_crs:
        found = crs.to_string() if crs is not None else "undefined"
        raise ValueError(
            f"Reference grid {grid_path} has CRS {found} but the project CRS is "
            f"{terminology.CRS}. Reproject the grid to the project CRS before "
            "using it as the reference grid."
        )

    res_x = abs(transform.a)
    res_y = abs(transform.e)
    target = terminology.REF_RESOLUTION_M
    if (
        abs(res_x - target) > _PIXEL_TOLERANCE_M
        or abs(res_y - target) > _PIXEL_TOLERANCE_M
        or abs(res_x - res_y) > _PIXEL_TOLERANCE_M
    ):
        raise ValueError(
            f"Reference grid {grid_path} has pixel size {res_x} x {res_y} m but "
            f"square {target} m pixels are required. Resample the grid to the "
            "reference resolution before using it as the reference grid."
        )

    return ReferenceGrid(crs=crs, transform=transform, width=width, height=height)


def reproject_raster_to_ref(
    src_path: Path,
    ref: ReferenceGrid,
    *,
    resampling: Resampling = Resampling.bilinear,
    src_nodata: float | None = None,
    dst_nodata: float = terminology.NODATA,
    dtype: str | None = None,
    num_threads: int | None = None,
    logger: logging.Logger | None = None,
) -> np.ndarray:
    """Reproject and resample a source raster onto the reference grid.

    The output shares the reference grid's CRS, transform, width and height
    exactly. An identity-CRS source is still resampled onto the grid; the
    operation is always logged.

    Args:
        src_path: Path to the source raster.
        ref: Target reference grid.
        resampling: Resampling method; bilinear by default.
        src_nodata: Source NoData. When ``None``, read from the file (and may
            itself be ``None`` if unset there).
        dst_nodata: Value filling cells outside the source coverage; defaults to
            the project float sentinel.
        dtype: Output dtype name; defaults to the source dtype.
        num_threads: Threads for the warp. ``None`` lets rasterio decide. For
            batch pre-processing pass ``os.cpu_count()`` so this composes safely
            inside later parallel code.
        logger: Logger for the one INFO line; defaults to the module logger.

    Returns:
        A 3-D array of shape ``(count, height, width)`` on the reference grid.

    Raises:
        ValueError: If the source raster has no CRS.
    """
    log = logger if logger is not None else logging.getLogger(__name__)

    with rasterio.open(src_path) as src:
        src_crs = src.crs
        if src_crs is None:
            raise ValueError(
                f"Source raster {src_path} has no CRS; cannot reproject onto the "
                "reference grid. Assign the correct CRS at source first."
            )
        count = src.count
        out_dtype = dtype if dtype is not None else src.dtypes[0]
        if src_nodata is None:
            src_nodata = src.nodata
        source = src.read()
        src_transform = src.transform

    if source.dtype != np.dtype(out_dtype):
        source = source.astype(out_dtype)

    destination = np.full((count, ref.height, ref.width), dst_nodata, dtype=out_dtype)

    reproject_kwargs: dict[str, object] = {
        "source": source,
        "destination": destination,
        "src_transform": src_transform,
        "src_crs": src_crs,
        "src_nodata": src_nodata,
        "dst_transform": ref.transform,
        "dst_crs": ref.crs,
        "dst_nodata": dst_nodata,
        "resampling": resampling,
    }
    if num_threads is not None:
        reproject_kwargs["num_threads"] = num_threads

    reproject(**reproject_kwargs)

    log.info(
        "Reprojecting raster onto reference grid: source CRS %s -> target %s, "
        "resampling=%s, input=%s",
        src_crs.to_string(),
        f"EPSG:{ref.crs.to_epsg()}",
        resampling.name,
        src_path,
    )

    return destination


def write_geotiff(
    path: Path,
    data: np.ndarray,
    ref: ReferenceGrid,
    *,
    dtype: str | None = None,
    nodata: float = terminology.NODATA,
    compress: str = "deflate",
    band_descriptions: Sequence[str] | None = None,
    logger: logging.Logger | None = None,
) -> Path:
    """Write a tiled, compressed GeoTIFF aligned to the reference grid.

    Args:
        path: Output path. The parent directory is created if absent.
        data: Array of shape ``(count, H, W)`` or ``(H, W)`` for a single band.
        ref: Reference grid the array must match spatially.
        dtype: Output dtype name; defaults to the array dtype.
        nodata: NoData sentinel; defaults to the project float sentinel.
        compress: GeoTIFF compression; ``"deflate"`` by default.
        band_descriptions: Optional per-band descriptions set in order. The
            canonical stack band order is supplied by the caller from
            terminology.
        logger: Logger for the one INFO line; defaults to the module logger.

    Returns:
        The output path.

    Raises:
        ValueError: If ``data`` is not 2-D or 3-D, if its spatial shape does not
            match ``ref.shape``, or if ``band_descriptions`` is given with a
            length other than the band count.
    """
    log = logger if logger is not None else logging.getLogger(__name__)

    array = np.asarray(data)
    if array.ndim == 2:
        array = array[np.newaxis, :, :]
    elif array.ndim != 3:
        raise ValueError(
            f"Expected a 2-D (H, W) or 3-D (count, H, W) array, got {array.ndim} dimensions."
        )

    count, height, width = array.shape
    if (height, width) != ref.shape:
        raise ValueError(
            f"Array spatial shape {(height, width)} does not match the reference "
            f"grid shape {ref.shape}."
        )
    if band_descriptions is not None and len(band_descriptions) != count:
        raise ValueError(
            f"band_descriptions has {len(band_descriptions)} entries but the array "
            f"has {count} band(s)."
        )

    out_dtype = dtype if dtype is not None else array.dtype.name
    profile = ref.profile(count=count, dtype=out_dtype, nodata=nodata, compress=compress)
    # Switch to BigTIFF automatically only once a file would exceed the 4 GB
    # classic-TIFF limit; smaller files stay classic TIFF and byte-identical.
    profile["bigtiff"] = "IF_SAFER"

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with rasterio.open(path, "w", **profile) as dst:
        dst.write(array.astype(out_dtype, copy=False))
        if band_descriptions is not None:
            for index, description in enumerate(band_descriptions, start=1):
                dst.set_band_description(index, description)

    log.info(
        "Wrote GeoTIFF %s: %d band(s), dtype=%s, nodata=%s",
        path,
        count,
        out_dtype,
        nodata,
    )
    return path


def rasterize_mask(
    geometries: Iterable[object],
    ref: ReferenceGrid,
    *,
    all_touched: bool = False,
    fill: int = 0,
    value: int = 1,
    dtype: str = "uint8",
) -> np.ndarray:
    """Burn geometries onto the reference grid, returning a 2-D mask.

    The geometries must already be in ``ref.crs``; this function does not
    reproject (``vector_io`` owns vector CRS handling).

    Args:
        geometries: An iterable of shapely geometries or a ``geopandas.GeoSeries``.
        ref: Reference grid to burn onto.
        all_touched: If ``True``, burn every pixel the geometry touches; if
            ``False``, only pixels whose centre is within the geometry.
        fill: Background value for unburned pixels.
        value: Value burned where geometries cover a pixel.
        dtype: Output dtype name.

    Returns:
        A 2-D array of shape ``ref.shape``.
    """
    shapes = [(geometry, value) for geometry in geometries]
    return rasterize(
        shapes,
        out_shape=ref.shape,
        transform=ref.transform,
        fill=fill,
        all_touched=all_touched,
        dtype=dtype,
    )


def distance_to_lines_on_ref(
    lines: Iterable[object],
    ref: ReferenceGrid,
    *,
    all_touched: bool = True,
    logger: logging.Logger | None = None,
) -> np.ndarray:
    """Compute metres to the nearest line for every pixel of the reference grid.

    The lines are rasterised onto the grid, then a Euclidean distance transform
    is scaled by the pixel size (per-axis sampling ``(y_resolution,
    x_resolution)``). Cells lying on a line are 0 m. This is metre-correct
    because the reference grid is an equal-area metric CRS (EPSG:3035) at the
    reference resolution. The lines must already be in ``ref.crs``.

    Args:
        lines: An iterable of shapely line geometries or a ``geopandas.GeoSeries``.
        ref: Reference grid to compute distances on.
        all_touched: If ``True``, burn every pixel a line touches; ``True`` by
            default so thin lines are not missed.
        logger: Logger for the one INFO line; defaults to the module logger.

    Returns:
        A 2-D ``float32`` array of metres to the nearest line.
    """
    log = logger if logger is not None else logging.getLogger(__name__)

    burned = rasterize_mask(lines, ref, all_touched=all_touched, fill=0, value=1, dtype="uint8")
    res_x, res_y = ref.resolution
    distance = ndimage.distance_transform_edt(burned == 0, sampling=(res_y, res_x))

    log.info(
        "Computed distance-to-lines on reference grid (%dx%d, %g m pixels)",
        ref.width,
        ref.height,
        res_x,
    )
    return distance.astype(np.float32)


def tiled_block_iterator(
    ref: ReferenceGrid,
    *,
    block_size: int = 2048,
) -> Iterator[Window]:
    """Yield windows tiling the whole reference grid.

    Windows cover every pixel exactly once with no overlap, including partial
    windows at the right and bottom edges.

    Args:
        ref: Reference grid to tile.
        block_size: Side length, in pixels, of the interior (non-edge) blocks.

    Yields:
        :class:`rasterio.windows.Window` tiles in row-major order.
    """
    for row_off in range(0, ref.height, block_size):
        block_height = min(block_size, ref.height - row_off)
        for col_off in range(0, ref.width, block_size):
            block_width = min(block_size, ref.width - col_off)
            yield Window(col_off, row_off, block_width, block_height)


def audit_raster(path: Path, *, with_stats: bool = False) -> RasterAudit:
    """Open a raster and return its :class:`RasterAudit`.

    Metadata is always populated. When ``with_stats`` is ``True`` per-band
    minimum, maximum, mean, standard deviation, valid count and NoData count are
    computed, masking the band NoData and any NaN. This function performs no
    logging; callers print ``str(audit)`` in notebooks or pass it to
    ``logger.info`` in scripts.

    Args:
        path: Path to the raster.
        with_stats: If ``True``, compute per-band statistics.

    Returns:
        The :class:`RasterAudit`, with ``band_stats`` set only when requested.
    """
    path = Path(path)
    with rasterio.open(path) as src:
        crs = src.crs
        transform = src.transform
        width = src.width
        height = src.height
        count = src.count
        dtype = src.dtypes[0]
        nodata = tuple(src.nodatavals)
        bounds = src.bounds

        band_stats: tuple[BandStats, ...] | None = None
        if with_stats:
            collected: list[BandStats] = []
            for index in range(1, count + 1):
                band = src.read(index, masked=True)
                valid = np.ma.masked_invalid(band).compressed()
                valid_count = int(valid.size)
                nodata_count = int(band.size - valid_count)
                if valid_count > 0:
                    minimum = float(valid.min())
                    maximum = float(valid.max())
                    mean = float(valid.mean())
                    std = float(valid.std())
                else:
                    minimum = maximum = mean = std = float("nan")
                collected.append(
                    BandStats(
                        index=index,
                        minimum=minimum,
                        maximum=maximum,
                        mean=mean,
                        std=std,
                        valid_count=valid_count,
                        nodata_count=nodata_count,
                    )
                )
            band_stats = tuple(collected)

    return RasterAudit(
        path=path,
        crs=crs,
        transform=transform,
        resolution=(abs(transform.a), abs(transform.e)),
        width=width,
        height=height,
        count=count,
        dtype=dtype,
        nodata=nodata,
        bounds=bounds,
        band_stats=band_stats,
    )


__all__ = [
    "BandStats",
    "RasterAudit",
    "ReferenceGrid",
    "audit_raster",
    "distance_to_lines_on_ref",
    "open_reference_grid",
    "rasterize_mask",
    "reproject_raster_to_ref",
    "tiled_block_iterator",
    "write_geotiff",
]
