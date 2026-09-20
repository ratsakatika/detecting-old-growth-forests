"""Spatial-autocorrelation primitives: pixel sampling, variograms and Moran's I.

This module supplies the reusable building blocks for the Stage 5
autocorrelation analysis, which quantifies how far spatial dependence reaches in
the predictors and the reference labels and so justifies the spatial split and
the block bootstrap. Three primitives are exposed: a uniform valid-pixel sampler
(:func:`sample_raster_points`), an empirical-plus-fitted variogram over several
theoretical models (:func:`fit_variograms`), and a distance-band Moran's I
correlogram (:func:`moran_correlogram`). The variable orchestration, the
principal-component reduction of the embeddings, parallelism and file output all
live in the Stage 5 script, not here.

The analysis parameters (maximum lag, lag count, candidate models, estimator,
sample size, Moran distance thresholds and permutation count) are module-level
constants. They live here rather than in :mod:`utils.terminology` because they
are specific to this analysis; the random seed, by contrast, is the project-wide
seed imported from terminology so it cannot drift.

Distances are metres throughout: the reference grid and the labels are on
EPSG:3035 (ETRS89 / LAEA Europe), an equal-area metric CRS, so a Euclidean
separation between two sampled points is a true ground distance in metres.

Importing this module only binds names and a module logger: it performs no
input/output, configures no logging handlers and inspects no environment, in
keeping with the repository rule that ``utils`` modules have no side effects on
import (see AGENTS.md).
"""

import logging
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Final

import esda
import numpy as np
import pandas as pd
import rasterio
import skgstat
from libpysal.weights import DistanceBand
from numpy.typing import NDArray
from rasterio.transform import xy as transform_xy

from utils import terminology
from utils.terminology import SEED

# Module logger. No handlers are configured here; callers own logging setup.
logger = logging.getLogger(__name__)

# --------------------------------------------------------------------------- #
# Analysis parameters. Specific to the autocorrelation analysis, so they live
# here rather than in utils.terminology; the seed is imported from terminology.
# --------------------------------------------------------------------------- #

# Largest separation, in metres, at which the empirical variogram is evaluated.
VGRAM_MAXLAG_M: Final[float] = 40_000.0
# Number of equal-width lag classes. Forty 1 km bins at a 40 km maximum lag
# resolve the sub-5 km ranges the spatial split is sized against.
VGRAM_N_LAGS: Final[int] = 40
# Theoretical models fitted to each empirical variogram.
VGRAM_MODELS: Final[tuple[str, ...]] = ("spherical", "exponential", "gaussian")
# Empirical semivariance estimator (Matheron's classical estimator).
VGRAM_ESTIMATOR: Final[str] = "matheron"
# Estimate a nugget when fitting; when False the fitted nugget is forced to zero.
VGRAM_USE_NUGGET: Final[bool] = True
# Pixels drawn per raster group; one sample feeds the variogram, Moran and PCA.
N_SAMPLE: Final[int] = 5_000
# Distance-band radii, in metres, for the Moran's I correlogram.
MORAN_THRESHOLDS_M: Final[tuple[int, ...]] = (5_000, 10_000, 15_000, 20_000)
# Conditional-permutation count for Moran's I pseudo-significance.
MORAN_PERMUTATIONS: Final[int] = 999

# Fraction of the maximum lag at or above which a fitted range is treated as
# having run into the maximum-lag boundary (so the fit is unreliable).
_RANGE_HITS_MAXLAG_FRACTION: Final[float] = 0.995

# Fraction of the maximum lag at or above which a fitted range is treated as
# non-identifiable ("range not resolved") when *reporting* it. Looser than
# _RANGE_HITS_MAXLAG_FRACTION because a variable with no spatial structure lets
# the optimiser park the range anywhere near the ceiling, a few percent short.
_RANGE_UNRESOLVED_MAXLAG_FRACTION: Final[float] = 0.90


def range_unresolved(
    effective_range_m: NDArray[np.float64] | pd.Series | float,
    maxlag_m: NDArray[np.float64] | pd.Series | float,
    *,
    maxlag_fraction: float = _RANGE_UNRESOLVED_MAXLAG_FRACTION,
    n_lags: int = VGRAM_N_LAGS,
) -> NDArray[np.bool_]:
    """Flag fitted ranges that are not resolved within the analysed window.

    A variogram range is reported as "range not resolved" -- rather than as a
    number -- when the fit runs into either boundary of the empirical window:

    * **Upper boundary (ceiling).** The effective range is at or above
      ``maxlag_fraction * maxlag_m`` (default 90%, ~36 km at a 40 km max lag).
      The optimiser has parked the range near the maximum lag, so the "range" is
      a boundary artefact rather than a resolved correlation length. This happens
      either because the curve never reaches a sill within the window (e.g. the
      spherical/gaussian fits of a pure-nugget variable), or because a single
      theoretical structure cannot describe a multi-scale empirical variogram
      whose sill *is* inside the window (e.g. seasonal productivity, which
      plateaus by ~31 km but combines a steep short-range jump with a gentler
      rise that no single model captures).
    * **Lower boundary (floor).** The effective range is below the width of the
      first lag class (``maxlag_m / n_lags``, ~1 km here). Nothing constrains a
      range that falls inside the first bin -- there is no empirical semivariance
      at shorter lags -- so a pure-nugget variable whose exponential fit collapses
      here yields a "range" that is a lower-boundary artefact, not structure.

    A non-finite range (a failed fit) is also flagged. A high nugget-to-sill ratio
    was considered as a further criterion and dropped as too aggressive; the ratio
    remains available in the fits table for callers that want it.

    Args:
        effective_range_m: Fitted effective range(s), in metres. Scalar, array
            or :class:`pandas.Series`.
        maxlag_m: The maximum lag used for the fit, in metres (same shape or
            broadcastable).
        maxlag_fraction: Range at or above ``maxlag_fraction * maxlag_m`` counts
            as unresolved (upper boundary).
        n_lags: Number of empirical lag classes, used to derive the first-bin
            width ``maxlag_m / n_lags`` below which a range counts as unresolved
            (lower boundary). Pass explicitly to stay correct if the binning
            changes; scalar or broadcastable to ``maxlag_m``.

    Returns:
        A boolean array, ``True`` where the range is not resolved.
    """
    effective_range = np.asarray(effective_range_m, dtype=np.float64)
    maxlag = np.asarray(maxlag_m, dtype=np.float64)
    lag_width = maxlag / np.asarray(n_lags, dtype=np.float64)
    hits_ceiling = effective_range >= maxlag_fraction * maxlag
    below_floor = effective_range < lag_width
    return ~np.isfinite(effective_range) | hits_ceiling | below_floor


def sample_raster_points(
    raster_path: str | Path,
    *,
    bands: Sequence[int] = (1,),
    n: int = N_SAMPLE,
    seed: int = SEED,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Draw a uniform sample of valid pixels with their coordinates and values.

    The requested one-based bands are read in full (the stacks are ``float32``,
    so a whole-band read is affordable). A pixel is valid only where every
    requested band is finite and not equal to the NoData value; the NoData value
    is the raster's own NoData when set, otherwise the project sentinel
    :data:`utils.terminology.NODATA`. Up to ``n`` valid pixels are then drawn
    uniformly without replacement; if fewer than ``n`` are valid, all valid
    pixels are returned and a warning is logged.

    Args:
        raster_path: Path to the raster to sample.
        bands: One-based band indices to read. A pixel must be valid in every
            one of them. Defaults to the first band only.
        n: Maximum number of valid pixels to draw. Must be positive.
        seed: Seed for the ``numpy`` generator driving the draw.

    Returns:
        A ``(coords, values)`` pair. ``coords`` is an ``(m, 2)`` ``float64``
        array of pixel-centre ``(x, y)`` coordinates in the raster CRS, and
        ``values`` is an ``(m, len(bands))`` ``float64`` array of the sampled
        band values, where ``m`` is the number of pixels returned.

    Raises:
        ValueError: If ``n`` is not positive.
    """
    if n <= 0:
        raise ValueError(f"n must be positive, got {n}.")

    band_indices = [int(band) for band in bands]
    with rasterio.open(raster_path) as src:
        transform = src.transform
        file_nodata = src.nodata
        data = src.read(indexes=band_indices)  # (n_bands, height, width), native dtype.

    nodata_value = float(file_nodata) if file_nodata is not None else float(terminology.NODATA)

    # Validity intersected band by band, so only one band-sized boolean array is
    # held at a time (the read array itself is the large allocation the docstring
    # sanctions).
    valid = np.ones(data.shape[1:], dtype=bool)
    for band in data:
        valid &= np.isfinite(band) & (band != nodata_value)

    rows, cols = np.nonzero(valid)
    n_valid = int(rows.size)
    rng = np.random.default_rng(seed)
    selection: NDArray[np.int64]
    if n_valid <= n:
        if n_valid < n:
            logger.warning(
                "Only %d valid pixels in %s across bands %s; requested %d, returning all.",
                n_valid,
                raster_path,
                band_indices,
                n,
            )
        selection = np.arange(n_valid, dtype=np.int64)
    else:
        selection = rng.choice(n_valid, size=n, replace=False)

    sel_rows = rows[selection]
    sel_cols = cols[selection]
    xs, ys = transform_xy(transform, sel_rows, sel_cols, offset="center")
    coords = np.column_stack((np.asarray(xs, dtype=np.float64), np.asarray(ys, dtype=np.float64)))
    values = np.asarray(data[:, sel_rows, sel_cols].T, dtype=np.float64)
    return coords, values


@dataclass(frozen=True, slots=True)
class VariogramResult:
    """Empirical variogram with one fitted model per candidate theoretical model.

    The empirical (experimental) variogram is computed once and shared across
    models; only the theoretical fit differs between models. The record is
    frozen: a result describes one variable and is never mutated.

    Attributes:
        fits: One row per model, with columns ``model``, ``effective_range_m``,
            ``sill``, ``nugget``, ``nugget_sill_ratio``, ``rmse`` and
            ``range_hits_maxlag``. ``sill`` is skgstat's fitted sill -- the
            *partial* sill (the rise above the nugget); the curve plateaus at
            ``nugget + sill``. ``nugget_sill_ratio`` is therefore
            ``nugget / (nugget + sill)``, bounded in ``[0, 1]``. A model whose fit
            failed carries ``NaN`` in the numeric columns and ``False`` in
            ``range_hits_maxlag``.
        lags_m: Empirical bin centres in metres, length :attr:`n_lags`.
        semivariance: Experimental semivariance aligned to :attr:`lags_m`.
        curve_lag_m: A fine ``0..maxlag`` grid for plotting fitted curves.
        fitted: Mapping of model name to fitted semivariance on
            :attr:`curve_lag_m`.
        n: Number of points the variogram was built from.
        maxlag_m: Maximum lag distance in metres.
        n_lags: Number of empirical lag classes.
    """

    fits: pd.DataFrame
    lags_m: NDArray[np.float64]
    semivariance: NDArray[np.float64]
    curve_lag_m: NDArray[np.float64]
    fitted: dict[str, NDArray[np.float64]]
    n: int
    maxlag_m: float
    n_lags: int


# Column order for the per-model fits table (also used to type an empty frame).
_FIT_COLUMNS: Final[tuple[str, ...]] = (
    "model",
    "effective_range_m",
    "sill",
    "nugget",
    "nugget_sill_ratio",
    "rmse",
    "range_hits_maxlag",
)


def _bin_centres(bin_edges: NDArray[np.float64]) -> NDArray[np.float64]:
    """Return lag-class centres from skgstat's upper bin edges.

    ``skgstat`` reports the upper bound of each lag class. The lower bound of the
    first class is 0 and of every later class is the previous upper bound, so the
    centre of each class is the midpoint of its lower and upper bounds. This is
    robust to uneven binning, not only the default even bins.

    Args:
        bin_edges: Upper bound of each lag class, ascending.

    Returns:
        The centre of each lag class.
    """
    lower_edges = np.concatenate(([0.0], bin_edges[:-1]))
    return (lower_edges + bin_edges) / 2.0


def fit_variograms(
    coords: NDArray[np.float64],
    values: NDArray[np.float64],
    *,
    models: Sequence[str] = VGRAM_MODELS,
    n_lags: int = VGRAM_N_LAGS,
    maxlag: float = VGRAM_MAXLAG_M,
    estimator: str = VGRAM_ESTIMATOR,
    use_nugget: bool = VGRAM_USE_NUGGET,
    n_curve: int = 100,
) -> VariogramResult:
    """Fit several theoretical variogram models sharing one empirical variogram.

    A single :class:`skgstat.Variogram` is built with the first model, which
    performs the costly pairwise-binning step once. Each further model is fitted
    by reassigning ``variogram.model`` and re-reading the parameters, so the
    binned experimental variogram is reused rather than recomputed. Should a
    given model's fit fail, that model contributes a ``NaN`` row rather than
    aborting the whole call.

    Args:
        coords: ``(m, 2)`` array of point coordinates in a metric CRS.
        values: Length-``m`` array of the variable's values at ``coords``.
        models: Theoretical models to fit.
        n_lags: Number of empirical lag classes.
        maxlag: Maximum lag distance in metres.
        estimator: Empirical semivariance estimator name.
        use_nugget: When ``True``, fit a nugget; when ``False``, the fitted
            nugget is forced to zero.
        n_curve: Number of points on the fine plotting grid.

    Returns:
        A :class:`VariogramResult` carrying the shared empirical variogram and
        each model's fit and fitted curve.

    Raises:
        ValueError: If ``coords`` and ``values`` differ in length, or if there
            are fewer than ``n_lags + 2`` points to fit.
    """
    coords = np.asarray(coords, dtype=np.float64)
    values = np.asarray(values, dtype=np.float64).reshape(-1)

    if len(coords) != len(values):
        raise ValueError(
            f"coords and values must have equal length, got {len(coords)} and {len(values)}."
        )
    minimum_points = n_lags + 2
    if len(values) < minimum_points:
        raise ValueError(
            f"need at least n_lags + 2 = {minimum_points} points to fit a variogram, "
            f"got {len(values)}."
        )

    curve_lag_m = np.linspace(0.0, float(maxlag), int(n_curve), dtype=np.float64)

    # Build the experimental variogram once with the first model; reassigning the
    # model below reuses this binning and only re-fits the theoretical model.
    variogram = skgstat.Variogram(
        coords,
        values,
        model=models[0],
        n_lags=n_lags,
        maxlag=maxlag,
        normalize=False,
        estimator=estimator,
        use_nugget=use_nugget,
    )
    lags_m = _bin_centres(np.asarray(variogram.bins, dtype=np.float64))
    semivariance = np.asarray(variogram.experimental, dtype=np.float64)

    range_threshold = float(maxlag) * _RANGE_HITS_MAXLAG_FRACTION
    rows: list[dict[str, object]] = []
    fitted: dict[str, NDArray[np.float64]] = {}
    for model in models:
        try:
            variogram.model = model
            description = variogram.describe()
            effective_range = float(description["effective_range"])
            # skgstat's "sill" is the partial sill (the rise above the nugget);
            # the curve plateaus at nugget + sill. Take the ratio over that total
            # so it is bounded in [0, 1] rather than the unbounded nugget/partial.
            sill = float(description["sill"])
            nugget = float(description["nugget"])
            total_sill = nugget + sill
            ratio = nugget / total_sill if total_sill > 0 else float("nan")
            curve = np.asarray(variogram.fitted_model(curve_lag_m), dtype=np.float64)
            rows.append(
                {
                    "model": model,
                    "effective_range_m": effective_range,
                    "sill": sill,
                    "nugget": nugget,
                    "nugget_sill_ratio": ratio,
                    "rmse": float(variogram.rmse),
                    "range_hits_maxlag": effective_range >= range_threshold,
                }
            )
            fitted[model] = curve
        except Exception as error:
            logger.warning(
                "Variogram fit failed for model %r: %s; recording NaN row.", model, error
            )
            rows.append(
                {
                    "model": model,
                    "effective_range_m": float("nan"),
                    "sill": float("nan"),
                    "nugget": float("nan"),
                    "nugget_sill_ratio": float("nan"),
                    "rmse": float("nan"),
                    "range_hits_maxlag": False,
                }
            )
            fitted[model] = np.full(int(n_curve), np.nan, dtype=np.float64)

    fits = pd.DataFrame(rows, columns=list(_FIT_COLUMNS))
    fits["range_hits_maxlag"] = fits["range_hits_maxlag"].astype(bool)

    return VariogramResult(
        fits=fits,
        lags_m=lags_m,
        semivariance=semivariance,
        curve_lag_m=curve_lag_m,
        fitted=fitted,
        n=int(len(values)),
        maxlag_m=float(maxlag),
        n_lags=int(n_lags),
    )


# Column order for the Moran correlogram table (also types an empty frame).
_MORAN_COLUMNS: Final[tuple[str, ...]] = (
    "morans_i",
    "expected_i",
    "z_sim",
    "p_sim",
    "p_norm",
    "n",
    "mean_neighbours",
)


def _run_moran(
    values: NDArray[np.float64],
    weights: DistanceBand,
    *,
    permutations: int,
    seed: int,
) -> esda.Moran:
    """Run Moran's I with a reproducible conditional-permutation draw.

    The locked ``esda`` (2.8.2) draws its conditional permutations from NumPy's
    global generator and exposes no ``seed`` argument, so the global generator
    state is set for the call and restored immediately afterwards. This makes
    ``p_sim`` reproducible without leaking the seed to any surrounding code.

    Args:
        values: The variable's values, one per spatial unit.
        weights: The binary distance-band spatial weights.
        permutations: Number of conditional permutations.
        seed: Seed for the permutation draw.

    Returns:
        The fitted :class:`esda.Moran` instance.
    """
    state = np.random.get_state()
    try:
        np.random.seed(seed)
        return esda.Moran(values, weights, permutations=permutations)
    finally:
        np.random.set_state(state)


def moran_correlogram(
    coords: NDArray[np.float64],
    values: NDArray[np.float64],
    *,
    thresholds_m: Sequence[int] = MORAN_THRESHOLDS_M,
    permutations: int = MORAN_PERMUTATIONS,
    seed: int = SEED,
) -> pd.DataFrame:
    """Compute Moran's I at a series of distance-band thresholds.

    For each threshold a binary distance-band weights graph is built and Moran's
    I is computed with conditional-permutation inference. Both the
    permutation-based pseudo p-value (``p_sim``) and the analytical
    normality-assumption p-value (``p_norm``) are reported. A threshold whose
    weights graph has no neighbours at all (typically one below the minimum
    pairwise distance) is logged and omitted.

    Args:
        coords: ``(m, 2)`` array of point coordinates in a metric CRS.
        values: Length-``m`` array of the variable's values at ``coords``.
        thresholds_m: Distance-band radii in metres.
        permutations: Number of conditional permutations per threshold.
        seed: Seed for the permutation draw.

    Returns:
        A :class:`pandas.DataFrame` indexed by ``threshold_m`` with columns
        ``morans_i``, ``expected_i``, ``z_sim``, ``p_sim``, ``p_norm``, ``n`` and
        ``mean_neighbours``. Skipped thresholds do not appear.
    """
    coords = np.asarray(coords, dtype=np.float64)
    values = np.asarray(values, dtype=np.float64).reshape(-1)

    records: list[dict[str, float]] = []
    index: list[int] = []
    for threshold in thresholds_m:
        weights = DistanceBand.from_array(
            coords, threshold=float(threshold), binary=True, silence_warnings=True
        )
        if weights.s0 == 0:
            logger.warning(
                "Distance band at %d m has no neighbours; skipping this threshold.",
                int(threshold),
            )
            continue

        moran = _run_moran(values, weights, permutations=permutations, seed=seed)
        cardinalities = np.fromiter(weights.cardinalities.values(), dtype=np.float64)
        records.append(
            {
                "morans_i": float(moran.I),
                "expected_i": float(moran.EI),
                "z_sim": float(moran.z_sim),
                "p_sim": float(moran.p_sim),
                "p_norm": float(moran.p_norm),
                "n": int(weights.n),
                "mean_neighbours": float(cardinalities.mean()),
            }
        )
        index.append(int(threshold))

    frame = pd.DataFrame.from_records(
        records, index=pd.Index(index, name="threshold_m", dtype="int64")
    )
    return frame.reindex(columns=list(_MORAN_COLUMNS))


__all__ = [
    "MORAN_PERMUTATIONS",
    "MORAN_THRESHOLDS_M",
    "N_SAMPLE",
    "VGRAM_ESTIMATOR",
    "VGRAM_MAXLAG_M",
    "VGRAM_MODELS",
    "VGRAM_N_LAGS",
    "VGRAM_USE_NUGGET",
    "VariogramResult",
    "fit_variograms",
    "moran_correlogram",
    "range_unresolved",
    "sample_raster_points",
]
