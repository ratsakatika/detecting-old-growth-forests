"""Spatial partitioning of parcels into cross-validation folds and bootstrap units.

Cross-validation folds and block-bootstrap units are the same problem: cut the
study area into spatially coherent, mass-balanced groups of parcels. This module
builds that partition once in :func:`assign_spatial_partition` and exposes a thin
:func:`assign_folds` wrapper for the training folds. It also provides the
reversible pixel-id encoding, the connected-components grouping, and cross-unit
distance diagnostics.

The partition lays the units out on an ``(n_cols, n_rows)`` grid. For a candidate
grid shape and rotation angle the unit centroids are ordered west to east and a
dynamic program cuts them into mass-balanced columns; each column is then ordered
south to north and cut into mass-balanced rows. The configuration (factorisation
and angle) with the lowest balance objective wins, and the cells are relabelled
``1..n_units``. Primary mass defaults to parcel area; an optional secondary mass
(old-growth area, or an old-growth count or area for prevalence) is balanced
either toward its equal share or toward the global ratio.

Importing this module only binds names; it performs no input or output and
mutates no global state.
"""

from __future__ import annotations

import logging
import math
from pathlib import Path
from typing import TYPE_CHECKING, Literal

import geopandas as gpd
import joblib
import numpy as np
import pandas as pd
import shapely
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components as _scipy_connected_components

from utils.terminology import FOLD_IDS, N_FOLDS, SEED

if TYPE_CHECKING:
    from collections.abc import Iterable, Sequence

    from numpy.typing import ArrayLike, NDArray

__all__ = [
    "assign_folds",
    "assign_spatial_partition",
    "connected_components",
    "cross_unit_distances",
    "decode_pixel_id",
    "encode_pixel_id",
    "load_fold_assignment",
]

logger = logging.getLogger(__name__)

_VECTOR_SUFFIXES = frozenset({".gpkg", ".geojson", ".json", ".shp", ".fgb"})
_DEFAULT_ROTATIONS: tuple[float, ...] = tuple(float(d) for d in range(0, 180, 5))


def encode_pixel_id(row: ArrayLike, col: ArrayLike, width: int) -> int | NDArray[np.int64]:
    """Encode ``(row, col)`` grid positions into a single int64 pixel id.

    The encoding is ``pixel_id = row * width + col`` and is exactly reversible by
    :func:`decode_pixel_id` for the same ``width``.

    Args:
        row: Zero-based row index, a scalar or a NumPy array.
        col: Zero-based column index, a scalar or a NumPy array.
        width: Number of columns in the grid; must be positive.

    Returns:
        A Python ``int`` when both inputs are scalars, otherwise an int64 array of
        the broadcast shape of ``row`` and ``col``.

    Raises:
        ValueError: If ``width`` is not positive.
    """
    if int(width) <= 0:
        raise ValueError(f"width must be positive, got {width}.")
    pixel_id = np.asarray(row).astype(np.int64) * np.int64(width) + np.asarray(col).astype(np.int64)
    return int(pixel_id) if pixel_id.ndim == 0 else pixel_id


def decode_pixel_id(
    pixel_id: ArrayLike, width: int
) -> tuple[int, int] | tuple[NDArray[np.int64], NDArray[np.int64]]:
    """Decode int64 pixel ids back into ``(row, col)`` grid positions.

    Inverse of :func:`encode_pixel_id` for the same ``width``.

    Args:
        pixel_id: Encoded pixel id, a scalar or a NumPy array.
        width: Number of columns in the grid; must be positive.

    Returns:
        A ``(row, col)`` tuple of Python ints for a scalar input, otherwise a
        tuple of two int64 arrays.

    Raises:
        ValueError: If ``width`` is not positive.
    """
    if int(width) <= 0:
        raise ValueError(f"width must be positive, got {width}.")
    encoded = np.asarray(pixel_id).astype(np.int64)
    row = encoded // np.int64(width)
    col = encoded % np.int64(width)
    if encoded.ndim == 0:
        return int(row), int(col)
    return row, col


def connected_components(parcels: gpd.GeoDataFrame, *, buffer_m: float = 0.0) -> pd.Series:
    """Group parcels into connected components by buffered intersection.

    Two parcels are connected when their geometries lie within ``buffer_m`` of
    each other (touching parcels when ``buffer_m`` is ``0``); components are the
    connected sets of that graph, found by union-find over a buffered STRtree
    intersection. This is the grouping :func:`assign_spatial_partition` uses to
    keep adjacent parcels in one unit.

    Args:
        parcels: Parcel polygons on the project CRS (EPSG:3035).
        buffer_m: Distance within which two parcels are treated as connected.

    Returns:
        An int ``pandas.Series`` of component ids indexed like ``parcels``.
    """
    geoms = parcels.geometry.to_numpy()
    n = len(geoms)
    probe = shapely.buffer(geoms, buffer_m / 2.0) if buffer_m > 0 else geoms
    left, right = shapely.STRtree(probe).query(probe, predicate="intersects")
    adjacency = coo_matrix((np.ones(left.size), (left, right)), shape=(n, n))
    _, labels = _scipy_connected_components(adjacency, directed=False)
    return pd.Series(labels.astype(int), index=parcels.index, name="component")


def _factor_pairs(n_units: int) -> list[tuple[int, int]]:
    """Return every ``(n_cols, n_rows)`` factor pair with ``n_cols >= n_rows``."""
    pairs = [(n_units // r, r) for r in range(1, math.isqrt(n_units) + 1) if n_units % r == 0]
    if n_units > 1 and pairs == [(n_units, 1)]:
        raise ValueError(f"n_units={n_units} is prime; pass grid_shape explicitly.")
    return pairs


def _rotate(
    cx: NDArray[np.float64], cy: NDArray[np.float64], angle_deg: float
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Rotate points about their centroid by ``angle_deg`` degrees anticlockwise."""
    theta = math.radians(angle_deg)
    cos_t, sin_t = math.cos(theta), math.sin(theta)
    dx, dy = cx - cx.mean(), cy - cy.mean()
    return dx * cos_t - dy * sin_t, dx * sin_t + dy * cos_t


def _dp_run(
    primary: NDArray[np.float64],
    secondary: NDArray[np.float64] | None,
    denominator: NDArray[np.float64] | None,
    k: int,
    *,
    secondary_objective: str,
    priority: str,
    tolerance: float,
    global_ratio: float,
) -> NDArray[np.int64]:
    """Optimal contiguous k-partition of ordered units minimising the objective.

    Returns a segment id (``0..k-1``) per unit. When ``priority`` is
    ``"primary_first"`` and no partition keeps every segment's primary share
    within ``tolerance``, it falls back to minimising primary deviation alone (so
    the caller's balance check then raises).
    """
    n = len(primary)
    cum_p = np.concatenate([np.zeros(1), np.cumsum(np.asarray(primary, dtype=np.float64))])
    total_p = float(cum_p[-1])
    target_p = total_p / k
    cum_s = (
        np.concatenate([np.zeros(1), np.cumsum(np.asarray(secondary, dtype=np.float64))])
        if secondary is not None
        else np.zeros(n + 1)
    )
    total_s = float(cum_s[-1]) if secondary is not None else 1.0
    cum_d = (
        np.concatenate([np.zeros(1), np.cumsum(np.asarray(denominator, dtype=np.float64))])
        if denominator is not None
        else np.zeros(n + 1)
    )

    dp = np.full((k + 1, n + 1), np.inf)
    dp[0, 0] = 0.0
    parent = np.zeros((k + 1, n + 1), dtype=np.int64)
    for j in range(1, k + 1):
        for i in range(j, n + 1):
            starts = np.arange(j - 1, i)
            seg_p = cum_p[i] - cum_p[starts]
            if secondary is None:
                secondary_dev = np.zeros_like(seg_p)
            elif secondary_objective == "share":
                secondary_dev = (((cum_s[i] - cum_s[starts]) / total_s) - 1.0 / k) ** 2
            else:
                seg_d = cum_d[i] - cum_d[starts]
                seg_s = cum_s[i] - cum_s[starts]
                seg_ratio = np.where(seg_d > 0, seg_s / np.where(seg_d > 0, seg_d, 1.0), 0.0)
                secondary_dev = (seg_ratio - global_ratio) ** 2
            if priority == "primary_first":
                cost = np.where(np.abs(seg_p / target_p - 1.0) <= tolerance, secondary_dev, np.inf)
            else:
                cost = (seg_p / total_p - 1.0 / k) ** 2 + secondary_dev
            candidate = dp[j - 1, starts] + cost
            best = int(np.argmin(candidate))
            dp[j, i] = candidate[best]
            parent[j, i] = starts[best]

    if not np.isfinite(dp[k, n]):  # no primary-feasible partition: minimise primary alone
        return _dp_run(
            primary,
            None,
            None,
            k,
            secondary_objective="share",
            priority="joint",
            tolerance=tolerance,
            global_ratio=global_ratio,
        )
    segment = np.empty(n, dtype=np.int64)
    cut = n
    for j in range(k, 0, -1):
        start = parent[j, cut]
        segment[start:cut] = j - 1
        cut = start
    return segment


def _dp_segments(
    primary: NDArray[np.float64],
    secondary: NDArray[np.float64] | None,
    denominator: NDArray[np.float64] | None,
    k: int,
    *,
    secondary_objective: str,
    priority: str,
    tolerance: float,
    global_ratio: float,
) -> NDArray[np.int64]:
    """Return a contiguous k-partition (one segment id per unit) of ordered units."""
    n = len(primary)
    if k <= 1:
        return np.zeros(n, dtype=np.int64)
    if n < k:  # too few units for k segments; leave trailing cells empty (search/check rejects it)
        return np.minimum(np.arange(n), k - 1).astype(np.int64)
    return _dp_run(
        primary,
        secondary,
        denominator,
        k,
        secondary_objective=secondary_objective,
        priority=priority,
        tolerance=tolerance,
        global_ratio=global_ratio,
    )


def _take(array: NDArray | None, index: NDArray[np.int64]) -> NDArray | None:
    """Index ``array`` by ``index``, propagating ``None``."""
    return None if array is None else array[index]


def _partition_units(
    cx: NDArray[np.float64],
    cy: NDArray[np.float64],
    primary: NDArray[np.float64],
    secondary: NDArray[np.float64] | None,
    denominator: NDArray[np.float64] | None,
    n_cols: int,
    n_rows: int,
    tiebreak: NDArray[np.float64],
    *,
    secondary_objective: str,
    priority: str,
    tolerance: float,
    global_ratio: float,
) -> NDArray[np.int64]:
    """Assign each unit a cell id ``1..n_cols*n_rows`` by column-then-row DP cuts."""
    cell = np.empty(len(cx), dtype=np.int64)
    column_order = np.lexsort((tiebreak, cx))
    column_segment = _dp_segments(
        primary[column_order],
        _take(secondary, column_order),
        _take(denominator, column_order),
        n_cols,
        secondary_objective=secondary_objective,
        priority=priority,
        tolerance=tolerance,
        global_ratio=global_ratio,
    )
    for c in range(n_cols):
        members = column_order[column_segment == c]
        row_order = members[np.lexsort((tiebreak[members], cy[members]))]
        row_segment = _dp_segments(
            primary[row_order],
            _take(secondary, row_order),
            _take(denominator, row_order),
            n_rows,
            secondary_objective=secondary_objective,
            priority=priority,
            tolerance=tolerance,
            global_ratio=global_ratio,
        )
        for r in range(n_rows):
            cell[row_order[row_segment == r]] = c * n_rows + r + 1
    return cell


def _cell_objective(
    unit_cell: NDArray[np.int64],
    primary: NDArray[np.float64],
    secondary: NDArray[np.float64] | None,
    denominator: NDArray[np.float64] | None,
    n_units: int,
    secondary_objective: str,
    global_ratio: float,
) -> float:
    """Summed squared deviation of primary share and secondary across the cells."""
    cell_p = np.bincount(unit_cell - 1, weights=primary, minlength=n_units)
    objective = float(np.sum((cell_p / primary.sum() - 1.0 / n_units) ** 2))
    if secondary is not None:
        cell_s = np.bincount(unit_cell - 1, weights=secondary, minlength=n_units)
        if secondary_objective == "share":
            objective += float(np.sum((cell_s / secondary.sum() - 1.0 / n_units) ** 2))
        else:
            cell_d = np.bincount(unit_cell - 1, weights=denominator, minlength=n_units)
            cell_ratio = np.where(cell_d > 0, cell_s / np.where(cell_d > 0, cell_d, 1.0), 0.0)
            objective += float(np.sum((cell_ratio - global_ratio) ** 2))
    return objective


def _check_balance(
    labels: NDArray[np.int64],
    quantity: NDArray[np.float64],
    n_units: int,
    name: str,
    tolerance: float,
) -> float:
    """Return the min-max ratio of ``quantity`` per unit, raising if it exceeds tolerance."""
    totals = np.bincount(labels, weights=quantity, minlength=n_units + 1)[1 : n_units + 1]
    equal_share = quantity.sum() / n_units
    ratio = float(totals.max() / totals.min()) if totals.min() > 0 else math.inf
    if np.max(np.abs(totals - equal_share)) / equal_share > tolerance:
        raise ValueError(
            f"{name} imbalance exceeds tolerance {tolerance:.2f}: per-unit min-max ratio "
            f"{ratio:.3f} (max relative deviation "
            f"{float(np.max(np.abs(totals - equal_share)) / equal_share):.3f})."
        )
    return ratio


def _validate_mass(values: ArrayLike, n_parcels: int, name: str) -> NDArray[np.float64]:
    """Return a validated non-negative per-parcel weight array with a positive total."""
    array = np.asarray(values, dtype=np.float64)
    if array.shape != (n_parcels,):
        raise ValueError(f"{name} must have length {n_parcels}, got shape {array.shape}.")
    if np.any(array < 0) or array.sum() <= 0:
        raise ValueError(f"{name} must be non-negative with a positive total.")
    return array


def assign_spatial_partition(
    parcels: gpd.GeoDataFrame,
    n_units: int,
    *,
    grid_shape: tuple[int, int] | None = None,
    mass: ArrayLike | None = None,
    secondary_mass: ArrayLike | None = None,
    secondary_denominator: ArrayLike | None = None,
    secondary_objective: Literal["share", "ratio"] = "share",
    priority: Literal["joint", "primary_first"] = "joint",
    group_connected: bool = True,
    buffer_m: float = 0.0,
    rotations: Iterable[float] = _DEFAULT_ROTATIONS,
    tolerance: float = 0.25,
    seed: int = SEED,
) -> pd.Series:
    """Partition parcels into ``n_units`` spatially coherent, mass-balanced units.

    Units (whole connected components of parcels, or individual parcels) are laid
    out on an ``(n_cols, n_rows)`` grid by a dynamic program that cuts the
    centroid-ordered units into mass-balanced columns and then rows. Every
    candidate grid factorisation and rotation angle is scored and the lowest
    balance objective wins.

    Args:
        parcels: Parcel polygons on the project CRS (EPSG:3035); the result is
            indexed like this frame.
        n_units: Number of units (grid cells) to produce.
        grid_shape: Explicit ``(n_cols, n_rows)`` whose product is ``n_units``;
            when ``None`` every factor pair with ``n_cols >= n_rows`` is searched.
        mass: Non-negative per-parcel primary weight; defaults to parcel area.
        secondary_mass: Optional non-negative per-parcel secondary weight (for
            example old-growth area). ``None`` ignores the secondary objective.
        secondary_denominator: Per-parcel non-negative denominator used only by
            the ``"ratio"`` objective; defaults to ``mass``.
        secondary_objective: ``"share"`` balances ``secondary_mass`` toward its
            equal per-unit share; ``"ratio"`` balances each unit's
            ``secondary_mass / secondary_denominator`` toward the global ratio.
        priority: ``"joint"`` minimises the summed squared deviation of primary
            share and secondary; ``"primary_first"`` minimises the secondary among
            segments whose primary share is within ``tolerance``, falling back to
            primary-only deviation (the final balance check then raises).
        group_connected: When ``True`` parcels within ``buffer_m`` are merged into
            connected components placed whole; when ``False`` each parcel is placed
            by its centroid.
        buffer_m: Distance within which parcels are connected; used only when
            ``group_connected`` is ``True``.
        rotations: Angles in degrees searched for the best axis; ``(0.0,)``
            disables rotation.
        tolerance: Maximum allowed relative deviation of any unit from its equal
            share of primary mass (and of secondary mass under ``"share"``).
        seed: Seed reserved for breaking exact centroid ties.

    Returns:
        An int ``pandas.Series`` of unit ids ``1..n_units`` indexed like
        ``parcels``.

    Raises:
        ValueError: If ``parcels`` is empty or lacks a CRS; if ``n_units`` is not
            positive; if ``grid_shape`` is inconsistent with ``n_units`` (or
            ``n_units`` is prime and ``grid_shape`` is ``None``); if a mass is the
            wrong length, negative or sums to zero; if there are fewer units than
            ``n_units``; or if the achieved balance exceeds ``tolerance``.
    """
    n_parcels = len(parcels)
    if n_parcels == 0:
        raise ValueError("parcels is empty; nothing to partition.")
    if parcels.crs is None:
        raise ValueError("parcels must have a defined CRS (EPSG:3035).")
    if n_units < 1:
        raise ValueError(f"n_units must be positive, got {n_units}.")

    if grid_shape is None:
        candidate_shapes = _factor_pairs(n_units)
    else:
        if grid_shape[0] < 1 or grid_shape[1] < 1 or grid_shape[0] * grid_shape[1] != n_units:
            raise ValueError(f"grid_shape {grid_shape} is inconsistent with n_units={n_units}.")
        candidate_shapes = [grid_shape]

    geoms = parcels.geometry.to_numpy()
    mass_arr = shapely.area(geoms) if mass is None else _validate_mass(mass, n_parcels, "mass")
    secondary_arr = (
        None
        if secondary_mass is None
        else _validate_mass(secondary_mass, n_parcels, "secondary_mass")
    )
    denominator_arr: NDArray[np.float64] | None = None
    if secondary_arr is not None and secondary_objective == "ratio":
        denominator_arr = (
            mass_arr
            if secondary_denominator is None
            else _validate_mass(secondary_denominator, n_parcels, "secondary_denominator")
        )

    centroids = shapely.centroid(geoms)
    cx_parcel = shapely.get_x(centroids).astype(np.float64)
    cy_parcel = shapely.get_y(centroids).astype(np.float64)

    if group_connected:
        component = connected_components(parcels, buffer_m=buffer_m).to_numpy()
        n_items = int(component.max()) + 1
        weight = np.bincount(component, weights=mass_arr, minlength=n_items).astype(np.float64)
        cx_unit = np.bincount(component, weights=mass_arr * cx_parcel, minlength=n_items) / weight
        cy_unit = np.bincount(component, weights=mass_arr * cy_parcel, minlength=n_items) / weight
        primary_unit = weight
        secondary_unit = (
            None
            if secondary_arr is None
            else np.bincount(component, weights=secondary_arr, minlength=n_items).astype(np.float64)
        )
        denominator_unit = (
            None
            if denominator_arr is None
            else np.bincount(component, weights=denominator_arr, minlength=n_items).astype(
                np.float64
            )
        )
    else:
        component = np.arange(n_parcels, dtype=np.int64)
        n_items = n_parcels
        cx_unit, cy_unit = cx_parcel, cy_parcel
        primary_unit = mass_arr.astype(np.float64)
        secondary_unit = None if secondary_arr is None else secondary_arr.astype(np.float64)
        denominator_unit = None if denominator_arr is None else denominator_arr.astype(np.float64)

    if n_items < n_units:
        raise ValueError(
            f"only {n_items} unit(s) to place but n_units={n_units}; "
            "lower n_units or reduce buffer_m."
        )

    if secondary_unit is not None and denominator_unit is not None:
        global_ratio = float(secondary_unit.sum() / denominator_unit.sum())
    else:
        global_ratio = 0.0
    tiebreak = np.random.default_rng(seed).random(n_items)

    best_objective = math.inf
    best_cell: NDArray[np.int64] = np.ones(n_items, dtype=np.int64)
    best_shape = candidate_shapes[0]
    best_angle = 0.0
    for n_cols, n_rows in candidate_shapes:
        for angle in rotations:
            rx, ry = _rotate(cx_unit, cy_unit, float(angle))
            unit_cell = _partition_units(
                rx,
                ry,
                primary_unit,
                secondary_unit,
                denominator_unit,
                n_cols,
                n_rows,
                tiebreak,
                secondary_objective=secondary_objective,
                priority=priority,
                tolerance=tolerance,
                global_ratio=global_ratio,
            )
            objective = _cell_objective(
                unit_cell,
                primary_unit,
                secondary_unit,
                denominator_unit,
                n_units,
                secondary_objective,
                global_ratio,
            )
            if objective < best_objective:
                best_objective, best_cell = objective, unit_cell
                best_shape, best_angle = (n_cols, n_rows), float(angle)

    labels = best_cell[component]
    mass_ratio = _check_balance(labels, mass_arr, n_units, "mass", tolerance)
    secondary_ratio = (
        _check_balance(labels, secondary_arr, n_units, "secondary_mass", tolerance)
        if secondary_arr is not None and secondary_objective == "share"
        else None
    )
    logger.info(
        "Spatial partition: %d parcels (%d %s) into %d units, grid %dx%d at %.0f deg; "
        "mass min-max ratio %.3f%s.",
        n_parcels,
        n_items,
        "components" if group_connected else "individual parcels",
        n_units,
        best_shape[0],
        best_shape[1],
        best_angle,
        mass_ratio,
        "" if secondary_ratio is None else f", secondary-mass ratio {secondary_ratio:.3f}",
    )
    return pd.Series(labels, index=parcels.index, name="unit_id", dtype=int)


def assign_folds(
    parcels: gpd.GeoDataFrame,
    ogf: ArrayLike,
    *,
    n_folds: int = N_FOLDS,
    buffer_m: float = 0.0,
    secondary_objective: Literal["share", "ratio"] = "share",
    priority: Literal["joint", "primary_first"] = "joint",
    prevalence_basis: Literal["count", "area"] = "count",
    rotations: Iterable[float] = _DEFAULT_ROTATIONS,
    tolerance: float = 0.25,
    seed: int = SEED,
) -> pd.Series:
    """Assign spatially-separated, class-balanced training folds to labelled parcels.

    A wrapper over :func:`assign_spatial_partition`. Primary mass is parcel area.
    Under the default ``"share"`` objective the secondary balances old-growth
    area toward its equal per-fold share (the manuscript design). Under
    ``"ratio"`` the secondary balances old-growth prevalence, by ``"count"``
    (old-growth parcel share) or ``"area"`` (old-growth area share). Six folds use
    a ``(3, 2)`` grid; other counts search factorisations.

    Args:
        parcels: Labelled parcel polygons on the project CRS (EPSG:3035).
        ogf: Per-parcel old-growth flag aligned to ``parcels`` (a boolean array or
            a ``pandas.Series`` indexed like ``parcels``).
        n_folds: Number of folds; defaults to the project fold count.
        buffer_m: Distance within which parcels are treated as connected.
        secondary_objective: ``"share"`` balances old-growth area; ``"ratio"``
            balances old-growth prevalence.
        priority: ``"joint"`` or ``"primary_first"`` (see
            :func:`assign_spatial_partition`).
        prevalence_basis: ``"count"`` or ``"area"`` basis for the ``"ratio"``
            objective; inert under ``"share"``.
        rotations: Angles in degrees searched for the best axis.
        tolerance: Maximum allowed relative deviation of any fold from its equal
            share.
        seed: Seed reserved for breaking exact centroid ties.

    Returns:
        An int ``pandas.Series`` of fold ids (values within
        :data:`utils.terminology.FOLD_IDS`) indexed like ``parcels``.

    Raises:
        ValueError: For the conditions documented on
            :func:`assign_spatial_partition`, or an unknown ``prevalence_basis``.
    """
    area = shapely.area(parcels.geometry.to_numpy())
    flags = ogf.reindex(parcels.index) if isinstance(ogf, pd.Series) else ogf
    is_ogf = np.asarray(pd.Series(flags).fillna(False), dtype=bool)

    denominator: NDArray[np.float64] | None = None
    if secondary_objective == "share":
        secondary = np.where(is_ogf, area, 0.0)
    elif prevalence_basis == "count":
        secondary = np.where(is_ogf, 1.0, 0.0)
        denominator = np.ones(len(parcels), dtype=np.float64)
    elif prevalence_basis == "area":
        secondary = np.where(is_ogf, area, 0.0)
        denominator = area
    else:
        raise ValueError(f"unknown prevalence_basis {prevalence_basis!r}.")

    grid_shape = (3, 2) if n_folds == N_FOLDS else None
    folds = assign_spatial_partition(
        parcels,
        n_folds,
        grid_shape=grid_shape,
        mass=area,
        secondary_mass=secondary,
        secondary_denominator=denominator,
        secondary_objective=secondary_objective,
        priority=priority,
        group_connected=True,
        buffer_m=buffer_m,
        rotations=rotations,
        tolerance=tolerance,
        seed=seed,
    )
    return folds.rename("fold_id")


def cross_unit_distances(
    parcels: gpd.GeoDataFrame,
    unit_col: str,
    *,
    mode: Literal["nearest", "pairwise"],
    quantiles: Sequence[float] = (0.05, 0.25, 0.5, 0.75, 0.95),
    n_jobs: int = -1,
    return_arrays: bool = False,
) -> pd.DataFrame | tuple[pd.DataFrame, dict[int, NDArray[np.float64]]]:
    """Summarise distances between each unit and the other units, in metres.

    Parcels whose ``unit_col`` is missing are dropped, so passing the full labels
    frame analyses only the assigned parcels. Distances are polygon-to-polygon
    (not centroid) on the project CRS (EPSG:3035).

    Args:
        parcels: Parcel polygons with an integer ``unit_col`` column.
        unit_col: Name of the unit-id column.
        mode: ``"nearest"`` measures each in-unit parcel to its closest
            out-of-unit parcel; ``"pairwise"`` measures every in-unit to
            out-of-unit parcel pair.
        quantiles: Quantiles to report (each in ``[0, 1]``).
        n_jobs: Worker count for the ``"pairwise"`` mode (joblib convention).
        return_arrays: When ``True`` also return the per-unit distance arrays.

    Returns:
        A ``pandas.DataFrame`` indexed by unit with columns ``n``, ``min`` and one
        per quantile (``q05`` and so on). When ``return_arrays`` is ``True``, a
        ``(frame, {unit: distances})`` tuple.

    Raises:
        ValueError: If ``mode`` is not ``"nearest"`` or ``"pairwise"``.
    """
    if mode not in ("nearest", "pairwise"):
        raise ValueError(f"mode must be 'nearest' or 'pairwise', got {mode!r}.")

    frame = parcels[parcels[unit_col].notna()]
    units = [int(u) for u in sorted(frame[unit_col].unique())]
    unit_of = frame[unit_col].to_numpy()
    geoms = frame.geometry.to_numpy()

    distances: dict[int, NDArray[np.float64]] = {}
    if mode == "nearest":
        for unit in units:
            inside = frame.loc[unit_of == unit, [frame.geometry.name]]
            outside = frame.loc[unit_of != unit, [frame.geometry.name]]
            joined = gpd.sjoin_nearest(inside, outside, distance_col="_distance_m")
            joined = joined[~joined.index.duplicated(keep="first")]  # one row per in-unit parcel
            distances[unit] = joined["_distance_m"].to_numpy()
    else:

        def _pairwise(unit: int) -> tuple[int, NDArray[np.float64]]:
            inside = unit_of == unit
            grid = shapely.distance(geoms[inside][:, None], geoms[~inside][None, :])
            return unit, grid.ravel()

        distances = dict(
            joblib.Parallel(n_jobs=n_jobs)(joblib.delayed(_pairwise)(u) for u in units)
        )

    quantile_cols = {f"q{int(round(q * 100)):02d}": q for q in quantiles}
    rows = {
        unit: {
            "n": distances[unit].size,
            "min": float(distances[unit].min()),
            **{name: float(np.quantile(distances[unit], q)) for name, q in quantile_cols.items()},
        }
        for unit in units
    }
    summary = pd.DataFrame.from_dict(rows, orient="index")[["n", "min", *quantile_cols]]
    summary.index.name = unit_col
    return (summary, distances) if return_arrays else summary


def load_fold_assignment(path: str | Path) -> pd.DataFrame:
    """Read a parcel-to-fold assignment and validate every fold id.

    Args:
        path: Path to a ``.csv``, ``.parquet`` or vector (``.gpkg`` and similar)
            file carrying ``parcel_id`` and ``fold_id`` columns.

    Returns:
        A ``pandas.DataFrame`` with ``parcel_id`` and integer ``fold_id`` columns.

    Raises:
        ValueError: If the file type is unsupported, a required column is missing,
            or any ``fold_id`` lies outside :data:`utils.terminology.FOLD_IDS`.
    """
    path = Path(path)
    suffix = path.suffix.lower()
    if suffix == ".csv":
        frame: pd.DataFrame = pd.read_csv(path)
    elif suffix == ".parquet":
        frame = pd.read_parquet(path)
    elif suffix in _VECTOR_SUFFIXES:
        frame = pd.DataFrame(gpd.read_file(path).drop(columns="geometry", errors="ignore"))
    else:
        raise ValueError(f"unsupported fold-assignment file type {suffix!r}.")

    missing = {"parcel_id", "fold_id"} - set(frame.columns)
    if missing:
        raise ValueError(f"fold-assignment file is missing column(s): {sorted(missing)}.")

    result = frame[["parcel_id", "fold_id"]].copy()
    result["fold_id"] = result["fold_id"].astype(int)
    invalid = sorted(set(result["fold_id"].unique()) - set(FOLD_IDS))
    if invalid:
        raise ValueError(f"fold_id values {invalid} are outside FOLD_IDS {FOLD_IDS}.")
    return result
