"""Block-bootstrap primitives for evaluation uncertainty and paired contrasts.

Stage 7 of the refactor takes the saved out-of-fold parcel predictions from the
nested cross-validation and turns its point estimates into confidence intervals
plus pre-specified paired contrasts between feature sets. This is an
*evaluation* bootstrap: it resamples saved predictions and recomputes pooled
parcel metrics, never retraining a model. The separate retraining-based
predicted-area bootstrap is deferred to the final mapping stage.

Four pure functions and one loader are exposed:

  - :func:`holm_adjusted_p_values` applies the Holm step-down family-wise
    correction (Holm 1979, Scand. J. Stat. 6:65-70) to a vector of raw
    p-values, returning the adjusted values in the input order.
  - :func:`percentile_interval` returns the two-sided percentile interval of a
    bootstrap sample at a given confidence level, ignoring NaN replicates.
  - :func:`two_sided_bootstrap_p` returns the add-one-smoothed achieved
    significance level for the null that the centre of a bootstrap difference
    distribution is zero (Davison and Hinkley 1997).
  - :func:`block_bootstrap_distribution` resamples block ids with replacement
    and recomputes a paired metric for every configuration on the same
    resampled rows, returning the ``(n_reps, n_configs)`` matrix.
  - :func:`load_block_assignment` reads the parcel-to-block assignment (and
    fold and label) of the labelled parcels from the partitioned labels layer.

The resampling unit for the cross-configuration analysis is the parcel, grouped
into the area-balanced block set built once on the labelled parcels (see
:data:`utils.terminology.BOOTSTRAP_N_UNITS`). Determinism is anchored on
``numpy.random.SeedSequence``: spawning one child sequence per replicate makes
the replicate matrix identical for serial and parallel execution.

Importing this module only binds names: it performs no input/output, configures
no logging and inspects no environment, in keeping with the repository rule
that ``utils`` modules have no side effects on import.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from pathlib import Path
from typing import TYPE_CHECKING

import geopandas as gpd
import numpy as np
import pandas as pd
from joblib import Parallel, delayed

if TYPE_CHECKING:
    from numpy.typing import ArrayLike, NDArray


# --------------------------------------------------------------------------- #
# Holm-Bonferroni step-down adjustment.
# --------------------------------------------------------------------------- #


def holm_adjusted_p_values(p_values: Sequence[float]) -> list[float]:
    """Return the Holm-Bonferroni step-down adjusted p-values in input order.

    With ``m = len(p_values)`` and the ascending sort ``p_(1) <= ... <= p_(m)``,
    the adjusted value at the ``k``-th smallest position (1-indexed) is the
    running maximum of ``min(1.0, (m - k + 1) * p_(k))``; the running maximum
    enforces monotonicity in the sorted order. The result is mapped back to the
    original input order so callers can pair it row-for-row with the contrast
    table. This is Holm (1979, *Scand. J. Stat.* 6:65-70), the uniformly more
    powerful step-down counterpart of Bonferroni for family-wise error control.

    Args:
        p_values: One-dimensional sequence of raw p-values, each in ``[0, 1]``.

    Returns:
        The Holm-adjusted p-values, one per input, in the input order.

    Raises:
        ValueError: If ``p_values`` is empty, or any value lies outside
            ``[0, 1]`` or is not finite.
    """
    array = np.asarray(p_values, dtype=np.float64)
    if array.ndim != 1:
        raise ValueError("p_values must be one-dimensional.")
    if array.size == 0:
        raise ValueError("p_values must not be empty.")
    if not np.isfinite(array).all():
        raise ValueError("p_values contains non-finite values.")
    if (array < 0.0).any() or (array > 1.0).any():
        raise ValueError("p_values must lie in [0, 1].")

    m = array.size
    order = np.argsort(array, kind="stable")
    sorted_p = array[order]
    weights = np.arange(m, 0, -1, dtype=np.float64)  # m, m-1, ..., 1
    raw = np.minimum(1.0, weights * sorted_p)
    adjusted_sorted = np.maximum.accumulate(raw)
    adjusted = np.empty(m, dtype=np.float64)
    adjusted[order] = adjusted_sorted
    return adjusted.tolist()


# --------------------------------------------------------------------------- #
# Percentile intervals and the two-sided bootstrap p-value.
# --------------------------------------------------------------------------- #


def percentile_interval(samples: ArrayLike, *, level: float = 0.95) -> tuple[float, float]:
    """Return the two-sided percentile interval of a bootstrap sample.

    The lower edge is the ``(1 - level) / 2`` quantile and the upper edge is the
    ``(1 + level) / 2`` quantile, ignoring NaNs. This is the classical
    percentile interval (Efron and Tibshirani 1993): coverage is approximate but
    requires no parametric assumption on the bootstrap distribution.

    Args:
        samples: One-dimensional bootstrap sample (possibly containing NaNs).
        level: Two-sided confidence level in ``(0, 1)``; the default of 0.95
            yields the 2.5 and 97.5 percentiles.

    Returns:
        The ``(lower, upper)`` percentile interval as Python floats.

    Raises:
        ValueError: If ``level`` is not in ``(0, 1)`` or ``samples`` contains
            no finite values.
    """
    if not 0.0 < level < 1.0:
        raise ValueError(f"level must lie in (0, 1), got {level}.")
    array = np.asarray(samples, dtype=np.float64).ravel()
    finite = array[np.isfinite(array)]
    if finite.size == 0:
        raise ValueError("samples contains no finite values.")
    lower_q = (1.0 - level) / 2.0
    upper_q = (1.0 + level) / 2.0
    lower, upper = np.quantile(finite, (lower_q, upper_q))
    return float(lower), float(upper)


def two_sided_bootstrap_p(differences: ArrayLike) -> float:
    """Return the achieved significance level for ``H0: centre = 0``.

    Computed with add-one smoothing (Davison and Hinkley 1997,
    *Bootstrap Methods and their Application*, p. 161): with ``n`` finite
    replicates,

    ``p = 2 * min((#(d <= 0) + 1) / (n + 1), (#(d >= 0) + 1) / (n + 1))``

    clipped to at most 1.0. The smoothing prevents an exact zero when no
    replicate falls on one side, which would otherwise be a misleading
    statement about an estimated tail probability. The clip handles the case
    where exact zeros are counted on both sides. This rule is consistent with
    whether the matching percentile interval excludes zero: when fewer than
    ``alpha (n + 1) / 2`` replicates lie on one side, ``p < alpha`` and the
    ``1 - alpha`` interval excludes zero.

    Args:
        differences: One-dimensional bootstrap differences (possibly with NaNs).

    Returns:
        The two-sided p-value, in ``(0, 1]``.

    Raises:
        ValueError: If ``differences`` contains no finite values.
    """
    array = np.asarray(differences, dtype=np.float64).ravel()
    finite = array[np.isfinite(array)]
    if finite.size == 0:
        raise ValueError("differences contains no finite values.")
    n = finite.size
    le = int(np.sum(finite <= 0.0))
    ge = int(np.sum(finite >= 0.0))
    left = (le + 1) / (n + 1)
    right = (ge + 1) / (n + 1)
    return float(min(1.0, 2.0 * min(left, right)))


# --------------------------------------------------------------------------- #
# Paired block bootstrap of an evaluation metric across configurations.
# --------------------------------------------------------------------------- #


def _replicate_metric(
    rng_state: np.random.SeedSequence,
    block_ids: NDArray[np.int64],
    unique_blocks: NDArray[np.int64],
    block_members: tuple[NDArray[np.int64], ...],
    y_true: NDArray[np.float64],
    scores: NDArray[np.float64],
    metric: Callable[[NDArray[np.float64], NDArray[np.float64]], float],
    min_classes: int,
) -> NDArray[np.float64]:
    """Draw one paired replicate and evaluate the metric for every score column.

    A single child :class:`~numpy.random.SeedSequence` consumed by
    :class:`~numpy.random.Generator` selects ``unique_blocks.size`` block ids
    with replacement; the resampled rows are the concatenation of the matching
    block members (multiplicity retained). When the resampled labels contain
    fewer than ``min_classes`` distinct classes, the row is filled with NaN.

    Args:
        rng_state: Child :class:`~numpy.random.SeedSequence` for this replicate.
        block_ids: Length-``n_obs`` block id per observation.
        unique_blocks: The distinct block ids in :attr:`block_ids`, sorted.
        block_members: For each entry of ``unique_blocks``, the int64 array of
            row indices belonging to that block.
        y_true: Length-``n_obs`` labels (kept float for metric compatibility).
        scores: ``(n_obs, n_configs)`` score matrix; columns are the
            configurations being compared.
        metric: Callable ``(y_true_subset, score_column_subset) -> float``.
        min_classes: Minimum distinct classes required in the resampled labels.

    Returns:
        A length-``n_configs`` float array of metric values (NaN if the
        replicate did not meet ``min_classes``).
    """
    rng = np.random.default_rng(rng_state)
    chosen = rng.integers(0, unique_blocks.size, size=unique_blocks.size)
    rows = np.concatenate([block_members[i] for i in chosen])
    yt = y_true[rows]
    if np.unique(yt).size < min_classes:
        return np.full(scores.shape[1], np.nan, dtype=np.float64)
    out = np.empty(scores.shape[1], dtype=np.float64)
    sub_scores = scores[rows]
    for column in range(scores.shape[1]):
        out[column] = float(metric(yt, sub_scores[:, column]))
    return out


def _chunk_replicates(
    seeds: Sequence[np.random.SeedSequence],
    block_ids: NDArray[np.int64],
    unique_blocks: NDArray[np.int64],
    block_members: tuple[NDArray[np.int64], ...],
    y_true: NDArray[np.float64],
    scores: NDArray[np.float64],
    metric: Callable[[NDArray[np.float64], NDArray[np.float64]], float],
    min_classes: int,
) -> NDArray[np.float64]:
    """Evaluate :func:`_replicate_metric` over a contiguous chunk of seeds.

    Used by :func:`block_bootstrap_distribution` to split the work across joblib
    workers without changing the per-replicate output.

    Args:
        seeds: Child seed sequences for the replicates in this chunk.
        block_ids: See :func:`_replicate_metric`.
        unique_blocks: See :func:`_replicate_metric`.
        block_members: See :func:`_replicate_metric`.
        y_true: See :func:`_replicate_metric`.
        scores: See :func:`_replicate_metric`.
        metric: See :func:`_replicate_metric`.
        min_classes: See :func:`_replicate_metric`.

    Returns:
        A ``(len(seeds), n_configs)`` array of metric values.
    """
    rows = [
        _replicate_metric(
            seed,
            block_ids,
            unique_blocks,
            block_members,
            y_true,
            scores,
            metric,
            min_classes,
        )
        for seed in seeds
    ]
    return np.vstack(rows)


def block_bootstrap_distribution(
    y_true: ArrayLike,
    scores: ArrayLike,
    block_ids: ArrayLike,
    metric: Callable[[NDArray[np.float64], NDArray[np.float64]], float],
    *,
    n_reps: int,
    seed: int,
    n_jobs: int = 1,
    min_classes: int = 2,
) -> NDArray[np.float64]:
    """Return the ``(n_reps, n_configs)`` paired block-bootstrap metric matrix.

    Each replicate draws ``unique_blocks.size`` blocks with replacement, takes
    the concatenated member rows (multiplicity retained), and evaluates
    ``metric`` for every score column on the *same* resampled rows. This
    pairing across configurations is essential: differences between columns are
    computed on identical resamples, removing replicate-level variability from
    the contrast.

    Determinism is anchored on :func:`numpy.random.SeedSequence`. One child
    sequence is spawned per replicate, so the row of the output matrix for a
    given replicate index is the same whether the function runs serially or
    splits chunks across joblib workers.

    Args:
        y_true: Length-``n_obs`` labels (one row per observation).
        scores: ``(n_obs, n_configs)`` score matrix; one column per
            configuration to bootstrap.
        block_ids: Length-``n_obs`` integer block id per observation.
        metric: Callable applied to ``(y_true_subset, score_column_subset)``
            for each column on each replicate.
        n_reps: Number of bootstrap replicates.
        seed: Seed for the parent :class:`~numpy.random.SeedSequence`.
        n_jobs: Joblib worker count; values ``<= 1`` run serially in this
            process. Even with ``n_jobs > 1`` the result is identical to the
            serial path because each replicate is driven by its own child seed.
        min_classes: Minimum distinct classes required in the resampled labels
            before the metric is computed; NaN is written for the whole row
            otherwise. Defaults to 2 (the smallest viable binary sample).

    Returns:
        A ``(n_reps, n_configs)`` float array of metric values; entries are NaN
        wherever ``min_classes`` was not met on the resample.

    Raises:
        ValueError: On dimension or length mismatches, or non-positive
            ``n_reps`` or ``min_classes``.
    """
    if n_reps <= 0:
        raise ValueError(f"n_reps must be positive, got {n_reps}.")
    if min_classes < 1:
        raise ValueError(f"min_classes must be positive, got {min_classes}.")

    yt = np.asarray(y_true, dtype=np.float64).ravel()
    score_matrix = np.asarray(scores, dtype=np.float64)
    if score_matrix.ndim == 1:
        score_matrix = score_matrix.reshape(-1, 1)
    if score_matrix.ndim != 2:
        raise ValueError(f"scores must be 1- or 2-dimensional, got shape {score_matrix.shape}.")
    if score_matrix.shape[0] != yt.shape[0]:
        raise ValueError(
            f"length mismatch: y_true has {yt.shape[0]} rows, scores has "
            f"{score_matrix.shape[0]}."
        )
    blocks = np.asarray(block_ids).astype(np.int64).ravel()
    if blocks.shape[0] != yt.shape[0]:
        raise ValueError(
            f"length mismatch: y_true has {yt.shape[0]} rows, block_ids has " f"{blocks.shape[0]}."
        )

    unique_blocks = np.unique(blocks)
    if unique_blocks.size == 0:
        raise ValueError("block_ids must contain at least one block.")
    block_members: tuple[NDArray[np.int64], ...] = tuple(
        np.flatnonzero(blocks == block_id).astype(np.int64) for block_id in unique_blocks
    )

    parent = np.random.SeedSequence(seed)
    seeds = parent.spawn(n_reps)

    if n_jobs <= 1:
        return _chunk_replicates(
            seeds,
            blocks,
            unique_blocks,
            block_members,
            yt,
            score_matrix,
            metric,
            min_classes,
        )

    # Even chunking so each worker draws contiguous replicates; per-replicate
    # outputs do not depend on chunk boundaries because each child seed is
    # independent.
    chunks = np.array_split(np.arange(n_reps), n_jobs)
    chunk_seeds = [[seeds[i] for i in chunk] for chunk in chunks if chunk.size]

    pieces: list[NDArray[np.float64]] = Parallel(n_jobs=n_jobs)(
        delayed(_chunk_replicates)(
            seed_chunk,
            blocks,
            unique_blocks,
            block_members,
            yt,
            score_matrix,
            metric,
            min_classes,
        )
        for seed_chunk in chunk_seeds
    )
    return np.vstack(pieces)


# --------------------------------------------------------------------------- #
# Block assignment loader.
# --------------------------------------------------------------------------- #


def load_block_assignment(labels_path: str | Path) -> pd.DataFrame:
    """Load the parcel-to-bootstrap-block assignment from the partitioned labels.

    The partitioned reference labels (``ogf_reference_labels_partitioned.gpkg``,
    written by the fold-assignment notebook) carry ``parcel_id`` as a zero-padded
    string and ``bootstrap_id``, ``fold_id`` and ``ogf`` as floats; all four are
    cast to int64 here so they align with the parquet parcel ids (explicit casts
    before any merge). Unlabelled parcels (``ogf`` missing) are dropped.

    Args:
        labels_path: Path to the partitioned labels GeoPackage.

    Returns:
        Columns ``parcel_id``, ``bootstrap_id``, ``fold_id`` and ``ogf`` (all
        int64) over the labelled parcels only, sorted by ``parcel_id``.
    """
    frame = gpd.read_file(labels_path)
    labelled = frame[frame["ogf"].notna()]
    out = pd.DataFrame(
        {
            "parcel_id": labelled["parcel_id"].astype(np.int64),
            "bootstrap_id": labelled["bootstrap_id"].astype(np.int64),
            "fold_id": labelled["fold_id"].astype(np.int64),
            "ogf": labelled["ogf"].astype(np.int64),
        }
    )
    return out.sort_values("parcel_id").reset_index(drop=True)


__all__ = [
    "block_bootstrap_distribution",
    "holm_adjusted_p_values",
    "load_block_assignment",
    "percentile_interval",
    "two_sided_bootstrap_p",
]
