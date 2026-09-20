"""Schema-validated table and metadata I/O for the repository.

This module is the single source of truth for how tabular results and metadata
reach disk (see AGENTS.md, "Schemas"). Every Parquet and CSV write passes
through :func:`validate_schema` first (hard rule 6), so no result table is
written with the wrong columns, dtypes or stray missing values.

The reusable validation and writer machinery lives here, together with the
concrete result schemas for the nested cross-validation pipeline (the pixel
table built by ``scripts/build_pixel_table.py`` and the per-fold and pooled
outputs of ``scripts/run_nested_cv.py``). Importing this module only binds
names: it performs no input/output and creates no directories. Parent
directories are created only when a writer is actually called.
"""

import json
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd


@dataclass(frozen=True, slots=True)
class ColumnSpec:
    """Expected dtype and nullability for a single table column.

    Attributes:
        dtype: One pandas dtype name (for example ``"int64"``) or a tuple of
            permitted dtype names. A column matches if its dtype name is among
            these. Matching is by exact name, so ``"int64"`` does not admit
            ``"int32"``.
        nullable: Whether the column may contain missing values. When ``False``,
            any ``NaN``/``None`` is reported as a problem.
    """

    dtype: str | tuple[str, ...]
    nullable: bool = True


# A schema maps each expected column name to its :class:`ColumnSpec`.
Schema = Mapping[str, ColumnSpec]


def _allowed_dtypes(spec: ColumnSpec) -> tuple[str, ...]:
    """Return the permitted dtype names for ``spec`` as a tuple.

    Args:
        spec: The column specification.

    Returns:
        The dtype names the column is allowed to take.
    """
    return (spec.dtype,) if isinstance(spec.dtype, str) else tuple(spec.dtype)


def validate_schema(df: pd.DataFrame, schema: Schema, *, allow_extra: bool = False) -> None:
    """Validate ``df`` against ``schema``, raising on any mismatch.

    All detected problems are collected and reported together, so a single call
    surfaces every issue rather than only the first. The checks are:

    - every column named in ``schema`` is present;
    - no columns beyond those in ``schema`` are present, unless ``allow_extra``;
    - each present column's dtype name is among its allowed dtype names;
    - each non-nullable column contains no missing values.

    Args:
        df: The table to validate.
        schema: Mapping of column name to :class:`ColumnSpec`.
        allow_extra: When ``True``, columns absent from ``schema`` are permitted.

    Raises:
        ValueError: If any column is missing, unexpected, of the wrong dtype, or
            holds missing values in a non-nullable column. The message lists
            every problem found.
    """
    problems: list[str] = []
    present = set(df.columns)
    expected = set(schema)

    missing = sorted(name for name in schema if name not in present)
    if missing:
        problems.append(f"missing required columns: {missing}")

    if not allow_extra:
        extra = sorted(str(name) for name in df.columns if name not in expected)
        if extra:
            problems.append(f"unexpected columns: {extra}")

    for name, spec in schema.items():
        if name not in present:
            continue
        column = df[name]
        actual = str(column.dtype)
        allowed = _allowed_dtypes(spec)
        if actual not in allowed:
            problems.append(
                f"column {name!r} has dtype {actual!r}; expected one of {list(allowed)}"
            )
        if not spec.nullable and column.isna().any():
            count = int(column.isna().sum())
            problems.append(f"non-nullable column {name!r} contains {count} missing value(s)")

    if problems:
        joined = "\n  ".join(problems)
        raise ValueError(f"Schema validation failed:\n  {joined}")


def write_parquet(
    df: pd.DataFrame, path: str | Path, schema: Schema, *, allow_extra: bool = False
) -> Path:
    """Validate ``df`` then write it to ``path`` as Parquet.

    Validation runs first (hard rule 6); the write is reached only if the table
    conforms. Parent directories are created when this function is called.

    Args:
        df: The table to write.
        path: Destination Parquet file.
        schema: Schema to validate against.
        allow_extra: Passed through to :func:`validate_schema`.

    Returns:
        The path written to.

    Raises:
        ValueError: If ``df`` fails :func:`validate_schema`.
    """
    validate_schema(df, schema, allow_extra=allow_extra)
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(target)
    return target


def write_csv(
    df: pd.DataFrame, path: str | Path, schema: Schema, *, allow_extra: bool = False
) -> Path:
    """Validate ``df`` then write it to ``path`` as CSV (without the index).

    Validation runs first (hard rule 6); the write is reached only if the table
    conforms. Parent directories are created when this function is called.

    Args:
        df: The table to write.
        path: Destination CSV file.
        schema: Schema to validate against.
        allow_extra: Passed through to :func:`validate_schema`.

    Returns:
        The path written to.

    Raises:
        ValueError: If ``df`` fails :func:`validate_schema`.
    """
    validate_schema(df, schema, allow_extra=allow_extra)
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(target, index=False)
    return target


def write_json(data: Mapping[str, Any], path: str | Path) -> Path:
    """Write ``data`` as pretty, deterministic UTF-8 JSON.

    Keys are sorted and the output is indented by two spaces, so repeated writes
    of equivalent data produce byte-identical files. Parent directories are
    created when this function is called.

    Args:
        data: The mapping to serialise.
        path: Destination JSON file.

    Returns:
        The path written to.
    """
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(data, indent=2, sort_keys=True, ensure_ascii=False)
    target.write_text(text + "\n", encoding="utf-8")
    return target


# --------------------------------------------------------------------------- #
# Nested cross-validation pipeline schemas.
#
# Every result write in scripts/build_pixel_table.py and scripts/run_nested_cv.py
# is validated against one of the schemas below (hard rule 6). Column names match
# the structures produced by utils.metrics, utils.hp_search and utils.models so
# that no result table invents a column or a dtype. Text columns admit both the
# object and the pandas ``string`` dtype so a frame is accepted whether it was
# built in memory or read back from Parquet.
# --------------------------------------------------------------------------- #

_TEXT_DTYPE: tuple[str, ...] = ("object", "string")

# Pixel index over every labelled, fully-covered pixel (build_pixel_table.py).
# ``pixel_id = row * reference_width + col``; rows are sorted by ``pixel_id`` and
# this ordering aligns one-to-one with the cached feature memmaps.
PIXEL_INDEX_SCHEMA: Schema = {
    "pixel_id": ColumnSpec("int64", nullable=False),
    "row": ColumnSpec("int32", nullable=False),
    "col": ColumnSpec("int32", nullable=False),
    "x": ColumnSpec("float64", nullable=False),
    "y": ColumnSpec("float64", nullable=False),
    "parcel_id": ColumnSpec("int64", nullable=False),
    "fold_id": ColumnSpec("int8", nullable=False),
    "ogf": ColumnSpec("int8", nullable=False),
    # Two forest-type stratifiers (stratification only; never a feature): the
    # CORINE land-cover type (broad coverage) and the inventory-derived type
    # (finer granularity, from the labels' species composition).
    "forest_type_corine": ColumnSpec(_TEXT_DTYPE, nullable=False),
    "forest_type_inventory": ColumnSpec(_TEXT_DTYPE, nullable=False),
}

# Per-feature-set validity mask: True where every band of that feature set is
# finite and not the NoData sentinel (build_pixel_table.py). Rows align with the
# pixel index and the feature memmap by ``pixel_id``.
PIXEL_VALID_SCHEMA: Schema = {
    "pixel_id": ColumnSpec("int64", nullable=False),
    "valid": ColumnSpec("bool", nullable=False),
}

# Out-of-fold per-pixel predictions for one outer fold (run_nested_cv.py).
PIXEL_PREDICTIONS_SCHEMA: Schema = {
    "pixel_id": ColumnSpec("int64", nullable=False),
    "outer_fold": ColumnSpec("int8", nullable=False),
    "y_true": ColumnSpec("int8", nullable=False),
    "p": ColumnSpec("float64", nullable=False),
}

# Cross-validated training dissimilarity indices for the area-of-applicability
# analysis (run_area_of_applicability.py): one row per final-fit training pixel,
# with its spatial fold, label and Meyer-Pebesma DI (nearest neighbour outside
# the pixel's own fold, divided by the mean pairwise training distance).
AOA_TRAINING_DI_SCHEMA: Schema = {
    "pixel_id": ColumnSpec("int64", nullable=False),
    "fold_id": ColumnSpec("int8", nullable=False),
    "ogf": ColumnSpec("int8", nullable=False),
    "di": ColumnSpec("float64", nullable=False),
}

# Out-of-fold per-parcel predictions (unweighted mean of pixel probabilities).
PARCEL_PREDICTIONS_SCHEMA: Schema = {
    "parcel_id": ColumnSpec("int64", nullable=False),
    "outer_fold": ColumnSpec("int8", nullable=False),
    "y_true": ColumnSpec("int8", nullable=False),
    "p_mean": ColumnSpec("float64", nullable=False),
    "n_pixels": ColumnSpec("int64", nullable=False),
}

# One row per Optuna trial in an outer fold's inner search (run_nested_cv.py).
# The five score columns are the trial's mean parcel metrics over the inner
# folds; ``pr_auc`` is the optimisation objective. ``inner_pixel_threshold`` and
# ``inner_parcel_threshold`` are the F1-maximising thresholds on the pooled inner
# validation predictions, applied later to the outer fold for an honest F1. The
# parameter columns cover both architecture-specific search spaces
# (utils.hp_search.XGB_SEARCH_SPACE and CNN_SEARCH_SPACE). Score, threshold and
# parameter columns are nullable so a failed trial can still be recorded and the
# irrelevant architecture's parameter columns remain blank.
HP_TRIALS_SCHEMA: Schema = {
    "trial_number": ColumnSpec("int64", nullable=False),
    "state": ColumnSpec(_TEXT_DTYPE, nullable=False),
    "pr_auc": ColumnSpec("float64"),
    "roc_auc": ColumnSpec("float64"),
    "f1": ColumnSpec("float64"),
    "precision": ColumnSpec("float64"),
    "recall": ColumnSpec("float64"),
    "inner_pixel_threshold": ColumnSpec("float64"),
    "inner_parcel_threshold": ColumnSpec("float64"),
    "n_estimators": ColumnSpec(("int64", "float64")),
    "learning_rate": ColumnSpec("float64"),
    "max_depth": ColumnSpec(("int64", "float64")),
    "min_child_weight": ColumnSpec("float64"),
    "subsample": ColumnSpec("float64"),
    "colsample_bytree": ColumnSpec("float64"),
    "reg_lambda": ColumnSpec("float64"),
    "reg_alpha": ColumnSpec("float64"),
    "gamma": ColumnSpec("float64"),
    "max_delta_step": ColumnSpec(("int64", "float64")),
    "n_channels": ColumnSpec(("int64", "float64")),
    "lr": ColumnSpec("float64"),
    "weight_decay": ColumnSpec("float64"),
    "dropout": ColumnSpec("float64"),
    "batch_size": ColumnSpec(("int64", "float64")),
    "cnn_epochs": ColumnSpec(("int64", "float64")),
}

# Hyperparameter boundary diagnostics (utils.hp_search.hp_boundary_diagnostics).
HP_BOUNDARY_SCHEMA: Schema = {
    "parameter": ColumnSpec(_TEXT_DTYPE, nullable=False),
    "value": ColumnSpec("float64", nullable=False),
    "low": ColumnSpec("float64", nullable=False),
    "high": ColumnSpec("float64", nullable=False),
    "at_lower_bound": ColumnSpec("bool", nullable=False),
    "at_upper_bound": ColumnSpec("bool", nullable=False),
}

# Gain-based feature importance for an outer fold (utils.models.xgboost).
FEATURE_IMPORTANCE_SCHEMA: Schema = {
    "feature": ColumnSpec(_TEXT_DTYPE, nullable=False),
    "importance": ColumnSpec("float64", nullable=False),
}

# Per-fold discrimination metrics, tidy (one row per outer fold and level).
# ``level`` is "pixel" or "parcel". ``threshold``/``f1``/``precision``/``recall``
# are the in-sample max-F1 operating point tuned on the outer fold itself; the
# ``*_inner_threshold`` columns are the honest operating point at ``inner_threshold``
# (tuned on the inner folds only). ``pr_auc``/``roc_auc`` are threshold-free.
PER_FOLD_METRICS_SCHEMA: Schema = {
    "outer_fold": ColumnSpec("int64", nullable=False),
    "level": ColumnSpec(_TEXT_DTYPE, nullable=False),
    "n": ColumnSpec("int64", nullable=False),
    "prevalence": ColumnSpec("float64", nullable=False),
    "pr_auc": ColumnSpec("float64", nullable=False),
    "roc_auc": ColumnSpec("float64", nullable=False),
    "threshold": ColumnSpec("float64", nullable=False),
    "f1": ColumnSpec("float64", nullable=False),
    "precision": ColumnSpec("float64", nullable=False),
    "recall": ColumnSpec("float64", nullable=False),
    "inner_threshold": ColumnSpec("float64", nullable=False),
    "f1_inner_threshold": ColumnSpec("float64", nullable=False),
    "precision_inner_threshold": ColumnSpec("float64", nullable=False),
    "recall_inner_threshold": ColumnSpec("float64", nullable=False),
}

# Across-fold summary (mean, sd, min, max) of each metric, per level, in long
# form (one row per level and metric). The manuscript reports mean [min-max].
# The ``metric`` column covers the threshold-free scores, the in-sample max-F1
# operating point and the inner-threshold operating point, so adding the honest
# F1 adds rows here rather than columns.
SUMMARY_METRICS_SCHEMA: Schema = {
    "level": ColumnSpec(_TEXT_DTYPE, nullable=False),
    "metric": ColumnSpec(_TEXT_DTYPE, nullable=False),
    "mean": ColumnSpec("float64", nullable=False),
    "sd": ColumnSpec("float64"),
    "min": ColumnSpec("float64", nullable=False),
    "max": ColumnSpec("float64", nullable=False),
}

# Pooled metrics over the concatenated out-of-fold predictions (long form),
# used for both the pixel and parcel pooled tables.
POOLED_METRICS_SCHEMA: Schema = {
    "metric": ColumnSpec(_TEXT_DTYPE, nullable=False),
    "value": ColumnSpec("float64", nullable=False),
}


__all__ = [
    "FEATURE_IMPORTANCE_SCHEMA",
    "HP_BOUNDARY_SCHEMA",
    "HP_TRIALS_SCHEMA",
    "PARCEL_PREDICTIONS_SCHEMA",
    "PER_FOLD_METRICS_SCHEMA",
    "PIXEL_INDEX_SCHEMA",
    "PIXEL_PREDICTIONS_SCHEMA",
    "PIXEL_VALID_SCHEMA",
    "POOLED_METRICS_SCHEMA",
    "SUMMARY_METRICS_SCHEMA",
    "ColumnSpec",
    "Schema",
    "validate_schema",
    "write_csv",
    "write_json",
    "write_parquet",
]
