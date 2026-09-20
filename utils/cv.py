"""Nested cross-validation core for the old-growth pixel models.

This module holds the scientific logic of the headline nested cross-validation,
kept apart from the orchestration script so the pure parts can be unit-tested on
synthetic arrays. It builds training and validation arrays from the cached
feature memmap, runs the Optuna inner search, retrains on the inner folds and
predicts the held-out outer fold, and aggregates the out-of-fold predictions
into per-fold, summary and pooled tables.

Design notes:

  - **Sampling.** Training pixels are sampled per parcel (up to
    ``n_per_parcel``, without replacement, seeded) so large parcels do not
    dominate; validation and test pixels are *all* pixels of the held-out
    parcels. Sampling selects row indices into the stored feature matrix; the
    matrix on disk is never subset (see ``scripts/build_pixel_table.py``).
  - **Selection metric.** Hyperparameters are selected on the mean parcel
    PR-AUC over the inner folds. XGBoost's internal ``aucpr`` eval metric is a
    separate, unused signal here: ``train_xgb_fold`` passes no eval set and
    there is no early stopping (``n_estimators`` is tuned), so ``aucpr`` never
    influences selection. All three reported metrics (PR-AUC, ROC-AUC, F1 with
    precision and recall) are recorded per trial so each is inspectable later.
  - **Determinism.** Per-fold, per-trial sampling RNGs are derived from the
    config seed and the (outer fold, trial, inner fold) coordinates, and the
    Optuna sampler is seeded per outer fold, so a run is reproducible.

Importing this module binds names only; it performs no input/output and
configures no logging at import time (the Optuna verbosity is set inside
:func:`run_nested_cv`).
"""

from __future__ import annotations

import logging
import time
from contextlib import AbstractContextManager, nullcontext
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Final

import numpy as np
import optuna
import pandas as pd
from optuna.samplers import TPESampler

from utils.hp_search import hp_boundary_diagnostics, suggest_xgb_params
from utils.io import (
    PER_FOLD_METRICS_SCHEMA,
    POOLED_METRICS_SCHEMA,
    SUMMARY_METRICS_SCHEMA,
)
from utils.metrics import (
    best_f1_threshold,
    metrics_at_threshold,
    parcel_aggregations,
    parcel_metrics,
    pixel_metrics,
)
from utils.models.xgboost import (
    feature_importance,
    make_xgb_classifier,
    predict_ogf_proba,
    train_xgb_fold,
)
from utils.sampling import SAMPLING_MODES, build_sample_weights
from utils.terminology import (
    ARCHITECTURES,
    FOLD_IDS,
    MODEL_INPUT_BANDS,
    MODEL_INPUTS,
    N_HP_TRIALS,
    N_PER_PARCEL,
    SEED,
)

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping, Sequence

    from numpy.typing import NDArray

logger = logging.getLogger(__name__)

# Permitted compute devices, passed straight through to XGBoost.
DEVICES: Final[tuple[str, ...]] = ("cuda", "cpu")

# Mean parcel metrics recorded per inner search (the inner objective's outputs).
_INNER_METRICS: Final[tuple[str, ...]] = ("pr_auc", "roc_auc", "f1", "precision", "recall")

# Metrics summarised across outer folds (the manuscript reports mean [min-max]):
# the threshold-free scores, the in-sample max-F1 operating point, and the
# honest inner-threshold operating point.
_SUMMARY_METRICS: Final[tuple[str, ...]] = (
    "pr_auc",
    "roc_auc",
    "threshold",
    "f1",
    "precision",
    "recall",
    "inner_threshold",
    "f1_inner_threshold",
    "precision_inner_threshold",
    "recall_inner_threshold",
)

# Fallback decision threshold when a pooled inner set is single-class.
_FALLBACK_THRESHOLD: Final[float] = 0.5

# Prediction levels.
_PIXEL: Final[str] = "pixel"
_PARCEL: Final[str] = "parcel"


@dataclass(frozen=True, slots=True)
class NestedCVConfig:
    """Validated configuration for one feature set's nested cross-validation.

    Attributes:
        feature_set: Canonical feature-set name (in ``FEATURE_SETS``); selects the
            cached feature matrix and its band order.
        architecture: Canonical architecture name (in ``ARCHITECTURES``).
        sampling_mode: One of :data:`utils.sampling.SAMPLING_MODES`.
        n_per_parcel: Maximum training pixels sampled per parcel.
        n_hp_trials: Optuna trials per outer fold.
        seed: Base random seed.
        device: ``"cuda"`` or ``"cpu"``.
        n_jobs: CPU threads per XGBoost fit.
        fold_ids: Outer folds to run (a subset of ``FOLD_IDS``).
    """

    feature_set: str
    architecture: str = "xgboost"
    sampling_mode: str = "none"
    n_per_parcel: int = N_PER_PARCEL
    n_hp_trials: int = N_HP_TRIALS
    seed: int = SEED
    device: str = "cpu"
    n_jobs: int = 1
    fold_ids: tuple[int, ...] = FOLD_IDS

    def __post_init__(self) -> None:
        """Validate the configuration once, on construction.

        Raises:
            ValueError: If any field is outside its permitted set or range.
        """
        if self.feature_set not in MODEL_INPUTS:
            raise ValueError(
                f"unknown feature set {self.feature_set!r}; expected one of {list(MODEL_INPUTS)}"
            )
        if self.architecture not in ARCHITECTURES:
            raise ValueError(
                f"unknown architecture {self.architecture!r}; "
                f"expected one of {list(ARCHITECTURES)}"
            )
        if self.sampling_mode not in SAMPLING_MODES:
            raise ValueError(
                f"unknown sampling mode {self.sampling_mode!r}; "
                f"expected one of {list(SAMPLING_MODES)}"
            )
        if self.device not in DEVICES:
            raise ValueError(f"device must be one of {list(DEVICES)}, got {self.device!r}")
        if self.n_per_parcel <= 0:
            raise ValueError(f"n_per_parcel must be positive, got {self.n_per_parcel}")
        if self.n_hp_trials <= 0:
            raise ValueError(f"n_hp_trials must be positive, got {self.n_hp_trials}")
        if self.seed < 0:
            raise ValueError(f"seed must be non-negative, got {self.seed}")
        if self.n_jobs < 1:
            raise ValueError(f"n_jobs must be positive, got {self.n_jobs}")
        if not self.fold_ids:
            raise ValueError("fold_ids must not be empty")
        invalid = sorted(set(self.fold_ids) - set(FOLD_IDS))
        if invalid:
            raise ValueError(f"fold_ids {invalid} are outside FOLD_IDS {FOLD_IDS}")


@dataclass(frozen=True, slots=True)
class FoldArrays:
    """Training and validation arrays for one (train folds, val fold) split.

    Attributes:
        X_train: Sampled training features, shape ``(n_train, n_bands)``.
        y_train: Sampled training labels (0/1), length ``n_train``.
        train_sample_weight: Per-row training weights, length ``n_train``.
        X_val: All validation features, shape ``(n_val, n_bands)``.
        y_val: All validation labels (0/1), length ``n_val``.
        val_pixel_ids: Validation pixel ids, length ``n_val``.
        val_parcel_ids: Validation parcel ids, length ``n_val``.
    """

    X_train: NDArray[np.float32]
    y_train: NDArray[np.int8]
    train_sample_weight: NDArray[np.float64]
    X_val: NDArray[np.float32]
    y_val: NDArray[np.int8]
    val_pixel_ids: NDArray[np.int64]
    val_parcel_ids: NDArray[np.int64]


@dataclass(frozen=True, slots=True)
class TrialRecord:
    """One Optuna trial's outcome, ready to persist (one row of ``hp_trials``).

    Attributes:
        trial_number: Zero-based trial index within the outer fold's study.
        state: Optuna trial state name (for example ``"COMPLETE"``).
        pr_auc: Mean parcel PR-AUC over the inner folds (the objective).
        roc_auc: Mean parcel ROC-AUC over the inner folds.
        f1: Mean parcel F1 over the inner folds.
        precision: Mean parcel precision over the inner folds.
        recall: Mean parcel recall over the inner folds.
        inner_pixel_threshold: F1-maximising threshold on the pooled inner pixel
            predictions, for honest application to the outer fold.
        inner_parcel_threshold: F1-maximising threshold on the pooled inner parcel
            predictions.
        params: The trial's tunable hyperparameters.
        cnn_epochs: Fixed outer-refit epoch count selected from the CNN inner
            fits, or ``None`` for XGBoost and failed CNN trials.
    """

    trial_number: int
    state: str
    pr_auc: float
    roc_auc: float
    f1: float
    precision: float
    recall: float
    inner_pixel_threshold: float
    inner_parcel_threshold: float
    params: Mapping[str, float | int]
    cnn_epochs: int | None = None

    def to_row(self) -> dict[str, object]:
        """Flatten to a single table row (scalar columns plus the parameters)."""
        row: dict[str, object] = {
            "trial_number": int(self.trial_number),
            "state": self.state,
            "pr_auc": self.pr_auc,
            "roc_auc": self.roc_auc,
            "f1": self.f1,
            "precision": self.precision,
            "recall": self.recall,
            "inner_pixel_threshold": self.inner_pixel_threshold,
            "inner_parcel_threshold": self.inner_parcel_threshold,
            "cnn_epochs": (float(self.cnn_epochs) if self.cnn_epochs is not None else float("nan")),
        }
        row.update(self.params)
        return row


@dataclass(frozen=True, slots=True)
class FoldResult:
    """Everything produced for one outer fold.

    Attributes:
        outer_fold: The held-out outer fold id.
        best_params: Selected hyperparameters.
        pixel_metrics: Outer-fold pixel metrics at the in-sample max-F1 threshold.
        parcel_metrics: Outer-fold parcel metrics at the in-sample max-F1 threshold.
        pixel_metrics_inner: Outer-fold pixel F1/precision/recall at the
            inner-tuned threshold (honest operating point).
        parcel_metrics_inner: Outer-fold parcel F1/precision/recall at the
            inner-tuned threshold.
        pixel_predictions: Per-pixel out-of-fold predictions
            (:data:`utils.io.PIXEL_PREDICTIONS_SCHEMA`).
        parcel_predictions: Per-parcel out-of-fold predictions
            (:data:`utils.io.PARCEL_PREDICTIONS_SCHEMA`).
        boundary: Hyperparameter boundary diagnostics.
        importance: Gain-based feature importance.
        best_inner_pr_auc: Winning mean inner-fold parcel PR-AUC.
        cnn_epochs: Fixed epoch count used for the CNN outer refit, or ``None``
            for XGBoost.
        n_train: Sampled training pixel count for the retrain.
        n_val: Outer-fold validation pixel count.
        seconds: Wall-clock seconds spent on the fold.
    """

    outer_fold: int
    best_params: dict[str, float | int]
    pixel_metrics: dict[str, float]
    parcel_metrics: dict[str, float]
    pixel_predictions: pd.DataFrame
    parcel_predictions: pd.DataFrame
    pixel_metrics_inner: dict[str, float] = field(default_factory=dict)
    parcel_metrics_inner: dict[str, float] = field(default_factory=dict)
    boundary: pd.DataFrame = field(default_factory=pd.DataFrame)
    importance: pd.DataFrame = field(default_factory=pd.DataFrame)
    best_inner_pr_auc: float = float("nan")
    cnn_epochs: int | None = None
    n_train: int = 0
    n_val: int = 0
    seconds: float = 0.0

    @classmethod
    def from_predictions(
        cls,
        outer_fold: int,
        pixel_predictions: pd.DataFrame,
        parcel_predictions: pd.DataFrame,
        *,
        best_params: dict[str, float | int] | None = None,
        boundary: pd.DataFrame | None = None,
        importance: pd.DataFrame | None = None,
        inner_pixel_threshold: float | None = None,
        inner_parcel_threshold: float | None = None,
        best_inner_pr_auc: float = float("nan"),
        cnn_epochs: int | None = None,
        n_train: int = 0,
        seconds: float = 0.0,
    ) -> FoldResult:
        """Build a :class:`FoldResult`, computing metrics from the predictions.

        This is the single place metrics are derived from predictions, so a
        freshly computed fold and a fold reloaded from disk on resume score
        identically. Both operating points are computed: the in-sample max-F1
        point and the honest point at the inner-tuned threshold (defaulting to
        0.5 when a threshold is not supplied).

        Args:
            outer_fold: The held-out outer fold id.
            pixel_predictions: Per-pixel predictions.
            parcel_predictions: Per-parcel predictions.
            best_params: Selected hyperparameters, if known.
            boundary: Boundary diagnostics, if known.
            importance: Feature importance, if known.
            inner_pixel_threshold: Inner-tuned pixel threshold; defaults to 0.5.
            inner_parcel_threshold: Inner-tuned parcel threshold; defaults to 0.5.
            best_inner_pr_auc: Winning mean inner-fold parcel PR-AUC, if known.
            cnn_epochs: Fixed CNN epoch count used for the outer refit, if
                applicable.
            n_train: Sampled training pixel count, if known.
            seconds: Wall-clock seconds for the fold, if known.

        Returns:
            The assembled :class:`FoldResult`.
        """
        pixel_y = pixel_predictions["y_true"].to_numpy()
        pixel_p = pixel_predictions["p"].to_numpy()
        parcel_y = parcel_predictions["y_true"].to_numpy()
        parcel_p = parcel_predictions["p_mean"].to_numpy()
        pixel_threshold = (
            inner_pixel_threshold if inner_pixel_threshold is not None else _FALLBACK_THRESHOLD
        )
        parcel_threshold = (
            inner_parcel_threshold if inner_parcel_threshold is not None else _FALLBACK_THRESHOLD
        )
        return cls(
            outer_fold=int(outer_fold),
            best_params=dict(best_params) if best_params is not None else {},
            pixel_metrics=pixel_metrics(pixel_y, pixel_p),
            parcel_metrics=parcel_metrics(parcel_y, parcel_p),
            pixel_metrics_inner=metrics_at_threshold(pixel_y, pixel_p, pixel_threshold),
            parcel_metrics_inner=metrics_at_threshold(parcel_y, parcel_p, parcel_threshold),
            pixel_predictions=pixel_predictions,
            parcel_predictions=parcel_predictions,
            boundary=boundary if boundary is not None else pd.DataFrame(),
            importance=importance if importance is not None else pd.DataFrame(),
            best_inner_pr_auc=float(best_inner_pr_auc),
            cnn_epochs=int(cnn_epochs) if cnn_epochs is not None else None,
            n_train=int(n_train),
            n_val=int(len(pixel_predictions)),
            seconds=float(seconds),
        )


@dataclass(frozen=True, slots=True)
class AggregatedMetrics:
    """Per-fold, summary and pooled metric tables for a completed run.

    Attributes:
        per_fold: One row per outer fold and level
            (:data:`utils.io.PER_FOLD_METRICS_SCHEMA`).
        summary: Across-fold mean, sd, min and max per level and metric
            (:data:`utils.io.SUMMARY_METRICS_SCHEMA`).
        pooled_pixel: Pooled pixel metrics over the concatenated predictions
            (:data:`utils.io.POOLED_METRICS_SCHEMA`).
        pooled_parcel: Pooled parcel metrics over the concatenated predictions.
    """

    per_fold: pd.DataFrame
    summary: pd.DataFrame
    pooled_pixel: pd.DataFrame
    pooled_parcel: pd.DataFrame


# --------------------------------------------------------------------------- #
# Pure helpers (unit-tested on synthetic arrays).
# --------------------------------------------------------------------------- #


def sample_train_rows(
    pixel_index_subset: pd.DataFrame, n_per_parcel: int, rng: np.random.Generator
) -> NDArray[np.int64]:
    """Sample up to ``n_per_parcel`` row positions per parcel, without replacement.

    Args:
        pixel_index_subset: A subset of the pixel index (any rows) with a
            ``parcel_id`` column. Positions are zero-based into this subset.
        n_per_parcel: Maximum rows to keep per parcel.
        rng: Seeded NumPy generator used for the per-parcel draws.

    Returns:
        Sorted, unique zero-based positions selecting at most ``n_per_parcel``
        rows for every parcel present.
    """
    parcel_ids = pixel_index_subset["parcel_id"].to_numpy()
    order = np.argsort(parcel_ids, kind="stable")
    sorted_ids = parcel_ids[order]
    boundaries = np.flatnonzero(np.diff(sorted_ids)) + 1
    groups = np.split(order, boundaries) if order.size else []

    selected: list[NDArray[np.int64]] = []
    for group in groups:
        if group.size <= n_per_parcel:
            selected.append(group)
        else:
            selected.append(rng.choice(group, size=n_per_parcel, replace=False))
    if not selected:
        return np.empty(0, dtype=np.int64)
    out = np.concatenate(selected).astype(np.int64)
    out.sort()
    return out


def build_fold_arrays(
    memmap: np.ndarray,
    pixel_index: pd.DataFrame,
    *,
    train_folds: Sequence[int],
    val_fold: int,
    sampling_mode: str,
    n_per_parcel: int,
    rng: np.random.Generator,
) -> FoldArrays:
    """Assemble sampled training and full validation arrays for one split.

    Invalid pixels (``valid == False`` in ``pixel_index``) are excluded from both
    sides. Training rows are sampled per parcel; validation keeps every pixel of
    the held-out parcels. Feature rows are copied out of ``memmap`` by advanced
    indexing, so only the needed rows enter RAM.

    Args:
        memmap: Feature matrix aligned row-for-row with ``pixel_index``.
        pixel_index: Pixel index carrying ``parcel_id``, ``fold_id``, ``ogf``,
            ``forest_type_corine``, ``pixel_id`` and a boolean ``valid`` column.
        train_folds: Fold ids contributing training pixels.
        val_fold: The single fold id whose pixels form the validation set.
        sampling_mode: One of :data:`utils.sampling.SAMPLING_MODES`.
        n_per_parcel: Maximum training pixels per parcel.
        rng: Seeded generator for the per-parcel sampling.

    Returns:
        The :class:`FoldArrays` for this split.
    """
    valid = pixel_index["valid"].to_numpy()
    fold = pixel_index["fold_id"].to_numpy()
    ogf = pixel_index["ogf"].to_numpy()
    parcel_ids = pixel_index["parcel_id"].to_numpy()
    pixel_ids = pixel_index["pixel_id"].to_numpy()
    # CORINE forest type drives the stratified_forest_type weighting (coverage).
    forest_type = pixel_index["forest_type_corine"].to_numpy()

    train_all = np.flatnonzero(np.isin(fold, np.asarray(train_folds)) & valid)
    sampled_local = sample_train_rows(pixel_index.iloc[train_all], n_per_parcel, rng)
    train_positions = train_all[sampled_local]
    val_positions = np.flatnonzero((fold == val_fold) & valid)

    y_train = ogf[train_positions].astype(np.int8)
    weight = build_sample_weights(
        sampling_mode, labels=y_train, forest_type=forest_type[train_positions]
    )
    return FoldArrays(
        X_train=np.asarray(memmap[train_positions], dtype=np.float32),
        y_train=y_train,
        train_sample_weight=weight,
        X_val=np.asarray(memmap[val_positions], dtype=np.float32),
        y_val=ogf[val_positions].astype(np.int8),
        val_pixel_ids=pixel_ids[val_positions].astype(np.int64),
        val_parcel_ids=parcel_ids[val_positions].astype(np.int64),
    )


def _parcel_labels(pixel_index: pd.DataFrame) -> dict[int, int]:
    """Map each parcel id to its single old-growth label."""
    unique = pixel_index.drop_duplicates("parcel_id")
    return {
        int(pid): int(label) for pid, label in zip(unique["parcel_id"], unique["ogf"], strict=True)
    }


def _parcel_truth(
    parcel_ids: NDArray[np.int64], parcel_label: Mapping[int, int]
) -> NDArray[np.int8]:
    """Look up the per-parcel label for an array of parcel ids."""
    return np.array([parcel_label[int(pid)] for pid in parcel_ids], dtype=np.int8)


def _pooled_f1_threshold(y_true: NDArray[np.int8], y_prob: NDArray[np.float64]) -> float:
    """F1-maximising threshold on a pooled set, falling back to 0.5 if single-class."""
    if np.unique(y_true).size < 2:
        return _FALLBACK_THRESHOLD
    return float(best_f1_threshold(y_true, y_prob).threshold)


def _pixel_prediction_frame(
    pixel_ids: NDArray[np.int64], outer_fold: int, y_true: NDArray[np.int8], p: NDArray[np.float64]
) -> pd.DataFrame:
    """Build a per-pixel prediction frame (:data:`utils.io.PIXEL_PREDICTIONS_SCHEMA`)."""
    return pd.DataFrame(
        {
            "pixel_id": pixel_ids.astype(np.int64),
            "outer_fold": np.full(pixel_ids.shape[0], outer_fold, dtype=np.int8),
            "y_true": y_true.astype(np.int8),
            "p": p.astype(np.float64),
        }
    )


def _parcel_prediction_frame(
    aggregations: pd.DataFrame, outer_fold: int, y_true: NDArray[np.int8]
) -> pd.DataFrame:
    """Build a per-parcel prediction frame (:data:`utils.io.PARCEL_PREDICTIONS_SCHEMA`)."""
    return pd.DataFrame(
        {
            "parcel_id": aggregations["parcel_id"].to_numpy().astype(np.int64),
            "outer_fold": np.full(len(aggregations), outer_fold, dtype=np.int8),
            "y_true": y_true.astype(np.int8),
            "p_mean": aggregations["p_mean"].to_numpy().astype(np.float64),
            "n_pixels": aggregations["n_pixels"].to_numpy().astype(np.int64),
        }
    )


def _per_fold_row(
    outer_fold: int,
    level: str,
    insample: Mapping[str, float],
    inner: Mapping[str, float],
) -> dict[str, object]:
    """One per-fold metrics row: in-sample max-F1 point plus inner-threshold point."""
    return {
        "outer_fold": outer_fold,
        "level": level,
        **insample,
        "inner_threshold": inner["threshold"],
        "f1_inner_threshold": inner["f1"],
        "precision_inner_threshold": inner["precision"],
        "recall_inner_threshold": inner["recall"],
    }


def aggregate_pooled_metrics(fold_results: Sequence[FoldResult]) -> AggregatedMetrics:
    """Build the per-fold, summary and pooled metric tables.

    The per-fold table reuses each fold's stored metrics; the summary takes the
    across-fold mean, sd (sample), min and max of every ranking metric; the
    pooled metrics are recomputed on the concatenation of every fold's
    out-of-fold predictions (the manuscript's pooled operating point).

    Args:
        fold_results: One :class:`FoldResult` per outer fold (any order).

    Returns:
        The :class:`AggregatedMetrics` tables.

    Raises:
        ValueError: If ``fold_results`` is empty.
    """
    if not fold_results:
        raise ValueError("fold_results is empty; nothing to aggregate.")

    ordered = sorted(fold_results, key=lambda result: result.outer_fold)

    per_fold_rows: list[dict[str, object]] = []
    for result in ordered:
        per_fold_rows.append(
            _per_fold_row(
                result.outer_fold, _PIXEL, result.pixel_metrics, result.pixel_metrics_inner
            )
        )
        per_fold_rows.append(
            _per_fold_row(
                result.outer_fold, _PARCEL, result.parcel_metrics, result.parcel_metrics_inner
            )
        )
    per_fold = pd.DataFrame(per_fold_rows)
    per_fold["outer_fold"] = per_fold["outer_fold"].astype("int64")
    per_fold["n"] = per_fold["n"].astype("int64")
    per_fold = per_fold[list(PER_FOLD_METRICS_SCHEMA)]

    summary_rows: list[dict[str, object]] = []
    for level in (_PIXEL, _PARCEL):
        level_rows = per_fold[per_fold["level"] == level]
        for metric in _SUMMARY_METRICS:
            values = level_rows[metric].to_numpy(dtype=np.float64)
            summary_rows.append(
                {
                    "level": level,
                    "metric": metric,
                    "mean": float(np.mean(values)),
                    "sd": float(np.std(values, ddof=1)) if values.size > 1 else float("nan"),
                    "min": float(np.min(values)),
                    "max": float(np.max(values)),
                }
            )
    summary = pd.DataFrame(summary_rows)[list(SUMMARY_METRICS_SCHEMA)]

    pooled_pixel = pd.concat([r.pixel_predictions for r in ordered], ignore_index=True)
    pooled_parcel = pd.concat([r.parcel_predictions for r in ordered], ignore_index=True)
    pooled_pixel_metrics = pixel_metrics(
        pooled_pixel["y_true"].to_numpy(), pooled_pixel["p"].to_numpy()
    )
    pooled_parcel_metrics = parcel_metrics(
        pooled_parcel["y_true"].to_numpy(), pooled_parcel["p_mean"].to_numpy()
    )

    return AggregatedMetrics(
        per_fold=per_fold,
        summary=summary,
        pooled_pixel=_metrics_to_long(pooled_pixel_metrics),
        pooled_parcel=_metrics_to_long(pooled_parcel_metrics),
    )


def _metrics_to_long(metrics: Mapping[str, float]) -> pd.DataFrame:
    """Convert a metrics dict to a long ``metric``/``value`` frame."""
    return pd.DataFrame(
        {
            "metric": list(metrics.keys()),
            "value": [float(value) for value in metrics.values()],
        }
    )[list(POOLED_METRICS_SCHEMA)]


# --------------------------------------------------------------------------- #
# Optuna inner objective and the nested-CV driver.
# --------------------------------------------------------------------------- #


def _study_seed(seed: int, outer_fold: int) -> int:
    """Derive a sampler seed for one outer fold's study."""
    return int(np.random.SeedSequence([seed, outer_fold]).generate_state(1)[0])


def _trial_record(trial: optuna.trial.FrozenTrial) -> TrialRecord:
    """Convert a finished Optuna trial into a :class:`TrialRecord`."""
    attrs = trial.user_attrs

    def score(name: str) -> float:
        value = attrs.get(name)
        return float(value) if value is not None else float("nan")

    return TrialRecord(
        trial_number=int(trial.number),
        state=trial.state.name,
        pr_auc=score("pr_auc"),
        roc_auc=score("roc_auc"),
        f1=score("f1"),
        precision=score("precision"),
        recall=score("recall"),
        inner_pixel_threshold=score("inner_pixel_threshold"),
        inner_parcel_threshold=score("inner_parcel_threshold"),
        params=dict(trial.params),
        cnn_epochs=(int(attrs["cnn_epochs"]) if attrs.get("cnn_epochs") is not None else None),
    )


def _inner_objective(
    trial: optuna.trial.Trial,
    *,
    config: NestedCVConfig,
    pixel_index: pd.DataFrame,
    memmap: np.ndarray,
    outer_fold: int,
    inner_folds: Sequence[int],
    parcel_label: Mapping[int, int],
    fit_guard: Callable[[], AbstractContextManager[object]],
) -> float:
    """Score one hyperparameter trial as the mean parcel PR-AUC over inner folds.

    For each inner validation fold, the model trains on the sampled pixels of the
    other inner folds, predicts that fold's pixels, aggregates to parcels and
    computes parcel metrics. The objective is the mean parcel PR-AUC; the other
    mean parcel metrics are stored as trial user attributes. The inner-fold
    predictions are also pooled (no extra fits) to derive the F1-maximising pixel
    and parcel thresholds, stored as user attributes for honest outer-fold F1.
    """
    params = suggest_xgb_params(trial)
    per_fold: list[dict[str, float]] = []
    pixel_truth_pool: list[NDArray[np.int8]] = []
    pixel_prob_pool: list[NDArray[np.float64]] = []
    parcel_truth_pool: list[NDArray[np.int8]] = []
    parcel_prob_pool: list[NDArray[np.float64]] = []
    for inner_val in inner_folds:
        train_inner = tuple(f for f in inner_folds if f != inner_val)
        rng = np.random.default_rng([config.seed, outer_fold, trial.number, inner_val])
        arrays = build_fold_arrays(
            memmap,
            pixel_index,
            train_folds=train_inner,
            val_fold=inner_val,
            sampling_mode=config.sampling_mode,
            n_per_parcel=config.n_per_parcel,
            rng=rng,
        )
        with fit_guard():
            probabilities = train_xgb_fold(
                params,
                arrays.X_train,
                arrays.y_train,
                arrays.X_val,
                sample_weight=arrays.train_sample_weight,
                device=config.device,
                n_jobs=config.n_jobs,
                seed=config.seed,
            )
        aggregations = parcel_aggregations(arrays.val_parcel_ids, probabilities)
        truth = _parcel_truth(aggregations["parcel_id"].to_numpy(), parcel_label)
        per_fold.append(parcel_metrics(truth, aggregations["p_mean"].to_numpy()))
        pixel_truth_pool.append(arrays.y_val)
        pixel_prob_pool.append(probabilities)
        parcel_truth_pool.append(truth)
        parcel_prob_pool.append(aggregations["p_mean"].to_numpy())

    means = {metric: float(np.mean([m[metric] for m in per_fold])) for metric in _INNER_METRICS}
    for metric, value in means.items():
        trial.set_user_attr(metric, value)
    trial.set_user_attr(
        "inner_pixel_threshold",
        _pooled_f1_threshold(np.concatenate(pixel_truth_pool), np.concatenate(pixel_prob_pool)),
    )
    trial.set_user_attr(
        "inner_parcel_threshold",
        _pooled_f1_threshold(np.concatenate(parcel_truth_pool), np.concatenate(parcel_prob_pool)),
    )
    return means["pr_auc"]


def run_nested_cv(
    config: NestedCVConfig,
    pixel_index: pd.DataFrame,
    memmap: np.ndarray,
    *,
    outer_folds: Sequence[int] | None = None,
    on_trial_end: Callable[[int, TrialRecord], None] | None = None,
    on_fold_end: Callable[[int, FoldResult], None] | None = None,
    fit_guard: Callable[[], AbstractContextManager[object]] | None = None,
) -> list[FoldResult]:
    """Run the nested cross-validation for one feature set.

    For each outer fold: run a seeded Optuna TPE study over the inner objective,
    pick the best hyperparameters (maximum mean parcel PR-AUC), retrain on the
    sampled pixels of all inner folds, predict every pixel of the outer fold, and
    assemble a :class:`FoldResult` with predictions, metrics, boundary
    diagnostics and feature importance. The callbacks let a driver persist
    incrementally, so a crash loses at most the in-flight trial.

    The inner folds for an outer fold are always all *other* folds in
    ``config.fold_ids``. The ``outer_folds`` argument restricts only which outer
    folds are executed this call (for a smoke run or to resume the remaining
    folds), without shrinking the inner search.

    Args:
        config: The run configuration.
        pixel_index: Pixel index with a boolean ``valid`` column already merged
            in, aligned row-for-row with ``memmap``.
        memmap: Feature matrix for ``config.feature_set``.
        outer_folds: Outer folds to execute, a subset of ``config.fold_ids``.
            Defaults to every fold in ``config.fold_ids``.
        on_trial_end: Called with ``(outer_fold, TrialRecord)`` as each trial
            completes.
        on_fold_end: Called with ``(outer_fold, FoldResult)`` as each fold
            completes.
        fit_guard: Optional context-manager factory entered around every model
            fit (used to cap concurrent GPU fits); defaults to a no-op.

    Returns:
        One :class:`FoldResult` per executed outer fold, in ascending fold order.

    Raises:
        ValueError: If ``outer_folds`` is not a subset of ``config.fold_ids``.
    """
    if outer_folds is None:
        execute = list(config.fold_ids)
    else:
        unknown = sorted(set(outer_folds) - set(config.fold_ids))
        if unknown:
            raise ValueError(
                f"outer_folds {unknown} are not in config.fold_ids {list(config.fold_ids)}"
            )
        execute = [fold for fold in config.fold_ids if fold in set(outer_folds)]

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    guard = fit_guard if fit_guard is not None else nullcontext
    parcel_label = _parcel_labels(pixel_index)
    feature_names = list(MODEL_INPUT_BANDS[config.feature_set])

    results: list[FoldResult] = []
    for outer_fold in execute:
        start = time.perf_counter()
        inner_folds = tuple(f for f in config.fold_ids if f != outer_fold)
        logger.info(
            "Outer fold %d: starting inner search (%d trials over inner folds %s).",
            outer_fold,
            config.n_hp_trials,
            list(inner_folds),
        )

        study = optuna.create_study(
            direction="maximize", sampler=TPESampler(seed=_study_seed(config.seed, outer_fold))
        )

        def objective(trial: optuna.trial.Trial, _outer: int = outer_fold) -> float:
            return _inner_objective(
                trial,
                config=config,
                pixel_index=pixel_index,
                memmap=memmap,
                outer_fold=_outer,
                inner_folds=tuple(f for f in config.fold_ids if f != _outer),
                parcel_label=parcel_label,
                fit_guard=guard,
            )

        def trial_callback(
            study: optuna.study.Study,
            trial: optuna.trial.FrozenTrial,
            _outer: int = outer_fold,
        ) -> None:
            record = _trial_record(trial)
            logger.info(
                "Outer fold %d trial %d (%s): parcel PR-AUC %.4f, ROC-AUC %.4f, F1 %.4f.",
                _outer,
                record.trial_number,
                record.state,
                record.pr_auc,
                record.roc_auc,
                record.f1,
            )
            if on_trial_end is not None:
                on_trial_end(_outer, record)

        study.optimize(objective, n_trials=config.n_hp_trials, callbacks=[trial_callback])
        best_params = dict(study.best_params)
        # The selected trial's inner-tuned thresholds, applied to the outer fold.
        best_attrs = study.best_trial.user_attrs
        inner_pixel_threshold = float(best_attrs.get("inner_pixel_threshold", _FALLBACK_THRESHOLD))
        inner_parcel_threshold = float(
            best_attrs.get("inner_parcel_threshold", _FALLBACK_THRESHOLD)
        )
        logger.info(
            "Outer fold %d: best mean parcel PR-AUC %.4f with params %s "
            "(inner thresholds pixel=%.3f parcel=%.3f).",
            outer_fold,
            study.best_value,
            best_params,
            inner_pixel_threshold,
            inner_parcel_threshold,
        )

        rng = np.random.default_rng([config.seed, outer_fold])
        arrays = build_fold_arrays(
            memmap,
            pixel_index,
            train_folds=inner_folds,
            val_fold=outer_fold,
            sampling_mode=config.sampling_mode,
            n_per_parcel=config.n_per_parcel,
            rng=rng,
        )
        model = make_xgb_classifier(
            best_params, device=config.device, n_jobs=config.n_jobs, seed=config.seed
        )
        with guard():
            model.fit(arrays.X_train, arrays.y_train, sample_weight=arrays.train_sample_weight)
        probabilities = predict_ogf_proba(model, arrays.X_val)

        pixel_predictions = _pixel_prediction_frame(
            arrays.val_pixel_ids, outer_fold, arrays.y_val, probabilities
        )
        aggregations = parcel_aggregations(arrays.val_parcel_ids, probabilities)
        parcel_truth = _parcel_truth(aggregations["parcel_id"].to_numpy(), parcel_label)
        parcel_predictions = _parcel_prediction_frame(aggregations, outer_fold, parcel_truth)

        fold_result = FoldResult.from_predictions(
            outer_fold,
            pixel_predictions,
            parcel_predictions,
            best_params=best_params,
            boundary=hp_boundary_diagnostics(best_params),
            importance=feature_importance(model, feature_names),
            inner_pixel_threshold=inner_pixel_threshold,
            inner_parcel_threshold=inner_parcel_threshold,
            best_inner_pr_auc=float(study.best_value),
            n_train=int(arrays.y_train.shape[0]),
            seconds=time.perf_counter() - start,
        )
        logger.info(
            "Outer fold %d done in %.1fs: parcel PR-AUC %.4f, parcel F1 %.4f (in-sample) / "
            "%.4f (inner thr %.3f).",
            outer_fold,
            fold_result.seconds,
            fold_result.parcel_metrics["pr_auc"],
            fold_result.parcel_metrics["f1"],
            fold_result.parcel_metrics_inner["f1"],
            fold_result.parcel_metrics_inner["threshold"],
        )
        if on_fold_end is not None:
            on_fold_end(outer_fold, fold_result)
        results.append(fold_result)

    return results


def fixed_hp_fold_result(
    config: NestedCVConfig,
    pixel_index: pd.DataFrame,
    memmap: np.ndarray,
    best_params: Mapping[str, float | int],
    *,
    outer_fold: int,
) -> FoldResult:
    """Refit one outer fold with fixed hyperparameters (no inner search) and score it.

    The sensitivity-analysis counterpart of the outer refit inside
    :func:`run_nested_cv`: it trains the XGBoost model with the supplied
    ``best_params`` on the sampled inner-fold pixels and scores the held-out outer
    fold, reusing the same sampling, seeding and parcel aggregation. With
    ``config.sampling_mode == "none"`` and the unmodified ``valid`` mask it
    reproduces the matching ``run_nested_cv`` fold exactly, so a caller can vary a
    single factor (the sampling mode, or the ``valid`` mask for a spatial gap) and
    attribute any change to that factor alone. No inner Optuna search runs, so the
    refit is a few seconds rather than a full nested-CV fold.

    Args:
        config: The run configuration; ``sampling_mode``, ``n_per_parcel``,
            ``seed``, ``device``, ``n_jobs`` and ``fold_ids`` are honoured.
        pixel_index: Pixel index with a boolean ``valid`` column already merged
            in, aligned row-for-row with ``memmap``. Pass a copy with ``valid``
            edited to drop training pixels for a spatial-gap arm.
        memmap: Feature matrix for ``config.feature_set``.
        best_params: The fixed XGBoost hyperparameters for this fold (for example
            loaded from ``best_params_fold{k}.json``).
        outer_fold: The held-out outer fold id; the inner folds are all other
            folds in ``config.fold_ids``.

    Returns:
        The :class:`FoldResult` for this fold (no boundary or importance tables;
        the sensitivity analyses do not use them).
    """
    inner_folds = tuple(f for f in config.fold_ids if f != outer_fold)
    parcel_label = _parcel_labels(pixel_index)
    rng = np.random.default_rng([config.seed, outer_fold])
    arrays = build_fold_arrays(
        memmap,
        pixel_index,
        train_folds=inner_folds,
        val_fold=outer_fold,
        sampling_mode=config.sampling_mode,
        n_per_parcel=config.n_per_parcel,
        rng=rng,
    )
    probabilities = train_xgb_fold(
        dict(best_params),
        arrays.X_train,
        arrays.y_train,
        arrays.X_val,
        sample_weight=arrays.train_sample_weight,
        device=config.device,
        n_jobs=config.n_jobs,
        seed=config.seed,
    )
    pixel_predictions = _pixel_prediction_frame(
        arrays.val_pixel_ids, outer_fold, arrays.y_val, probabilities
    )
    aggregations = parcel_aggregations(arrays.val_parcel_ids, probabilities)
    parcel_truth = _parcel_truth(aggregations["parcel_id"].to_numpy(), parcel_label)
    parcel_predictions = _parcel_prediction_frame(aggregations, outer_fold, parcel_truth)
    return FoldResult.from_predictions(
        outer_fold,
        pixel_predictions,
        parcel_predictions,
        best_params=dict(best_params),
        n_train=int(arrays.y_train.shape[0]),
    )


# --------------------------------------------------------------------------- #
# Receptive-field CNN path (additive; reuses the shared helpers above).
#
# The torch-dependent imports (utils.patches, utils.models.cnn) live inside these
# functions so importing utils.cv stays torch-free and the XGBoost path is wholly
# unaffected. The outer-fold loop is duplicated rather than refactored, the
# accepted tidiness cost of leaving the XGBoost driver byte-for-byte unchanged.
# --------------------------------------------------------------------------- #


def _loader_seed(seed: int, *coords: int) -> int:
    """Derive an integer seed for a patch loader's shuffling generator.

    The torch counterpart of :func:`_study_seed`: a stable, well-mixed seed from
    the config seed and the run coordinates, so a loader's shuffle is reproducible
    and independent across folds and trials.

    Args:
        seed: Base configuration seed.
        coords: Run coordinates (for example outer fold, trial, inner fold).

    Returns:
        A non-negative integer seed.
    """
    return int(np.random.SeedSequence([seed, *coords]).generate_state(1)[0])


def _cnn_epochs_from_fits(best_epochs: Sequence[int], warmup_epochs: int) -> int:
    """Mean inner-fold training length to use for the outer refit.

    Each inner fold contributes ``best_epoch + 1`` (the number of epochs actually
    run to reach the restored best). Folds that never improved
    (``best_epoch == -1``) carry no usable length and are excluded. The mean is
    rounded to an integer and floored at ``warmup_epochs + 1`` so the refit always
    completes its warm-up; with no usable fold the floor itself is returned.

    Args:
        best_epochs: One ``best_epoch`` per inner fold.
        warmup_epochs: Linear warm-up length, the lower clamp's basis.

    Returns:
        The refit epoch count.
    """
    lengths = [int(epoch) + 1 for epoch in best_epochs if epoch >= 0]
    floor = int(warmup_epochs) + 1
    if not lengths:
        return floor
    return max(floor, int(round(float(np.mean(lengths)))))


def _inner_objective_cnn(
    trial: optuna.trial.Trial,
    *,
    config: NestedCVConfig,
    pixel_index: pd.DataFrame,
    patch_cache: np.ndarray,
    patch_size: int,
    grid_shape: tuple[int, int],
    outer_fold: int,
    inner_folds: Sequence[int],
    parcel_label: Mapping[int, int],
    fit_guard: Callable[[], AbstractContextManager[object]],
) -> float:
    """Score one CNN trial as the mean parcel PR-AUC over the inner folds.

    The patch analogue of :func:`_inner_objective`: for each inner validation fold
    it builds patch loaders over the same pixels the XGBoost path would use, trains
    a receptive-field CNN with early stopping, takes the restored-best validation
    probabilities as the pixel predictions, aggregates them to parcels and computes
    parcel metrics. The objective is the mean parcel PR-AUC; the other mean parcel
    metrics and the pooled inner F1 thresholds are stored as trial user attributes,
    exactly as the XGBoost objective does. It additionally records ``cnn_epochs``,
    the mean inner training length, so the outer refit can train for the same
    number of epochs without a validation set.

    Patches are read from ``patch_cache`` (rows aligned with ``pixel_index``);
    ``grid_shape`` is the source grid ``(height, width)`` used for the edge drop.
    """
    from utils.hp_search import suggest_cnn_params
    from utils.models.cnn import shutdown_loader_workers, train_cnn_fold
    from utils.patches import build_patch_loaders
    from utils.terminology import CNN_WARMUP_EPOCHS

    params = suggest_cnn_params(trial)
    n_bands = int(patch_cache.shape[1])
    per_fold: list[dict[str, float]] = []
    pixel_truth_pool: list[NDArray[np.int8]] = []
    pixel_prob_pool: list[NDArray[np.float64]] = []
    parcel_truth_pool: list[NDArray[np.int8]] = []
    parcel_prob_pool: list[NDArray[np.float64]] = []
    best_epochs: list[int] = []
    for inner_val in inner_folds:
        train_inner = tuple(f for f in inner_folds if f != inner_val)
        rng = np.random.default_rng([config.seed, outer_fold, trial.number, inner_val])
        train_loader, val_loader, val_index = build_patch_loaders(
            patch_cache,
            pixel_index,
            grid_shape=grid_shape,
            train_folds=train_inner,
            val_fold=inner_val,
            sampling_mode=config.sampling_mode,
            n_per_parcel=config.n_per_parcel,
            patch_size=patch_size,
            batch_size=int(params["batch_size"]),
            rng=rng,
            # n_jobs is the per-fit CPU budget (as for XGBoost); reserve one core
            # for the training loop and let the rest load patches.
            num_workers=config.n_jobs - 1,
            seed=_loader_seed(config.seed, outer_fold, trial.number, inner_val),
        )
        try:
            with fit_guard():
                fit = train_cnn_fold(
                    params,
                    train_loader,
                    val_loader,
                    n_bands=n_bands,
                    patch_size=patch_size,
                    device=config.device,
                    seed=config.seed,
                )
        finally:
            shutdown_loader_workers(train_loader)
            shutdown_loader_workers(val_loader)
        probabilities = fit.val_probs
        aggregations = parcel_aggregations(val_index["parcel_id"].to_numpy(), probabilities)
        truth = _parcel_truth(aggregations["parcel_id"].to_numpy(), parcel_label)
        per_fold.append(parcel_metrics(truth, aggregations["p_mean"].to_numpy()))
        pixel_truth_pool.append(val_index["y_true"].to_numpy().astype(np.int8))
        pixel_prob_pool.append(probabilities)
        parcel_truth_pool.append(truth)
        parcel_prob_pool.append(aggregations["p_mean"].to_numpy())
        best_epochs.append(fit.best_epoch)

    means = {metric: float(np.mean([m[metric] for m in per_fold])) for metric in _INNER_METRICS}
    for metric, value in means.items():
        trial.set_user_attr(metric, value)
    trial.set_user_attr(
        "inner_pixel_threshold",
        _pooled_f1_threshold(np.concatenate(pixel_truth_pool), np.concatenate(pixel_prob_pool)),
    )
    trial.set_user_attr(
        "inner_parcel_threshold",
        _pooled_f1_threshold(np.concatenate(parcel_truth_pool), np.concatenate(parcel_prob_pool)),
    )
    trial.set_user_attr("cnn_epochs", _cnn_epochs_from_fits(best_epochs, CNN_WARMUP_EPOCHS))
    return means["pr_auc"]


def run_nested_cv_cnn(
    config: NestedCVConfig,
    pixel_index: pd.DataFrame,
    patch_cache: np.ndarray,
    patch_size: int,
    *,
    grid_shape: tuple[int, int],
    outer_folds: Sequence[int] | None = None,
    on_trial_end: Callable[[int, TrialRecord], None] | None = None,
    on_fold_end: Callable[[int, FoldResult], None] | None = None,
    fit_guard: Callable[[], AbstractContextManager[object]] | None = None,
) -> list[FoldResult]:
    """Run the receptive-field CNN nested cross-validation for one feature set.

    The patch-based counterpart of :func:`run_nested_cv`, sharing its outer-loop
    structure, Optuna seeding, callbacks and aggregation so the two architectures
    are scored by identical logic. For each outer fold it runs a seeded inner study
    over :func:`_inner_objective_cnn`, reads the selected trial's ``cnn_epochs``,
    refits on all inner folds for exactly that many epochs with
    :func:`utils.models.cnn.train_cnn_fixed` (no early stopping; the outer fold is
    used only to score), predicts the outer fold and assembles a
    :class:`FoldResult`. There is no feature-importance table for the CNN, so that
    field is left empty rather than fabricated.

    Args:
        config: The run configuration (``architecture`` should be one of the
            ``cnn_*`` names; the patch size is supplied separately).
        pixel_index: Pixel index with a boolean ``valid`` column already merged in,
            its rows aligned with ``patch_cache``.
        patch_cache: Patch cache of shape ``(n_pixels, n_bands, patch_size,
            patch_size)`` for ``config.feature_set`` (from
            ``scripts/build_patch_cache.py``).
        patch_size: Receptive-field patch side length; one of
            :data:`utils.terminology.CNN_PATCH_SIZES`.
        grid_shape: Source grid ``(height, width)``, used for the patch edge drop.
        outer_folds: Outer folds to execute, a subset of ``config.fold_ids``.
            Defaults to every fold in ``config.fold_ids``.
        on_trial_end: Called with ``(outer_fold, TrialRecord)`` as each trial
            completes.
        on_fold_end: Called with ``(outer_fold, FoldResult)`` as each fold
            completes.
        fit_guard: Optional context-manager factory entered around every model fit
            (used to cap concurrent GPU fits); defaults to a no-op.

    Returns:
        One :class:`FoldResult` per executed outer fold, in ascending fold order.

    Raises:
        ValueError: If ``outer_folds`` is not a subset of ``config.fold_ids``.
    """
    from utils.hp_search import cnn_boundary_diagnostics
    from utils.models.cnn import predict_proba, shutdown_loader_workers, train_cnn_fixed
    from utils.patches import build_patch_loaders
    from utils.terminology import CNN_WARMUP_EPOCHS

    if outer_folds is None:
        execute = list(config.fold_ids)
    else:
        unknown = sorted(set(outer_folds) - set(config.fold_ids))
        if unknown:
            raise ValueError(
                f"outer_folds {unknown} are not in config.fold_ids {list(config.fold_ids)}"
            )
        execute = [fold for fold in config.fold_ids if fold in set(outer_folds)]

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    guard = fit_guard if fit_guard is not None else nullcontext
    parcel_label = _parcel_labels(pixel_index)
    n_bands = int(patch_cache.shape[1])

    results: list[FoldResult] = []
    for outer_fold in execute:
        start = time.perf_counter()
        inner_folds = tuple(f for f in config.fold_ids if f != outer_fold)
        logger.info(
            "Outer fold %d: starting CNN inner search (%d trials over inner folds %s).",
            outer_fold,
            config.n_hp_trials,
            list(inner_folds),
        )

        study = optuna.create_study(
            direction="maximize", sampler=TPESampler(seed=_study_seed(config.seed, outer_fold))
        )

        def objective(trial: optuna.trial.Trial, _outer: int = outer_fold) -> float:
            return _inner_objective_cnn(
                trial,
                config=config,
                pixel_index=pixel_index,
                patch_cache=patch_cache,
                patch_size=patch_size,
                grid_shape=grid_shape,
                outer_fold=_outer,
                inner_folds=tuple(f for f in config.fold_ids if f != _outer),
                parcel_label=parcel_label,
                fit_guard=guard,
            )

        def trial_callback(
            study: optuna.study.Study,
            trial: optuna.trial.FrozenTrial,
            _outer: int = outer_fold,
        ) -> None:
            record = _trial_record(trial)
            logger.info(
                "Outer fold %d CNN trial %d (%s): parcel PR-AUC %.4f, ROC-AUC %.4f, F1 %.4f.",
                _outer,
                record.trial_number,
                record.state,
                record.pr_auc,
                record.roc_auc,
                record.f1,
            )
            if on_trial_end is not None:
                on_trial_end(_outer, record)

        study.optimize(objective, n_trials=config.n_hp_trials, callbacks=[trial_callback])
        best_params = dict(study.best_params)
        best_attrs = study.best_trial.user_attrs
        inner_pixel_threshold = float(best_attrs.get("inner_pixel_threshold", _FALLBACK_THRESHOLD))
        inner_parcel_threshold = float(
            best_attrs.get("inner_parcel_threshold", _FALLBACK_THRESHOLD)
        )
        cnn_epochs = int(best_attrs.get("cnn_epochs", CNN_WARMUP_EPOCHS + 1))
        logger.info(
            "Outer fold %d: best mean parcel PR-AUC %.4f over %d epochs with params %s "
            "(inner thresholds pixel=%.3f parcel=%.3f).",
            outer_fold,
            study.best_value,
            cnn_epochs,
            best_params,
            inner_pixel_threshold,
            inner_parcel_threshold,
        )

        rng = np.random.default_rng([config.seed, outer_fold])
        train_loader, val_loader, val_index = build_patch_loaders(
            patch_cache,
            pixel_index,
            grid_shape=grid_shape,
            train_folds=inner_folds,
            val_fold=outer_fold,
            sampling_mode=config.sampling_mode,
            n_per_parcel=config.n_per_parcel,
            patch_size=patch_size,
            batch_size=int(best_params["batch_size"]),
            rng=rng,
            # n_jobs is the per-fit CPU budget (as for XGBoost); reserve one core
            # for the training loop and let the rest load patches.
            num_workers=config.n_jobs - 1,
            seed=_loader_seed(config.seed, outer_fold),
        )
        n_train = len(train_loader.dataset)  # type: ignore[arg-type]
        try:
            with guard():
                model = train_cnn_fixed(
                    best_params,
                    train_loader,
                    n_bands=n_bands,
                    patch_size=patch_size,
                    n_epochs=cnn_epochs,
                    device=config.device,
                    seed=config.seed,
                )
            probabilities = predict_proba(model, val_loader, device=config.device)
        finally:
            shutdown_loader_workers(train_loader)
            shutdown_loader_workers(val_loader)

        val_pixel_ids = val_index["pixel_id"].to_numpy().astype(np.int64)
        val_parcel_ids = val_index["parcel_id"].to_numpy().astype(np.int64)
        y_val = val_index["y_true"].to_numpy().astype(np.int8)

        pixel_predictions = _pixel_prediction_frame(val_pixel_ids, outer_fold, y_val, probabilities)
        aggregations = parcel_aggregations(val_parcel_ids, probabilities)
        parcel_truth = _parcel_truth(aggregations["parcel_id"].to_numpy(), parcel_label)
        parcel_predictions = _parcel_prediction_frame(aggregations, outer_fold, parcel_truth)

        fold_result = FoldResult.from_predictions(
            outer_fold,
            pixel_predictions,
            parcel_predictions,
            best_params=best_params,
            boundary=cnn_boundary_diagnostics(best_params),
            inner_pixel_threshold=inner_pixel_threshold,
            inner_parcel_threshold=inner_parcel_threshold,
            best_inner_pr_auc=float(study.best_value),
            cnn_epochs=cnn_epochs,
            n_train=int(n_train),
            seconds=time.perf_counter() - start,
        )
        logger.info(
            "Outer fold %d CNN done in %.1fs: parcel PR-AUC %.4f, parcel F1 %.4f (in-sample) / "
            "%.4f (inner thr %.3f).",
            outer_fold,
            fold_result.seconds,
            fold_result.parcel_metrics["pr_auc"],
            fold_result.parcel_metrics["f1"],
            fold_result.parcel_metrics_inner["f1"],
            fold_result.parcel_metrics_inner["threshold"],
        )
        if on_fold_end is not None:
            on_fold_end(outer_fold, fold_result)
        results.append(fold_result)

    return results


def fixed_hp_fold_result_cnn(
    config: NestedCVConfig,
    pixel_index: pd.DataFrame,
    patch_cache: np.ndarray,
    best_params: Mapping[str, float | int],
    *,
    patch_size: int,
    grid_shape: tuple[int, int],
    cnn_epochs: int,
    outer_fold: int,
    fit_guard: Callable[[], AbstractContextManager[object]] | None = None,
) -> FoldResult:
    """Refit one outer fold of the receptive-field CNN with fixed hyperparameters.

    The CNN counterpart of :func:`fixed_hp_fold_result`: it reuses the outer-refit
    logic of :func:`run_nested_cv_cnn` (build the patch loaders for the inner/outer
    split, train for a fixed ``cnn_epochs`` with no early stopping, predict the
    held-out fold) but takes the per-fold ``best_params`` and ``cnn_epochs`` directly
    instead of running an inner Optuna search. With ``config.sampling_mode == "none"``
    and the unmodified ``valid`` mask it reproduces the matching ``run_nested_cv_cnn``
    fold exactly, so a caller can vary a single factor (here the ``valid`` mask for a
    spatial gap) and attribute any change to that factor alone.

    Args:
        config: The run configuration; ``sampling_mode``, ``n_per_parcel``, ``seed``,
            ``device``, ``n_jobs`` and ``fold_ids`` are honoured.
        pixel_index: Pixel index with a boolean ``valid`` column merged in, aligned
            row-for-row with ``patch_cache``. Pass a copy with ``valid`` edited to drop
            training pixels for a spatial-gap arm.
        patch_cache: Patch cache of shape ``(n_pixels, n_bands, patch_size,
            patch_size)`` for ``config.feature_set``.
        best_params: The fixed CNN hyperparameters for this fold (``n_channels``,
            ``lr``, ``weight_decay``, ``dropout``, ``batch_size``), for example loaded
            from ``best_params_fold{k}.json``.
        patch_size: Receptive-field patch side length.
        grid_shape: Source grid ``(height, width)`` for the patch edge drop.
        cnn_epochs: Fixed outer-refit epoch count for this fold (from the fold's
            training recipe); trained with no early stopping.
        outer_fold: The held-out outer fold id; the inner folds are all other folds in
            ``config.fold_ids``.
        fit_guard: Optional context-manager factory entered around the fit (to cap
            concurrent GPU fits); defaults to a no-op.

    Returns:
        The :class:`FoldResult` for this fold (no importance table).
    """
    from utils.hp_search import cnn_boundary_diagnostics
    from utils.models.cnn import predict_proba, shutdown_loader_workers, train_cnn_fixed
    from utils.patches import build_patch_loaders

    guard = fit_guard if fit_guard is not None else nullcontext
    parcel_label = _parcel_labels(pixel_index)
    n_bands = int(patch_cache.shape[1])
    inner_folds = tuple(f for f in config.fold_ids if f != outer_fold)
    params = dict(best_params)

    rng = np.random.default_rng([config.seed, outer_fold])
    train_loader, val_loader, val_index = build_patch_loaders(
        patch_cache,
        pixel_index,
        grid_shape=grid_shape,
        train_folds=inner_folds,
        val_fold=outer_fold,
        sampling_mode=config.sampling_mode,
        n_per_parcel=config.n_per_parcel,
        patch_size=patch_size,
        batch_size=int(params["batch_size"]),
        rng=rng,
        num_workers=config.n_jobs - 1,
        seed=_loader_seed(config.seed, outer_fold),
    )
    n_train = len(train_loader.dataset)  # type: ignore[arg-type]
    try:
        with guard():
            model = train_cnn_fixed(
                params,
                train_loader,
                n_bands=n_bands,
                patch_size=patch_size,
                n_epochs=cnn_epochs,
                device=config.device,
                seed=config.seed,
            )
        probabilities = predict_proba(model, val_loader, device=config.device)
    finally:
        shutdown_loader_workers(train_loader)
        shutdown_loader_workers(val_loader)

    val_pixel_ids = val_index["pixel_id"].to_numpy().astype(np.int64)
    val_parcel_ids = val_index["parcel_id"].to_numpy().astype(np.int64)
    y_val = val_index["y_true"].to_numpy().astype(np.int8)

    pixel_predictions = _pixel_prediction_frame(val_pixel_ids, outer_fold, y_val, probabilities)
    aggregations = parcel_aggregations(val_parcel_ids, probabilities)
    parcel_truth = _parcel_truth(aggregations["parcel_id"].to_numpy(), parcel_label)
    parcel_predictions = _parcel_prediction_frame(aggregations, outer_fold, parcel_truth)
    return FoldResult.from_predictions(
        outer_fold,
        pixel_predictions,
        parcel_predictions,
        best_params=params,
        boundary=cnn_boundary_diagnostics(params),
        cnn_epochs=int(cnn_epochs),
        n_train=int(n_train),
    )


__all__ = [
    "DEVICES",
    "AggregatedMetrics",
    "FoldArrays",
    "FoldResult",
    "NestedCVConfig",
    "TrialRecord",
    "_inner_objective_cnn",
    "aggregate_pooled_metrics",
    "build_fold_arrays",
    "fixed_hp_fold_result",
    "fixed_hp_fold_result_cnn",
    "run_nested_cv",
    "run_nested_cv_cnn",
    "sample_train_rows",
]
