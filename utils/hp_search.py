"""XGBoost hyperparameter search space and boundary diagnostics.

A single declarative search space (:data:`XGB_SEARCH_SPACE`) drives both the
Optuna suggestion function and the boundary report, so the bounds live in one
place. The space is mostly continuous (log-uniform where scale-free), which
suits the tree-structured Parzen-estimator sampler better than the coarse
categorical grids of the legacy script.

The module takes an Optuna ``trial`` by duck typing and does not import Optuna,
so it has no heavy dependency and no import side effects. It performs no
input/output or logging.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Final

import pandas as pd


@dataclass(frozen=True, slots=True)
class ParamSpec:
    """One hyperparameter's range.

    Attributes:
        kind: ``"int"`` or ``"float"``.
        low: Inclusive lower bound.
        high: Inclusive upper bound.
        log: Sample on a log scale (float only).
        step: Integer step (int only); ``None`` means 1.
    """

    kind: str
    low: float
    high: float
    log: bool = False
    step: int | None = None


XGB_SEARCH_SPACE: Final[dict[str, ParamSpec]] = {
    "n_estimators": ParamSpec("int", 300, 3000, step=100),
    "learning_rate": ParamSpec("float", 2e-3, 0.1, log=True),
    "max_depth": ParamSpec("int", 3, 12),
    "min_child_weight": ParamSpec("float", 0.5, 30.0, log=True),
    "subsample": ParamSpec("float", 0.6, 1.0),
    "colsample_bytree": ParamSpec("float", 0.01, 1.0, log=True),
    "reg_lambda": ParamSpec("float", 0.5, 15.0, log=True),
    "reg_alpha": ParamSpec("float", 1e-2, 2.0, log=True),
    "gamma": ParamSpec("float", 0.0, 0.6),
    "max_delta_step": ParamSpec("int", 0, 5),
}

CNN_SEARCH_SPACE: Final[dict[str, ParamSpec]] = {
    "n_channels": ParamSpec("int", 16, 256, log=True),
    "lr": ParamSpec("float", 1e-5, 1e-2, log=True),
    "weight_decay": ParamSpec("float", 1e-6, 1e-1, log=True),
    "dropout": ParamSpec("float", 0.0, 0.6),
    "batch_size": ParamSpec("int", 2048, 16384, log=True),
}


def suggest_xgb_params(trial: Any) -> dict[str, float | int]:
    """Sample one hyperparameter set from an Optuna trial.

    Args:
        trial: An Optuna ``Trial`` (duck-typed; only the ``suggest_*`` methods
            are used).

    Returns:
        A dict with one value per key in :data:`XGB_SEARCH_SPACE`.
    """
    params: dict[str, float | int] = {}
    for name, spec in XGB_SEARCH_SPACE.items():
        if spec.kind == "int":
            params[name] = trial.suggest_int(
                name, int(spec.low), int(spec.high), step=spec.step or 1
            )
        else:
            params[name] = trial.suggest_float(name, spec.low, spec.high, log=spec.log)
    return params


def hp_boundary_diagnostics(
    best_params: dict[str, float | int], *, rel_tol: float = 0.02
) -> pd.DataFrame:
    """Flag selected hyperparameters sitting at the edge of their range.

    A parameter at a bound suggests the range should be widened.

    Args:
        best_params: The chosen hyperparameters; keys not in the space are
            ignored, and missing keys are skipped.
        rel_tol: Fraction of the range treated as "at the bound".

    Returns:
        A frame with columns ``parameter``, ``value``, ``low``, ``high``,
        ``at_lower_bound`` and ``at_upper_bound``.
    """
    rows = []
    for name, spec in XGB_SEARCH_SPACE.items():
        if name not in best_params:
            continue
        value = float(best_params[name])
        margin = rel_tol * (spec.high - spec.low)
        rows.append(
            {
                "parameter": name,
                "value": value,
                "low": float(spec.low),
                "high": float(spec.high),
                "at_lower_bound": bool(value <= spec.low + margin),
                "at_upper_bound": bool(value >= spec.high - margin),
            }
        )
    return pd.DataFrame(rows)


def suggest_cnn_params(trial: Any) -> dict[str, float | int]:
    """Sample one CNN hyperparameter set from an Optuna trial.

    Mirrors :func:`suggest_xgb_params` over :data:`CNN_SEARCH_SPACE`.

    Args:
        trial: An Optuna ``Trial`` (duck-typed; only the ``suggest_*`` methods
            are used).

    Returns:
        A dict with one value per key in :data:`CNN_SEARCH_SPACE`.
    """
    params: dict[str, float | int] = {}
    for name, spec in CNN_SEARCH_SPACE.items():
        if spec.kind == "int":
            params[name] = trial.suggest_int(
                name, int(spec.low), int(spec.high), step=spec.step or 1
            )
        else:
            params[name] = trial.suggest_float(name, spec.low, spec.high, log=spec.log)
    return params


def _boundary_diagnostics(
    best_params: dict[str, float | int],
    space: dict[str, ParamSpec],
    *,
    rel_tol: float = 0.02,
) -> pd.DataFrame:
    """Flag selected hyperparameters sitting at the edge of an arbitrary space.

    Shared core for :func:`cnn_boundary_diagnostics`; the XGBoost
    :func:`hp_boundary_diagnostics` keeps its own copy so it is byte-for-byte
    unchanged.

    Args:
        best_params: The chosen hyperparameters; keys not in ``space`` are
            ignored, and missing keys are skipped.
        space: The search space whose bounds the values are checked against.
        rel_tol: Fraction of the range treated as "at the bound".

    Returns:
        A frame with columns ``parameter``, ``value``, ``low``, ``high``,
        ``at_lower_bound`` and ``at_upper_bound``.
    """
    rows = []
    for name, spec in space.items():
        if name not in best_params:
            continue
        value = float(best_params[name])
        margin = rel_tol * (spec.high - spec.low)
        rows.append(
            {
                "parameter": name,
                "value": value,
                "low": float(spec.low),
                "high": float(spec.high),
                "at_lower_bound": bool(value <= spec.low + margin),
                "at_upper_bound": bool(value >= spec.high - margin),
            }
        )
    return pd.DataFrame(rows)


def cnn_boundary_diagnostics(
    best_params: dict[str, float | int], *, rel_tol: float = 0.02
) -> pd.DataFrame:
    """Flag selected CNN hyperparameters sitting at the edge of their range.

    The CNN counterpart of :func:`hp_boundary_diagnostics`, over
    :data:`CNN_SEARCH_SPACE`.

    Args:
        best_params: The chosen hyperparameters; keys not in the space are
            ignored, and missing keys are skipped.
        rel_tol: Fraction of the range treated as "at the bound".

    Returns:
        A frame with columns ``parameter``, ``value``, ``low``, ``high``,
        ``at_lower_bound`` and ``at_upper_bound``.
    """
    return _boundary_diagnostics(best_params, CNN_SEARCH_SPACE, rel_tol=rel_tol)


__all__ = [
    "CNN_SEARCH_SPACE",
    "XGB_SEARCH_SPACE",
    "ParamSpec",
    "cnn_boundary_diagnostics",
    "hp_boundary_diagnostics",
    "suggest_cnn_params",
    "suggest_xgb_params",
]
