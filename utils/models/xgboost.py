"""XGBoost pixel model for old-growth detection.

A thin, explicit wrapper over :class:`xgboost.XGBClassifier`: build a
classifier from a tuned parameter dict, predict the positive-class
(old-growth) probability, read gain-based feature importance, and train one
cross-validation fold. Design choices:

  - ``device`` is supplied by the caller (``"cuda"`` or ``"cpu"``); the module
    never probes the hardware, so it is GPU-agnostic and testable on CPU.
  - Class imbalance is handled by per-row ``sample_weight`` (from
    ``utils.sampling``) passed to ``fit``, not by ``scale_pos_weight``; one
    mechanism then serves the ``none``/``inverse_prevalence``/
    ``stratified_forest_type`` schemes.
  - ``n_estimators`` is a tuned hyperparameter and there is no early stopping,
    so the inner-validation fold is not used both to stop boosting and to score.
  - Importance is gain-based and column-aligned.

Importing this module binds names only; the functions are pure and perform no
input/output or logging.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Final

import numpy as np
import numpy.typing as npt
import pandas as pd
from xgboost import XGBClassifier

from utils.terminology import SEED

# Tunable hyperparameters this wrapper expects in a parameter dict. The fixed
# settings (objective, eval metric, tree method, importance type) are not tuned
# and are applied below.
TUNABLE_PARAMS: Final[tuple[str, ...]] = (
    "n_estimators",
    "learning_rate",
    "max_depth",
    "min_child_weight",
    "subsample",
    "colsample_bytree",
    "reg_lambda",
    "reg_alpha",
    "gamma",
    "max_delta_step",
)


def make_xgb_classifier(
    params: Mapping[str, float | int],
    *,
    device: str = "cpu",
    n_jobs: int = 1,
    seed: int = SEED,
) -> XGBClassifier:
    """Build an :class:`xgboost.XGBClassifier` from a tuned parameter dict.

    Args:
        params: Values for every name in :data:`TUNABLE_PARAMS`.
        device: ``"cuda"`` or ``"cpu"``, passed straight to XGBoost.
        n_jobs: CPU threads for data preparation and the CPU tree method.
        seed: Random seed.

    Returns:
        An unfitted classifier with fixed objective ``binary:logistic``,
        ``aucpr`` eval metric (used only if an eval set is supplied), the
        ``hist`` tree method and gain-based importance.

    Raises:
        ValueError: If any tunable parameter is missing from ``params``.
    """
    missing = [name for name in TUNABLE_PARAMS if name not in params]
    if missing:
        raise ValueError(f"missing XGBoost parameters: {missing}")

    return XGBClassifier(
        objective="binary:logistic",
        eval_metric="aucpr",
        tree_method="hist",
        importance_type="gain",
        device=device,
        n_estimators=int(params["n_estimators"]),
        learning_rate=float(params["learning_rate"]),
        max_depth=int(params["max_depth"]),
        min_child_weight=float(params["min_child_weight"]),
        subsample=float(params["subsample"]),
        colsample_bytree=float(params["colsample_bytree"]),
        reg_lambda=float(params["reg_lambda"]),
        reg_alpha=float(params["reg_alpha"]),
        gamma=float(params["gamma"]),
        max_delta_step=int(params["max_delta_step"]),
        n_jobs=int(n_jobs),
        random_state=int(seed),
        verbosity=0,
    )


def predict_ogf_proba(model: XGBClassifier, X: npt.ArrayLike) -> npt.NDArray[np.float64]:
    """Return the positive-class (old-growth) probability for each row.

    Args:
        model: A fitted classifier.
        X: Feature matrix of shape ``(n_samples, n_features)``.

    Returns:
        A one-dimensional ``float64`` array of old-growth probabilities.
    """
    proba = model.predict_proba(X)
    return proba[:, 1].astype(np.float64)


def feature_importance(model: XGBClassifier, feature_names: Sequence[str]) -> pd.DataFrame:
    """Return gain-based feature importance, highest first.

    Args:
        model: A fitted classifier.
        feature_names: Names in the same order as the training columns.

    Returns:
        A frame with columns ``feature`` and ``importance`` (float64), sorted by
        importance descending.

    Raises:
        ValueError: If ``feature_names`` length does not match the model.
    """
    importances = np.asarray(model.feature_importances_, dtype=np.float64)
    if len(feature_names) != importances.shape[0]:
        raise ValueError(
            f"feature_names length {len(feature_names)} does not match the "
            f"model's {importances.shape[0]} features."
        )
    return pd.DataFrame({"feature": list(feature_names), "importance": importances}).sort_values(
        "importance", ascending=False, ignore_index=True
    )


def train_xgb_fold(
    params: Mapping[str, float | int],
    X_train: npt.ArrayLike,
    y_train: npt.ArrayLike,
    X_val: npt.ArrayLike,
    *,
    sample_weight: npt.ArrayLike | None = None,
    device: str = "cpu",
    n_jobs: int = 1,
    seed: int = SEED,
) -> npt.NDArray[np.float64]:
    """Train one fold and return validation old-growth probabilities.

    Args:
        params: Tuned hyperparameters (see :data:`TUNABLE_PARAMS`).
        X_train: Training feature matrix.
        y_train: Training binary labels.
        X_val: Validation feature matrix.
        sample_weight: Optional per-row training weights (from
            ``utils.sampling``); ``None`` means uniform.
        device: ``"cuda"`` or ``"cpu"``.
        n_jobs: CPU threads.
        seed: Random seed.

    Returns:
        A one-dimensional ``float64`` array of validation probabilities.
    """
    model = make_xgb_classifier(params, device=device, n_jobs=n_jobs, seed=seed)
    model.fit(X_train, y_train, sample_weight=sample_weight)
    return predict_ogf_proba(model, X_val)


__all__ = [
    "TUNABLE_PARAMS",
    "feature_importance",
    "make_xgb_classifier",
    "predict_ogf_proba",
    "train_xgb_fold",
]
