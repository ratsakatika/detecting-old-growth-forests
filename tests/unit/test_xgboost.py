"""Unit tests for utils.models.xgboost."""

import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
from sklearn.metrics import average_precision_score
from xgboost import XGBClassifier

from utils.models import xgboost as xgbmod
from utils.models.xgboost import (
    TUNABLE_PARAMS,
    feature_importance,
    make_xgb_classifier,
    predict_ogf_proba,
    train_xgb_fold,
)

# Small, fast, valid parameter set for tests.
PARAMS = {
    "n_estimators": 50,
    "learning_rate": 0.1,
    "max_depth": 3,
    "min_child_weight": 1.0,
    "subsample": 1.0,
    "colsample_bytree": 1.0,
    "reg_lambda": 1.0,
    "reg_alpha": 0.1,
    "gamma": 0.0,
    "max_delta_step": 0,
}


def _separable(seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """Two clearly separated 3-D clusters; a correct model scores AP near 1."""
    rng = np.random.default_rng(seed)
    neg = rng.normal(0.0, 1.0, (60, 3))
    pos = rng.normal(4.0, 1.0, (60, 3))
    X = np.vstack([neg, pos]).astype(np.float32)
    y = np.array([0] * 60 + [1] * 60, dtype=np.int8)
    return X, y


def test_import_writes_no_files(tmp_path: Path) -> None:
    repo_root = Path(xgbmod.__file__).resolve().parents[2]
    result = subprocess.run(
        [sys.executable, "-c", "import utils.models.xgboost"],
        cwd=tmp_path,
        env={**os.environ, "PYTHONPATH": str(repo_root)},
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert list(tmp_path.iterdir()) == []


def test_tunable_params_match_test_fixture() -> None:
    assert set(TUNABLE_PARAMS) == set(PARAMS)


def test_make_requires_all_params() -> None:
    with pytest.raises(ValueError, match="missing XGBoost parameters"):
        make_xgb_classifier({"n_estimators": 10})


def test_make_sets_fixed_and_tunable_fields() -> None:
    model = make_xgb_classifier(PARAMS, device="cpu", n_jobs=2, seed=7)
    assert isinstance(model, XGBClassifier)
    assert model.objective == "binary:logistic"
    assert model.tree_method == "hist"
    assert model.importance_type == "gain"
    assert model.n_estimators == 50
    assert model.max_depth == 3
    assert model.n_jobs == 2
    assert model.random_state == 7


def test_predict_ogf_proba_shape_dtype_range() -> None:
    X, y = _separable()
    model = make_xgb_classifier(PARAMS)
    model.fit(X, y)
    p = predict_ogf_proba(model, X)
    assert p.shape == (120,)
    assert str(p.dtype) == "float64"
    assert p.min() >= 0.0 and p.max() <= 1.0


def test_feature_importance_sorted_and_validated() -> None:
    X, y = _separable()
    model = make_xgb_classifier(PARAMS)
    model.fit(X, y)
    fi = feature_importance(model, ["a", "b", "c"])
    assert list(fi.columns) == ["feature", "importance"]
    assert fi["importance"].is_monotonic_decreasing
    assert len(fi) == 3
    with pytest.raises(ValueError, match="feature_names length"):
        feature_importance(model, ["a", "b"])


def test_train_xgb_fold_learns_separable() -> None:
    X, y = _separable(0)
    X_val, y_val = _separable(1)
    p = train_xgb_fold(PARAMS, X, y, X_val, device="cpu")
    assert p.shape == (120,)
    # Separable clusters: a correct pipeline scores well above the ~0.5 random AP.
    assert average_precision_score(y_val, p) > 0.95


def test_train_xgb_fold_accepts_sample_weight() -> None:
    X, y = _separable(0)
    X_val, _ = _separable(1)
    weights = np.where(y == 1, 2.0, 0.5).astype(np.float64)
    p = train_xgb_fold(PARAMS, X, y, X_val, sample_weight=weights, device="cpu")
    assert p.shape == (120,)
    assert p.min() >= 0.0 and p.max() <= 1.0
