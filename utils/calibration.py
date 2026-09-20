"""Probability calibration for the final old-growth map.

The discrimination metrics that rank model configurations are invariant to any
monotonic recalibration (``utils.metrics``), so calibration is a separate, final
step applied only when a probability is reported on the map. This module fits and
compares five one-dimensional calibrators on held-out (out-of-fold) predictions
and selects one to apply to the wall-to-wall surface:

  - ``isotonic``: non-parametric monotonic regression (the most flexible, but it
    can overfit a small calibration set).
  - ``platt``: logistic regression on the logit of the score (a slope and bias;
    Platt 1999).
  - ``beta``: logistic regression on ``[ln p, -ln(1-p)]`` (Kull et al. 2017,
    doi:10.1214/17-EJS1338SI), a three-parameter family that handles the skew of
    a rare positive class better than Platt.
  - ``temperature``: a single positive scalar dividing the logit (Guo et al.
    2017); the minimal parametric correction.
  - ``spline``: a cubic smoothing spline through the binned reliability curve.

Selection is by cross-validated expected calibration error (ECE), with the Brier
score as the tie-break, so the flexible non-parametric calibrators are not
rewarded for fitting the selection set in-sample. The chosen method is then refit
on all of the held-out predictions by the caller via :func:`fit_calibrator`.

Importing this module binds names only; the functions are pure and perform no
input/output or logging.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Final

import numpy as np
import numpy.typing as npt
import pandas as pd
from scipy.interpolate import UnivariateSpline
from scipy.optimize import minimize_scalar
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold

from utils.terminology import SEED

# The calibrators compared, in a fixed reporting order.
CALIBRATION_METHODS: Final[tuple[str, ...]] = (
    "isotonic",
    "platt",
    "beta",
    "temperature",
    "spline",
)

# The calibration-quality metrics written to the comparison table, lower is better.
CALIBRATION_METRICS: Final[tuple[str, ...]] = ("ece", "mce", "brier", "log_loss")

_EPS: Final[float] = 1e-6  # keeps logits and the log-loss finite at p in {0, 1}
_N_ECE_BINS: Final[int] = 10


@dataclass(frozen=True, slots=True)
class Calibrator:
    """A fitted one-dimensional probability calibrator.

    Attributes:
        method: The calibrator name, one of :data:`CALIBRATION_METHODS`.
        predict: Maps raw probabilities in ``[0, 1]`` to calibrated probabilities
            in ``[0, 1]`` (clipped); the array is one-dimensional and the same
            length as its input.
    """

    method: str
    predict: Callable[[npt.NDArray[np.float64]], npt.NDArray[np.float64]] = field(repr=False)


def _validate(
    y_true: npt.ArrayLike, y_prob: npt.ArrayLike
) -> tuple[npt.NDArray[np.int64], npt.NDArray[np.float64]]:
    """Coerce and check binary labels and probabilities for calibration."""
    yt = np.asarray(y_true)
    yp = np.asarray(y_prob, dtype=np.float64)
    if yt.ndim != 1 or yp.ndim != 1:
        raise ValueError("y_true and y_prob must be one-dimensional.")
    if yt.shape[0] != yp.shape[0]:
        raise ValueError(f"length mismatch: y_true has {yt.shape[0]}, y_prob has {yp.shape[0]}.")
    if yt.shape[0] == 0:
        raise ValueError("inputs are empty.")
    if not np.isfinite(yp).all():
        raise ValueError("y_prob contains non-finite values.")
    if yp.min() < 0.0 or yp.max() > 1.0:
        raise ValueError("y_prob must lie in [0, 1].")
    labels = np.unique(yt)
    if not np.isin(labels, (0, 1)).all():
        raise ValueError(f"y_true must be binary 0/1; found {labels.tolist()}.")
    return yt.astype(np.int64), yp


def _logit(p: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    """Logit with the probabilities clipped away from 0 and 1."""
    q = np.clip(p, _EPS, 1.0 - _EPS)
    return np.log(q / (1.0 - q))


def _clip01(p: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    """Clip to ``[0, 1]`` as a plain ``float64`` array."""
    return np.clip(np.asarray(p, dtype=np.float64), 0.0, 1.0)


def _fit_isotonic(y: npt.NDArray[np.int64], p: npt.NDArray[np.float64]) -> Calibrator:
    iso = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
    iso.fit(p, y.astype(np.float64))

    def predict(q: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        return _clip01(iso.predict(np.clip(q, 0.0, 1.0)))

    return Calibrator("isotonic", predict)


def _fit_logit_features(
    method: str,
    y: npt.NDArray[np.int64],
    features: npt.NDArray[np.float64],
    build: Callable[[npt.NDArray[np.float64]], npt.NDArray[np.float64]],
) -> Calibrator:
    """Logistic regression on engineered score features (Platt and Beta share this)."""
    model = LogisticRegression(C=1e6, solver="lbfgs", max_iter=1000)
    model.fit(features, y)

    def predict(q: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        proba = model.predict_proba(build(np.asarray(q, dtype=np.float64)))[:, 1]
        return _clip01(proba)

    return Calibrator(method, predict)


def _fit_platt(y: npt.NDArray[np.int64], p: npt.NDArray[np.float64]) -> Calibrator:
    return _fit_logit_features(
        "platt", y, _logit(p).reshape(-1, 1), lambda q: _logit(q).reshape(-1, 1)
    )


def _beta_features(p: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    q = np.clip(p, _EPS, 1.0 - _EPS)
    return np.column_stack([np.log(q), -np.log(1.0 - q)])


def _fit_beta(y: npt.NDArray[np.int64], p: npt.NDArray[np.float64]) -> Calibrator:
    return _fit_logit_features("beta", y, _beta_features(p), _beta_features)


def _fit_temperature(y: npt.NDArray[np.int64], p: npt.NDArray[np.float64]) -> Calibrator:
    z = _logit(p)

    def nll(log_t: float) -> float:
        temperature = float(np.exp(log_t))
        scaled = 1.0 / (1.0 + np.exp(-z / temperature))
        scaled = np.clip(scaled, _EPS, 1.0 - _EPS)
        return float(-np.mean(y * np.log(scaled) + (1 - y) * np.log(1.0 - scaled)))

    result = minimize_scalar(nll, bounds=(np.log(0.05), np.log(20.0)), method="bounded")
    temperature = float(np.exp(result.x))

    def predict(q: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        return _clip01(1.0 / (1.0 + np.exp(-_logit(np.asarray(q, dtype=np.float64)) / temperature)))

    return Calibrator("temperature", predict)


def _fit_spline(y: npt.NDArray[np.int64], p: npt.NDArray[np.float64]) -> Calibrator:
    """Cubic smoothing spline through the quantile-binned reliability curve."""
    n_bins = int(min(50, max(4, np.unique(p).size)))
    edges = np.quantile(p, np.linspace(0.0, 1.0, n_bins + 1))
    edges = np.unique(edges)
    if edges.size < 4:  # too few distinct scores for a cubic spline; fall back to isotonic
        return Calibrator("spline", _fit_isotonic(y, p).predict)
    idx = np.clip(np.digitize(p, edges[1:-1]), 0, edges.size - 2)
    frame = pd.DataFrame({"bin": idx, "p": p, "y": y.astype(np.float64)})
    grouped = frame.groupby("bin", sort=True).mean()
    xs = grouped["p"].to_numpy()
    ys = grouped["y"].to_numpy()
    order = np.argsort(xs)
    xs, ys = xs[order], ys[order]
    xs, unique = np.unique(xs, return_index=True)
    ys = ys[unique]
    if xs.size < 4:  # pragma: no cover - binning rarely collapses below a cubic's support
        return Calibrator("spline", _fit_isotonic(y, p).predict)
    degree = int(min(3, xs.size - 1))
    spline = UnivariateSpline(xs, ys, k=degree, s=xs.size * 0.0005, ext="const")

    def predict(q: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        return _clip01(spline(np.clip(np.asarray(q, dtype=np.float64), 0.0, 1.0)))

    return Calibrator("spline", predict)


_FITTERS: Final[
    dict[str, Callable[[npt.NDArray[np.int64], npt.NDArray[np.float64]], Calibrator]]
] = {
    "isotonic": _fit_isotonic,
    "platt": _fit_platt,
    "beta": _fit_beta,
    "temperature": _fit_temperature,
    "spline": _fit_spline,
}


def fit_calibrator(method: str, y_true: npt.ArrayLike, y_prob: npt.ArrayLike) -> Calibrator:
    """Fit one calibrator on held-out probabilities.

    Args:
        method: One of :data:`CALIBRATION_METHODS`.
        y_true: One-dimensional binary labels (0/1 or boolean).
        y_prob: One-dimensional raw probabilities in ``[0, 1]``.

    Returns:
        The fitted :class:`Calibrator`.

    Raises:
        ValueError: If ``method`` is unknown or the inputs fail validation.
    """
    if method not in _FITTERS:
        raise ValueError(
            f"unknown calibration method {method!r}; expected one of {CALIBRATION_METHODS}."
        )
    yt, yp = _validate(y_true, y_prob)
    return _FITTERS[method](yt, yp)


def calibration_metrics(y_true: npt.ArrayLike, y_prob: npt.ArrayLike) -> dict[str, float]:
    """Calibration-quality metrics for a probability vector.

    Args:
        y_true: One-dimensional binary labels.
        y_prob: One-dimensional probabilities in ``[0, 1]``.

    Returns:
        ``ece`` (expected calibration error, 10 equal-width bins), ``mce`` (the
        maximum bin gap), ``brier`` and ``log_loss``. Lower is better for all.
    """
    yt, yp = _validate(y_true, y_prob)
    bin_index = np.clip(
        np.digitize(yp, np.linspace(0.0, 1.0, _N_ECE_BINS + 1)[1:-1]), 0, _N_ECE_BINS - 1
    )
    ece = 0.0
    mce = 0.0
    n = yt.shape[0]
    for b in range(_N_ECE_BINS):
        mask = bin_index == b
        count = int(mask.sum())
        if count == 0:
            continue
        gap = abs(float(yt[mask].mean()) - float(yp[mask].mean()))
        ece += (count / n) * gap
        mce = max(mce, gap)
    clipped = np.clip(yp, _EPS, 1.0 - _EPS)
    log_loss = float(-np.mean(yt * np.log(clipped) + (1 - yt) * np.log(1.0 - clipped)))
    return {
        "ece": float(ece),
        "mce": float(mce),
        "brier": float(np.mean((yp - yt) ** 2)),
        "log_loss": log_loss,
    }


def compare_calibrators(
    y_true: npt.ArrayLike,
    y_prob: npt.ArrayLike,
    *,
    methods: Sequence[str] = CALIBRATION_METHODS,
    n_splits: int = 5,
    seed: int = SEED,
) -> pd.DataFrame:
    """Cross-validated calibration metrics for each method plus the raw scores.

    For every method a stratified ``n_splits``-fold cross-validation is run over
    the held-out predictions: the calibrator is fit on the training split and
    scored on the validation split, and the per-fold metrics are averaged. The
    cross-validation avoids rewarding the flexible non-parametric calibrators for
    fitting the selection set in-sample. The ``uncalibrated`` row reports the raw
    scores for reference.

    Args:
        y_true: One-dimensional binary labels.
        y_prob: One-dimensional raw probabilities in ``[0, 1]``.
        methods: The calibrators to compare.
        n_splits: Cross-validation folds for the metric estimate.
        seed: Cross-validation shuffle seed.

    Returns:
        One row per method (and ``uncalibrated``) with the columns of
        :data:`CALIBRATION_METRICS`, in ``methods`` order.
    """
    yt, yp = _validate(y_true, y_prob)
    splitter = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    rows: list[dict[str, float | str]] = []
    for method in methods:
        fold_metrics: list[dict[str, float]] = []
        for train_idx, val_idx in splitter.split(yp, yt):
            calibrator = _FITTERS[method](yt[train_idx], yp[train_idx])
            fold_metrics.append(calibration_metrics(yt[val_idx], calibrator.predict(yp[val_idx])))
        rows.append(
            {
                "method": method,
                **{m: float(np.mean([fm[m] for fm in fold_metrics])) for m in CALIBRATION_METRICS},
            }
        )
    rows.append({"method": "uncalibrated", **calibration_metrics(yt, yp)})
    return pd.DataFrame(rows, columns=["method", *CALIBRATION_METRICS])


def select_best_calibrator(
    y_true: npt.ArrayLike,
    y_prob: npt.ArrayLike,
    *,
    methods: Sequence[str] = CALIBRATION_METHODS,
    n_splits: int = 5,
    seed: int = SEED,
) -> tuple[str, pd.DataFrame]:
    """Pick the calibrator with the lowest cross-validated ECE (Brier tie-break).

    Args:
        y_true: One-dimensional binary labels.
        y_prob: One-dimensional raw probabilities in ``[0, 1]``.
        methods: The calibrators to compare.
        n_splits: Cross-validation folds for the metric estimate.
        seed: Cross-validation shuffle seed.

    Returns:
        The selected method name and the full comparison frame from
        :func:`compare_calibrators` (the caller writes the frame and refits the
        selected method on all of ``y_prob`` via :func:`fit_calibrator`).
    """
    comparison = compare_calibrators(y_true, y_prob, methods=methods, n_splits=n_splits, seed=seed)
    candidates = comparison[comparison["method"].isin(methods)]
    best = candidates.sort_values(["ece", "brier"], kind="stable").iloc[0]["method"]
    return str(best), comparison


__all__ = [
    "CALIBRATION_METHODS",
    "CALIBRATION_METRICS",
    "Calibrator",
    "calibration_metrics",
    "compare_calibrators",
    "fit_calibrator",
    "select_best_calibrator",
]
