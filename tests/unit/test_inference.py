"""Unit tests for utils.inference (the wall-to-wall building blocks)."""

import numpy as np
import pytest
from xgboost import XGBClassifier

from utils.inference import (
    PIXEL_AREA_HA,
    gather_pixel_features,
    ogf_area_ha,
    predict_pixels,
)


def test_gather_pixel_features_matches_grid_indexing() -> None:
    rng = np.random.default_rng(0)
    memmap = rng.standard_normal((5, 4, 6)).astype(np.float32)  # (n_bands, H, W)
    width = 6
    pixel_ids = np.array([0, 7, 23, 11], dtype=np.int64)  # row*width + col
    gathered = gather_pixel_features(memmap, pixel_ids)
    assert gathered.shape == (4, 5)
    assert gathered.flags["C_CONTIGUOUS"]
    for i, pid in enumerate(pixel_ids):
        row, col = divmod(int(pid), width)
        np.testing.assert_array_equal(gathered[i], memmap[:, row, col])


def test_gather_pixel_features_validation() -> None:
    memmap = np.zeros((3, 4, 4), dtype=np.float32)
    with pytest.raises(ValueError, match="one-dimensional"):
        gather_pixel_features(memmap, np.zeros((2, 2), dtype=np.int64))
    with pytest.raises(ValueError, match="out of range"):
        gather_pixel_features(memmap, np.array([16], dtype=np.int64))
    with pytest.raises(ValueError, match="n_bands"):
        gather_pixel_features(np.zeros((4, 4), dtype=np.float32), np.array([0]))


def _fit_tiny_model() -> tuple[XGBClassifier, np.ndarray]:
    rng = np.random.default_rng(1)
    X = rng.standard_normal((400, 5)).astype(np.float32)
    y = (X[:, 0] + 0.3 * rng.standard_normal(400) > 0).astype(np.int64)
    model = XGBClassifier(n_estimators=20, max_depth=3, device="cpu", verbosity=0)
    model.fit(X, y)
    return model, X


def test_predict_pixels_chunking_is_exact_and_bounded() -> None:
    model, X = _fit_tiny_model()
    full = predict_pixels(model, X, chunk_size=10_000)
    chunked = predict_pixels(model, X, chunk_size=37)
    assert full.shape == (400,)
    assert full.min() >= 0.0 and full.max() <= 1.0
    np.testing.assert_allclose(full, chunked, rtol=1e-6, atol=1e-6)


def test_predict_pixels_validation() -> None:
    model, _ = _fit_tiny_model()
    with pytest.raises(ValueError, match="two-dimensional"):
        predict_pixels(model, np.zeros(5, dtype=np.float32))
    with pytest.raises(ValueError, match="chunk_size"):
        predict_pixels(model, np.zeros((2, 5), dtype=np.float32), chunk_size=0)


def test_ogf_area_ha() -> None:
    assert PIXEL_AREA_HA == pytest.approx(0.01)
    mask = np.array([[True, False], [True, True]])
    assert ogf_area_ha(mask) == pytest.approx(3 * 0.01)
    assert ogf_area_ha(np.array([0, 1, 1, 0]), pixel_area_ha=2.0) == pytest.approx(4.0)
    assert ogf_area_ha(np.zeros((3, 3), dtype=bool)) == pytest.approx(0.0)


def test_parcel_consistent_area_ha() -> None:
    from utils.inference import PIXEL_AREA_HA, parcel_consistent_area_ha

    # Parcel 0 (mean 0.85) clears the threshold; parcel 1 (mean 0.20) does not.
    parcel_ids = np.array([0, 0, 1, 1, 1])
    probs = np.array([0.9, 0.8, 0.1, 0.2, 0.3])
    area = parcel_consistent_area_ha(parcel_ids, probs, threshold=0.5)
    assert area == pytest.approx(2 * PIXEL_AREA_HA)  # only parcel 0's two pixels
    assert (
        parcel_consistent_area_ha(np.array([], dtype=np.int64), np.array([]), threshold=0.5) == 0.0
    )


def test_parcel_consistent_area_ha_length_mismatch() -> None:
    from utils.inference import parcel_consistent_area_ha

    with pytest.raises(ValueError, match="same shape"):
        parcel_consistent_area_ha(np.array([0, 1]), np.array([0.5]), threshold=0.5)
