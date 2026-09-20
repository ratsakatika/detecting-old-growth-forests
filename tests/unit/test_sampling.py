"""Unit tests for utils.sampling."""

import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from utils import sampling
from utils.sampling import SAMPLING_MODES, build_sample_weights


def test_import_writes_no_files(tmp_path: Path) -> None:
    # A fresh interpreter importing the module in an empty directory must create
    # nothing. Run in a subprocess so the check cannot perturb this session.
    repo_root = Path(sampling.__file__).resolve().parents[1]
    result = subprocess.run(
        [sys.executable, "-c", "import utils.sampling"],
        cwd=tmp_path,
        env={**os.environ, "PYTHONPATH": str(repo_root)},
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert list(tmp_path.iterdir()) == []


def test_sampling_modes_are_the_three_expected() -> None:
    assert SAMPLING_MODES == ("none", "inverse_prevalence", "stratified_forest_type")


# --- none ---


def test_none_is_uniform_and_mean_one() -> None:
    weights = build_sample_weights("none", labels=[0, 1, 0, 1, 1])
    assert str(weights.dtype) == "float64"
    np.testing.assert_allclose(weights, np.ones(5))
    assert weights.mean() == pytest.approx(1.0)


# --- inverse_prevalence ---


def test_inverse_prevalence_equalises_class_totals() -> None:
    # 1 positive, 3 negatives. Positive raw 1.0, each negative raw 1/3; after
    # the mean-1 normalisation (factor 4/2 = 2): positive 2.0, each negative 2/3.
    labels = [1, 0, 0, 0]
    weights = build_sample_weights("inverse_prevalence", labels=labels)

    np.testing.assert_allclose(weights, [2.0, 2.0 / 3.0, 2.0 / 3.0, 2.0 / 3.0])
    assert weights.mean() == pytest.approx(1.0)
    # Each class carries equal total weight.
    labels_arr = np.asarray(labels)
    assert weights[labels_arr == 1].sum() == pytest.approx(weights[labels_arr == 0].sum())


def test_inverse_prevalence_single_class_is_uniform() -> None:
    weights = build_sample_weights("inverse_prevalence", labels=[1, 1, 1])
    np.testing.assert_allclose(weights, np.ones(3))


# --- stratified_forest_type ---


def test_stratified_equalises_cell_totals() -> None:
    # Cells: (1,b) size 2, (0,b) size 1, (0,c) size 3. Raw 1/size; after the
    # mean-1 normalisation (factor 6/3 = 2): [1, 1, 2, 2/3, 2/3, 2/3].
    labels = [1, 1, 0, 0, 0, 0]
    forest_type = ["b", "b", "b", "c", "c", "c"]
    weights = build_sample_weights("stratified_forest_type", labels=labels, forest_type=forest_type)

    np.testing.assert_allclose(weights, [1.0, 1.0, 2.0, 2.0 / 3.0, 2.0 / 3.0, 2.0 / 3.0])
    assert weights.mean() == pytest.approx(1.0)

    df = np.rec.fromarrays(
        [np.asarray(labels), np.asarray(forest_type), weights], names=["label", "ft", "w"]
    )
    totals = {
        key: df.w[(df.label == lab) & (df.ft == ft)].sum()
        for key, (lab, ft) in {"1b": (1, "b"), "0b": (0, "b"), "0c": (0, "c")}.items()
    }
    assert totals["1b"] == pytest.approx(totals["0b"]) == pytest.approx(totals["0c"])


def test_stratified_requires_forest_type() -> None:
    with pytest.raises(ValueError, match="forest_type is required"):
        build_sample_weights("stratified_forest_type", labels=[0, 1])


def test_stratified_rejects_length_mismatch() -> None:
    with pytest.raises(ValueError, match="length mismatch"):
        build_sample_weights("stratified_forest_type", labels=[0, 1, 1], forest_type=["b", "c"])


def test_stratified_rejects_two_dimensional_forest_type() -> None:
    with pytest.raises(ValueError, match="one-dimensional"):
        build_sample_weights(
            "stratified_forest_type",
            labels=[0, 1, 0, 1],
            forest_type=np.zeros((2, 2)),
        )


# --- shared behaviour and validation ---


@pytest.mark.parametrize("mode", SAMPLING_MODES)
def test_every_mode_returns_mean_one_float64(mode: str) -> None:
    labels = [0, 0, 1, 1, 1, 0]
    forest_type = ["b", "c", "b", "c", "m", "m"]
    weights = build_sample_weights(mode, labels=labels, forest_type=forest_type)
    assert str(weights.dtype) == "float64"
    assert weights.shape == (6,)
    assert weights.mean() == pytest.approx(1.0)


def test_unknown_mode_raises() -> None:
    with pytest.raises(ValueError, match="unknown mode"):
        build_sample_weights("balanced", labels=[0, 1])


def test_accepts_boolean_labels() -> None:
    weights = build_sample_weights(
        "inverse_prevalence", labels=np.array([True, False, False, False])
    )
    np.testing.assert_allclose(weights, [2.0, 2.0 / 3.0, 2.0 / 3.0, 2.0 / 3.0])


@pytest.mark.parametrize(
    ("labels", "match"),
    [
        ([], "empty"),
        ([0, 2, 1], "binary"),
    ],
)
def test_label_validation_errors(labels: object, match: str) -> None:
    with pytest.raises(ValueError, match=match):
        build_sample_weights("none", labels=labels)


def test_label_validation_rejects_two_dimensional() -> None:
    with pytest.raises(ValueError, match="one-dimensional"):
        build_sample_weights("none", labels=np.zeros((2, 2)))
