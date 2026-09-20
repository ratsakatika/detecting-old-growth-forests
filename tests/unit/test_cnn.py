"""Unit tests for utils.models.cnn."""

import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import torch
from sklearn.metrics import average_precision_score
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from utils.models import cnn
from utils.models.cnn import (
    CNNFitResult,
    ReceptiveFieldCNN,
    make_cnn,
    predict_proba,
    shutdown_loader_workers,
    train_cnn_fixed,
    train_cnn_fold,
)
from utils.terminology import CNN_PATCH_SIZES

_FULL_PARAMS = {"n_channels": 16, "dropout": 0.0, "lr": 1e-2, "weight_decay": 1e-4}


def _synthetic_loaders(
    *,
    n_train: int,
    n_val: int,
    n_bands: int,
    patch_size: int,
    signal: float,
    seed: int,
    learnable: bool = True,
    batch_size: int = 32,
) -> tuple[DataLoader, DataLoader, np.ndarray]:
    """Build train/val loaders yielding (patch, label, weight).

    When ``learnable``, the centre pixel of band 0 carries ``signal * label``
    plus noise so a receptive-field model can separate the classes; otherwise
    every cell is pure noise (an unlearnable task).
    """
    rng = np.random.default_rng(seed)

    def make(n: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        labels = (np.arange(n) % 2).astype(np.float32)  # balanced 0/1
        patches = rng.standard_normal((n, n_bands, patch_size, patch_size)).astype(np.float32)
        if learnable:
            centre = patch_size // 2
            patches[:, 0, centre, centre] = signal * labels + 0.3 * rng.standard_normal(n).astype(
                np.float32
            )
        weights = np.ones(n, dtype=np.float32)
        return (
            torch.from_numpy(patches),
            torch.from_numpy(labels),
            torch.from_numpy(weights),
        )

    xtr, ytr, wtr = make(n_train)
    xva, yva, wva = make(n_val)
    train_loader = DataLoader(
        TensorDataset(xtr, ytr, wtr),
        batch_size=batch_size,
        shuffle=True,
        generator=torch.Generator().manual_seed(seed),
    )
    val_loader = DataLoader(TensorDataset(xva, yva, wva), batch_size=n_val, shuffle=False)
    return train_loader, val_loader, yva.numpy()


# --------------------------------------------------------------------------- #
# Import side effects.
# --------------------------------------------------------------------------- #


def test_import_writes_no_files(tmp_path: Path) -> None:
    repo_root = Path(cnn.__file__).resolve().parents[2]
    result = subprocess.run(
        [sys.executable, "-c", "import utils.models.cnn"],
        cwd=tmp_path,
        env={**os.environ, "PYTHONPATH": str(repo_root)},
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert list(tmp_path.iterdir()) == []


# --------------------------------------------------------------------------- #
# Architecture.
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("patch_size", CNN_PATCH_SIZES)
def test_layer_count_matches_patch_size(patch_size: int) -> None:
    model = ReceptiveFieldCNN(n_bands=4, patch_size=patch_size, n_channels=8, dropout=0.1)
    conv_layers = [m for m in model.features if isinstance(m, nn.Conv2d)]
    assert len(conv_layers) == (patch_size - 1) // 2
    # Every spatial conv is a valid (no-padding) 3x3.
    for conv in conv_layers:
        assert conv.kernel_size == (3, 3)
        assert conv.padding == (0, 0)
    # The head is a 1x1 convolution to a single channel, not a Linear layer.
    assert isinstance(model.head, nn.Conv2d)
    assert model.head.kernel_size == (1, 1)
    assert model.head.out_channels == 1


@pytest.mark.parametrize("patch_size", CNN_PATCH_SIZES)
def test_features_collapse_to_single_pixel(patch_size: int) -> None:
    model = ReceptiveFieldCNN(n_bands=5, patch_size=patch_size, n_channels=8, dropout=0.0)
    model.eval()
    x = torch.randn(4, 5, patch_size, patch_size)
    feat = model.features(x)
    assert tuple(feat.shape) == (4, 8, 1, 1)


@pytest.mark.parametrize("patch_size", CNN_PATCH_SIZES)
def test_forward_output_shape_is_batch(patch_size: int) -> None:
    model = ReceptiveFieldCNN(n_bands=3, patch_size=patch_size, n_channels=8, dropout=0.0)
    model.eval()
    out = model(torch.randn(7, 3, patch_size, patch_size))
    assert tuple(out.shape) == (7,)


@pytest.mark.parametrize("bad_size", [1, 2, 4, 6, 8])
def test_invalid_patch_size_raises(bad_size: int) -> None:
    with pytest.raises(ValueError, match="patch_size"):
        ReceptiveFieldCNN(n_bands=4, patch_size=bad_size, n_channels=8, dropout=0.1)


def test_invalid_dropout_raises() -> None:
    with pytest.raises(ValueError, match="dropout"):
        ReceptiveFieldCNN(n_bands=4, patch_size=3, n_channels=8, dropout=1.0)


def test_invalid_n_channels_raises() -> None:
    with pytest.raises(ValueError, match="n_channels"):
        ReceptiveFieldCNN(n_bands=4, patch_size=3, n_channels=0, dropout=0.1)


def test_forward_rejects_wrong_spatial_size() -> None:
    model = ReceptiveFieldCNN(n_bands=4, patch_size=3, n_channels=8, dropout=0.0)
    model.eval()
    with pytest.raises(ValueError, match="spatial size"):
        model(torch.randn(2, 4, 5, 5))  # 5x5 into a 3x3 model


# --------------------------------------------------------------------------- #
# make_cnn.
# --------------------------------------------------------------------------- #


def test_make_cnn_requires_arch_params() -> None:
    with pytest.raises(ValueError, match="n_channels"):
        make_cnn({"dropout": 0.1}, n_bands=4, patch_size=3)


def test_make_cnn_builds_and_is_deterministic() -> None:
    # The internal seed should make weight init reproducible regardless of the
    # ambient RNG state, so two builds with the same seed match in eval forward.
    torch.manual_seed(111)
    m1 = make_cnn(_FULL_PARAMS, n_bands=4, patch_size=5, seed=123)
    torch.manual_seed(999)
    m2 = make_cnn(_FULL_PARAMS, n_bands=4, patch_size=5, seed=123)
    m1.eval()
    m2.eval()
    x = torch.randn(6, 4, 5, 5)
    torch.testing.assert_close(m1(x), m2(x))
    assert isinstance(m1, ReceptiveFieldCNN)


# --------------------------------------------------------------------------- #
# train_cnn_fold.
# --------------------------------------------------------------------------- #


def test_train_cnn_fold_requires_optimiser_params() -> None:
    train_loader, val_loader, _ = _synthetic_loaders(
        n_train=16, n_val=8, n_bands=3, patch_size=3, signal=3.0, seed=0
    )
    with pytest.raises(ValueError, match="lr"):
        train_cnn_fold(
            {"n_channels": 8, "dropout": 0.0},
            train_loader,
            val_loader,
            n_bands=3,
            patch_size=3,
        )


def test_train_cnn_fold_learns_separable_task() -> None:
    train_loader, val_loader, y_val = _synthetic_loaders(
        n_train=256, n_val=128, n_bands=3, patch_size=3, signal=3.0, seed=0
    )
    result = train_cnn_fold(
        _FULL_PARAMS,
        train_loader,
        val_loader,
        n_bands=3,
        patch_size=3,
        max_epochs=15,
        patience=15,  # do not early-stop in this test
        warmup_epochs=2,
        device="cpu",
        seed=0,
    )
    assert isinstance(result, CNNFitResult)
    assert result.val_probs.shape == (128,)
    assert str(result.val_probs.dtype) == "float64"
    assert np.all((result.val_probs >= 0.0) & (result.val_probs <= 1.0))
    assert 0 <= result.best_epoch < result.n_epochs_run + 1
    assert result.n_epochs_run <= 15
    assert len(result.history) == result.n_epochs_run
    # The task is strongly separable, so the model should rank it well and assign
    # higher probability to positives than negatives.
    assert average_precision_score(y_val, result.val_probs) > 0.6
    assert result.val_probs[y_val == 1].mean() > result.val_probs[y_val == 0].mean()


def test_train_cnn_fold_early_stops_when_no_improvement() -> None:
    # A huge min_delta means no epoch after the first counts as an improvement,
    # so after `patience` non-improving epochs training stops deterministically.
    train_loader, val_loader, _ = _synthetic_loaders(
        n_train=64, n_val=32, n_bands=3, patch_size=3, signal=3.0, seed=1
    )
    result = train_cnn_fold(
        _FULL_PARAMS,
        train_loader,
        val_loader,
        n_bands=3,
        patch_size=3,
        max_epochs=50,
        patience=2,
        warmup_epochs=1,
        min_delta=10.0,
        device="cpu",
        seed=1,
    )
    assert result.best_epoch == 0
    assert result.n_epochs_run == 3  # epoch 0 best, epochs 1 and 2 fail patience
    assert result.n_epochs_run < 50


# --------------------------------------------------------------------------- #
# train_cnn_fixed (fixed-length outer refit).
# --------------------------------------------------------------------------- #


class _CountingLoader:
    """Wrap a loader to count how many times it is iterated (once per epoch)."""

    def __init__(self, loader: DataLoader) -> None:
        self._loader = loader
        self.epochs = 0

    def __iter__(self):  # type: ignore[no-untyped-def]
        self.epochs += 1
        return iter(self._loader)

    def __len__(self) -> int:
        return len(self._loader)


def test_train_cnn_fixed_runs_all_epochs_and_predicts() -> None:
    train_loader, val_loader, _ = _synthetic_loaders(
        n_train=64, n_val=32, n_bands=3, patch_size=3, signal=3.0, seed=0
    )
    counting = _CountingLoader(train_loader)
    model = train_cnn_fixed(
        _FULL_PARAMS,
        counting,
        n_bands=3,
        patch_size=3,
        n_epochs=4,
        warmup_epochs=1,
        device="cpu",
        seed=0,
    )
    # Every epoch runs, with no early stopping.
    assert counting.epochs == 4
    assert isinstance(model, ReceptiveFieldCNN)
    probs = predict_proba(model, val_loader, device="cpu")
    assert probs.shape == (32,)
    assert np.all(np.isfinite(probs))
    assert np.all((probs >= 0.0) & (probs <= 1.0))


def test_train_cnn_fixed_requires_optimiser_params() -> None:
    train_loader, _, _ = _synthetic_loaders(
        n_train=16, n_val=8, n_bands=3, patch_size=3, signal=3.0, seed=0
    )
    with pytest.raises(ValueError, match="lr"):
        train_cnn_fixed(
            {"n_channels": 8, "dropout": 0.0},
            train_loader,
            n_bands=3,
            patch_size=3,
            n_epochs=2,
        )


# --------------------------------------------------------------------------- #
# predict_proba and loader cleanup.
# --------------------------------------------------------------------------- #


def test_predict_proba_is_ordered_deterministic_and_in_range() -> None:
    _, val_loader, _ = _synthetic_loaders(
        n_train=8, n_val=20, n_bands=3, patch_size=3, signal=3.0, seed=2
    )
    model = make_cnn(_FULL_PARAMS, n_bands=3, patch_size=3, seed=2)
    first = predict_proba(model, val_loader, device="cpu")
    second = predict_proba(model, val_loader, device="cpu")
    assert first.shape == (20,)
    assert np.all((first >= 0.0) & (first <= 1.0))
    np.testing.assert_array_equal(first, second)  # eval is deterministic


def test_shutdown_loader_workers_is_noop_on_fresh_loader() -> None:
    _, val_loader, _ = _synthetic_loaders(
        n_train=8, n_val=8, n_bands=3, patch_size=3, signal=1.0, seed=3
    )
    # No iterator has been created yet; the call must be a safe no-op.
    shutdown_loader_workers(val_loader)


def test_predict_proba_on_empty_loader_returns_empty() -> None:
    empty = DataLoader(
        TensorDataset(torch.empty(0, 3, 3, 3), torch.empty(0), torch.empty(0)),
        batch_size=4,
    )
    model = make_cnn(_FULL_PARAMS, n_bands=3, patch_size=3, seed=0)
    out = predict_proba(model, empty, device="cpu")
    assert out.shape == (0,)
    assert str(out.dtype) == "float64"


def test_shutdown_loader_workers_shuts_down_active_iterator() -> None:
    # Duck-typed stub standing in for a loader with a live worker iterator, so
    # the shutdown branch runs without spawning real worker processes.
    class _StubIterator:
        def __init__(self) -> None:
            self.shutdown_called = False

        def _shutdown_workers(self) -> None:
            self.shutdown_called = True

    class _StubLoader:
        def __init__(self) -> None:
            self._iterator = _StubIterator()

    loader = _StubLoader()
    iterator = loader._iterator
    shutdown_loader_workers(loader)
    assert iterator.shutdown_called is True
    assert loader._iterator is None
