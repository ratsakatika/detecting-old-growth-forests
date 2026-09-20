"""Receptive-field CNN pixel model for old-growth detection.

A from-scratch convolutional pixel classifier and its fold trainer. The model
takes a ``(C, P, P)`` patch centred on a pixel and predicts the positive-class
(old-growth) probability for that centre pixel, using the patch as the centre
pixel's spatial context. Design choices, with no dependence on the archive
design (which collapsed the spatial map with an average pool and so discarded
the neighbourhood arrangement):

  - **Receptive field equals the patch, by construction.** Valid (no-padding)
    3x3 convolutions shrink the patch by one pixel ring per layer, so
    ``K = (P - 1) // 2`` layers reduce ``P x P`` to exactly ``1 x 1``. The head
    is a 1x1 convolution to a single logit, not a flattened linear layer, so the
    network is fully convolutional and never averages over positions.
  - **Imbalance via per-row sample weights** (from ``utils.sampling``), the same
    mechanism as the XGBoost model; there is no ``pos_weight``, so a CNN-vs-
    XGBoost gap is not a difference in imbalance handling.
  - **Epochs chosen by early stopping** on the inner-validation loss with a
    patience window, keeping the best-epoch weights. This is the CNN analogue of
    tuning XGBoost ``n_estimators``: the validation fold is never used both to
    stop training and to score, because the epoch is chosen on the inner-
    validation loss while the reported metrics come from the outer fold.
  - **Optimiser AdamW; schedule a short linear warm-up then cosine decay**
    (preferred over one-cycle because early stopping makes the run length
    variable and cosine degrades gracefully when training stops early).
  - **Mixed precision** is used on CUDA only (halves memory, enables large
    batches); on CPU it is a no-op, so the module is testable on CPU.
  - ``device`` is supplied by the caller; the module never probes the hardware.

The fixed training budgets (``CNN_MAX_EPOCHS``, ``CNN_PATIENCE``,
``CNN_WARMUP_EPOCHS``, ``CNN_EARLY_STOP_MIN_DELTA``) are imported from
:mod:`utils.terminology`, the single source of truth for project constants, so
they cannot drift. They are provisional and to be confirmed from the Stage 6 CNN
pilot and benchmark.

This module consumes ``DataLoader``s that yield ``(patch, label, weight)``
batches; it does not build patches or read rasters, so the patch-supply
mechanism is independent of it. Importing the module binds names and a module
logger only: it performs no input/output and configures no logging handlers.
"""

from __future__ import annotations

import contextlib
import logging
import math
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Final

import numpy as np
import numpy.typing as npt
import torch
from torch import nn
from torch.utils.data import DataLoader

from utils.terminology import (
    CNN_EARLY_STOP_MIN_DELTA,
    CNN_MAX_EPOCHS,
    CNN_PATCH_SIZES,
    CNN_PATIENCE,
    CNN_WARMUP_EPOCHS,
    SEED,
)

# Module logger. No handlers are configured here; callers own logging setup.
logger = logging.getLogger(__name__)


# Architecture hyperparameters this wrapper expects in a parameter dict.
CNN_ARCH_PARAMS: Final[tuple[str, ...]] = ("n_channels", "dropout")
# Optimiser hyperparameters used by the fold trainer.
CNN_TRAIN_PARAMS: Final[tuple[str, ...]] = ("lr", "weight_decay")

# Guard against a zero weight sum when normalising a weighted mean.
_WEIGHT_EPS: Final[float] = 1e-12


class ReceptiveFieldCNN(nn.Module):
    """Fully convolutional centre-pixel classifier with a patch-sized field.

    With ``K = (patch_size - 1) // 2`` blocks of [valid 3x3 Conv2d -> BatchNorm2d
    -> ReLU], the first mapping ``n_bands`` to ``n_channels`` and each removing
    one pixel ring, the ``P x P`` patch reduces to ``n_channels x 1 x 1``. A
    ``Dropout2d`` then a 1x1 convolution to one channel give the centre-pixel
    logit. The output is one logit per patch.

    Args:
        n_bands: Number of input feature bands (channels of the patch).
        patch_size: Patch side length; one of :data:`CNN_PATCH_SIZES`.
        n_channels: Hidden channel width ``H`` (``> 0``).
        dropout: Channel-dropout probability in ``[0, 1)``.

    Raises:
        ValueError: If ``patch_size`` is not a supported odd size, ``n_channels``
            is not a positive integer, or ``dropout`` is outside ``[0, 1)``.
    """

    def __init__(self, n_bands: int, patch_size: int, n_channels: int, dropout: float) -> None:
        """Build the network for ``n_bands`` inputs at the given ``patch_size``."""
        super().__init__()
        if patch_size not in CNN_PATCH_SIZES:
            raise ValueError(f"patch_size must be one of {CNN_PATCH_SIZES}; got {patch_size}.")
        if int(n_channels) != n_channels or n_channels <= 0:
            raise ValueError(f"n_channels must be a positive integer; got {n_channels}.")
        if not 0.0 <= dropout < 1.0:
            raise ValueError(f"dropout must be in [0, 1); got {dropout}.")

        self.n_bands = int(n_bands)
        self.patch_size = int(patch_size)
        self.n_channels = int(n_channels)
        n_layers = (self.patch_size - 1) // 2

        blocks: list[nn.Module] = []
        in_channels = self.n_bands
        for _ in range(n_layers):
            blocks.append(nn.Conv2d(in_channels, self.n_channels, kernel_size=3, padding=0))
            blocks.append(nn.BatchNorm2d(self.n_channels))
            blocks.append(nn.ReLU(inplace=True))
            in_channels = self.n_channels
        self.features = nn.Sequential(*blocks)
        # Channel dropout on the (H, 1, 1) feature: zeroes whole channels, the
        # correct behaviour for convolutional features where neighbouring
        # spatial elements are correlated.
        self.dropout = nn.Dropout2d(p=float(dropout))
        # 1x1 convolution to a single logit (not a flattened Linear head).
        self.head = nn.Conv2d(self.n_channels, 1, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Map a batch of patches to one logit each.

        Args:
            x: Patch batch of shape ``(B, n_bands, patch_size, patch_size)``.

        Returns:
            A ``(B,)`` tensor of logits.

        Raises:
            ValueError: If the input spatial size does not match ``patch_size``.
        """
        if x.shape[-1] != self.patch_size or x.shape[-2] != self.patch_size:
            raise ValueError(
                f"expected spatial size {self.patch_size}x{self.patch_size}; "
                f"got {tuple(x.shape[-2:])}."
            )
        x = self.features(x)  # (B, H, 1, 1)
        x = self.dropout(x)
        x = self.head(x)  # (B, 1, 1, 1)
        return x.flatten(start_dim=1).squeeze(-1)  # (B,)


@dataclass(frozen=True, slots=True)
class CNNFitResult:
    """Outcome of training one fold.

    Attributes:
        val_probs: Old-growth probabilities for the validation loader, in loader
            order, ``float64`` of length ``n_val``.
        best_epoch: Zero-based epoch whose weights were restored (lowest
            validation loss); ``-1`` if training produced no improvement.
        best_val_loss: The lowest weighted validation loss reached.
        n_epochs_run: Number of epochs actually executed (<= ``max_epochs``).
        history: Per-epoch ``(epoch, val_loss)`` pairs.
    """

    val_probs: npt.NDArray[np.float64]
    best_epoch: int
    best_val_loss: float
    n_epochs_run: int
    history: tuple[tuple[int, float], ...]


def make_cnn(
    params: Mapping[str, float | int],
    *,
    n_bands: int,
    patch_size: int,
    seed: int = SEED,
) -> ReceptiveFieldCNN:
    """Build a :class:`ReceptiveFieldCNN` from a tuned parameter dict.

    Args:
        params: Must contain every name in :data:`CNN_ARCH_PARAMS`
            (``n_channels``, ``dropout``); other keys are ignored.
        n_bands: Number of input feature bands.
        patch_size: Patch side length; one of :data:`CNN_PATCH_SIZES`.
        seed: Random seed for weight initialisation (set globally before build).

    Returns:
        An unfitted :class:`ReceptiveFieldCNN`.

    Raises:
        ValueError: If any architecture parameter is missing from ``params``.
    """
    missing = [name for name in CNN_ARCH_PARAMS if name not in params]
    if missing:
        raise ValueError(f"missing CNN architecture parameters: {missing}")
    _seed_everything(int(seed))
    return ReceptiveFieldCNN(
        n_bands=int(n_bands),
        patch_size=int(patch_size),
        n_channels=int(params["n_channels"]),
        dropout=float(params["dropout"]),
    )


def train_cnn_fold(
    params: Mapping[str, float | int],
    train_loader: DataLoader,
    val_loader: DataLoader,
    *,
    n_bands: int,
    patch_size: int,
    max_epochs: int = CNN_MAX_EPOCHS,
    patience: int = CNN_PATIENCE,
    warmup_epochs: int = CNN_WARMUP_EPOCHS,
    min_delta: float = CNN_EARLY_STOP_MIN_DELTA,
    device: str = "cpu",
    seed: int = SEED,
) -> CNNFitResult:
    """Train one fold with early stopping and return validation probabilities.

    Builds a model from ``params`` (so the same dict drives architecture and
    optimiser), trains on ``train_loader`` minimising the weighted binary
    cross-entropy, stops when the weighted validation loss has not improved for
    ``patience`` epochs, restores the best-epoch weights, and predicts the
    validation loader.

    Both loaders must yield ``(patch, label, weight)`` batches: ``patch`` of
    shape ``(B, n_bands, patch_size, patch_size)``, ``label`` and ``weight`` of
    shape ``(B,)``. ``train_loader`` should shuffle; ``val_loader`` must not, so
    the returned probabilities align with the caller's validation order.

    Args:
        params: Must contain :data:`CNN_ARCH_PARAMS` and :data:`CNN_TRAIN_PARAMS`
            (``n_channels``, ``dropout``, ``lr``, ``weight_decay``).
        train_loader: Shuffled training batches.
        val_loader: Unshuffled validation batches.
        n_bands: Number of input feature bands.
        patch_size: Patch side length; one of :data:`CNN_PATCH_SIZES`.
        max_epochs: Upper bound on epochs.
        patience: Epochs without improvement before stopping.
        warmup_epochs: Linear warm-up length before cosine decay.
        min_delta: Minimum validation-loss decrease counted as improvement.
        device: ``"cuda"`` or ``"cpu"``.
        seed: Random seed.

    Returns:
        A :class:`CNNFitResult`.

    Raises:
        ValueError: If any optimiser parameter is missing from ``params``.
    """
    missing = [name for name in CNN_TRAIN_PARAMS if name not in params]
    if missing:
        raise ValueError(f"missing CNN optimiser parameters: {missing}")

    model = make_cnn(params, n_bands=n_bands, patch_size=patch_size, seed=seed).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(params["lr"]),
        weight_decay=float(params["weight_decay"]),
    )
    steps_per_epoch = max(1, len(train_loader))
    scheduler = _make_warmup_cosine_scheduler(
        optimizer,
        warmup_epochs=warmup_epochs,
        max_epochs=max_epochs,
        steps_per_epoch=steps_per_epoch,
    )
    use_amp = device == "cuda"
    scaler = _make_grad_scaler(use_amp)
    criterion = nn.BCEWithLogitsLoss(reduction="none")

    best_val_loss = math.inf
    best_epoch = -1
    best_state: dict[str, torch.Tensor] | None = None
    epochs_without_improvement = 0
    history: list[tuple[int, float]] = []
    n_epochs_run = 0

    for epoch in range(int(max_epochs)):
        model.train()
        for patches, labels, weights in train_loader:
            patches = patches.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True).float()
            weights = weights.to(device, non_blocking=True).float()
            optimizer.zero_grad(set_to_none=True)
            with _autocast(use_amp):
                logits = model(patches)
                per_row = criterion(logits, labels)
                loss = (per_row * weights).sum() / weights.sum().clamp_min(_WEIGHT_EPS)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()

        val_loss = _weighted_val_loss(model, val_loader, criterion, device, use_amp)
        history.append((epoch, val_loss))
        n_epochs_run = epoch + 1
        logger.debug("CNN epoch %d: val_loss %.6f", epoch, val_loss)

        if val_loss < best_val_loss - float(min_delta):
            best_val_loss = val_loss
            best_epoch = epoch
            best_state = {k: v.detach().to("cpu").clone() for k, v in model.state_dict().items()}
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= int(patience):
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    val_probs = predict_proba(model, val_loader, device=device)
    return CNNFitResult(
        val_probs=val_probs,
        best_epoch=best_epoch,
        best_val_loss=float(best_val_loss),
        n_epochs_run=n_epochs_run,
        history=tuple(history),
    )


def train_cnn_fixed(
    params: Mapping[str, float | int],
    train_loader: DataLoader,
    *,
    n_bands: int,
    patch_size: int,
    n_epochs: int,
    warmup_epochs: int = CNN_WARMUP_EPOCHS,
    device: str = "cpu",
    seed: int = SEED,
) -> ReceptiveFieldCNN:
    """Train for a fixed number of epochs with no early stopping or validation.

    This is the outer-refit counterpart to :func:`train_cnn_fold`: once the inner
    search has chosen both the hyperparameters and the training length, the model
    is retrained on all inner-fold pixels for exactly ``n_epochs`` and then used
    to predict the held-out outer fold. There is no validation set and no early
    stopping; every epoch runs, and the warm-up-then-cosine schedule spans exactly
    ``n_epochs`` so the learning rate completes its decay. The optimiser, weighted
    binary cross-entropy and CUDA mixed precision match :func:`train_cnn_fold`, so
    the two training paths differ only in their stopping rule.

    Args:
        params: Must contain :data:`CNN_ARCH_PARAMS` and :data:`CNN_TRAIN_PARAMS`
            (``n_channels``, ``dropout``, ``lr``, ``weight_decay``).
        train_loader: Shuffled training batches yielding ``(patch, label,
            weight)``.
        n_bands: Number of input feature bands.
        patch_size: Patch side length; one of :data:`CNN_PATCH_SIZES`.
        n_epochs: Exact number of epochs to train.
        warmup_epochs: Linear warm-up length before cosine decay.
        device: ``"cuda"`` or ``"cpu"``.
        seed: Random seed.

    Returns:
        The fitted :class:`ReceptiveFieldCNN`, ready for :func:`predict_proba`.

    Raises:
        ValueError: If any optimiser parameter is missing from ``params``.
    """
    missing = [name for name in CNN_TRAIN_PARAMS if name not in params]
    if missing:
        raise ValueError(f"missing CNN optimiser parameters: {missing}")

    model = make_cnn(params, n_bands=n_bands, patch_size=patch_size, seed=seed).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(params["lr"]),
        weight_decay=float(params["weight_decay"]),
    )
    steps_per_epoch = max(1, len(train_loader))
    scheduler = _make_warmup_cosine_scheduler(
        optimizer,
        warmup_epochs=warmup_epochs,
        max_epochs=n_epochs,
        steps_per_epoch=steps_per_epoch,
    )
    use_amp = device == "cuda"
    scaler = _make_grad_scaler(use_amp)
    criterion = nn.BCEWithLogitsLoss(reduction="none")

    for epoch in range(int(n_epochs)):
        model.train()
        for patches, labels, weights in train_loader:
            patches = patches.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True).float()
            weights = weights.to(device, non_blocking=True).float()
            optimizer.zero_grad(set_to_none=True)
            with _autocast(use_amp):
                logits = model(patches)
                per_row = criterion(logits, labels)
                loss = (per_row * weights).sum() / weights.sum().clamp_min(_WEIGHT_EPS)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
        logger.debug("CNN fixed-length epoch %d/%d complete.", epoch + 1, n_epochs)

    return model


def predict_proba(
    model: nn.Module, loader: DataLoader, *, device: str = "cpu"
) -> npt.NDArray[np.float64]:
    """Return old-growth probabilities for every row of a loader, in order.

    Args:
        model: A trained model.
        loader: A loader whose first per-batch element is the patch tensor; it
            must not shuffle if the order is to be meaningful.
        device: ``"cuda"`` or ``"cpu"``.

    Returns:
        A one-dimensional ``float64`` array of probabilities in loader order.
    """
    model = model.to(device)
    model.eval()
    use_amp = device == "cuda"
    chunks: list[npt.NDArray[np.float64]] = []
    with torch.no_grad():
        for batch in loader:
            patches = batch[0].to(device, non_blocking=True)
            with _autocast(use_amp):
                logits = model(patches)
            chunks.append(torch.sigmoid(logits.float()).cpu().numpy().astype(np.float64))
    if not chunks:
        return np.empty(0, dtype=np.float64)
    return np.concatenate(chunks)


def shutdown_loader_workers(loader: DataLoader) -> None:
    """Shut down a loader's worker processes and release their file descriptors.

    Ported from ``archive/scripts/027_*.py``: across many short-lived loaders (one
    per fold and trial) the worker file descriptors otherwise accumulate until
    the process hits its open-file limit. Safe to call when no iterator exists.

    Args:
        loader: The loader whose workers should be shut down.
    """
    iterator = getattr(loader, "_iterator", None)
    if iterator is not None and hasattr(iterator, "_shutdown_workers"):
        iterator._shutdown_workers()
        loader._iterator = None


# --------------------------------------------------------------------------- #
# Internal helpers.
# --------------------------------------------------------------------------- #


def _seed_everything(seed: int) -> None:
    """Seed NumPy and torch (CPU and CUDA) for reproducible runs."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _autocast(enabled: bool) -> Any:
    """Return a CUDA autocast context when enabled, else a no-op context."""
    if enabled:
        return torch.autocast(device_type="cuda", enabled=True)  # pragma: no cover (CUDA)
    return contextlib.nullcontext()


def _make_grad_scaler(enabled: bool) -> Any:
    """Build a gradient scaler, tolerant of the torch AMP API version."""
    amp = getattr(torch, "amp", None)
    if amp is not None and hasattr(amp, "GradScaler"):
        try:
            return amp.GradScaler("cuda", enabled=enabled)
        except TypeError:  # pragma: no cover (old torch.amp)
            return amp.GradScaler(enabled=enabled)
    return torch.cuda.amp.GradScaler(enabled=enabled)  # pragma: no cover (pre-2.x torch)


def _make_warmup_cosine_scheduler(
    optimizer: torch.optim.Optimizer,
    *,
    warmup_epochs: int,
    max_epochs: int,
    steps_per_epoch: int,
) -> torch.optim.lr_scheduler.LambdaLR:
    """Linear warm-up for ``warmup_epochs`` then cosine decay to zero.

    The multiplier rises linearly from ~0 to 1 over the warm-up, then follows a
    cosine from 1 to 0 across the remaining steps of the nominal horizon. Early
    stopping may end training before the horizon, in which case the rate simply
    has not reached zero, which is harmless.
    """
    total_steps = max(1, int(max_epochs) * int(steps_per_epoch))
    warmup_steps = max(0, int(warmup_epochs) * int(steps_per_epoch))

    def lr_lambda(step: int) -> float:
        if warmup_steps > 0 and step < warmup_steps:
            return (step + 1) / warmup_steps
        denom = max(1, total_steps - warmup_steps)
        progress = (step - warmup_steps) / denom
        progress = min(1.0, max(0.0, progress))
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def _weighted_val_loss(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: str,
    use_amp: bool,
) -> float:
    """Weighted-mean BCE over a validation loader (same criterion as training)."""
    model.eval()
    total = 0.0
    weight_sum = 0.0
    with torch.no_grad():
        for patches, labels, weights in loader:
            patches = patches.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True).float()
            weights = weights.to(device, non_blocking=True).float()
            with _autocast(use_amp):
                logits = model(patches)
                per_row = criterion(logits, labels)
            total += float((per_row.float() * weights).sum().item())
            weight_sum += float(weights.sum().item())
    return total / max(weight_sum, _WEIGHT_EPS)


__all__ = [
    "CNN_ARCH_PARAMS",
    "CNN_TRAIN_PARAMS",
    "CNNFitResult",
    "ReceptiveFieldCNN",
    "make_cnn",
    "predict_proba",
    "shutdown_loader_workers",
    "train_cnn_fixed",
    "train_cnn_fold",
]
