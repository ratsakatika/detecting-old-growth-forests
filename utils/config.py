"""Typed, validated configuration objects for model runs.

These dataclasses pin down the *shape* of a run's configuration — the canonical
architecture and feature-set names, the fold, and the sampling and
hyper-parameter-search budgets — and validate it on construction. They execute
no modelling, calibration or data loading; ``calibrate`` on
:class:`InferenceConfig` is only a flag recorded for the final mapping output
(see AGENTS.md, "Calibration").

Canonical names, fold IDs and the default budgets all come from
``utils.terminology`` rather than being hard-coded, so configuration cannot
encode an invented name. Importing this module only binds names: it performs no
input/output, and the sole filesystem write happens inside :func:`write_config`.
"""

from dataclasses import asdict, dataclass, is_dataclass
from pathlib import Path
from typing import Any

from utils.io import write_json
from utils.terminology import (
    ARCHITECTURES,
    FOLD_IDS,
    MODEL_INPUTS,
    N_HP_TRIALS,
    N_PER_PARCEL,
    SEED,
)


def _validate_architecture(architecture: str) -> None:
    """Raise if ``architecture`` is not a canonical architecture name.

    Args:
        architecture: Candidate architecture name.

    Raises:
        ValueError: If it is not in :data:`utils.terminology.ARCHITECTURES`.
    """
    if architecture not in ARCHITECTURES:
        raise ValueError(
            f"unknown architecture {architecture!r}; expected one of {list(ARCHITECTURES)}"
        )


def _validate_feature_set(feature_set: str) -> None:
    """Raise if ``feature_set`` is not a canonical feature-set name.

    Args:
        feature_set: Candidate feature-set name.

    Raises:
        ValueError: If it is not in :data:`utils.terminology.FEATURE_SETS`.
    """
    if feature_set not in MODEL_INPUTS:
        raise ValueError(
            f"unknown feature set {feature_set!r}; expected one of {list(MODEL_INPUTS)}"
        )


@dataclass(frozen=True, slots=True)
class RunConfig:
    """Validated configuration for a single nested-CV model run.

    Attributes:
        architecture: Canonical architecture name (in ``ARCHITECTURES``).
        feature_set: Canonical feature-set name (in ``FEATURE_SETS``).
        fold_id: Outer fold identifier (in ``FOLD_IDS``, i.e. 1..6).
        seed: Random seed; must be non-negative. Defaults to ``SEED``.
        n_hp_trials: Optuna trials per fold; must be positive. Defaults to
            ``N_HP_TRIALS``.
        n_per_parcel: Pixels sampled per parcel; must be positive. Defaults to
            ``N_PER_PARCEL``.
    """

    architecture: str
    feature_set: str
    fold_id: int
    seed: int = SEED
    n_hp_trials: int = N_HP_TRIALS
    n_per_parcel: int = N_PER_PARCEL

    def __post_init__(self) -> None:
        """Validate the configuration once, on construction.

        Raises:
            ValueError: If any field is outside its permitted set or range.
        """
        _validate_architecture(self.architecture)
        _validate_feature_set(self.feature_set)
        if self.fold_id not in FOLD_IDS:
            raise ValueError(f"fold_id must be one of {list(FOLD_IDS)}, got {self.fold_id}")
        if self.seed < 0:
            raise ValueError(f"seed must be non-negative, got {self.seed}")
        if self.n_hp_trials <= 0:
            raise ValueError(f"n_hp_trials must be positive, got {self.n_hp_trials}")
        if self.n_per_parcel <= 0:
            raise ValueError(f"n_per_parcel must be positive, got {self.n_per_parcel}")


@dataclass(frozen=True, slots=True)
class InferenceConfig:
    """Validated configuration for a final wall-to-wall inference run.

    Attributes:
        architecture: Canonical architecture name (in ``ARCHITECTURES``).
        feature_set: Canonical feature-set name (in ``FEATURE_SETS``).
        seed: Random seed; must be non-negative. Defaults to ``SEED``.
        calibrate: Whether to calibrate probabilities for the final map. This is
            only a recorded flag; calibration itself is not implemented here.
    """

    architecture: str
    feature_set: str
    seed: int = SEED
    calibrate: bool = True

    def __post_init__(self) -> None:
        """Validate the configuration once, on construction.

        Raises:
            ValueError: If a name is not canonical or the seed is negative.
        """
        _validate_architecture(self.architecture)
        _validate_feature_set(self.feature_set)
        if self.seed < 0:
            raise ValueError(f"seed must be non-negative, got {self.seed}")


def config_to_dict(config: object) -> dict[str, Any]:
    """Convert a dataclass configuration instance to a plain dictionary.

    Args:
        config: A dataclass *instance* (not a class).

    Returns:
        A plain ``dict`` of the configuration's fields.

    Raises:
        TypeError: If ``config`` is not a dataclass instance.
    """
    if not is_dataclass(config) or isinstance(config, type):
        raise TypeError(f"expected a dataclass instance, got {type(config).__name__}")
    return asdict(config)


def write_config(config: object, path: str | Path) -> Path:
    """Write a configuration to ``path`` as deterministic JSON.

    Args:
        config: A dataclass configuration instance.
        path: Destination JSON file.

    Returns:
        The path written to.

    Raises:
        TypeError: If ``config`` is not a dataclass instance.
    """
    return write_json(config_to_dict(config), path)


__all__ = [
    "InferenceConfig",
    "RunConfig",
    "config_to_dict",
    "write_config",
]
