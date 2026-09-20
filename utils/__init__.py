"""Shared helpers for the old-growth-forests publication repository.

This package collects the importable modules that the scripts and notebooks
build on, such as canonical names and constants, input/output schemas, logging,
paths and plotting style.

Submodules are imported explicitly by name (for example
``from utils.terminology import SEED``) rather than being re-exported here, so
``__all__`` is intentionally empty for now.
"""

__all__: list[str] = []
