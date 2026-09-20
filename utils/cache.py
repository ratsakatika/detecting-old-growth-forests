"""Versioned cache-lineage and atomic replacement helpers.

Cache manifests identify their exact source bytes, while temporary sibling
files ensure an interrupted rebuild cannot overwrite the last complete cache.
Importing this module binds names only and performs no file-system operations.
"""

from __future__ import annotations

import json
import os
from collections.abc import Mapping
from pathlib import Path
from typing import Any, Final
from uuid import uuid4

import numpy as np

from utils.io import write_json
from utils.manifest import sha256_file

CACHE_MANIFEST_VERSION: Final[int] = 1


def is_pre_lineage_manifest(path: str | Path) -> bool:
    """Return whether a manifest is absent or predates the current lineage version.

    A pre-lineage cache is one written before content fingerprinting existed (no
    ``cache_manifest_version``, or an older one). Such a cache may be *adopted*:
    its array is re-described with current lineage rather than re-extracted. A
    current-version manifest is never pre-lineage, so a genuine signature mismatch
    on a current cache still forces a rebuild.

    Args:
        path: Path to the manifest JSON.

    Returns:
        ``True`` if the manifest is missing, unreadable, or carries no current
        ``cache_manifest_version``; ``False`` for a current-version manifest.
    """
    try:
        manifest = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return True
    return manifest.get("cache_manifest_version") != CACHE_MANIFEST_VERSION


def file_signature(path: str | Path) -> dict[str, str | int]:
    """Return the content digest and byte size of one cache input."""
    target = Path(path)
    return {
        "sha256": sha256_file(target),
        "size_bytes": int(target.stat().st_size),
    }


def temporary_sibling(path: str | Path) -> Path:
    """Return a unique hidden temporary path beside ``path``."""
    target = Path(path)
    return target.with_name(f".{target.name}.{uuid4().hex}.tmp")


def replace_file(temp_path: str | Path, target_path: str | Path) -> Path:
    """Atomically replace ``target_path`` with a completed sibling file."""
    source = Path(temp_path)
    target = Path(target_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    os.replace(source, target)
    return target


def atomic_write_json(data: Mapping[str, Any], path: str | Path) -> Path:
    """Write deterministic JSON to a temporary sibling, then replace atomically."""
    target = Path(path)
    temp = temporary_sibling(target)
    try:
        write_json(data, temp)
        replace_file(temp, target)
    finally:
        temp.unlink(missing_ok=True)
    return target


def npy_matches(path: str | Path, shape: tuple[int, ...], *, dtype: str) -> bool:
    """Return whether a readable ``.npy`` has exactly ``shape`` and ``dtype``."""
    try:
        array = np.load(Path(path), mmap_mode="r")
    except (OSError, ValueError):
        return False
    try:
        return tuple(int(value) for value in array.shape) == shape and str(array.dtype) == dtype
    finally:
        del array


__all__ = [
    "CACHE_MANIFEST_VERSION",
    "atomic_write_json",
    "file_signature",
    "is_pre_lineage_manifest",
    "npy_matches",
    "replace_file",
    "temporary_sibling",
]
