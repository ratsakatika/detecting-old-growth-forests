"""Unit tests for versioned cache-lineage and atomic replacement helpers."""

import hashlib
import json
from pathlib import Path

import numpy as np

from utils.cache import (
    CACHE_MANIFEST_VERSION,
    atomic_write_json,
    file_signature,
    is_pre_lineage_manifest,
    npy_matches,
    temporary_sibling,
)


def test_file_signature_records_digest_and_size(tmp_path: Path) -> None:
    target = tmp_path / "input.bin"
    target.write_bytes(b"cache lineage")

    signature = file_signature(target)

    assert signature == {
        "sha256": hashlib.sha256(b"cache lineage").hexdigest(),
        "size_bytes": len(b"cache lineage"),
    }


def test_temporary_sibling_is_hidden_unique_and_adjacent(tmp_path: Path) -> None:
    target = tmp_path / "cache.npy"

    first = temporary_sibling(target)
    second = temporary_sibling(target)

    assert first.parent == target.parent
    assert first.name.startswith(".cache.npy.")
    assert first != second


def test_atomic_write_json_replaces_complete_document(tmp_path: Path) -> None:
    target = tmp_path / "cache.json"
    target.write_text('{"old": true}\n', encoding="utf-8")

    atomic_write_json({"new": {"value": 1}}, target)

    assert json.loads(target.read_text(encoding="utf-8")) == {"new": {"value": 1}}
    assert not list(tmp_path.glob(".*.tmp"))


def test_npy_matches_checks_shape_and_dtype(tmp_path: Path) -> None:
    target = tmp_path / "cache.npy"
    np.save(target, np.zeros((2, 3), dtype=np.float32))

    assert npy_matches(target, (2, 3), dtype="float32")
    assert not npy_matches(target, (3, 2), dtype="float32")
    assert not npy_matches(target, (2, 3), dtype="float64")


def test_is_pre_lineage_manifest_branches(tmp_path: Path) -> None:
    manifest = tmp_path / "cache.json"
    # Absent: treated as pre-lineage (adoptable).
    assert is_pre_lineage_manifest(manifest)
    # Unreadable / malformed JSON: also pre-lineage.
    manifest.write_text("not json", encoding="utf-8")
    assert is_pre_lineage_manifest(manifest)
    # Present but no version (the old on-disk format): pre-lineage.
    manifest.write_text(json.dumps({"feature_set": "baseline"}), encoding="utf-8")
    assert is_pre_lineage_manifest(manifest)
    # An older version: still pre-lineage relative to the current one.
    manifest.write_text(json.dumps({"cache_manifest_version": 0}), encoding="utf-8")
    assert is_pre_lineage_manifest(manifest)
    # The current version: not pre-lineage, so a mismatch here forces a rebuild.
    manifest.write_text(
        json.dumps({"cache_manifest_version": CACHE_MANIFEST_VERSION}), encoding="utf-8"
    )
    assert not is_pre_lineage_manifest(manifest)
