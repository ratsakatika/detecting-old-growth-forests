"""Unit tests for the input-file manifest helpers in utils.manifest."""

import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

import utils.manifest
from utils.manifest import (
    FileManifestEntry,
    build_input_manifest,
    sha256_file,
    write_input_manifest,
)


def _write_file(path: Path, content: bytes) -> Path:
    """Create ``path`` (with parents) holding ``content``."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(content)
    return path


def test_importing_manifest_creates_nothing(tmp_path: Path) -> None:
    # Import the module in a fresh interpreter whose working directory is an
    # empty temp dir; if anything were created at import time it would appear
    # here. A subprocess avoids polluting this process's already-imported module.
    repo_root = Path(utils.manifest.__file__).resolve().parents[1]
    code = (
        "import pathlib\n"
        "import utils.manifest\n"
        "print(sorted(p.name for p in pathlib.Path('.').iterdir()))\n"
    )
    result = subprocess.run(
        [sys.executable, "-c", code],
        cwd=tmp_path,
        env={**os.environ, "PYTHONPATH": str(repo_root)},
        capture_output=True,
        text=True,
        check=True,
    )

    assert result.stdout.strip() == "[]"


def test_sha256_file_matches_hashlib(tmp_path: Path) -> None:
    content = b"old-growth forest probability\n"
    target = _write_file(tmp_path / "small.bin", content)

    assert sha256_file(target) == hashlib.sha256(content).hexdigest()


def test_sha256_file_streams_in_small_chunks(tmp_path: Path) -> None:
    # A tiny chunk_size forces many reads; the digest must still be correct.
    content = b"x" * 4096
    target = _write_file(tmp_path / "chunked.bin", content)

    assert sha256_file(target, chunk_size=7) == hashlib.sha256(content).hexdigest()


def test_build_input_manifest_records_fields(tmp_path: Path) -> None:
    content = b"reference labels raster bytes"
    target = _write_file(tmp_path / "input.tif", content)

    manifest = build_input_manifest([target])

    key = target.as_posix()
    assert set(manifest) == {key}
    entry = manifest[key]
    assert isinstance(entry, FileManifestEntry)
    assert entry.path == key
    assert entry.sha256 == hashlib.sha256(content).hexdigest()
    assert entry.size_bytes == len(content)
    assert entry.modified_time == target.stat().st_mtime


def test_repo_root_produces_relative_posix_keys(tmp_path: Path) -> None:
    target = _write_file(tmp_path / "data" / "raw" / "input.tif", b"abc")

    manifest = build_input_manifest([target], repo_root=tmp_path)

    assert set(manifest) == {"data/raw/input.tif"}
    assert manifest["data/raw/input.tif"].path == "data/raw/input.tif"


def test_missing_files_reported_together(tmp_path: Path) -> None:
    present = _write_file(tmp_path / "here.tif", b"ok")
    absent_a = tmp_path / "gone_a.tif"
    absent_b = tmp_path / "gone_b.tif"

    with pytest.raises(FileNotFoundError) as excinfo:
        build_input_manifest([present, absent_a, absent_b])

    message = str(excinfo.value)
    assert str(absent_a) in message
    assert str(absent_b) in message


def test_directories_are_rejected(tmp_path: Path) -> None:
    directory = tmp_path / "a_directory"
    directory.mkdir()

    with pytest.raises(IsADirectoryError) as excinfo:
        build_input_manifest([directory])

    assert str(directory) in str(excinfo.value)


def test_write_input_manifest_writes_sorted_deterministic_json(tmp_path: Path) -> None:
    first_file = _write_file(tmp_path / "data" / "a.tif", b"alpha")
    second_file = _write_file(tmp_path / "data" / "b.tif", b"beta")

    out = tmp_path / "results" / "run" / "input_manifest.json"
    returned = write_input_manifest([first_file, second_file], out, repo_root=tmp_path)

    assert returned == out
    assert out.is_file()

    text = out.read_text(encoding="utf-8")
    payload = json.loads(text)
    assert set(payload) == {"data/a.tif", "data/b.tif"}
    assert payload["data/a.tif"]["sha256"] == hashlib.sha256(b"alpha").hexdigest()
    assert payload["data/a.tif"]["size_bytes"] == len(b"alpha")
    # Top-level keys appear in sorted order in the serialised text.
    assert text.index('"data/a.tif"') < text.index('"data/b.tif"')

    # Reordering the inputs yields a byte-identical file (deterministic output).
    other = tmp_path / "results" / "run" / "input_manifest_again.json"
    write_input_manifest([second_file, first_file], other, repo_root=tmp_path)
    assert other.read_text(encoding="utf-8") == text


def test_manifest_key_falls_back_to_absolute_when_outside_repo_root(tmp_path: Path) -> None:
    outside = _write_file(tmp_path / "outside" / "input.tif", b"bytes")
    repo_root = tmp_path / "repo"
    repo_root.mkdir()

    manifest = build_input_manifest([outside], repo_root=repo_root)

    # The file is not under repo_root, so the key stays the absolute POSIX path.
    assert set(manifest) == {outside.as_posix()}
