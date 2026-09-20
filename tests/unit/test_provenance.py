"""Unit tests for the download-provenance helpers in utils.provenance."""

import hashlib
import json
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import pytest

import utils.provenance
from utils.provenance import (
    ProvenanceCheck,
    format_provenance_result,
    provenance_path_for,
    verify_provenance,
    write_provenance,
)


def _write_file(path: Path, content: bytes) -> Path:
    """Create ``path`` (with parents) holding ``content``."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(content)
    return path


def test_importing_provenance_creates_nothing(tmp_path: Path) -> None:
    # Import the module in a fresh interpreter whose working directory is an
    # empty temp dir; anything created at import time would appear here.
    repo_root = Path(utils.provenance.__file__).resolve().parents[1]
    code = (
        "import pathlib\n"
        "import utils.provenance\n"
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


def test_provenance_path_for_appends_suffix_in_same_directory() -> None:
    data_path = Path("data/raw/natura2000_end2024.gpkg")

    sidecar = provenance_path_for(data_path)

    assert sidecar == Path("data/raw/natura2000_end2024.gpkg.provenance.json")
    assert sidecar.parent == data_path.parent


def test_provenance_path_for_accepts_string() -> None:
    sidecar = provenance_path_for("data/raw/input.tif")

    assert sidecar == Path("data/raw/input.tif.provenance.json")


def test_write_provenance_records_expected_fields(tmp_path: Path) -> None:
    content = b"worldcover composite bytes"
    target = _write_file(tmp_path / "data" / "raw" / "input.tif", content)

    sidecar = write_provenance(target, "https://example.org/input.tif")

    assert sidecar == provenance_path_for(target)
    assert sidecar.is_file()

    record = json.loads(sidecar.read_text(encoding="utf-8"))
    assert record["filename"] == "input.tif"
    assert record["source"] == "https://example.org/input.tif"
    assert record["size_bytes"] == len(content)
    assert record["sha256"] == hashlib.sha256(content).hexdigest()
    # The timestamp is ISO-8601 and timezone-aware (UTC offset present).
    parsed = datetime.fromisoformat(record["downloaded_at"])
    assert parsed.tzinfo is not None
    assert parsed.utcoffset() is not None


def test_write_provenance_merges_extra_metadata(tmp_path: Path) -> None:
    target = _write_file(tmp_path / "input.tif", b"abc")

    sidecar = write_provenance(
        target,
        "earth-engine://asset",
        extra={"licence": "CC-BY-4.0", "epsg": 3035},
    )

    record = json.loads(sidecar.read_text(encoding="utf-8"))
    assert record["licence"] == "CC-BY-4.0"
    assert record["epsg"] == 3035
    # Reserved fields are still present and correct.
    assert record["source"] == "earth-engine://asset"
    assert record["filename"] == "input.tif"


def test_write_provenance_rejects_extra_overwriting_reserved_keys(tmp_path: Path) -> None:
    target = _write_file(tmp_path / "input.tif", b"abc")

    with pytest.raises(ValueError, match="reserved"):
        write_provenance(target, "src", extra={"source": "spoofed", "size_bytes": 0})

    # No sidecar should have been written on the failed call.
    assert not provenance_path_for(target).exists()


def test_write_provenance_can_omit_hash(tmp_path: Path) -> None:
    target = _write_file(tmp_path / "input.tif", b"abc")

    sidecar = write_provenance(target, "src", hash_data=False)

    record = json.loads(sidecar.read_text(encoding="utf-8"))
    assert "sha256" not in record
    # Size is still recorded even when the digest is skipped.
    assert record["size_bytes"] == 3


def test_write_provenance_honours_custom_provenance_path(tmp_path: Path) -> None:
    target = _write_file(tmp_path / "input.tif", b"abc")
    custom = tmp_path / "sidecars" / "input.json"

    returned = write_provenance(target, "src", provenance_path=custom)

    assert returned == custom
    assert custom.is_file()
    assert not provenance_path_for(target).exists()


def test_verify_provenance_verified_on_match(tmp_path: Path) -> None:
    content = b"unchanged bytes"
    target = _write_file(tmp_path / "input.tif", content)
    write_provenance(target, "src")

    result = verify_provenance(target)

    assert isinstance(result, ProvenanceCheck)
    assert result.ok is True
    assert result.status == "verified"
    assert result.expected_sha256 == hashlib.sha256(content).hexdigest()
    assert result.actual_sha256 == result.expected_sha256


def test_verify_provenance_mismatch_on_changed_content(tmp_path: Path) -> None:
    target = _write_file(tmp_path / "input.tif", b"original content here")
    write_provenance(target, "src")

    # Overwrite with different content of the *same* length so the size check
    # passes and the digest comparison is what catches the change.
    replacement = b"tampered content !!!!"
    assert len(replacement) == len(b"original content here")
    target.write_bytes(replacement)

    result = verify_provenance(target)

    assert result.ok is False
    assert result.status == "mismatch"
    assert result.expected_sha256 == hashlib.sha256(b"original content here").hexdigest()
    assert result.actual_sha256 == hashlib.sha256(replacement).hexdigest()


def test_verify_provenance_size_mismatch_skips_hashing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    target = _write_file(tmp_path / "input.tif", b"abc")
    write_provenance(target, "src")

    # Change the file's size; the cheap size check must catch this before any
    # hashing happens. Patch sha256_file to blow up if it is ever called.
    target.write_bytes(b"abcdef")

    def _explode(*args: object, **kwargs: object) -> str:
        raise AssertionError("sha256_file must not be called on a size mismatch")

    monkeypatch.setattr(utils.provenance, "sha256_file", _explode)

    result = verify_provenance(target)

    assert result.ok is False
    assert result.status == "mismatch"
    assert result.actual_sha256 is None


def test_verify_provenance_data_missing(tmp_path: Path) -> None:
    target = _write_file(tmp_path / "input.tif", b"abc")
    write_provenance(target, "src")
    target.unlink()

    result = verify_provenance(target)

    assert result.ok is False
    assert result.status == "data_missing"
    assert result.actual_sha256 is None


def test_verify_provenance_provenance_missing(tmp_path: Path) -> None:
    target = _write_file(tmp_path / "input.tif", b"abc")
    # No sidecar written.

    result = verify_provenance(target)

    assert result.ok is False
    assert result.status == "provenance_missing"
    assert result.expected_sha256 is None
    assert result.actual_sha256 is None


def test_verify_provenance_no_recorded_hash(tmp_path: Path) -> None:
    target = _write_file(tmp_path / "input.tif", b"abc")
    write_provenance(target, "src", hash_data=False)

    result = verify_provenance(target)

    assert result.ok is False
    assert result.status == "no_recorded_hash"
    assert result.expected_sha256 is None


def test_verify_provenance_honours_custom_provenance_path(tmp_path: Path) -> None:
    content = b"custom sidecar bytes"
    target = _write_file(tmp_path / "input.tif", content)
    custom = tmp_path / "sidecars" / "input.json"
    write_provenance(target, "src", provenance_path=custom)

    # Without the custom path, the default sidecar is absent.
    assert verify_provenance(target).status == "provenance_missing"

    result = verify_provenance(target, provenance_path=custom)
    assert result.ok is True
    assert result.status == "verified"


def test_verify_provenance_ignores_downloaded_at(tmp_path: Path) -> None:
    # The same, unchanged file recorded again at a different download time must
    # still verify: only size_bytes and sha256 matter, never downloaded_at.
    content = b"date-independent bytes"
    target = _write_file(tmp_path / "input.tif", content)
    sidecar = write_provenance(target, "src")

    record = json.loads(sidecar.read_text(encoding="utf-8"))
    original_timestamp = record["downloaded_at"]
    record["downloaded_at"] = "2099-12-31T23:59:59+00:00"
    assert record["downloaded_at"] != original_timestamp
    sidecar.write_text(json.dumps(record), encoding="utf-8")

    result = verify_provenance(target)

    assert result.ok is True
    assert result.status == "verified"


def test_format_provenance_result_recorded_message() -> None:
    # On a first download the check is ignored; a "recorded" message is returned.
    check = ProvenanceCheck(
        ok=False,
        status="provenance_missing",
        message="irrelevant",
        expected_sha256=None,
        actual_sha256=None,
    )

    message = format_provenance_result(check, recorded=True)

    assert message == ("Recorded this file's fingerprint, so it can be verified on future runs.")


def test_format_provenance_result_verified() -> None:
    check = ProvenanceCheck(
        ok=True,
        status="verified",
        message="matches",
        expected_sha256="abc",
        actual_sha256="abc",
    )

    message = format_provenance_result(check)

    assert message == "Verified: your file matches the version used in the study."


def test_format_provenance_result_mismatch_shows_both_checksums() -> None:
    check = ProvenanceCheck(
        ok=False,
        status="mismatch",
        message="differs",
        expected_sha256="expected-digest",
        actual_sha256="actual-digest",
    )

    message = format_provenance_result(check)

    assert "WARNING" in message
    # Plain explanation that this is a benign version difference.
    assert "updated" in message
    assert "results may differ" in message
    # Both checksums appear so the user can see what changed.
    assert "expected-digest" in message
    assert "actual-digest" in message


def test_format_provenance_result_no_recorded_hash() -> None:
    check = ProvenanceCheck(
        ok=False,
        status="no_recorded_hash",
        message="no checksum",
        expected_sha256=None,
        actual_sha256=None,
    )

    message = format_provenance_result(check)

    assert "checksum" in message
    assert "could not be checked" in message


@pytest.mark.parametrize("status", ["data_missing", "provenance_missing", "something_else"])
def test_format_provenance_result_unverifiable_statuses(status: str) -> None:
    check = ProvenanceCheck(
        ok=False,
        status=status,
        message="underlying detail",
        expected_sha256=None,
        actual_sha256=None,
    )

    message = format_provenance_result(check)

    assert "WARNING" in message
    # The status and underlying message are surfaced so the user can act.
    assert status in message
    assert "underlying detail" in message
