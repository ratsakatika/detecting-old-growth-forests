"""Download-provenance sidecars for files placed in ``data/raw/``.

Raw inputs are large and are not committed; their provenance is. For each raw
file this module writes a small ``<name>.provenance.json`` sidecar recording
where the file came from, when it was fetched, its size and (by default) its
SHA256 digest. The author commits the sidecars; a user who later downloads the
same file can verify it matches the recorded digest (see AGENTS.md,
"Reproducibility floor").

The module is deliberately ignorant of *how* a file was obtained — direct HTTP,
a Google Earth Engine export or an API call all look the same here. It operates
on a file that already exists on disk plus a free-text ``source`` string, so
every dataset cell in ``notebooks/001_public_data_download.ipynb`` can use it
regardless of download method.

Importing this module only binds names: it performs no input/output and creates
no directories. JSON writing is delegated to :func:`utils.io.write_json` and
hashing to :func:`utils.manifest.sha256_file`; neither is reimplemented here.
"""

import json
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Final

from utils.io import write_json
from utils.manifest import sha256_file

# Suffix appended to a data file's name to form its sidecar's name.
_PROVENANCE_SUFFIX: Final[str] = ".provenance.json"

# Keys the sidecar owns and computes itself. Caller-supplied ``extra`` metadata
# may never overwrite these, so a stray key cannot silently rewrite the recorded
# source, timestamp, size or digest.
_RESERVED_KEYS: Final[tuple[str, ...]] = (
    "filename",
    "source",
    "downloaded_at",
    "size_bytes",
    "sha256",
)


@dataclass(frozen=True, slots=True)
class ProvenanceCheck:
    """Outcome of verifying a data file against its provenance sidecar.

    The record is frozen: a check describes a single comparison and is never
    mutated. Callers decide how to report it (hard rule 3); this type carries no
    side effects.

    Attributes:
        ok: ``True`` only when the file is present and its SHA256 matches the
            recorded digest.
        status: One of ``"verified"``, ``"mismatch"``, ``"data_missing"``,
            ``"provenance_missing"`` or ``"no_recorded_hash"``.
        message: A human-readable explanation of the outcome.
        expected_sha256: The digest recorded in the sidecar, or ``None`` when no
            digest was recorded or no sidecar was found.
        actual_sha256: The digest computed from the file on disk, or ``None``
            when the file was not hashed (missing file, no recorded digest, or a
            cheap size-only mismatch).
    """

    ok: bool
    status: str
    message: str
    expected_sha256: str | None
    actual_sha256: str | None


def provenance_path_for(data_path: str | Path) -> Path:
    """Return the sidecar path for a data file.

    The sidecar sits beside the data file and is named after it, with
    :data:`_PROVENANCE_SUFFIX` appended — for example
    ``natura2000_end2024.gpkg`` maps to
    ``natura2000_end2024.gpkg.provenance.json``.

    Args:
        data_path: The data file the sidecar describes.

    Returns:
        The sidecar path, in the same directory as ``data_path``.
    """
    path = Path(data_path)
    return path.with_name(path.name + _PROVENANCE_SUFFIX)


def write_provenance(
    data_path: str | Path,
    source: str,
    *,
    extra: dict[str, Any] | None = None,
    hash_data: bool = True,
    provenance_path: str | Path | None = None,
) -> Path:
    """Write a provenance sidecar for an already-downloaded file.

    The record always carries the file's name, the free-text ``source``, a
    timezone-aware UTC download timestamp and the file's size in bytes; it also
    carries the SHA256 digest unless ``hash_data`` is ``False``. Any ``extra``
    metadata is merged in, but it may not overwrite the reserved keys.

    Args:
        data_path: The downloaded file to describe. It must already exist.
        source: Free-text description of where the file came from (a URL, an
            Earth Engine asset, an API endpoint, and so on).
        extra: Optional additional metadata to record alongside the reserved
            fields. Must not contain any reserved key.
        hash_data: When ``True`` (the default), compute and record the SHA256
            digest via :func:`utils.manifest.sha256_file`. When ``False``, the
            ``sha256`` key is omitted.
        provenance_path: Optional explicit sidecar path. When ``None``, it is
            derived with :func:`provenance_path_for`.

    Returns:
        The path of the written sidecar.

    Raises:
        ValueError: If ``extra`` contains any reserved key.
    """
    path = Path(data_path)

    if extra is not None:
        clashes = sorted(key for key in extra if key in _RESERVED_KEYS)
        if clashes:
            joined = ", ".join(clashes)
            raise ValueError(f"extra may not overwrite reserved provenance keys: {joined}.")

    record: dict[str, Any] = dict(extra) if extra is not None else {}
    record["filename"] = path.name
    record["source"] = source
    record["downloaded_at"] = datetime.now(UTC).isoformat()
    record["size_bytes"] = int(path.stat().st_size)
    if hash_data:
        record["sha256"] = sha256_file(path)

    target = provenance_path_for(path) if provenance_path is None else Path(provenance_path)
    return write_json(record, target)


def verify_provenance(
    data_path: str | Path,
    *,
    provenance_path: str | Path | None = None,
) -> ProvenanceCheck:
    """Verify a data file against its recorded provenance sidecar.

    The checks short-circuit in order, cheapest first, so a large file is only
    hashed when nothing simpler has already settled the outcome:

    1. sidecar absent -> ``"provenance_missing"``;
    2. data file absent -> ``"data_missing"``;
    3. sidecar records no digest -> ``"no_recorded_hash"``;
    4. recorded size differs from the file's size -> ``"mismatch"`` (the file is
       not hashed, since differing sizes already prove differing contents);
    5. otherwise the file is hashed and compared -> ``"verified"`` or
       ``"mismatch"``.

    Args:
        data_path: The file to verify.
        provenance_path: Optional explicit sidecar path. When ``None``, it is
            derived with :func:`provenance_path_for`.

    Returns:
        A :class:`ProvenanceCheck` describing the outcome.
    """
    path = Path(data_path)
    sidecar = provenance_path_for(path) if provenance_path is None else Path(provenance_path)

    if not sidecar.is_file():
        return ProvenanceCheck(
            ok=False,
            status="provenance_missing",
            message=f"No provenance sidecar found at {sidecar}.",
            expected_sha256=None,
            actual_sha256=None,
        )

    record = json.loads(sidecar.read_text(encoding="utf-8"))
    expected = record.get("sha256")

    if not path.is_file():
        return ProvenanceCheck(
            ok=False,
            status="data_missing",
            message=f"Data file is missing: {path}.",
            expected_sha256=expected,
            actual_sha256=None,
        )

    if expected is None:
        return ProvenanceCheck(
            ok=False,
            status="no_recorded_hash",
            message=(
                f"Provenance sidecar {sidecar} records no SHA256; " "cannot verify file contents."
            ),
            expected_sha256=None,
            actual_sha256=None,
        )

    recorded_size = record.get("size_bytes")
    actual_size = int(path.stat().st_size)
    if recorded_size is not None and int(recorded_size) != actual_size:
        return ProvenanceCheck(
            ok=False,
            status="mismatch",
            message=(
                f"Size mismatch for {path.name}: recorded {int(recorded_size)} bytes, "
                f"found {actual_size} bytes; contents differ."
            ),
            expected_sha256=expected,
            actual_sha256=None,
        )

    actual = sha256_file(path)
    if actual == expected:
        return ProvenanceCheck(
            ok=True,
            status="verified",
            message=f"{path.name} matches the recorded SHA256.",
            expected_sha256=expected,
            actual_sha256=actual,
        )

    return ProvenanceCheck(
        ok=False,
        status="mismatch",
        message=f"SHA256 mismatch for {path.name}: the file differs from the recorded digest.",
        expected_sha256=expected,
        actual_sha256=actual,
    )


def format_provenance_result(check: ProvenanceCheck, *, recorded: bool = False) -> str:
    """Render a verification outcome as a single message for a notebook user.

    This produces plain, non-alarming wording suitable for printing beneath a
    download cell. It does not print anything itself (hard rule 3); the caller
    decides how to display the returned string.

    Args:
        check: The outcome to describe.
        recorded: Set to ``True`` on a first download, when the provenance
            sidecar has just been written and there was nothing to verify
            against. In that case ``check`` is ignored and a "recorded" message
            is returned.

    Returns:
        A human-friendly message. For a benign version difference (``"mismatch"``)
        the message is a multi-line warning that shows the expected and actual
        SHA256 digests; for the various "could not verify" outcomes it names the
        status and includes the underlying message.
    """
    if recorded:
        return "Recorded this file's fingerprint, so it can be verified on future runs."

    if check.status == "verified":
        return "Verified: your file matches the version used in the study."

    if check.status == "mismatch":
        return (
            "WARNING: this file does not match the version used in the study.\n"
            "The online source has most likely been updated since the study was run, "
            "so your results may differ slightly.\n"
            f"  expected SHA256: {check.expected_sha256}\n"
            f"  actual SHA256:   {check.actual_sha256}"
        )

    if check.status == "no_recorded_hash":
        return (
            "Note: the provenance record has no checksum, so the file's contents "
            "could not be checked."
        )

    return f"WARNING: could not complete verification ({check.status}).\n{check.message}"


__all__ = [
    "ProvenanceCheck",
    "format_provenance_result",
    "provenance_path_for",
    "verify_provenance",
    "write_provenance",
]
