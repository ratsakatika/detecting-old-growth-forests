"""Input-file manifest helpers for reproducibility.

Long-running scripts record an ``input_manifest.json`` capturing, for every
input file read, its SHA256 hash, size in bytes and modification time (see
AGENTS.md, "Reproducibility floor"). This lets a later run verify that its
inputs did not change. The helpers here build and write that manifest.

Hashing is streamed in chunks so arbitrarily large rasters never have to be
read into memory at once. Importing this module only binds names: it performs
no filesystem reads or writes, and it does not depend on ``archive/``.
"""

import hashlib
from collections.abc import Iterable
from dataclasses import asdict, dataclass
from pathlib import Path

from utils.io import write_json


@dataclass(frozen=True, slots=True)
class FileManifestEntry:
    """Recorded fingerprint of a single input file.

    Attributes:
        path: Manifest key for the file — a repo-root-relative POSIX path when a
            repository root was supplied, otherwise the file's POSIX path.
        sha256: Hexadecimal SHA256 digest of the file's contents.
        size_bytes: File size in bytes.
        modified_time: Last-modification time as a POSIX timestamp.
    """

    path: str
    sha256: str
    size_bytes: int
    modified_time: float


def sha256_file(path: str | Path, *, chunk_size: int = 1024 * 1024) -> str:
    """Compute the SHA256 digest of a file by streaming it in chunks.

    The file is read in blocks of ``chunk_size`` bytes so that large inputs are
    never loaded into memory whole.

    Args:
        path: File to hash.
        chunk_size: Number of bytes to read per block. Defaults to 1 MiB.

    Returns:
        The hexadecimal SHA256 digest.
    """
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _manifest_key(path: Path, repo_root: Path | None) -> str:
    """Return the manifest key for ``path``.

    Args:
        path: The input file path.
        repo_root: Repository root to make the key relative to, or ``None``.

    Returns:
        A repo-root-relative POSIX path when ``repo_root`` is given and ``path``
        lies beneath it; otherwise the POSIX form of ``path``.
    """
    if repo_root is not None:
        try:
            return path.resolve().relative_to(repo_root.resolve()).as_posix()
        except ValueError:
            pass
    return path.as_posix()


def build_input_manifest(
    paths: Iterable[str | Path], *, repo_root: str | Path | None = None
) -> dict[str, FileManifestEntry]:
    """Build a manifest of fingerprints for the given input files.

    Every path is validated before any hashing, so all missing paths (or all
    directories) are reported together rather than one at a time.

    Args:
        paths: Input files to fingerprint.
        repo_root: When given, manifest keys are repo-root-relative POSIX paths
            where possible; otherwise keys are the POSIX form of each path.

    Returns:
        A mapping from manifest key to :class:`FileManifestEntry`.

    Raises:
        FileNotFoundError: If any path does not exist; the message lists them all.
        IsADirectoryError: If any path is a directory; the message lists them all.
    """
    root = Path(repo_root) if repo_root is not None else None
    candidates = [Path(p) for p in paths]

    missing = sorted(str(p) for p in candidates if not p.exists())
    if missing:
        joined = "\n  ".join(missing)
        raise FileNotFoundError(f"Input files not found:\n  {joined}")

    directories = sorted(str(p) for p in candidates if p.is_dir())
    if directories:
        joined = "\n  ".join(directories)
        raise IsADirectoryError(f"Expected files but found directories:\n  {joined}")

    manifest: dict[str, FileManifestEntry] = {}
    for candidate in candidates:
        stat = candidate.stat()
        key = _manifest_key(candidate, root)
        manifest[key] = FileManifestEntry(
            path=key,
            sha256=sha256_file(candidate),
            size_bytes=int(stat.st_size),
            modified_time=float(stat.st_mtime),
        )
    return manifest


def write_input_manifest(
    paths: Iterable[str | Path],
    output_path: str | Path,
    *,
    repo_root: str | Path | None = None,
) -> Path:
    """Build an input manifest and write it as deterministic JSON.

    The manifest is serialised with :func:`utils.io.write_json`, which sorts keys
    and indents the output, so repeated writes of equivalent inputs produce
    byte-identical files.

    Args:
        paths: Input files to fingerprint.
        output_path: Destination JSON file.
        repo_root: Forwarded to :func:`build_input_manifest`.

    Returns:
        The path written to.
    """
    manifest = build_input_manifest(paths, repo_root=repo_root)
    serialisable = {key: asdict(entry) for key, entry in manifest.items()}
    return write_json(serialisable, output_path)


__all__ = [
    "FileManifestEntry",
    "build_input_manifest",
    "sha256_file",
    "write_input_manifest",
]
