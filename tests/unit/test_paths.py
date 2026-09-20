"""Unit tests for the canonical project paths in utils.paths."""

import dataclasses
import importlib
from pathlib import Path

import pytest

import utils.paths
from utils.paths import (
    AOI_RELATIVE,
    REFERENCE_GRID_RELATIVE,
    ProjectPaths,
    ensure_writable_dirs,
    find_repo_root,
    get_project_paths,
    validate_project_paths,
)


def _make_repo(root: Path) -> Path:
    """Create the marker files that identify a repository root."""
    (root / "AGENTS.md").write_text("marker\n", encoding="utf-8")
    (root / "pyproject.toml").write_text("[tool.ruff]\n", encoding="utf-8")
    return root


def test_find_repo_root_finds_marked_repo(tmp_path: Path) -> None:
    repo = _make_repo(tmp_path)
    nested = repo / "utils" / "deep"
    nested.mkdir(parents=True)

    assert find_repo_root(nested) == repo.resolve()
    # Starting from a file inside the repo resolves to the same root.
    marker_file = repo / "AGENTS.md"
    assert find_repo_root(marker_file) == repo.resolve()


def test_find_repo_root_raises_outside_repo(tmp_path: Path) -> None:
    # tmp_path carries no markers and lives under /tmp, so no ancestor qualifies.
    with pytest.raises(FileNotFoundError):
        find_repo_root(tmp_path)


def test_get_project_paths_returns_canonical_reference_grid(tmp_path: Path) -> None:
    paths = get_project_paths(tmp_path)

    assert isinstance(paths, ProjectPaths)
    assert paths.repo_root == tmp_path
    assert paths.reference_grid == tmp_path / REFERENCE_GRID_RELATIVE
    assert paths.reference_grid == (
        tmp_path / "data/processed/rasters/worldcover_composites_10m/"
        "stacked_merged_clipped_to_aoi_3035_10m.tif"
    )
    assert paths.aoi == tmp_path / AOI_RELATIVE
    # A few derived directories should sit under the data tree / root.
    assert paths.raw == tmp_path / "data" / "raw"
    assert paths.processed == tmp_path / "data" / "processed"
    assert paths.labels == paths.repo_root / "data" / "processed" / "vectors" / "labels"
    assert paths.figures == tmp_path / "figures"


def test_get_project_paths_accepts_string_root(tmp_path: Path) -> None:
    paths = get_project_paths(str(tmp_path))
    assert paths.repo_root == tmp_path


def test_importing_paths_succeeds_without_agents_md(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Reloading re-executes the module body. If discovery ran at import time it
    # would raise here, since tmp_path has no AGENTS.md in any ancestor.
    monkeypatch.chdir(tmp_path)
    module = importlib.reload(utils.paths)
    assert hasattr(module, "get_project_paths")


def test_validate_project_paths_reports_missing_paths_clearly(tmp_path: Path) -> None:
    repo = _make_repo(tmp_path)
    paths = get_project_paths(repo)

    with pytest.raises(FileNotFoundError) as excinfo:
        validate_project_paths(paths)

    message = str(excinfo.value)
    # The unmet directories should be named so the failure is actionable.
    assert "data" in message
    assert "results" in message


def test_validate_project_paths_passes_for_complete_layout(tmp_path: Path) -> None:
    repo = _make_repo(tmp_path)
    paths = get_project_paths(repo)
    for name in (
        "data",
        "raw",
        "processed",
        "cache",
        "labels",
        "figures",
        "results",
        "tests",
        "docs",
        "scripts",
        "notebooks",
        "utils",
    ):
        getattr(paths, name).mkdir(parents=True, exist_ok=True)

    # Should not raise.
    validate_project_paths(paths)


def test_validate_project_paths_rejects_wrong_reference_grid(tmp_path: Path) -> None:
    repo = _make_repo(tmp_path)
    paths = get_project_paths(repo)
    tampered = dataclasses.replace(paths, reference_grid=repo / "data" / "wrong.tif")

    with pytest.raises(ValueError, match="reference_grid"):
        validate_project_paths(tampered)


def test_ensure_writable_dirs_creates_only_allowed_directories(tmp_path: Path) -> None:
    paths = get_project_paths(tmp_path)

    ensure_writable_dirs(paths)

    for created in (paths.processed, paths.cache, paths.labels, paths.figures, paths.results):
        assert created.is_dir()

    # Read-only inputs must never be created.
    assert not paths.raw.exists()


def test_project_paths_has_no_archive_attribute(tmp_path: Path) -> None:
    # archive/ is agent-only reference material, never a runtime path field
    # (see AGENTS.md, "Agent-only reference material").
    paths = get_project_paths(tmp_path)
    assert not hasattr(paths, "archive")
    assert "archive" not in {field.name for field in dataclasses.fields(paths)}


def test_project_paths_is_frozen(tmp_path: Path) -> None:
    paths = get_project_paths(tmp_path)
    with pytest.raises(AttributeError):
        paths.repo_root = tmp_path  # type: ignore[misc]
