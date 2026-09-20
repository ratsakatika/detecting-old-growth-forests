"""Tests for utils/tessera_version.py on a synthetic repository tree."""

import json
from pathlib import Path

import pytest

from utils import tessera_version as tv


def _sidecar(path: Path, label: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"dataset": label, "sha256": "0" * 64}), encoding="utf-8")


def _touch(path: Path, text: str = "x") -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return path


def _live_v11(repo: Path) -> dict[str, Path]:
    """A v1.1 installation: one of every kind of dependent artefact plus bystanders."""
    tile = _touch(repo / "data/raw/rasters/tessera/grid_24.65_45.75_2020.tiff", "v1.1 tile")
    _sidecar(
        repo / "data/raw/rasters/tessera/grid_24.65_45.75_2020.tiff.provenance.json",
        tv.dataset_label("v1.1", 2020),
    )
    _touch(repo / "data/raw/rasters/tessera/.gitkeep", "")
    return {
        "tile": tile,
        "arrays": _touch(repo / "data/cache/tessera/global_0.1_degree_representation/2020/g/g.npy"),
        "mosaic": _touch(repo / "data/processed/rasters/tessera_10m/tessera_2020_3035_10m.tif"),
        "stack": _touch(repo / "data/processed/rasters/stacks_10m/baseline_tessera_3035_10m.tif"),
        "patch": _touch(repo / "data/cache/patch_cache/baseline_tessera_p3.npy"),
        "run": _touch(repo / "results/main_nested_cv/20260607__xgboost__baseline_tessera/m.csv"),
        "matrix": _touch(repo / "results/performance_matrix/20260709/performance_matrix.csv"),
        "figure": _touch(repo / "figures/010_mapping/calibration.pdf"),
        # Bystanders that must never move.
        "other_stack": _touch(repo / "data/processed/rasters/stacks_10m/baseline_3035_10m.tif"),
        "other_run": _touch(repo / "results/main_nested_cv/20260607__xgboost__baseline/m.csv"),
        "other_patch": _touch(repo / "data/cache/patch_cache/baseline_alphaearth_p3.npy"),
        "labels_fig": _touch(repo / "figures/006_aoi_and_reference_label_statistics/s1.pdf"),
    }


def test_installed_version_reads_the_sidecars(tmp_path: Path) -> None:
    assert tv.installed_version(tmp_path) is None
    _live_v11(tmp_path)
    assert tv.installed_version(tmp_path) == "v1.1"
    _sidecar(
        tmp_path / "data/raw/rasters/tessera/grid_24.75_45.75_2020.tiff.provenance.json",
        tv.dataset_label("v2", 2020),
    )
    with pytest.raises(RuntimeError, match="mixes"):
        tv.installed_version(tmp_path)


def test_dataset_label_rejects_unknown_versions() -> None:
    assert tv.dataset_label("v2", 2020) == "TESSERA embeddings (v2, 2020)"
    with pytest.raises(ValueError, match="unknown"):
        tv.dataset_label("v3", 2020)


def test_switch_parks_the_installed_version_and_restores_the_requested_one(
    tmp_path: Path,
) -> None:
    live = _live_v11(tmp_path)
    slot_v2 = tv.archive_slot(tmp_path, "v2")
    v2_tile = _touch(slot_v2 / "data/raw/rasters/tessera/grid_24.65_45.75_2020.tiff", "v2 tile")
    v2_arrays = _touch(slot_v2 / "data/cache/tessera/tessera_metadata.json", "{}")

    summary = tv.switch_version(tmp_path, "v1.1", "v2")

    assert summary == {"parked": 9, "restored": 2}  # the sidecar moves with its tile
    slot_v11 = tv.archive_slot(tmp_path, "v1.1")
    for key in ("arrays", "mosaic", "stack", "patch", "run", "matrix", "figure"):
        assert not live[key].exists(), key
        assert (slot_v11 / live[key].relative_to(tmp_path)).is_file(), key
    assert (slot_v11 / live["tile"].relative_to(tmp_path)).read_text() == "v1.1 tile"
    assert list((slot_v11 / "data/raw/rasters/tessera").glob("*.provenance.json"))
    for key in ("other_stack", "other_run", "other_patch", "labels_fig"):
        assert live[key].is_file(), key
    assert (tmp_path / "data/raw/rasters/tessera/.gitkeep").is_file()
    assert (tmp_path / live["tile"].relative_to(tmp_path)).read_text() == "v2 tile"
    assert (tmp_path / v2_arrays.relative_to(slot_v2)).is_file()
    assert not slot_v2.exists()  # emptied and pruned
    assert not v2_tile.exists()

    # Back again: v2 is parked in full, v1.1 comes back byte for byte.
    assert tv.switch_version(tmp_path, "v2", "v1.1") == {"parked": 2, "restored": 9}
    assert live["tile"].read_text() == "v1.1 tile"
    assert live["patch"].is_file() and live["matrix"].is_file()
    assert (
        slot_v2 / "data/raw/rasters/tessera/grid_24.65_45.75_2020.tiff"
    ).read_text() == "v2 tile"
    assert not slot_v11.exists()


def test_switch_is_a_no_op_for_the_same_version_and_restores_without_parking(
    tmp_path: Path,
) -> None:
    assert tv.switch_version(tmp_path, "v1.1", "v1.1") == {"parked": 0, "restored": 0}
    _touch(tv.archive_slot(tmp_path, "v2") / "results/final/README.txt")
    assert tv.switch_version(tmp_path, None, "v2") == {"parked": 0, "restored": 1}
    assert (tmp_path / "results/final/README.txt").is_file()


def test_switch_aborts_before_moving_when_a_destination_exists(tmp_path: Path) -> None:
    live = _live_v11(tmp_path)
    _touch(tv.archive_slot(tmp_path, "v1.1") / "figures/010_mapping/calibration.pdf", "stale")
    with pytest.raises(FileExistsError, match="nothing was moved"):
        tv.switch_version(tmp_path, "v1.1", "v2")
    assert all(path.is_file() for path in live.values())
    with pytest.raises(ValueError, match="unknown"):
        tv.switch_version(tmp_path, "v1.1", "v9")
