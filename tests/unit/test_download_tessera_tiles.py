"""Functional tests for scripts/download_tessera_tiles.py.

A fake HTTP session serves a synthetic tile (int8 array, per-pixel scales and a
landmask GeoTIFF) so the download, GeoTIFF export, resume and manifest logic run
end to end without the network.
"""

import json
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import pytest
import rasterio
from rasterio.transform import from_origin

from scripts import download_tessera_tiles as dt

_NAME = "grid_24.65_45.75"
_YEAR = 2020
_SHAPE = (6, 5)


def _tile_bytes(tmp_path: Path) -> dict[str, bytes]:
    """Serialised objects of one synthetic tile plus its expected pixel values."""
    rng = np.random.default_rng(0)
    quantised = rng.integers(-127, 128, size=(*_SHAPE, dt._N_BANDS), dtype=np.int8)
    scales = rng.uniform(0.01, 0.05, size=_SHAPE).astype(np.float32)
    tmp_path.mkdir(exist_ok=True)
    npy, scl, lmk = tmp_path / "q.npy", tmp_path / "s.npy", tmp_path / "l.tiff"
    np.save(npy, quantised)
    np.save(scl, scales)
    with rasterio.open(
        lmk,
        "w",
        driver="GTiff",
        height=_SHAPE[0],
        width=_SHAPE[1],
        count=1,
        dtype="uint8",
        crs="EPSG:32635",
        transform=from_origin(470_000.0, 5_070_000.0, 10.0, 10.0),
    ) as dst:
        dst.write(np.ones(_SHAPE, dtype=np.uint8), 1)
    expected = (quantised.astype(np.float32) * scales[:, :, None]).transpose(2, 0, 1)
    np.save(tmp_path / "expected.npy", expected)
    return {"npy": npy.read_bytes(), "scales": scl.read_bytes(), "landmask": lmk.read_bytes()}


class _Response:
    def __init__(self, status: int, body: bytes = b"") -> None:
        self.status_code = status
        self.headers = {"Content-Length": str(len(body))}
        self._body = body

    def raise_for_status(self) -> None:
        if self.status_code >= 400:
            raise dt.requests.HTTPError(f"status {self.status_code}")

    def iter_content(self, size: int):
        for start in range(0, len(self._body), size):
            yield self._body[start : start + size]


class _FakeSession:
    """Serves ``objects`` (url -> bytes); everything else is a 404."""

    def __init__(self, objects: dict[str, bytes]) -> None:
        self.objects = objects
        self.gets: list[str] = []

    def head(self, url: str, timeout: object, allow_redirects: bool) -> _Response:
        return _Response(200, self.objects[url]) if url in self.objects else _Response(404)

    @contextmanager
    def get(self, url: str, stream: bool, timeout: object):
        self.gets.append(url)
        yield _Response(200, self.objects[url]) if url in self.objects else _Response(404)


@pytest.fixture
def served(tmp_path: Path) -> tuple[_FakeSession, np.ndarray]:
    objects = _tile_bytes(tmp_path / "src")
    urls = dt.tile_urls(dt.DATASET_PATHS["v2"], _YEAR, _NAME)
    session = _FakeSession({urls[key]: objects[key] for key in objects})
    return session, np.load(tmp_path / "src" / "expected.npy")


def test_tile_name_round_trip() -> None:
    assert dt.tile_name(24.65, 45.75) == _NAME
    assert dt.tile_centre(_NAME) == (24.65, 45.75)


def test_read_tile_list_sorts_dedups_and_rejects_malformed(tmp_path: Path) -> None:
    listing = tmp_path / "tiles.txt"
    listing.write_text("# comment\ngrid_25.05_45.55\n\ngrid_24.65_45.75\ngrid_24.65_45.75\n")
    assert dt.read_tile_list(listing) == ["grid_24.65_45.75", "grid_25.05_45.55"]
    listing.write_text("grid_24.65_45.75\ntile_1\n")
    with pytest.raises(ValueError, match="malformed"):
        dt.read_tile_list(listing)


def test_urls_and_paths_follow_the_mirror_and_embeddings_dir_layout(tmp_path: Path) -> None:
    urls = dt.tile_urls("v2-2B-L~beta1", _YEAR, _NAME)
    assert urls["npy"].endswith(f"/npy/v2-2B-L~beta1/{_YEAR}/{_NAME}/{_NAME}.npy")
    assert urls["scales"].endswith(f"/{_NAME}/{_NAME}_scales.npy")
    assert urls["landmask"].endswith(f"/landmasks/v1.1/{_NAME}.tiff")
    nested = dt.tile_paths(tmp_path, _YEAR, _NAME)
    tile_dir = tmp_path / dt._EMBEDDINGS_DIR / str(_YEAR) / _NAME
    assert nested["npy"] == tile_dir / f"{_NAME}.npy"
    assert nested["landmask"] == tmp_path / dt._LANDMASKS_DIR / f"{_NAME}.tiff"
    assert nested["tiff"] == tile_dir / f"{_NAME}_{_YEAR}.tiff"
    flat = dt.tile_paths(tmp_path, _YEAR, _NAME, tmp_path / "flat")
    assert flat["tiff"] == tmp_path / "flat" / f"{_NAME}_{_YEAR}.tiff"


def test_dataset_tags_split_version_and_variant() -> None:
    tags = dt.dataset_tags("v1.1-cam")
    assert (tags["TESSERA_DATASET_VERSION"], tags["TESSERA_DATASET_VARIANT"]) == ("1.1", "cam")
    assert dt.dataset_tags("v2-2B-L~beta1")["TESSERA_DATASET_VERSION_PATH"] == "v2"


def test_fetch_tiles_downloads_exports_and_resumes(
    tmp_path: Path, served: tuple[_FakeSession, np.ndarray]
) -> None:
    session, expected = served
    cache, flat = tmp_path / "cache", tmp_path / "flat"

    def fetch(names: list[str], checksum: bool) -> list[dict[str, object]]:
        return dt.fetch_tiles(
            names,
            dataset_version="v2",
            year=_YEAR,
            cache_dir=cache,
            tiff_dir=flat,
            checksum=checksum,
            workers=1,
        )

    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(dt, "make_session", lambda: session)
        patch.setattr(dt, "free_gb", lambda path: 100.0)
        records = fetch([_NAME, "grid_0.05_0.05"], checksum=True)
        again = fetch([_NAME], checksum=False)

    assert [r["status"] for r in records] == [
        "downloaded",
        "missing on mirror: npy, scales, landmask",
    ]
    assert again[0]["status"] == "already exported"
    assert len(session.gets) == 3  # the second call downloaded nothing

    tiff = flat / f"{_NAME}_{_YEAR}.tiff"
    with rasterio.open(tiff) as src:
        assert src.count == dt._N_BANDS
        assert src.crs.to_epsg() == 32635
        assert src.descriptions[:2] == ("Tessera_Band_0", "Tessera_Band_1")
        assert src.tags()["TESSERA_DATASET_VERSION_PATH"] == "v2"
        assert src.tags()["TESSERA_TILE_LON"] == "24.65"
        assert src.profile["compress"] == "lzw" and src.profile["tiled"]
        np.testing.assert_array_equal(src.read(), expected)

    manifest = json.loads((cache / "download_manifest.json").read_text(encoding="utf-8"))
    assert manifest["dataset_path"] == "v2-2B-L~beta1"
    by_tile = {r["tile"]: r for r in manifest["tiles"]}
    assert set(by_tile[_NAME]["sha256"]) == {"npy", "scales"}
    assert by_tile[_NAME]["tiff_bytes"] == tiff.stat().st_size
    metadata = json.loads((cache / "tessera_metadata.json").read_text(encoding="utf-8"))
    assert metadata["dataset_variant"] == "2B-L~beta1"
    assert (cache / "logs" / "run.log").is_file()


def test_fetch_tiles_rejects_bad_arguments(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="dataset_version"):
        dt.fetch_tiles([_NAME], dataset_version="v3", year=_YEAR, cache_dir=tmp_path)
    (tmp_path / "tessera_metadata.json").write_text(
        '{"dataset_path": "v1.1-cam"}', encoding="utf-8"
    )
    with pytest.raises(ValueError, match="holds v1.1-cam arrays"):
        dt.fetch_tiles([_NAME], dataset_version="v2", year=_YEAR, cache_dir=tmp_path)
    with pytest.raises(ValueError, match="purge"):
        dt.fetch_tiles(
            [_NAME], dataset_version="v2", year=_YEAR, cache_dir=tmp_path, export=False, purge=True
        )


def test_main_purges_arrays_after_export(
    tmp_path: Path, served: tuple[_FakeSession, np.ndarray]
) -> None:
    session, _ = served
    listing = tmp_path / "tiles.txt"
    listing.write_text(f"{_NAME}\n")
    cache = tmp_path / "cache"
    argv = ["--dataset-version", "v2", "--tiles", str(listing), "--output", str(cache)]

    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(dt, "make_session", lambda: session)
        patch.setattr(dt, "free_gb", lambda path: 100.0)
        assert dt.main([*argv, "--export-tiff", "--purge-npy", "--workers", "1"]) == 0

    paths = dt.tile_paths(cache, _YEAR, _NAME)
    assert paths["tiff"].is_file()
    assert not paths["npy"].exists() and not paths["scales"].exists()
    assert paths["landmask"].is_file()


def test_fetch_embedding_returns_the_dequantised_tile_or_none(
    tmp_path: Path, served: tuple[_FakeSession, np.ndarray]
) -> None:
    session, expected = served
    fetched = dt.fetch_embedding(
        _NAME, session=session, cache_dir=tmp_path, dataset_path="v2-2B-L~beta1", year=_YEAR
    )
    assert fetched is not None
    embedding, crs, transform = fetched
    assert embedding.shape == (*_SHAPE, dt._N_BANDS) and embedding.dtype == np.float32
    np.testing.assert_array_equal(embedding.transpose(2, 0, 1), expected)
    assert crs.to_epsg() == 32635 and transform.a == 10.0
    assert dt.tile_paths(tmp_path, _YEAR, _NAME)["npy"].is_file()
    assert (
        dt.fetch_embedding(
            "grid_0.05_0.05",
            session=session,
            cache_dir=tmp_path,
            dataset_path="v2-2B-L~beta1",
            year=_YEAR,
        )
        is None
    )
