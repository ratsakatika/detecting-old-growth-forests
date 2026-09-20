"""Download TESSERA embedding tiles directly from the Source Cooperative mirror.

Fetches one year of TESSERA tiles by URL template, so the download does not
depend on the GeoTessera registry manifest (which can lag the object store), for
either published version:

* ``v1.1`` - the Cambridge build (``npy/v1.1-cam``), the tiles used in the study;
* ``v2`` - the 2B-L distilled student (``npy/v2-2B-L~beta1``).

Each tile's quantised array, per-pixel scales and landmask are stored in the
GeoTessera ``embeddings_dir`` layout under ``--output`` (by default
``data/cache/tessera``, which holds one version at a time: a cache whose
metadata names another version is refused), and with ``--export-tiff`` the tile
is written
as the GeoTIFF ``geotessera download --format tiff`` would produce: the int8
array multiplied by its scales, georeferenced from the landmask, tiled, LZW
compressed, with ``Tessera_Band_<i>`` descriptions and the dataset tags.
``--tiff-dir`` puts those GeoTIFFs flat in one directory as
``grid_<lon>_<lat>_<year>.tiff``, the study's raw layout; otherwise they sit
beside their arrays.

Tiles come from ``--tiles`` (one ``grid_<lon>_<lat>`` per line) or, by default,
from every 0.1-degree cell intersecting the Carpathian massif. Notebook 001 calls
:func:`fetch_tiles` for the study-area tiles.

Sizes at 10 m: about 115 MB per tile as arrays and about 345 MB as a GeoTIFF, so
the ~2,000 Carpathian tiles need roughly 240 GB raw and 700 GB as GeoTIFFs; run
``--dry-run`` first, and use ``--purge-npy`` to drop the arrays once a tile's
GeoTIFF is written.

Usage::

    python -m scripts.download_tessera_tiles --dataset-version v2 --dry-run
    python -m scripts.download_tessera_tiles --dataset-version v2 --tiles tiles.txt --export-tiff
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import shutil
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Final

import numpy as np
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from utils.carpathians import (
    carpathians_massif_path,
    degree_tiles_intersecting,
    load_carpathian_massif,
)
from utils.logging import make_run_logger
from utils.paths import get_project_paths
from utils.tessera_version import DATASET_PATHS

_MIRROR: Final[str] = "https://data.source.coop/tessera/tessera"
_LANDMASK_VERSION_PATH: Final[str] = "v1.1"  # the landmask tiles are shared by every version
_DEFAULT_VERSION: Final[str] = "v2"
_DEFAULT_YEAR: Final[int] = 2020
_DEFAULT_CACHE: Final[Path] = Path("data/cache/tessera")
_EMBEDDINGS_DIR: Final[str] = "global_0.1_degree_representation"
_LANDMASKS_DIR: Final[str] = "global_0.1_degree_tiff_all"
_TILE_STEP_DEG: Final[float] = 0.1
_USER_AGENT: Final[str] = "old-growth-forests-publication download_tessera_tiles"
_MIN_FREE_GB: Final[float] = 20.0
_CHUNK: Final[int] = 1 << 20
_N_BANDS: Final[int] = 128

_LOG = logging.getLogger("download_tessera_tiles")


def tile_name(lon: float, lat: float) -> str:
    """Return the registry name of the tile centred on ``(lon, lat)``."""
    return f"grid_{lon:.2f}_{lat:.2f}"


def tile_centre(name: str) -> tuple[float, float]:
    """Parse ``grid_<lon>_<lat>`` back into its centre coordinates."""
    _, lon, lat = name.split("_")
    return float(lon), float(lat)


def massif_tiles(repo_root: Path) -> list[str]:
    """Names of the 0.1-degree tiles whose cell intersects the Carpathian massif."""
    massif = load_carpathian_massif(carpathians_massif_path(repo_root), logger=_LOG).to_crs(4326)
    corners = degree_tiles_intersecting(massif, step_deg=_TILE_STEP_DEG, logger=_LOG)
    return sorted(tile_name(lon + 0.05, lat + 0.05) for lon, lat in corners)


def read_tile_list(path: Path) -> list[str]:
    """Read tile names from a text file, one ``grid_<lon>_<lat>`` per line."""
    names = [line.strip() for line in path.read_text(encoding="utf-8").splitlines()]
    names = [n for n in names if n and not n.startswith("#")]
    bad = [n for n in names if not n.startswith("grid_") or n.count("_") != 2]
    if bad:
        raise ValueError(f"{path}: {len(bad)} malformed tile name(s), e.g. {bad[0]!r}")
    return sorted(set(names))


def tile_urls(dataset_path: str, year: int, name: str) -> dict[str, str]:
    """URLs of the three objects a tile needs: embedding, scales and landmask."""
    return {
        "npy": f"{_MIRROR}/npy/{dataset_path}/{year}/{name}/{name}.npy",
        "scales": f"{_MIRROR}/npy/{dataset_path}/{year}/{name}/{name}_scales.npy",
        "landmask": f"{_MIRROR}/landmasks/{_LANDMASK_VERSION_PATH}/{name}.tiff",
    }


def tile_paths(output: Path, year: int, name: str, tiff_dir: Path | None = None) -> dict[str, Path]:
    """Local destinations of a tile's objects (``embeddings_dir`` layout; flat GeoTIFF optional)."""
    tile_dir = output / _EMBEDDINGS_DIR / str(year) / name
    return {
        "npy": tile_dir / f"{name}.npy",
        "scales": tile_dir / f"{name}_scales.npy",
        "landmask": output / _LANDMASKS_DIR / f"{name}.tiff",
        "tiff": (tiff_dir if tiff_dir is not None else tile_dir) / f"{name}_{year}.tiff",
    }


def make_session() -> requests.Session:
    """An HTTP session that retries transient mirror errors with back-off."""
    retry = Retry(
        total=6,
        connect=6,
        read=6,
        backoff_factor=1.5,
        status_forcelist=(429, 500, 502, 503, 504),
        allowed_methods=frozenset({"HEAD", "GET"}),
        raise_on_status=False,
    )
    session = requests.Session()
    session.mount("https://", HTTPAdapter(max_retries=retry, pool_maxsize=16))
    session.headers["User-Agent"] = _USER_AGENT
    return session


def remote_size(session: requests.Session, url: str) -> int | None:
    """Size of a mirror object in bytes, or ``None`` when it does not exist."""
    for attempt in range(4):
        try:
            response = session.head(url, timeout=(30, 60), allow_redirects=True)
        except requests.RequestException as error:
            if attempt == 3:
                raise
            _LOG.warning("HEAD %s failed (%s); retrying.", url, error)
            time.sleep(2.0 * (attempt + 1))
            continue
        if response.status_code == 404:
            return None
        response.raise_for_status()
        return int(response.headers["Content-Length"])
    return None


def download(session: requests.Session, url: str, dest: Path, expected: int) -> bool:
    """Stream ``url`` to ``dest`` via a ``.part`` file; skip when already complete.

    Returns ``True`` when a download happened and ``False`` when the existing
    file already had the expected size.
    """
    if dest.is_file() and dest.stat().st_size == expected:
        return False
    dest.parent.mkdir(parents=True, exist_ok=True)
    part = dest.with_name(dest.name + ".part")
    for attempt in range(4):
        try:
            with session.get(url, stream=True, timeout=(30, 120)) as response:
                response.raise_for_status()
                with open(part, "wb") as handle:
                    for chunk in response.iter_content(_CHUNK):
                        handle.write(chunk)
            if part.stat().st_size != expected:
                raise OSError(f"{dest.name}: got {part.stat().st_size} bytes, expected {expected}")
            part.replace(dest)
            return True
        except (requests.RequestException, OSError) as error:
            part.unlink(missing_ok=True)
            if attempt == 3:
                raise
            _LOG.warning("GET %s failed (%s); retrying.", url, error)
            time.sleep(3.0 * (attempt + 1))
    raise RuntimeError("unreachable")


def sha256_of(path: Path) -> str:
    """SHA-256 hex digest of a file, streamed."""
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(_CHUNK), b""):
            digest.update(chunk)
    return digest.hexdigest()


def dataset_tags(dataset_path: str) -> dict[str, str]:
    """The dataset tags GeoTessera writes into every exported GeoTIFF."""
    version_path, _, variant = dataset_path.partition("-")
    version = {"v1": "1.0", "v1.1": "1.1", "v2": "2.0"}.get(version_path, version_path)
    try:
        from geotessera import __version__ as geotessera_version
    except ImportError:  # the export needs no GeoTessera code, only its conventions
        geotessera_version = "not installed"
    return {
        "TESSERA_DATASET_VERSION": version,
        "TESSERA_DATASET_VERSION_PATH": version_path,
        "TESSERA_DATASET_VARIANT": variant,
        "TESSERA_DESCRIPTION": "GeoTessera satellite embedding tile",
        "GEOTESSERA_VERSION": geotessera_version,
    }


def dequantise(paths: dict[str, Path], name: str) -> tuple[np.ndarray, Any, Any]:
    """Return a cached tile as ``(embedding, crs, transform)``.

    The embedding is the int8 array multiplied by its per-pixel float32 scales,
    shape ``(height, width, 128)`` float32; the CRS and transform come from the
    landmask tile, which shares the embedding grid.
    """
    import rasterio

    quantised = np.load(paths["npy"], mmap_mode="r")
    scales = np.load(paths["scales"])
    if quantised.ndim != 3 or quantised.shape[2] != _N_BANDS:
        raise ValueError(f"{name}: unexpected embedding shape {quantised.shape}")
    if scales.shape != quantised.shape[:2]:
        raise ValueError(f"{name}: scales {scales.shape} do not match {quantised.shape[:2]}")
    with rasterio.open(paths["landmask"]) as landmask:
        crs, transform = landmask.crs, landmask.transform
        if (landmask.height, landmask.width) != quantised.shape[:2]:
            raise ValueError(f"{name}: landmask grid differs from the embedding grid")
    return quantised.astype(np.float32) * scales[:, :, None], crs, transform


def fetch_embedding(
    name: str, *, session: requests.Session, cache_dir: Path, dataset_path: str, year: int
) -> tuple[np.ndarray, Any, Any] | None:
    """Download a tile's objects when they are not cached and return it dequantised.

    Returns ``(embedding, crs, transform)`` as :func:`dequantise` does, or
    ``None`` when the tile is not on the mirror.
    """
    urls = tile_urls(dataset_path, year, name)
    paths = tile_paths(cache_dir, year, name)
    sizes = {key: remote_size(session, url) for key, url in urls.items()}
    if any(size is None for size in sizes.values()):
        return None
    for key in ("npy", "scales", "landmask"):
        download(session, urls[key], paths[key], sizes[key])  # type: ignore[arg-type]
    return dequantise(paths, name)


def export_tiff(paths: dict[str, Path], name: str, year: int, tags: dict[str, str]) -> None:
    """Write the tile's GeoTIFF exactly as ``geotessera download --format tiff`` does.

    The quantised int8 array is multiplied by its per-pixel float32 scales; the
    CRS and transform come from the landmask tile; the file is a tiled, LZW
    compressed float32 GeoTIFF with one ``Tessera_Band_<i>`` description per
    band and the dataset tags.
    """
    import rasterio

    embedding, crs, transform = dequantise(paths, name)
    lon, lat = tile_centre(name)
    height, width = embedding.shape[:2]
    part = paths["tiff"].with_name(paths["tiff"].name + ".part")
    with rasterio.open(
        part,
        "w",
        driver="GTiff",
        height=height,
        width=width,
        count=_N_BANDS,
        dtype="float32",
        crs=crs,
        transform=transform,
        compress="lzw",
        tiled=True,
        blockxsize=256,
        blockysize=256,
    ) as dst:
        for band in range(_N_BANDS):
            dst.write(embedding[:, :, band], band + 1)
            dst.set_band_description(band + 1, f"Tessera_Band_{band}")
        dst.update_tags(
            **tags,
            TESSERA_YEAR=str(year),
            TESSERA_TILE_LAT=f"{lat:.2f}",
            TESSERA_TILE_LON=f"{lon:.2f}",
        )
    part.replace(paths["tiff"])


def free_gb(path: Path) -> float:
    """Free disk space at ``path`` in gigabytes."""
    return shutil.disk_usage(path).free / 1e9


def process_tile(
    name: str,
    *,
    session: requests.Session,
    output: Path,
    dataset_path: str,
    year: int,
    export: bool,
    purge: bool,
    checksum: bool,
    tags: dict[str, str],
    tiff_dir: Path | None = None,
) -> dict[str, object]:
    """Download one tile's objects, export its GeoTIFF if asked, and describe the result."""
    urls = tile_urls(dataset_path, year, name)
    paths = tile_paths(output, year, name, tiff_dir)
    record: dict[str, object] = {"tile": name, "year": year, "dataset_path": dataset_path}
    if export and paths["tiff"].is_file() and (purge or paths["npy"].is_file()):
        record["status"] = "already exported"
        return record
    sizes = {key: remote_size(session, url) for key, url in urls.items()}
    missing = [key for key, size in sizes.items() if size is None]
    if missing:
        record["status"] = f"missing on mirror: {', '.join(missing)}"
        return record
    fetched = []
    for key in ("npy", "scales", "landmask"):
        if download(session, urls[key], paths[key], sizes[key]):  # type: ignore[arg-type]
            fetched.append(key)
    record["bytes"] = int(sum(sizes.values()))  # type: ignore[arg-type]
    record["urls"] = urls
    if checksum:
        record["sha256"] = {key: sha256_of(paths[key]) for key in ("npy", "scales")}
    if export:
        export_tiff(paths, name, year, tags)
        record["tiff_bytes"] = paths["tiff"].stat().st_size
        if purge:
            paths["npy"].unlink()
            paths["scales"].unlink()
    record["status"] = "downloaded" if fetched else "already present"
    return record


def dry_run(
    names: list[str],
    session: requests.Session,
    dataset_path: str,
    year: int,
    output: Path,
    workers: int,
) -> int:
    """Check every tile's objects on the mirror and report the total against free disk."""
    present: list[str] = []
    absent: list[str] = []
    total = 0

    def probe(name: str) -> tuple[str, dict[str, int | None]]:
        return name, {
            k: remote_size(session, u) for k, u in tile_urls(dataset_path, year, name).items()
        }

    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(probe, name) for name in names]
        for done, future in enumerate(as_completed(futures), 1):
            name, sizes = future.result()
            if any(size is None for size in sizes.values()):
                absent.append(name)
            else:
                present.append(name)
                total += sum(sizes.values())  # type: ignore[arg-type]
            if done % 200 == 0:
                _LOG.info("Probed %d/%d tiles.", done, len(names))
    raw_gb = total / 1e9
    tiff_gb = len(present) * 0.345
    _LOG.info(
        "%d of %d tiles are on the mirror for %s; %d missing%s.",
        len(present),
        len(names),
        year,
        len(absent),
        f" (e.g. {', '.join(absent[:5])})" if absent else "",
    )
    _LOG.info(
        "Raw download %.0f GB; GeoTIFF export would add about %.0f GB; free disk at %s: %.0f GB.",
        raw_gb,
        tiff_gb,
        output,
        free_gb(output if output.exists() else output.parent),
    )
    return 0 if present else 1


def fetch_tiles(
    names: list[str],
    *,
    dataset_version: str,
    year: int,
    cache_dir: Path,
    tiff_dir: Path | None = None,
    export: bool = True,
    purge: bool = False,
    checksum: bool = False,
    workers: int = 4,
    logger: logging.Logger | None = None,
) -> list[dict[str, object]]:
    """Download ``names`` for one dataset version and year, exporting GeoTIFFs if asked.

    Arrays and landmasks go under ``cache_dir`` in the ``embeddings_dir`` layout
    (one version per cache: a ``tessera_metadata.json`` naming another dataset
    path is refused, since both versions share tile names and byte sizes);
    GeoTIFFs go flat into ``tiff_dir`` when given, else beside their arrays. Tiles
    already present with the expected sizes are skipped, so an interrupted run
    resumes. Every tile gets a record in ``<cache_dir>/download_manifest.json``.

    Args:
        names: Tile names, ``grid_<lon>_<lat>``.
        dataset_version: ``"v1.1"`` or ``"v2"`` (see :data:`DATASET_PATHS`).
        year: Embedding year.
        cache_dir: Root for the arrays, landmasks, manifest and log.
        tiff_dir: Flat GeoTIFF directory (the study layout), or ``None``.
        export: Write GeoTIFFs.
        purge: Delete the arrays once a tile's GeoTIFF is written.
        checksum: Record SHA-256 digests of the arrays.
        workers: Concurrent tiles.
        logger: Logger; by default a run logger writing to stdout and
            ``<cache_dir>/logs/run.log``.

    Returns:
        One record per tile, with a ``status`` of ``downloaded``, ``already
        present``, ``already exported``, ``missing on mirror: ...`` or ``failed: ...``.
    """
    if dataset_version not in DATASET_PATHS:
        raise ValueError(
            f"dataset_version must be one of {sorted(DATASET_PATHS)}, got {dataset_version!r}"
        )
    if purge and not export:
        raise ValueError("purge requires export")
    dataset_path = DATASET_PATHS[dataset_version]
    metadata_path = cache_dir / "tessera_metadata.json"
    if metadata_path.is_file():
        cached = json.loads(metadata_path.read_text(encoding="utf-8")).get("dataset_path")
        if cached != dataset_path:
            raise ValueError(
                f"{cache_dir} holds {cached} arrays, not {dataset_path}; use another cache_dir"
            )
    cache_dir.mkdir(parents=True, exist_ok=True)
    log = logger or make_run_logger(_LOG.name, cache_dir)
    if tiff_dir is not None:
        tiff_dir.mkdir(parents=True, exist_ok=True)
    tags = dataset_tags(dataset_path)
    session = make_session()
    manifest_path = cache_dir / "download_manifest.json"
    records: dict[str, dict[str, object]] = {}
    if manifest_path.is_file():
        records = {
            str(r["tile"]): r
            for r in json.loads(manifest_path.read_text(encoding="utf-8"))["tiles"]
        }
    lock = threading.Lock()

    def save_manifest() -> None:
        payload = {
            "dataset_version": dataset_version,
            "dataset_path": dataset_path,
            "year": year,
            "source_url_prefix": f"{_MIRROR}/npy/{dataset_path}/",
            "updated_at": datetime.now(UTC).isoformat(),
            "tiles": sorted(records.values(), key=lambda r: str(r["tile"])),
        }
        tmp = manifest_path.with_suffix(".json.part")
        tmp.write_text(json.dumps(payload, indent=1), encoding="utf-8")
        tmp.replace(manifest_path)

    def work(name: str) -> dict[str, object]:
        if free_gb(cache_dir) < _MIN_FREE_GB:
            raise RuntimeError(f"less than {_MIN_FREE_GB:.0f} GB free under {cache_dir}; stopping")
        return process_tile(
            name,
            session=session,
            output=cache_dir,
            dataset_path=dataset_path,
            year=year,
            export=export,
            purge=purge,
            checksum=checksum,
            tags=tags,
            tiff_dir=tiff_dir,
        )

    log.info(
        "%d tiles for %s (%s) %d from %s.", len(names), dataset_version, dataset_path, year, _MIRROR
    )
    started = time.time()
    failures = 0
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(work, name): name for name in names}
        for done, future in enumerate(as_completed(futures), 1):
            name = futures[future]
            try:
                record = future.result()
            except Exception as error:  # one bad tile must not end the run
                failures += 1
                record = {"tile": name, "year": year, "status": f"failed: {error}"}
                log.error("%s failed: %s", name, error)
            with lock:  # a short-circuit status keeps the details recorded earlier
                records[name] = {**records.get(name, {}), **record}
                if done % 10 == 0 or done == len(names):
                    save_manifest()
            if done % 25 == 0 or done == len(names):
                log.info(
                    "%d/%d tiles (%d failed) in %.0f s; %.0f GB free.",
                    done,
                    len(names),
                    failures,
                    time.time() - started,
                    free_gb(cache_dir),
                )
    save_manifest()
    metadata_path.write_text(
        json.dumps(
            {
                "dataset_version": dataset_version,
                "dataset_path": dataset_path,
                "dataset_variant": tags["TESSERA_DATASET_VARIANT"],
                "dataset_version_path": tags["TESSERA_DATASET_VERSION_PATH"],
                "embeddings_subdir": _EMBEDDINGS_DIR,
                "format": "tiff" if export else "npy",
                "tiff_dir": str(tiff_dir) if tiff_dir is not None else None,
                "generated_at": datetime.now(UTC).isoformat(),
                "source_url_prefix": f"{_MIRROR}/npy/{dataset_path}/",
                "year": year,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    results = [records[name] for name in names if name in records]
    counts: dict[str, int] = {}
    for record in results:
        key = str(record["status"]).split(":")[0]
        counts[key] = counts.get(key, 0) + 1
    log.info("Done: %s", counts)
    return results


def main(argv: list[str] | None = None) -> int:
    """Command-line entry point."""
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--dataset-version",
        default=_DEFAULT_VERSION,
        choices=sorted(DATASET_PATHS),
        help="TESSERA version (default %(default)s).",
    )
    parser.add_argument("--year", type=int, default=_DEFAULT_YEAR)
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Root for the arrays and landmasks (default data/cache/tessera).",
    )
    parser.add_argument(
        "--tiff-dir",
        type=Path,
        default=None,
        help="Write the GeoTIFFs flat into this directory (implies --export-tiff).",
    )
    parser.add_argument(
        "--tiles",
        type=Path,
        default=None,
        help="Text file of tile names; default: every cell over the Carpathian massif.",
    )
    parser.add_argument("--limit", type=int, default=None, help="Process only the first N tiles.")
    parser.add_argument("--export-tiff", action="store_true", help="Write each tile's GeoTIFF.")
    parser.add_argument(
        "--purge-npy",
        action="store_true",
        help="Delete the arrays once the GeoTIFF is written (needs --export-tiff).",
    )
    parser.add_argument(
        "--checksum", action="store_true", help="Record SHA-256 digests of the arrays."
    )
    parser.add_argument(
        "--workers", type=int, default=4, help="Concurrent tiles (default %(default)s)."
    )
    parser.add_argument("--dry-run", action="store_true", help="Check availability and sizes only.")
    args = parser.parse_args(argv)
    export = args.export_tiff or args.tiff_dir is not None
    if args.purge_npy and not export:
        parser.error("--purge-npy requires --export-tiff or --tiff-dir")

    project = get_project_paths()
    dataset_path = DATASET_PATHS[args.dataset_version]
    cache_dir = args.output or project.repo_root / _DEFAULT_CACHE
    if not cache_dir.is_absolute():
        cache_dir = project.repo_root / cache_dir
    tiff_dir = args.tiff_dir
    if tiff_dir is not None and not tiff_dir.is_absolute():
        tiff_dir = project.repo_root / tiff_dir

    names = read_tile_list(args.tiles) if args.tiles else massif_tiles(project.repo_root)
    if args.limit:
        names = names[: args.limit]
    if args.dry_run:
        logging.basicConfig(
            level=logging.INFO, format="%(asctime)s %(levelname)-7s %(name)s %(message)s"
        )
        _LOG.info(
            "%d tiles for %s (%s) %d.", len(names), args.dataset_version, dataset_path, args.year
        )
        return dry_run(names, make_session(), dataset_path, args.year, cache_dir, args.workers)

    results = fetch_tiles(
        names,
        dataset_version=args.dataset_version,
        year=args.year,
        cache_dir=cache_dir,
        tiff_dir=tiff_dir,
        export=export,
        purge=args.purge_npy,
        checksum=args.checksum,
        workers=args.workers,
    )
    return 1 if any(str(r["status"]).startswith(("failed", "missing")) for r in results) else 0


if __name__ == "__main__":
    raise SystemExit(main())
