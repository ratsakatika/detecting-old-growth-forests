"""Download CORINE CLC2018 for the Carpathian countries and build the forest layer.

Requests the CLC2018 vector product from the CLMS API, one NUTS-0 request per
country (a bounding-box request returns empty geometry for this product, CLMS
error DR-002), polls asynchronously, downloads the prepared GeoPackages and then
filters the forest classes (``utils.terminology.CORINE_FOREST_CLASSES``: 311
broadleaf, 312 coniferous, 313 mixed), clips them to the Carpathian massif and
saves one combined layer.

Coverage caveat: CLC2018 covers the EEA39 only — Ukraine has just a pilot
mapping around Kyiv, so the Ukrainian Carpathians are absent from this layer.
The AOA forest mask therefore uses ESA WorldCover instead
(``scripts/download_carpathians_worldcover.py``); this CORINE layer serves
EU-side cross-checks and forest-type stratification.

Resumable: task IDs are cached in ``data/cache/api/clms_tasks_carpathians.json``
so the script can be stopped and rerun; already-downloaded countries are kept; a
manually downloaded ``corine_clc2018_<country>.zip`` placed in the raw directory
is adopted. The EEA results endpoint rejects non-browser User-Agents, so file
downloads use a browser User-Agent (the results URL is pre-signed; no token).
Requires ``CLMS_SERVICE_KEY_PATH`` in ``.env``. Run::

    .venv/bin/python -m scripts.download_carpathians_corine
"""

import argparse
import json
import logging
import os
import time
import zipfile
from collections.abc import Sequence
from pathlib import Path
from typing import Final

import geopandas as gpd
import pandas as pd
import requests

from utils import terminology
from utils.carpathians import load_carpathian_massif
from utils.cli import positive_int
from utils.env_capture import write_environment_snapshot
from utils.logging import make_run_logger
from utils.manifest import write_input_manifest, write_json
from utils.paths import ProjectPaths, ensure_writable_dirs, get_project_paths
from utils.provenance import provenance_path_for, write_provenance
from utils.vector_io import audit_vector, repair_geometries

logger = logging.getLogger(__name__)

_STEPS: Final[tuple[str, ...]] = ("download", "forest")

# CLC2018 vector product identifiers from the CLMS dataset catalogue (as
# notebook 001's Romania download).
_DATASET_UID: Final[str] = "0407d497d3c44bcd93ce8fd5bf78596a"
_DOWNLOAD_INFO_ID: Final[str] = "1bda2fbd-3230-42ba-98cf-69c96ac063bc"
_API: Final[str] = "https://land.copernicus.eu/api"

# NUTS-0 codes of the countries the massif touches, with their raw filenames.
# Ukraine cannot be requested (no CLC2018 coverage).
_COUNTRIES: Final[dict[str, str]] = {
    "RO": "corine_clc2018_romania.gpkg",
    "SK": "corine_clc2018_slovakia.gpkg",
    "PL": "corine_clc2018_poland.gpkg",
    "HU": "corine_clc2018_hungary.gpkg",
    "CZ": "corine_clc2018_czechia.gpkg",
    "RS": "corine_clc2018_serbia.gpkg",
}
# CLC attribute column candidates, matched case-insensitively (the per-country
# exports use "Code_18").
_CODE_CANDIDATES: Final[tuple[str, ...]] = ("code_18", "code18", "clc_code", "clc", "code")

# The EEA FME results endpoint rejects non-browser User-Agents with 403; the
# results URL is pre-signed, so no bearer token is needed for the file itself.
_BROWSER_UA: Final[dict[str, str]] = {
    "User-Agent": "Mozilla/5.0 (X11; Linux x86_64; rv:127.0) Gecko/20100101 Firefox/127.0"
}


def _clms_auth(repo_root: Path) -> dict[str, str]:
    """Exchange the CLMS service key for a bearer-token header set."""
    import jwt  # PyJWT
    from dotenv import load_dotenv

    load_dotenv(repo_root / ".env")
    key_path = os.environ.get("CLMS_SERVICE_KEY_PATH")
    if not key_path:
        raise RuntimeError("Set CLMS_SERVICE_KEY_PATH in your .env file (see env.example).")
    key = json.loads(Path(key_path).read_text())
    now = int(time.time())
    grant = jwt.encode(
        {
            "iss": key["client_id"],
            "sub": key["user_id"],
            "aud": key["token_uri"],
            "iat": now,
            "exp": now + 3600,
        },
        key["private_key"],
        algorithm="RS256",
    )
    response = requests.post(
        key["token_uri"],
        data={"grant_type": "urn:ietf:params:oauth:grant-type:jwt-bearer", "assertion": grant},
        headers={"Accept": "application/json"},
        timeout=60,
    )
    response.raise_for_status()
    return {
        "Authorization": f"Bearer {response.json()['access_token']}",
        "Accept": "application/json",
    }


def _keep_gpkg(zip_path: Path, out_file: Path) -> None:
    """Extract the single GeoPackage from a CLMS archive, then delete the archive."""
    with zipfile.ZipFile(zip_path) as archive:
        member = next((m for m in archive.namelist() if m.lower().endswith(".gpkg")), None)
        if member is None:
            raise RuntimeError(f"No .gpkg in {zip_path.name}. Contents: {archive.namelist()}")
        with archive.open(member) as src, open(out_file, "wb") as dst:
            for chunk in iter(lambda: src.read(1 << 20), b""):
                dst.write(chunk)
    zip_path.unlink()


def _download_result(url: str, out_file: Path) -> None:
    """Stream a pre-signed CLMS result zip and keep its GeoPackage."""
    zip_path = out_file.with_suffix(".zip")
    with requests.get(url, headers=_BROWSER_UA, stream=True, timeout=600) as response:
        response.raise_for_status()
        with open(zip_path, "wb") as handle:
            for chunk in response.iter_content(chunk_size=1 << 20):
                handle.write(chunk)
    _keep_gpkg(zip_path, out_file)


def step_download(
    raw_dir: Path,
    task_cache: Path,
    *,
    repo_root: Path,
    poll_seconds: int,
    max_polls: int,
    log: logging.Logger,
) -> list[str]:
    """Fetch all country GeoPackages; returns the list of still-missing NUTS codes."""
    raw_dir.mkdir(parents=True, exist_ok=True)
    task_cache.parent.mkdir(parents=True, exist_ok=True)
    tasks: dict[str, object] = json.loads(task_cache.read_text()) if task_cache.exists() else {}

    pending = {n: raw_dir / f for n, f in _COUNTRIES.items() if not (raw_dir / f).exists()}
    for nuts, out_file in sorted(pending.items()):
        manual = out_file.with_suffix(".zip")
        if manual.exists():
            log.info("%s: adopting manual download %s", nuts, manual.name)
            _keep_gpkg(manual, out_file)
    pending = {n: f for n, f in pending.items() if not f.exists()}
    if not pending:
        log.info("All %d CORINE country GeoPackages present.", len(_COUNTRIES))
    else:
        auth = _clms_auth(repo_root)
        for nuts in sorted(pending):
            if nuts in tasks:
                continue
            body = {
                "Datasets": [
                    {
                        "DatasetID": _DATASET_UID,
                        "DatasetDownloadInformationID": _DOWNLOAD_INFO_ID,
                        "NUTS": nuts,
                        "OutputFormat": "GPKG",
                        "OutputGCS": terminology.CRS,
                    }
                ]
            }
            try:
                submit = requests.post(
                    f"{_API}/@datarequest_post",
                    headers={**auth, "Content-Type": "application/json"},
                    json=body,
                    timeout=60,
                )
                submit.raise_for_status()
                tasks[nuts] = submit.json()["TaskIds"][0]["TaskID"]
                task_cache.write_text(json.dumps(tasks, indent=2))
                log.info("%s: submitted task %s", nuts, tasks[nuts])
            except Exception as exc:
                log.error("%s: submit failed: %s", nuts, exc)

        for attempt in range(max_polls):
            still = {n: f for n, f in pending.items() if not f.exists() and n in tasks}
            if not still:
                break
            time.sleep(poll_seconds)
            try:
                status = requests.get(f"{_API}/@datarequest_search", headers=auth, timeout=60)
                if status.status_code == 401:
                    auth = _clms_auth(repo_root)
                    continue
                status.raise_for_status()
            except Exception as exc:
                log.warning("Poll error (will retry): %s", exc)
                continue
            entries = status.json()
            for nuts, out_file in sorted(still.items()):
                entry = entries.get(str(tasks[nuts]), {})
                entry_status = str(entry.get("Status", "")).lower()
                message = entry.get("Message") or entry.get("message") or ""
                if (
                    "reject" in entry_status
                    or "fail" in entry_status
                    or "code:" in str(message).lower()
                ):
                    log.error("%s: rejected: %s", nuts, str(message or entry)[:300])
                    tasks.pop(nuts, None)
                    task_cache.write_text(json.dumps(tasks, indent=2))
                    continue
                url = next(
                    (
                        value
                        for key, value in entry.items()
                        if isinstance(value, str)
                        and value.startswith("http")
                        and "download" in key.lower()
                    ),
                    None,
                )
                if url:
                    log.info("%s: ready, downloading", nuts)
                    try:
                        _download_result(url, out_file)
                        log.info("%s: kept %s", nuts, out_file.name)
                    except Exception as exc:
                        log.error("%s: download failed (will retry next poll): %s", nuts, exc)
            done = sum((raw_dir / f).exists() for f in _COUNTRIES.values())
            log.info("Poll %d: %d/%d countries present", attempt + 1, done, len(_COUNTRIES))

    for nuts, filename in sorted(_COUNTRIES.items()):
        out_file = raw_dir / filename
        if out_file.exists() and not provenance_path_for(out_file).exists():
            write_provenance(
                out_file,
                source=f"CLMS API: CORINE Land Cover 2018, NUTS {nuts}",
                extra={"dataset": f"CORINE Land Cover 2018 ({filename.split('_')[-1][:-5]})"},
            )
    return [n for n, f in _COUNTRIES.items() if not (raw_dir / f).exists()]


def step_forest(
    raw_dir: Path, out_path: Path, *, massif_path: Path | None, force: bool, log: logging.Logger
) -> Path:
    """Filter forest classes, clip to the massif and save the combined layer."""
    if out_path.exists() and not force:
        log.info("Forest layer already present: %s", audit_vector(out_path))
        return out_path

    massif = load_carpathian_massif(massif_path, logger=log)
    massif_geometry = massif.geometry.iloc[0]
    bbox = tuple(massif.total_bounds)
    forest_classes = terminology.CORINE_FOREST_CLASSES

    parts = []
    for nuts, filename in sorted(_COUNTRIES.items()):
        path = raw_dir / filename
        if not path.exists():
            log.warning("%s missing; forest layer will lack it", filename)
            continue
        frame = gpd.read_file(path, bbox=bbox)
        lower_map = {c.lower(): c for c in frame.columns}
        code_col = next((lower_map[c] for c in _CODE_CANDIDATES if c in lower_map), None)
        if code_col is None:
            raise RuntimeError(f"{path.name}: no CLC code column among {_CODE_CANDIDATES}.")
        codes = pd.to_numeric(frame[code_col], errors="coerce")
        frame = frame[codes.isin(forest_classes)].copy()
        frame["clc_code"] = codes[codes.isin(forest_classes)].astype(int)
        frame["forest_type"] = frame["clc_code"].map(forest_classes)
        frame["country"] = nuts
        frame = frame.to_crs(terminology.CRS)
        frame = gpd.clip(frame, massif_geometry)
        parts.append(frame[["clc_code", "forest_type", "country", "geometry"]])
        log.info("%s: %d forest polygons in massif", nuts, len(frame))
    if not parts:
        raise RuntimeError("No CORINE country files available; run the download step first.")

    forest = gpd.GeoDataFrame(pd.concat(parts, ignore_index=True), crs=terminology.CRS)
    forest = repair_geometries(forest, logger=log)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    forest.to_file(out_path, driver="GPKG")
    forest_km2 = float(forest.geometry.area.sum()) / 1e6
    massif_km2 = float(massif.geometry.area.iloc[0]) / 1e6
    log.info(
        "Wrote %s: %.0f km2 forest = %.0f%% of massif (Ukraine absent from CLC2018)",
        audit_vector(out_path),
        forest_km2,
        100 * forest_km2 / massif_km2,
    )
    return out_path


def _parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--steps",
        default=",".join(_STEPS),
        help=f"Comma-separated subset of {_STEPS} to run (default: all, in order).",
    )
    parser.add_argument(
        "--massif",
        type=Path,
        default=None,
        help="Carpathian massif GeoPackage (default: the processed stage-020 location).",
    )
    parser.add_argument(
        "--poll-seconds", type=positive_int, default=30, help="CLMS poll interval (default 30)."
    )
    parser.add_argument(
        "--max-polls", type=positive_int, default=400, help="Maximum CLMS polls (default 400)."
    )
    parser.add_argument(
        "--force", action="store_true", help="Rebuild the forest layer even if present."
    )
    return parser.parse_args(argv)


def run(args: argparse.Namespace, paths: ProjectPaths) -> int:
    """Execute the selected steps; returns a process exit code."""
    steps = tuple(s.strip() for s in args.steps.split(",") if s.strip())
    unknown = [s for s in steps if s not in _STEPS]
    if unknown:
        raise ValueError(f"Unknown steps {unknown}; choose from {_STEPS}.")

    raw_dir = paths.raw / "vectors" / "corine_land_cover"
    out_dir = paths.processed / "vectors" / "corine_land_cover"
    out_path = out_dir / "corine_forest_carpathians_3035.gpkg"
    task_cache = paths.cache / "api" / "clms_tasks_carpathians.json"
    global logger
    logger = make_run_logger(__name__, out_dir)

    missing: list[str] = []
    if "download" in steps:
        missing = step_download(
            raw_dir,
            task_cache,
            repo_root=paths.repo_root,
            poll_seconds=int(args.poll_seconds),
            max_polls=int(args.max_polls),
            log=logger,
        )
        if missing:
            logger.error("Countries still missing after polling: %s (rerun to resume)", missing)
    if "forest" in steps:
        step_forest(raw_dir, out_path, massif_path=args.massif, force=args.force, log=logger)

    write_input_manifest(
        [raw_dir / f for f in _COUNTRIES.values() if (raw_dir / f).exists()],
        out_dir / "input_manifest.json",
        repo_root=paths.repo_root,
    )
    write_environment_snapshot(out_dir / "env_snapshot.txt")
    write_json(
        {
            "steps": list(steps),
            "countries": list(_COUNTRIES),
            "countries_missing": missing,
            "forest_classes": {str(k): v for k, v in terminology.CORINE_FOREST_CLASSES.items()},
            "output": out_path.name,
            "note": "Ukraine has no CLC2018 coverage; the AOA forest mask uses WorldCover.",
        },
        out_dir / "run_metadata.json",
    )
    return 1 if missing else 0


def main(argv: Sequence[str] | None = None) -> int:
    """Command-line entry point."""
    args = _parse_args(argv)
    paths = get_project_paths()
    ensure_writable_dirs(paths)
    return run(args, paths)


if __name__ == "__main__":
    raise SystemExit(main())
