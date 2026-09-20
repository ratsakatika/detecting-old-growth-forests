"""Meyer-Pebesma area of applicability of the deployed model over the Carpathians.

Computes the dissimilarity index (DI), the AOA threshold and the Carpathian AOA
rasters for the deployed ``xgboost__baseline_tessera`` model, per the resolved
resolved decisions (recorded in the area-of-applicability notebook):

1. Training points are the exact final-fit sample (``run_final_inference``'s
   selection: valid labelled pixels, at most ``N_PER_PARCEL`` per parcel, seed
   ``[SEED, 0]``).
2. The final model is refitted once (GPU) from its stored hyper-parameters
   (never serialised) and its gain importances weight the feature space.
3. Features are standardised by training mean/sd and importance-weighted; the
   DI denominator is the mean pairwise training distance (random-subset
   estimate with a subset-size sensitivity table).
4. Cross-validated training DIs use plain fold exclusion over the six spatial
   blocks; the AOA threshold is the paper's outlier-removed maximum
   (upper whisker, Q3 + 1.5 x IQR), with q95 reported as sensitivity only.
5. The DI, binary AOA and nearest-training-label rasters are computed
   block-wise over the Carpathian grid from the stage-020 predictor rasters.
6. Reporting: WorldCover forest in/out-of-AOA summary, and the windowed
   PR-AUC (primary) / Brier (sensitivity) DI-performance curve from the
   pooled out-of-fold predictions.

Outputs land in ``results/area_of_applicability/baseline_tessera/`` (smoke runs
go to ``results/preflight/``). Resumable at the mapping stage via a per-window
checkpoint. Ported ideas from the archive prototype
``archive/scripts/034_carpathian_tessera_zarr_aoa.py`` (blockwise mapping,
denominator diagnostics, forest summary); features, year, weights, folds and
threshold follow the publication pipeline instead. Run::

    screen -S carp_aoa
    .venv/bin/python -m scripts.run_area_of_applicability --grid-res 100
"""

import argparse
import json
import logging
import time
from collections.abc import Sequence
from pathlib import Path
from typing import Any, Final

import numpy as np
import pandas as pd
import pyarrow.dataset as pads
import rasterio
from rasterio.windows import Window
from scipy.ndimage import distance_transform_edt
from sklearn.isotonic import IsotonicRegression

from scripts.run_sampling_sensitivity import _load_pixel_data
from utils import raster_io, terminology
from utils.aoa import (
    AOA_INSIDE,
    AOA_OUTSIDE,
    AOA_UINT8_NODATA,
    cv_nearest_distances,
    di_threshold,
    fit_feature_scaling,
    mean_pairwise_distance,
    nearest_distances,
    performance_by_di_bins,
    performance_by_di_window,
)
from utils.carpathians import carpathian_grid, carpathians_massif_path, load_carpathian_massif
from utils.cli import positive_int
from utils.cv import sample_train_rows
from utils.env_capture import write_environment_snapshot
from utils.io import (
    AOA_TRAINING_DI_SCHEMA,
    FEATURE_IMPORTANCE_SCHEMA,
    ColumnSpec,
    Schema,
    validate_schema,
    write_csv,
    write_json,
    write_parquet,
)
from utils.logging import make_run_logger
from utils.manifest import write_input_manifest
from utils.models.xgboost import feature_importance, make_xgb_classifier
from utils.paths import ProjectPaths, ensure_writable_dirs, get_project_paths
from utils.sampling import build_sample_weights

logger = logging.getLogger(__name__)

_ARCHITECTURE: Final[str] = "xgboost"
_FEATURE_SET: Final[str] = "baseline_tessera"
_RESULTS_ROLE: Final[str] = "area_of_applicability"
_MAPPING_ROLE: Final[str] = "mapping"
_SOURCE_ROLE: Final[str] = "main_nested_cv"
_BUFFER_ROLE: Final[str] = "spatial_buffer"
_TERRAIN_STEMS: Final[tuple[str, ...]] = ("elevation", "slope_deg", "heat_load_index")
_CHECKPOINT: Final[str] = "map_windows_done.json"
# Distant-deployment slice of the spatial-buffer gap sweep (matches the buffered
# performance matrix) and the largest gap used for the distance decay. 10 km is
# where residual spatial autocorrelation is spent: residual variogram effective
# ranges are ~5-7 km, the buffered gap sweep plateaus at ~10 km, and the
# xy-coordinates control reaches its no-skill floor by ~10 km. Beyond that the
# buffer also starts gutting the small AOI training set (capacity artefact).
_BUFFER_GAP_KM: Final[int] = 10
_DECAY_MAX_GAP_KM: Final[int] = 10

DENOMINATOR_SCHEMA: Final[Schema] = {
    "sample_size": ColumnSpec("int64", nullable=False),
    "mean_distance": ColumnSpec("float64", nullable=False),
}
CURVE_SCHEMA: Final[Schema] = {
    "di_lo": ColumnSpec("float64", nullable=False),
    "di_mid": ColumnSpec("float64", nullable=False),
    "di_hi": ColumnSpec("float64", nullable=False),
    "n": ColumnSpec("int64", nullable=False),
    "prevalence": ColumnSpec("float64", nullable=False),
    "pr_auc": ColumnSpec("float64"),
    "pr_auc_lift": ColumnSpec("float64"),
    "roc_auc": ColumnSpec("float64"),
    "f1": ColumnSpec("float64"),
    "brier": ColumnSpec("float64", nullable=False),
    "pr_auc_fit": ColumnSpec("float64"),
    "roc_auc_fit": ColumnSpec("float64"),
    "f1_fit": ColumnSpec("float64"),
    "brier_fit": ColumnSpec("float64", nullable=False),
}
DI_TABLE_SCHEMA: Final[Schema] = {
    "di_bin": ColumnSpec(("object", "string"), nullable=False),
    "di_lo": ColumnSpec("float64", nullable=False),
    "di_hi": ColumnSpec("float64", nullable=False),
    "n": ColumnSpec("int64", nullable=False),
    "prevalence": ColumnSpec("float64", nullable=False),
    "pr_auc": ColumnSpec("float64"),
    "pr_auc_lift": ColumnSpec("float64"),
    "roc_auc": ColumnSpec("float64"),
    "f1": ColumnSpec("float64"),
    "brier": ColumnSpec("float64", nullable=False),
}
DECAY_SCHEMA: Final[Schema] = {
    "gap_km": ColumnSpec("int64", nullable=False),
    "pr_auc_buffered": ColumnSpec("float64", nullable=False),
    "pr_auc_control": ColumnSpec("float64", nullable=False),
    "pr_auc_proximity_adjusted": ColumnSpec("float64", nullable=False),
    "pr_auc_adjusted_fit": ColumnSpec("float64", nullable=False),
}
SENSITIVITY_SCHEMA: Final[Schema] = {
    "forest_mask": ColumnSpec(("object", "string"), nullable=False),
    "threshold_rule": ColumnSpec(("object", "string"), nullable=False),
    "threshold": ColumnSpec("float64", nullable=False),
    "forest_Mha": ColumnSpec("float64", nullable=False),
    "pct_forest_inside": ColumnSpec("float64", nullable=False),
}
FOREST_SUMMARY_SCHEMA: Final[Schema] = {
    "stratum": ColumnSpec(("object", "string"), nullable=False),
    "valid_cells": ColumnSpec("int64", nullable=False),
    "valid_area_ha": ColumnSpec("float64", nullable=False),
    "inside_aoa_area_ha": ColumnSpec("float64", nullable=False),
    "pct_valid_area_inside_aoa": ColumnSpec("float64", nullable=False),
    "forest_area_ha": ColumnSpec("float64", nullable=False),
    "forest_inside_aoa_area_ha": ColumnSpec("float64", nullable=False),
    "forest_outside_aoa_area_ha": ColumnSpec("float64", nullable=False),
    "pct_forest_area_inside_aoa": ColumnSpec("float64", nullable=False),
}


def _latest_run(root: Path, *, required: str) -> Path:
    """Latest ``*__xgboost__baseline_tessera`` run directory containing a file."""
    pattern = f"*__{_ARCHITECTURE}__{_FEATURE_SET}"
    for candidate in sorted(root.glob(pattern), reverse=True):
        if (candidate / required).exists():
            return candidate
    raise FileNotFoundError(f"No {pattern} run with {required} under {root}.")


def _carpathian_sources(paths: ProjectPaths, res: int) -> dict[str, Path]:
    root = paths.processed / "rasters" / "carpathians"
    sources = {
        stem: root / f"baseline_{res}m" / f"{stem}_3035_{res}m.tif" for stem in _TERRAIN_STEMS
    }
    sources["distance_to_roads"] = root / f"baseline_{res}m" / f"distance_to_roads_3035_{res}m.tif"
    # Prefer the consolidated single-file mosaic: windowed reads through the
    # ~2000-tile VRT decode thousands of tiny blocks per window and are slow.
    tessera_tif = root / f"tessera_{res}m" / f"tessera_2020_3035_{res}m.tif"
    sources["tessera"] = (
        tessera_tif
        if tessera_tif.exists()
        else root / f"tessera_{res}m" / f"tessera_2020_3035_{res}m.vrt"
    )
    sources["worldcover"] = (
        root / f"worldcover_{res}m" / f"worldcover_tree_fraction_3035_{res}m.tif"
    )
    missing = [str(p) for p in sources.values() if not p.exists()]
    if missing:
        raise FileNotFoundError(
            "Carpathian predictor rasters missing; run scripts/launch_carpathian_downloads.sh "
            f"first. Missing: {missing}"
        )
    return sources


def _sample_training_features_from_map(
    paths: ProjectPaths, res: int, massif_path: Path | None, xs: np.ndarray, ys: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Sample the 134 map-side features at the training coordinates.

    The DI is computed in one feature space on both sides: the training points'
    features are re-sampled from the same Carpathian rasters the map uses (the
    TESSERA version notebook 001 installed, at the analysis resolution), so the
    map and the training support share the product and the resampling. The
    model refit — and hence the gain weights — stays on the 10 m training
    matrix.

    Returns:
        ``(features, valid_mask)`` where invalid rows fall outside the grid or
        touch NoData in any band.
    """
    sources = _carpathian_sources(paths, res)
    massif = load_carpathian_massif(massif_path, logger=logger)
    grid = carpathian_grid(massif, float(res), logger=logger)
    cols = np.floor((xs - grid.bounds.left) / res).astype(np.int64)
    rows = np.floor((grid.bounds.top - ys) / res).astype(np.int64)
    inside = (cols >= 0) & (cols < grid.width) & (rows >= 0) & (rows < grid.height)

    n_bands_total = len(terminology.FEATURE_SET_BANDS[_FEATURE_SET])
    out = np.full((len(xs), n_bands_total), np.nan, dtype=np.float32)
    position = 0
    for key in (*_TERRAIN_STEMS, "distance_to_roads", "tessera"):
        with rasterio.open(sources[key]) as src:
            data = src.read()
        out[inside, position : position + data.shape[0]] = data[
            :, rows[inside], cols[inside]
        ].T.astype(np.float32)
        position += data.shape[0]
        del data
    if position != n_bands_total:
        raise ValueError(f"Assembled {position} bands; expected {n_bands_total}.")
    finite = np.isfinite(out).all(axis=1) & (out != terminology.NODATA).all(axis=1)
    return out, inside & finite


def build_training_support(
    args: argparse.Namespace, paths: ProjectPaths, out_dir: Path
) -> dict[str, Any]:
    """Reproduce the final-fit sample, refit the model and fit the DI machinery."""
    pixel_index, memmap, _ = _load_pixel_data(paths, _FEATURE_SET)
    valid = pixel_index["valid"].to_numpy()
    train_all = np.flatnonzero(valid)
    rng = np.random.default_rng([terminology.SEED, 0])
    sampled = train_all[
        sample_train_rows(pixel_index.iloc[train_all], terminology.N_PER_PARCEL, rng)
    ]
    if args.smoke:
        sampled = np.random.default_rng(terminology.SEED).choice(
            sampled, size=min(20_000, len(sampled)), replace=False
        )
        sampled.sort()
    features = np.asarray(memmap[sampled], dtype=np.float32)
    labels = pixel_index["ogf"].to_numpy()[sampled].astype(np.int8)
    folds = pixel_index["fold_id"].to_numpy()[sampled].astype(np.int8)
    pixel_ids = pixel_index["pixel_id"].to_numpy()[sampled]
    logger.info(
        "Final-fit training sample: %s pixels (%s OGF), %d folds",
        f"{len(sampled):,}",
        f"{int(labels.sum()):,}",
        len(np.unique(folds)),
    )

    # DI feature space: the same rasters the map is computed from.
    xs = pixel_index["x"].to_numpy()[sampled]
    ys = pixel_index["y"].to_numpy()[sampled]
    di_features, di_valid = _sample_training_features_from_map(
        paths, int(args.grid_res), args.massif, xs, ys
    )
    logger.info(
        "Training features re-sampled from the Carpathian rasters: " "%s of %s pixels valid",
        f"{int(di_valid.sum()):,}",
        f"{len(sampled):,}",
    )
    if float(di_valid.mean()) < 0.95:
        raise RuntimeError(
            "More than 5% of training pixels have no valid Carpathian features; "
            "check raster coverage before trusting the DI."
        )

    mapping_run = _latest_run(paths.results / _MAPPING_ROLE, required="final_best_params.json")
    best_params = json.loads((mapping_run / "final_best_params.json").read_text())
    if args.smoke:
        best_params = {**best_params, "n_estimators": 50}
    weights = build_sample_weights("none", labels=labels)
    model = make_xgb_classifier(
        best_params, device=args.device, n_jobs=int(args.n_jobs), seed=terminology.SEED
    )
    start = time.perf_counter()
    model.fit(features, labels, sample_weight=weights)
    logger.info(
        "Refitted the final model in %.1f s (device=%s)", time.perf_counter() - start, args.device
    )

    band_names = list(terminology.FEATURE_SET_BANDS[_FEATURE_SET])
    importances = feature_importance(model, band_names)
    validate_schema(importances, FEATURE_IMPORTANCE_SCHEMA)
    write_csv(importances, out_dir / "feature_weights.csv", FEATURE_IMPORTANCE_SCHEMA)
    gains = importances.set_index("feature").loc[band_names, "importance"].to_numpy()

    di_matrix = di_features[di_valid]
    di_labels = labels[di_valid]
    di_folds = folds[di_valid]
    di_pixel_ids = pixel_ids[di_valid]
    scaling = fit_feature_scaling(di_matrix, gains)
    weighted = scaling.transform(di_matrix)

    sizes = sorted({min(s, len(weighted)) for s in args.denominator_sizes})
    diagnostics = []
    for size in sizes:
        mean, used = mean_pairwise_distance(
            weighted, sample_size=size, seed=terminology.SEED, device=args.device
        )
        diagnostics.append({"sample_size": used, "mean_distance": mean})
        logger.info("DI denominator (subset %s): %.6f", f"{used:,}", mean)
    diagnostics_frame = pd.DataFrame(diagnostics)
    write_csv(diagnostics_frame, out_dir / "denominator_diagnostics.csv", DENOMINATOR_SCHEMA)
    denominator = float(diagnostics_frame["mean_distance"].iloc[-1])

    cv_distances = cv_nearest_distances(
        weighted,
        di_folds,
        device=args.device,
        query_block=int(args.query_block),
        reference_block=int(args.reference_block),
    )
    train_di = cv_distances / denominator
    frame = pd.DataFrame(
        {
            "pixel_id": di_pixel_ids,
            "fold_id": di_folds,
            "ogf": di_labels,
            "di": train_di.astype(np.float64),
        }
    )
    validate_schema(frame, AOA_TRAINING_DI_SCHEMA)
    write_parquet(frame, out_dir / "cv_training_di.parquet", AOA_TRAINING_DI_SCHEMA)

    grid_for_count = carpathian_grid(
        load_carpathian_massif(args.massif, logger=logger), float(args.grid_res), logger=logger
    )
    cols_100 = np.floor((xs[di_valid] - grid_for_count.bounds.left) / float(args.grid_res))
    rows_100 = np.floor((grid_for_count.bounds.top - ys[di_valid]) / float(args.grid_res))
    n_distinct = int(len(np.unique(rows_100 * grid_for_count.width + cols_100)))
    logger.info(
        "Training rows %s collapse to %s distinct %sm cells (map-side vectors)",
        f"{int(di_valid.sum()):,}",
        f"{n_distinct:,}",
        int(args.grid_res),
    )

    threshold_stats = di_threshold(train_di)
    threshold_stats["n_distinct_map_cells"] = n_distinct
    threshold_stats["denominator"] = denominator
    threshold_stats["denominator_sample_size"] = int(diagnostics_frame["sample_size"].iloc[-1])
    write_json(threshold_stats, out_dir / "cv_di_threshold.json")
    logger.info(
        "AOA threshold %.4f (whisker %.4f; %.2f%% of training DIs above; q95 %.4f)",
        threshold_stats["aoa_threshold"],
        threshold_stats["upper_whisker"],
        threshold_stats["pct_above_whisker"],
        threshold_stats["q95_sensitivity"],
    )

    np.savez_compressed(
        out_dir / "training_stats.npz",
        means=scaling.means,
        sds=scaling.sds,
        weights=scaling.weights,
        denominator=np.array([denominator]),
    )
    return {
        "weighted": weighted,
        "labels": di_labels,
        "train_di_frame": frame,
        "threshold": float(threshold_stats["aoa_threshold"]),
        "denominator": denominator,
        "n_training": int(len(sampled)),
        "n_di_training": int(di_valid.sum()),
        "mapping_run": mapping_run,
    }


def map_di(
    args: argparse.Namespace,
    paths: ProjectPaths,
    out_dir: Path,
    support: dict[str, Any],
) -> dict[str, Path]:
    """Compute the DI, AOA and nearest-label rasters block-wise over the grid."""
    res = int(args.grid_res)
    massif = load_carpathian_massif(args.massif, logger=logger)
    grid = carpathian_grid(massif, float(res), logger=logger)
    sources = _carpathian_sources(paths, res)
    band_names = list(terminology.FEATURE_SET_BANDS[_FEATURE_SET])

    out_paths = {
        "di": out_dir / f"carpathian_di_3035_{res}m.tif",
        "aoa": out_dir / f"carpathian_aoa_3035_{res}m.tif",
        "nearest": out_dir / f"carpathian_nearest_label_3035_{res}m.tif",
    }
    checkpoint_path = out_dir / _CHECKPOINT
    done: set[str] = (
        set(json.loads(checkpoint_path.read_text()))
        if checkpoint_path.exists() and not args.force
        else set()
    )
    windows = list(raster_io.tiled_block_iterator(grid, block_size=int(args.block_size)))
    if args.smoke:
        windows = windows[: min(2, len(windows))]

    mode = "r+" if done and all(p.exists() for p in out_paths.values()) else "w"
    float_profile = grid.profile(count=1, dtype="float32", nodata=terminology.NODATA)
    byte_profile = grid.profile(count=1, dtype="uint8", nodata=AOA_UINT8_NODATA)
    for profile in (float_profile, byte_profile):
        profile["bigtiff"] = "IF_SAFER"

    weighted_train: np.ndarray = support["weighted"]
    labels: np.ndarray = support["labels"]
    threshold: float = support["threshold"]
    denominator: float = support["denominator"]

    opened = {name: rasterio.open(path) for name, path in sources.items() if name != "worldcover"}
    di_dst = rasterio.open(out_paths["di"], mode, **float_profile)
    aoa_dst = rasterio.open(out_paths["aoa"], mode, **byte_profile)
    near_dst = rasterio.open(out_paths["nearest"], mode, **byte_profile)
    if mode == "w":
        di_dst.set_band_description(1, "dissimilarity_index")
        aoa_dst.set_band_description(1, "aoa_inside")
        near_dst.set_band_description(1, "nearest_training_label_ogf")
    try:
        for window in windows:
            key = f"{int(window.row_off)}_{int(window.col_off)}"
            if key in done:
                continue
            stack = _read_window_stack(opened, window, band_names)
            valid = np.all(stack != terminology.NODATA, axis=0) & np.all(np.isfinite(stack), axis=0)
            di_block = np.full(stack.shape[1:], terminology.NODATA, dtype=np.float32)
            aoa_block = np.full(stack.shape[1:], AOA_UINT8_NODATA, dtype=np.uint8)
            near_block = np.full(stack.shape[1:], AOA_UINT8_NODATA, dtype=np.uint8)
            n_valid = int(valid.sum())
            if n_valid:
                queries = stack[:, valid].T
                scaling_weighted = _transform_queries(out_dir, queries)
                distances, indices = nearest_distances(
                    scaling_weighted,
                    weighted_train,
                    device=args.device,
                    query_block=int(args.query_block),
                    reference_block=int(args.reference_block),
                    return_indices=True,
                )
                di_values = distances / denominator
                di_block[valid] = di_values
                aoa_block[valid] = np.where(di_values <= threshold, AOA_INSIDE, AOA_OUTSIDE)
                near_block[valid] = labels[indices]
            di_dst.write(di_block, 1, window=window)
            aoa_dst.write(aoa_block, 1, window=window)
            near_dst.write(near_block, 1, window=window)
            done.add(key)
            checkpoint_path.write_text(json.dumps(sorted(done)))
            logger.info(
                "Window %s: %s valid px, DI median %.3f",
                key,
                f"{n_valid:,}",
                float(np.median(di_block[valid])) if n_valid else float("nan"),
            )
    finally:
        for dataset in (*opened.values(), di_dst, aoa_dst, near_dst):
            dataset.close()
    return out_paths


def _read_window_stack(
    opened: dict[str, rasterio.DatasetReader], window: Window, band_names: list[str]
) -> np.ndarray:
    """Assemble the 134-band window stack in canonical band order."""
    parts = [opened[stem].read(1, window=window) for stem in _TERRAIN_STEMS]
    parts.append(opened["distance_to_roads"].read(window=window))
    parts.append(opened["tessera"].read(window=window))
    terrain = np.stack(parts[:3], axis=0)
    stack = np.concatenate([terrain, parts[3], parts[4]], axis=0).astype(np.float32)
    if stack.shape[0] != len(band_names):
        raise ValueError(f"Assembled {stack.shape[0]} bands; expected {len(band_names)}.")
    return stack


def _transform_queries(out_dir: Path, queries: np.ndarray) -> np.ndarray:
    """Standardise and weight query features with the saved training statistics."""
    stats = np.load(out_dir / "training_stats.npz")
    scaled = (queries.astype(np.float64) - stats["means"]) / stats["sds"]
    return (scaled * stats["weights"]).astype(np.float32)


def report(
    args: argparse.Namespace,
    paths: ProjectPaths,
    out_dir: Path,
    support: dict[str, Any],
    rasters: dict[str, Path],
) -> None:
    """Forest in/out-of-AOA summary and the DI-performance curve."""
    res = int(args.grid_res)
    sources = _carpathian_sources(paths, res)
    with rasterio.open(rasters["aoa"]) as src:
        aoa = src.read(1)
    with rasterio.open(rasters["nearest"]) as src:
        nearest = src.read(1)
    with rasterio.open(sources["worldcover"]) as src:
        fraction = src.read(1)

    cell_ha = (res * res) / 10_000.0
    valid = aoa != AOA_UINT8_NODATA
    forest = valid & (fraction != terminology.NODATA) & (fraction >= 0.5)
    inside = aoa == AOA_INSIDE

    def _row(stratum: str, mask: np.ndarray) -> dict[str, object]:
        n_valid = int((valid & mask).sum())
        n_inside = int((valid & inside & mask).sum())
        n_forest = int((forest & mask).sum())
        n_forest_in = int((forest & inside & mask).sum())
        return {
            "stratum": stratum,
            "valid_cells": n_valid,
            "valid_area_ha": n_valid * cell_ha,
            "inside_aoa_area_ha": n_inside * cell_ha,
            "pct_valid_area_inside_aoa": 100.0 * n_inside / n_valid if n_valid else float("nan"),
            "forest_area_ha": n_forest * cell_ha,
            "forest_inside_aoa_area_ha": n_forest_in * cell_ha,
            "forest_outside_aoa_area_ha": (n_forest - n_forest_in) * cell_ha,
            "pct_forest_area_inside_aoa": 100.0 * n_forest_in / n_forest
            if n_forest
            else float("nan"),
        }

    everywhere = np.ones_like(valid, dtype=bool)
    summary = pd.DataFrame(
        [
            _row("overall", everywhere),
            _row("nearest_non_ogf", nearest == 0),
            _row("nearest_ogf", nearest == 1),
        ]
    )
    write_csv(summary, out_dir / "forest_aoa_summary.csv", FOREST_SUMMARY_SCHEMA)
    logger.info(
        "Forest inside AOA: %.1f%% of %.0f ha",
        summary.loc[0, "pct_forest_area_inside_aoa"],
        summary.loc[0, "forest_area_ha"],
    )

    # Headline sensitivity, first-class: threshold rule (paper whisker vs q95)
    # crossed with the forest-mask source. CORINE covers only the EU-side
    # countries (Ukraine has no CLC2018), so its rows describe a smaller domain.
    with rasterio.open(rasters["di"]) as src:
        di = src.read(1)
    threshold_stats = json.loads((out_dir / "cv_di_threshold.json").read_text())
    rules = (
        ("upper_whisker", float(threshold_stats["aoa_threshold"])),
        ("q95", float(threshold_stats["q95_sensitivity"])),
    )
    forest_masks = {"worldcover": forest}
    clc_path = (
        paths.processed / "vectors" / "corine_land_cover" / "corine_forest_carpathians_3035.gpkg"
    )
    if clc_path.exists():
        import geopandas as gpd

        massif = load_carpathian_massif(args.massif, logger=logger)
        grid = carpathian_grid(massif, float(res), logger=logger)
        clc = gpd.read_file(clc_path)
        forest_masks["corine_eu"] = valid & (
            raster_io.rasterize_mask(clc.geometry, grid, all_touched=False) == 1
        )
    sensitivity_rows = []
    for mask_name, mask in forest_masks.items():
        mask_di = di[mask & (di != terminology.NODATA) & np.isfinite(di)]
        for rule_name, rule_value in rules:
            sensitivity_rows.append(
                {
                    "forest_mask": mask_name,
                    "threshold_rule": rule_name,
                    "threshold": rule_value,
                    "forest_Mha": float(mask.sum()) * cell_ha / 1e6,
                    "pct_forest_inside": 100.0 * float((mask_di <= rule_value).mean()),
                }
            )
    sensitivity = pd.DataFrame(sensitivity_rows)
    write_csv(sensitivity, out_dir / "headline_sensitivity.csv", SENSITIVITY_SCHEMA)
    whisker_value = rules[0][1]
    wc_di = di[forest & (di != terminology.NODATA) & np.isfinite(di)]
    slope = (
        100.0
        * float(((wc_di <= whisker_value + 0.01).mean()) - ((wc_di <= whisker_value - 0.01).mean()))
        / 2.0
    )
    threshold_stats["headline_slope_pp_per_001_di"] = slope
    write_json(threshold_stats, out_dir / "cv_di_threshold.json")
    logger.info("Headline sensitivity written (slope %.1f pp per 0.01 DI at the threshold)", slope)

    source_run = _latest_run(paths.results / _SOURCE_ROLE, required="predictions_fold1.parquet")
    oof = pd.concat(
        [
            pd.read_parquet(source_run / f"predictions_fold{fold}.parquet")
            for fold in terminology.FOLD_IDS
        ],
        ignore_index=True,
    )
    train_di_frame: pd.DataFrame = support["train_di_frame"]
    joined = train_di_frame.merge(oof[["pixel_id", "y_true", "p"]], on="pixel_id", how="inner")
    logger.info(
        "DI-performance curve: %s of %s training pixels matched to OOF predictions",
        f"{len(joined):,}",
        f"{len(train_di_frame):,}",
    )
    # Each sample is classified at its own fold's inner-CV pixel threshold — the
    # leakage-free operating point a per-fold deployment would use.
    per_fold = pd.read_csv(source_run / "per_fold_metrics.csv")
    pixel_rows = per_fold[per_fold["level"] == "pixel"].set_index("outer_fold")
    inner_thresholds = pixel_rows["inner_threshold"].to_dict()
    sample_thresholds = joined["fold_id"].map(inner_thresholds).to_numpy(dtype=np.float64)
    if not np.isfinite(sample_thresholds).all():
        raise ValueError("Missing inner-CV pixel threshold for at least one fold.")

    window = min(int(args.curve_window), max(len(joined) // 4, 2))
    curve = performance_by_di_window(
        joined["di"].to_numpy(),
        joined["y_true"].to_numpy(),
        joined["p"].to_numpy(),
        thresholds=sample_thresholds,
        window=window,
    )
    write_csv(curve, out_dir / "di_performance_curve.csv", CURVE_SCHEMA)

    threshold_value = float(support["threshold"])
    edges = np.array([0.0, 0.20, 0.25, 0.30, 0.35, 0.40, threshold_value, np.inf])
    table = performance_by_di_bins(
        joined["di"].to_numpy(),
        joined["y_true"].to_numpy(),
        joined["p"].to_numpy(),
        edges=edges,
        thresholds=sample_thresholds,
    )
    write_csv(table, out_dir / "di_performance_table.csv", DI_TABLE_SCHEMA)

    # Buffered transfer: the same DI windows under the spatial-buffer regime.
    # The fold-exclusion curve above is calibrated with cross-fold training
    # data still nearby (median 4.4 km; ~88% of pixels within 10 km); the gap
    # run's buffered arm at _BUFFER_GAP_KM provides per-pixel predictions from
    # fixed-HP refits with no training data within that distance of the test
    # parcels - the regime that applies to almost all of the Carpathians.
    gap_run = _latest_run(paths.results / _BUFFER_ROLE, required="gap_analysis.csv")
    checkpoints = [
        gap_run / "fold_checkpoints" / f"fold{fold}_pixel.parquet" for fold in terminology.FOLD_IDS
    ]
    missing_ckpt = [str(path) for path in checkpoints if not path.exists()]
    if missing_ckpt:
        raise FileNotFoundError(f"Spatial-buffer fold checkpoints missing: {missing_ckpt}")
    buffered = (
        pads.dataset([str(path) for path in checkpoints])
        .to_table(
            filter=(pads.field("gap_km") == _BUFFER_GAP_KM) & (pads.field("arm") == "buffered"),
            columns=["pixel_id", "y_true", "p"],
        )
        .to_pandas()
    )
    joined_buf = train_di_frame.merge(buffered, on="pixel_id", how="inner")
    logger.info(
        "Buffered transfer: %s of %s pixels matched at gap %d km (%s)",
        f"{len(joined_buf):,}",
        f"{len(train_di_frame):,}",
        _BUFFER_GAP_KM,
        gap_run.name,
    )
    thresholds_buf = joined_buf["fold_id"].map(inner_thresholds).to_numpy(dtype=np.float64)
    curve_buf = performance_by_di_window(
        joined_buf["di"].to_numpy(),
        joined_buf["y_true"].to_numpy(),
        joined_buf["p"].to_numpy(),
        thresholds=thresholds_buf,
        window=window,
    )
    write_csv(curve_buf, out_dir / "di_performance_curve_buffered.csv", CURVE_SCHEMA)
    table_buf = performance_by_di_bins(
        joined_buf["di"].to_numpy(),
        joined_buf["y_true"].to_numpy(),
        joined_buf["p"].to_numpy(),
        edges=edges,
        thresholds=thresholds_buf,
    )
    write_csv(table_buf, out_dir / "di_performance_table_buffered.csv", DI_TABLE_SCHEMA)

    # Distance decay for the blended expected-performance surface. The
    # proximity-only decay (buffered - control + control-at-zero) removes the
    # training-set-capacity component: the control arm removes the same number
    # of parcels at random, so its drop is pure capacity. Truncated at
    # _DECAY_MAX_GAP_KM (the spatially robust distance; see the constant's
    # comment); the blend clamps at the far value beyond it.
    gap_metrics = pd.read_csv(gap_run / "gap_analysis.csv")
    pixel_pr = gap_metrics[(gap_metrics["level"] == "pixel") & (gap_metrics["metric"] == "pr_auc")]
    buffered_pr = pixel_pr[pixel_pr["arm"] == "buffered"].set_index("gap_km")["pooled"]
    control_pr = pixel_pr[pixel_pr["arm"] == "control"].set_index("gap_km")["pooled"]
    adjusted = (buffered_pr - control_pr + control_pr.loc[0]).sort_index().loc[:_DECAY_MAX_GAP_KM]
    iso = IsotonicRegression(increasing=False, out_of_bounds="clip")
    adjusted_fit = iso.fit_transform(adjusted.index.to_numpy(dtype=float), adjusted.to_numpy())
    decay = pd.DataFrame(
        {
            "gap_km": adjusted.index.to_numpy(dtype=np.int64),
            "pr_auc_buffered": buffered_pr.sort_index().loc[:_DECAY_MAX_GAP_KM].to_numpy(),
            "pr_auc_control": control_pr.sort_index().loc[:_DECAY_MAX_GAP_KM].to_numpy(),
            "pr_auc_proximity_adjusted": adjusted.to_numpy(),
            "pr_auc_adjusted_fit": adjusted_fit,
        }
    )
    write_csv(decay, out_dir / "distance_decay.csv", DECAY_SCHEMA)

    # Distance-to-training raster (km) so the blended map can weight the two
    # transfers continuously - no step at the AOI boundary.
    massif = load_carpathian_massif(args.massif, logger=logger)
    grid = carpathian_grid(massif, float(res), logger=logger)
    coords = pd.read_parquet(
        paths.results / _SOURCE_ROLE / "pixel_index.parquet", columns=["pixel_id", "x", "y"]
    ).merge(train_di_frame[["pixel_id"]], on="pixel_id", how="inner")
    cols = ((coords["x"].to_numpy() - grid.transform.c) / grid.transform.a).astype(np.int64)
    rows_px = ((coords["y"].to_numpy() - grid.transform.f) / grid.transform.e).astype(np.int64)
    inside = (cols >= 0) & (cols < grid.width) & (rows_px >= 0) & (rows_px < grid.height)
    training_cells = np.zeros(grid.shape, dtype=bool)
    training_cells[rows_px[inside], cols[inside]] = True
    distance_km = (
        distance_transform_edt(~training_cells, sampling=(grid.resolution[1], grid.resolution[0]))
        / 1000.0
    ).astype(np.float32)
    profile = grid.profile(count=1, dtype="float32", nodata=terminology.NODATA)
    profile["bigtiff"] = "IF_SAFER"
    dist_path = out_dir / f"carpathian_dist_to_training_3035_{res}m.tif"
    with rasterio.open(dist_path, "w", **profile) as dst:
        dst.write(distance_km, 1)
    logger.info(
        "Distance-to-training raster written: %s (training cells: %s)",
        dist_path.name,
        f"{int(training_cells.sum()):,}",
    )


def _parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--grid-res", type=positive_int, default=100, help="Analysis grid (m).")
    parser.add_argument("--device", default="cuda", help="torch/XGBoost device (default cuda).")
    parser.add_argument("--n-jobs", type=positive_int, default=8, help="XGBoost CPU threads.")
    parser.add_argument(
        "--massif", type=Path, default=None, help="Massif GeoPackage (default: stage-020 location)."
    )
    parser.add_argument(
        "--denominator-sizes",
        type=lambda s: tuple(int(v) for v in s.split(",")),
        default=(10_000, 25_000, 50_000),
        help="Subset sizes for the DI denominator (largest is used; default 10k,25k,50k).",
    )
    parser.add_argument(
        "--curve-window",
        type=positive_int,
        default=20_000,
        help="Samples per DI-performance window (default 20000).",
    )
    parser.add_argument("--block-size", type=positive_int, default=2048, help="Map block (px).")
    parser.add_argument(
        "--query-block", type=positive_int, default=16_384, help="NN query rows per block."
    )
    parser.add_argument(
        "--reference-block", type=positive_int, default=65_536, help="NN reference rows per block."
    )
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Tiny run into results/preflight/ (20k training pixels, 50 trees, 2 windows).",
    )
    parser.add_argument(
        "--force", action="store_true", help="Recompute the map even when windows are checkpointed."
    )
    return parser.parse_args(argv)


def _run(args: argparse.Namespace, paths: ProjectPaths, out_dir: Path) -> dict[str, object]:
    support = build_training_support(args, paths, out_dir)
    rasters = map_di(args, paths, out_dir, support)
    report(args, paths, out_dir, support, rasters)

    write_input_manifest(
        [
            carpathians_massif_path(paths.repo_root) if args.massif is None else Path(args.massif),
            *(_carpathian_sources(paths, int(args.grid_res)).values()),
            support["mapping_run"] / "final_best_params.json",
            _latest_run(paths.results / _BUFFER_ROLE, required="gap_analysis.csv")
            / "gap_analysis.csv",
        ],
        out_dir / "input_manifest.json",
        repo_root=paths.repo_root,
    )
    write_environment_snapshot(out_dir / "env_snapshot.txt")
    return {
        "architecture": _ARCHITECTURE,
        "feature_set": _FEATURE_SET,
        "grid_res_m": int(args.grid_res),
        "device": args.device,
        "smoke": bool(args.smoke),
        "n_training": support["n_training"],
        "n_di_training": support["n_di_training"],
        "aoa_threshold": support["threshold"],
        "denominator": support["denominator"],
        "mapping_run": support["mapping_run"].name,
        "outputs": {name: path.name for name, path in rasters.items()},
        "feature_space": (
            "the Carpathian rasters at the analysis resolution for BOTH training points "
            "and the map (the TESSERA version installed by notebook 001); gain weights "
            "come from the deployed model refit on the 10 m training matrix."
        ),
    }


def main(argv: Sequence[str] | None = None) -> int:
    """Command-line entry point."""
    args = _parse_args(argv)
    paths = get_project_paths()
    ensure_writable_dirs(paths)
    out_dir = (
        paths.results / "preflight" / f"{_RESULTS_ROLE}_smoke"
        if args.smoke
        else paths.results / _RESULTS_ROLE / _FEATURE_SET
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    global logger
    logger = make_run_logger(__name__, out_dir)

    summary = _run(args, paths, out_dir)
    write_json(summary, out_dir / "run_metadata.json")
    logger.info("Area of applicability complete: %s", out_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
