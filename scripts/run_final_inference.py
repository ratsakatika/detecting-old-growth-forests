"""Final old-growth model, wall-to-wall map and calibrated parcel product (Stage 10).

The selected model (XGBoost + ``baseline_tessera``) is the best within the AOI, so
it is used for mapping. The nested cross-validation already gives the conservative
out-of-fold performance estimate; for the map the model is tuned once and trained
on all labelled parcels:

  1. Hyperparameters are tuned by a single Optuna search whose objective is the
     pooled out-of-fold parcel PR-AUC over the six spatial folds (each trial refits
     every fold via :func:`utils.cv.fixed_hp_fold_result`). The final model is then
     trained on all labelled pixels with the selected hyperparameters.
  2. The model is applied to every valid pixel of the parcel-map area by
     slicing the co-registered grid memmap. Five calibrators are compared on the
     pixel out-of-fold predictions and the selected one recalibrates the surface;
     the published probability GeoTIFF is calibrated (the raw surface is kept
     alongside it). The binary GeoTIFF is computed *directly on the calibrated
     surface*: the raw F1-max threshold is mapped through the calibrator to
     ``pixel_threshold_calibrated`` and the extent is ``calibrated >= that``. This
     is the same extent as the raw decision (monotonic calibration), but expressing
     it on the calibrated axis lets a user reproduce the binary from the published
     probability using the saved threshold.
  3. The pixel probabilities (raw) are aggregated to the full parcel geometry (the
     repaired ``parcel_map_clean`` parcels, before notebook 005 subtracts roads
     and disturbances) by their unweighted mean.
  4. Five calibrators are compared on the out-of-fold parcel predictions and the
     selected one is applied; only the selected calibrated probability is stored.
     The parcel verdict is likewise computed *on the calibrated parcel probability*
     at ``parcel_threshold_calibrated`` (the raw parcel F1-max threshold mapped
     through the parcel calibrator), so it too is reproducible from the published
     probability. ``--calibrator`` forces one method for both calibrations. Both
     calibrated thresholds are written to the run metadata.

The new product is written to a new GeoPackage (``ogf_reference_labels_mapped.gpkg``)
without overwriting the labels; the label attributes are joined back by parcel id
with a geometry-containment integrity check. The model runs on GPU (CUDA) by
default. Output goes to ``results/mapping/<run_id>/``.
"""

import argparse
import json
import logging
import os
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Final

import geopandas as gpd
import numpy as np
import numpy.typing as npt
import optuna
import pandas as pd
import rasterio
from rasterio.features import rasterize
from xgboost import XGBClassifier

from scripts.run_sampling_sensitivity import _latest_run, _load_pixel_data
from utils.calibration import (
    CALIBRATION_METHODS,
    CALIBRATION_METRICS,
    Calibrator,
    compare_calibrators,
    fit_calibrator,
    select_best_calibrator,
)
from utils.cv import (
    NestedCVConfig,
    aggregate_pooled_metrics,
    fixed_hp_fold_result,
    sample_train_rows,
)
from utils.env_capture import write_environment_snapshot
from utils.hp_search import suggest_xgb_params
from utils.inference import (
    gather_pixel_features,
    ogf_area_ha,
    parcel_consistent_area_ha,
    predict_pixels,
)
from utils.io import ColumnSpec, Schema, write_csv, write_json
from utils.logging import make_run_logger
from utils.manifest import write_input_manifest
from utils.metrics import parcel_aggregations
from utils.models.xgboost import make_xgb_classifier
from utils.paths import ProjectPaths, ensure_writable_dirs, get_project_paths
from utils.raster_io import ReferenceGrid, open_reference_grid, write_geotiff
from utils.sampling import build_sample_weights
from utils.terminology import FOLD_IDS, N_HP_TRIALS, N_PER_PARCEL, NODATA, SEED

logger = logging.getLogger(__name__)

_LOGGER_NAME: Final[str] = "run_final_inference"
_ARCHITECTURE: Final[str] = "xgboost"
_FEATURE_SET: Final[str] = "baseline_tessera"
_RESULTS_ROLE: Final[str] = "mapping"
_SOURCE_ROLE: Final[str] = "main_nested_cv"

_GRID_MEMMAP: Final[Path] = Path("data/cache/grid_memmap") / f"{_FEATURE_SET}.npy"
_PARCEL_TEMPLATE: Final[Path] = Path("data/processed/vectors/forest_records/parcel_map_clean.gpkg")
_LABELS: Final[Path] = Path("data/processed/vectors/labels/ogf_reference_labels_partitioned.gpkg")
_MAPPED_GPKG: Final[Path] = Path("data/processed/vectors/labels/ogf_reference_labels_mapped.gpkg")
_PARCEL_PRED_TEMPLATE: Final[str] = "parcel_predictions_fold{fold}.parquet"
_PIXEL_PRED_TEMPLATE: Final[str] = "predictions_fold{fold}.parquet"
_DEFAULT_N_JOBS: Final[int] = min(8, os.cpu_count() or 1)

# The published wall-to-wall surface is calibrated; the raw model surface is retained alongside
# it under this name so a resume can recalibrate (different method) without re-running inference.
_RAW_PROB_TIF: Final[str] = "ogf_probability_raw_3035_10m.tif"
_PROB_TIF: Final[str] = "ogf_probability_3035_10m.tif"
_BINARY_TIF: Final[str] = "ogf_binary_3035_10m.tif"

CALIBRATION_SCHEMA: Final[Schema] = {
    "method": ColumnSpec("object"),
    **{metric: ColumnSpec("float64") for metric in CALIBRATION_METRICS},
}


@dataclass(frozen=True, slots=True)
class FinalModelInputs:
    """Everything the orchestration needs after loading; keeps ``_run`` readable."""

    pixel_index: pd.DataFrame
    memmap: np.ndarray
    oof_parcels: pd.DataFrame
    oof_pixels: pd.DataFrame
    source_run: Path


def _config(device: str, n_jobs: int, seed: int, n_per_parcel: int) -> NestedCVConfig:
    return NestedCVConfig(
        feature_set=_FEATURE_SET,
        architecture=_ARCHITECTURE,
        sampling_mode="none",
        n_per_parcel=n_per_parcel,
        n_hp_trials=N_HP_TRIALS,
        seed=seed,
        device=device,
        n_jobs=n_jobs,
        fold_ids=FOLD_IDS,
    )


def cv_tuned_search(
    config: NestedCVConfig,
    pixel_index: pd.DataFrame,
    memmap: np.ndarray,
    *,
    n_trials: int,
    run_logger: logging.Logger,
) -> tuple[dict[str, float | int], float]:
    """Tune hyperparameters by the pooled out-of-fold parcel PR-AUC over the folds."""

    def objective(trial: optuna.trial.Trial) -> float:
        params = suggest_xgb_params(trial)
        trial.set_user_attr("params", params)
        results = [
            fixed_hp_fold_result(config, pixel_index, memmap, params, outer_fold=fold)
            for fold in config.fold_ids
        ]
        pooled = aggregate_pooled_metrics(results).pooled_parcel
        return float(pooled.loc[pooled["metric"] == "pr_auc", "value"].iloc[0])

    study = optuna.create_study(
        direction="maximize", sampler=optuna.samplers.TPESampler(seed=config.seed)
    )
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
    best_params = dict(study.best_trial.user_attrs["params"])
    run_logger.info(
        "CV-tuned search: best pooled parcel PR-AUC %.4f over %d trials.",
        study.best_value,
        n_trials,
    )
    return best_params, float(study.best_value)


def train_final_model(
    config: NestedCVConfig,
    pixel_index: pd.DataFrame,
    memmap: np.ndarray,
    best_params: dict[str, float | int],
) -> XGBClassifier:
    """Train the final XGBoost model on all labelled pixels with the selected HPs."""
    valid = pixel_index["valid"].to_numpy()
    train_all = np.flatnonzero(valid)
    rng = np.random.default_rng([config.seed, 0])
    sampled = train_all[sample_train_rows(pixel_index.iloc[train_all], config.n_per_parcel, rng)]
    y_train = pixel_index["ogf"].to_numpy()[sampled].astype(np.int8)
    weights = build_sample_weights(
        config.sampling_mode,
        labels=y_train,
        forest_type=pixel_index["forest_type_corine"].to_numpy()[sampled],
    )
    model = make_xgb_classifier(
        best_params, device=config.device, n_jobs=config.n_jobs, seed=config.seed
    )
    model.fit(np.asarray(memmap[sampled], dtype=np.float32), y_train, sample_weight=weights)
    return model


def valid_pixel_ids(
    grid_memmap: npt.NDArray[np.float32], domain_mask: npt.NDArray[np.bool_]
) -> npt.NDArray[np.int64]:
    """Flat ids of pixels that are inside the domain mask and have valid features.

    The domain is the parcel-map area (``parcel_grid >= 0``), so every parcel's valid pixels
    are predicted and aggregated; pixels with the NoData sentinel in band 0 are excluded.
    """
    if domain_mask.shape != grid_memmap.shape[1:]:
        raise ValueError("domain mask shape does not match the grid.")
    valid = (np.asarray(grid_memmap[0]) != NODATA) & domain_mask
    return np.flatnonzero(valid.reshape(-1)).astype(np.int64)


def wall_to_wall_over_ids(
    model: XGBClassifier,
    grid_memmap: npt.NDArray[np.float32],
    pixel_ids: npt.NDArray[np.int64],
    *,
    chunk_size: int,
) -> npt.NDArray[np.float64]:
    """Predict probabilities for the given pixel ids in memory-bounded chunks."""
    out = np.empty(pixel_ids.size, dtype=np.float64)
    for start in range(0, pixel_ids.size, chunk_size):
        stop = min(start + chunk_size, pixel_ids.size)
        features = gather_pixel_features(grid_memmap, pixel_ids[start:stop])
        out[start:stop] = predict_pixels(model, features, chunk_size=chunk_size)
    return out


def aggregate_to_parcels(
    pixel_ids: npt.NDArray[np.int64],
    probabilities: npt.NDArray[np.float64],
    parcel_id_grid: npt.NDArray[np.int64],
) -> pd.DataFrame:
    """Mean pixel probability per parcel, dropping pixels outside any parcel."""
    parcel_ids = parcel_id_grid.reshape(-1)[pixel_ids]
    inside = parcel_ids >= 0
    return parcel_aggregations(parcel_ids[inside], probabilities[inside])


def fit_and_compare(
    y_true: npt.NDArray[np.int64],
    p_oof: npt.NDArray[np.float64],
    *,
    seed: int,
    method: str | None,
) -> tuple[str, pd.DataFrame, Calibrator]:
    """Compare the five calibrators on held-out predictions; select (or force) one and refit it.

    When ``method`` is ``None`` the lowest cross-validated ECE wins (Brier tie-break); otherwise
    ``method`` is used regardless of its rank, but the full comparison is still computed so the
    analysis (notebook 010) shows every calibrator. The chosen calibrator is refit on all of the
    held-out predictions and returned so the caller can both transform a surface and map a
    threshold through it.
    """
    if method is None:
        best, comparison = select_best_calibrator(y_true, p_oof, seed=seed)
    else:
        if method not in CALIBRATION_METHODS:
            raise ValueError(
                f"unknown calibrator {method!r}; expected one of {CALIBRATION_METHODS}."
            )
        comparison = compare_calibrators(y_true, p_oof, seed=seed)
        best = method
    return best, comparison, fit_calibrator(best, y_true, p_oof)


def calibrate_parcels(
    oof_parcels: pd.DataFrame,
    raw_probabilities: npt.NDArray[np.float64],
    *,
    seed: int,
    method: str | None = None,
) -> tuple[str, pd.DataFrame, npt.NDArray[np.float64], Calibrator]:
    """Compare the five calibrators on the OOF parcels; apply the selected (or forced) one.

    Returns the selected method, the full comparison, the calibrated probabilities, and the fitted
    calibrator itself (so the caller can also map the F1-max threshold through it).
    """
    best, comparison, calibrator = fit_and_compare(
        oof_parcels["y_true"].to_numpy(), oof_parcels["p_mean"].to_numpy(), seed=seed, method=method
    )
    return best, comparison, calibrator.predict(raw_probabilities), calibrator


def _parcel_id_grid(template: gpd.GeoDataFrame, ref: ReferenceGrid) -> npt.NDArray[np.int64]:
    """Rasterise the parcel template to a per-pixel parcel-id grid (-1 outside parcels)."""
    shapes = list(zip(template.geometry, template["parcel_id"].astype(np.int64), strict=True))
    if not shapes:
        return np.full(ref.shape, -1, dtype=np.int64)
    burned = rasterize(shapes, out_shape=ref.shape, transform=ref.transform, fill=-1, dtype="int64")
    return burned.astype(np.int64)


def build_mapped_frame(
    template: gpd.GeoDataFrame,
    labels: gpd.GeoDataFrame,
    parcel_probs: pd.DataFrame,
    calibrated: npt.NDArray[np.float64],
    *,
    threshold_calibrated: float,
) -> gpd.GeoDataFrame:
    """Join probabilities and label attributes onto the full parcel template.

    The verdict is taken directly on the published (calibrated) parcel probability at
    ``threshold_calibrated`` (the raw parcel F1-max threshold mapped through the parcel
    calibrator). Monotonic calibration makes this the same parcel set as the raw decision, but
    on the calibrated axis a user can reproduce ``ogf_ratsakatika`` from the published
    ``ogf_ratsakatika_probability`` using the saved threshold. The raw mean is retained too.
    """
    label_attrs = labels.drop(columns="geometry").rename(columns={"ogf": "reference_label"})
    probs = parcel_probs.assign(ogf_ratsakatika_probability=calibrated).rename(
        columns={"p_mean": "ogf_ratsakatika_probability_raw"}
    )
    merged = template.merge(label_attrs, on="parcel_id", how="left").merge(
        probs[["parcel_id", "ogf_ratsakatika_probability_raw", "ogf_ratsakatika_probability"]],
        on="parcel_id",
        how="left",
    )
    merged["ogf_ratsakatika"] = merged["ogf_ratsakatika_probability"] >= threshold_calibrated
    return gpd.GeoDataFrame(merged, geometry="geometry", crs=template.crs)


def _containment_ok(
    template: gpd.GeoDataFrame, labels: gpd.GeoDataFrame, *, sample: int = 200
) -> bool:
    """Check that label geometries sit within their same-id template geometry."""
    joined = labels.sample(min(sample, len(labels)), random_state=SEED).merge(
        template[["parcel_id", "geometry"]], on="parcel_id", suffixes=("_label", "_template")
    )
    return bool(
        np.all(
            [
                tmpl.buffer(1.0).contains(lab)
                for lab, tmpl in zip(
                    joined["geometry_label"], joined["geometry_template"], strict=True
                )
            ]
        )
    )


def _parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cuda", choices=["cpu", "cuda"])
    parser.add_argument("--n-jobs", type=int, default=_DEFAULT_N_JOBS)
    parser.add_argument("--n-trials", type=int, default=N_HP_TRIALS)
    parser.add_argument("--n-per-parcel", type=int, default=N_PER_PARCEL)
    parser.add_argument("--chunk-size", type=int, default=2_000_000)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument(
        "--calibrator",
        default="platt",
        choices=[*CALIBRATION_METHODS, "auto"],
        help="Calibrator forced for BOTH the parcel product and the pixel surface. Defaults to "
        "'platt' (a deliberate override: the published products are Platt-calibrated, and Platt is "
        "deterministic and monotonic, so re-runs are reproducible). Pass 'auto' to instead "
        "pick the "
        "lowest cross-validated ECE per level (this selects isotonic for the pixel surface and "
        "spline for the parcels on the current data, which shifts the calibrated probability by up "
        "to ~0.05 and is NOT what the published map uses). The full comparison is always computed.",
    )
    parser.add_argument(
        "--resume",
        default=None,
        help="Reuse an existing results/mapping/<run_id>, skipping the search and inference "
        "steps already saved there (final_best_params.json and the probability GeoTIFF).",
    )
    return parser.parse_args(argv)


def _pooled_threshold(run_dir: Path, level: str) -> float:
    """The F1-max operating threshold on the uncalibrated pooled OOF predictions."""
    frame = pd.read_csv(run_dir / f"pooled_{level}_metrics.csv")
    return float(frame.loc[frame["metric"] == "threshold", "value"].iloc[0])


def _load_inputs(paths: ProjectPaths) -> FinalModelInputs:
    """Resolve the source run, pixel data and out-of-fold parcel and pixel predictions."""
    source_run = _latest_run(paths.results / _SOURCE_ROLE, _FEATURE_SET)
    pixel_index, memmap, _ = _load_pixel_data(paths, _FEATURE_SET)
    oof_parcels = pd.concat(
        [
            pd.read_parquet(source_run / _PARCEL_PRED_TEMPLATE.format(fold=fold))
            for fold in FOLD_IDS
        ],
        ignore_index=True,
    )
    oof_pixels = pd.concat(
        [pd.read_parquet(source_run / _PIXEL_PRED_TEMPLATE.format(fold=fold)) for fold in FOLD_IDS],
        ignore_index=True,
    )
    return FinalModelInputs(pixel_index, memmap, oof_parcels, oof_pixels, source_run)


def _run(args: argparse.Namespace, paths: ProjectPaths, run_dir: Path) -> dict:
    """Tune, train, infer, aggregate, calibrate and write the mapped product."""
    run_logger = make_run_logger(_LOGGER_NAME, run_dir)
    config = _config(args.device, args.n_jobs, args.seed, args.n_per_parcel)
    # 'platt' (the default) forces Platt for both calibrations; 'auto' (method=None) restores the
    # lowest-ECE per-level auto-selection. The published map is Platt, so that is the default.
    calibrator = None if args.calibrator == "auto" else args.calibrator
    data = _load_inputs(paths)
    run_logger.info(
        "Loaded %d labelled pixels; source run %s.", len(data.pixel_index), data.source_run.name
    )

    ref = open_reference_grid()
    # Area uses the F1-max threshold on the uncalibrated pooled pixel OOF predictions.
    pixel_threshold = _pooled_threshold(data.source_run, "pixel")

    # The inference domain is the parcel-map area itself (every parcel's valid pixels), so each
    # parcel aggregates over all of its predictions. A CORINE forest mask would drop valid parcel
    # pixels (leaving some parcels unpredicted) and bias the surviving parcel means upward.
    # parcel_id is a zero-padded string in the GeoPackages but an integer everywhere else, so
    # coerce before joining and rasterising.
    template = gpd.read_file(paths.repo_root / _PARCEL_TEMPLATE)[["parcel_id", "geometry"]]
    template["parcel_id"] = template["parcel_id"].astype("int64")
    labels = gpd.read_file(paths.repo_root / _LABELS)
    labels["parcel_id"] = labels["parcel_id"].astype("int64")
    if not _containment_ok(template, labels):
        raise ValueError(
            "label geometries are not contained in their template parcels; id mismatch."
        )
    parcel_grid = _parcel_id_grid(template, ref)

    # Resume support: reuse a saved hyperparameter search if present.
    best_params_path = run_dir / "final_best_params.json"
    if best_params_path.is_file():
        best_params = json.loads(best_params_path.read_text("utf-8"))
        best_value = float("nan")
        run_logger.info("Resume: reusing the saved hyperparameter search.")
    else:
        best_params, best_value = cv_tuned_search(
            config, data.pixel_index, data.memmap, n_trials=args.n_trials, run_logger=run_logger
        )
        write_json(best_params, best_params_path)

    # Resume support: reuse the saved raw wall-to-wall surface, else train and infer. Older runs
    # stored the raw surface under the published name (before pixel calibration was added), so fall
    # back to it; new runs keep the raw surface separate so the published one can be calibrated.
    raw_path = run_dir / _RAW_PROB_TIF
    prob_path = run_dir / _PROB_TIF
    reuse_src = raw_path if raw_path.is_file() else (prob_path if prob_path.is_file() else None)
    if reuse_src is not None:
        run_logger.info("Resume: reusing the saved wall-to-wall surface (%s).", reuse_src.name)
        with rasterio.open(reuse_src) as src:
            flat = src.read(1).reshape(-1)
        pixel_ids = np.flatnonzero(flat != NODATA).astype(np.int64)
        probabilities = flat[pixel_ids].astype(np.float64)
    else:
        model = train_final_model(config, data.pixel_index, data.memmap, best_params)
        grid = np.load(paths.repo_root / _GRID_MEMMAP, mmap_mode="r")
        pixel_ids = valid_pixel_ids(grid, parcel_grid >= 0)
        run_logger.info("Wall-to-wall over %d valid parcel-map pixels.", pixel_ids.size)
        probabilities = wall_to_wall_over_ids(model, grid, pixel_ids, chunk_size=args.chunk_size)

    # Pixel-level calibration of the published surface. The calibrator is fit on the pixel
    # out-of-fold predictions and applied to the wall-to-wall raw probabilities; the operating
    # threshold is the raw F1-max threshold mapped through the same monotonic calibrator, so the
    # binary extent is unchanged (calibration only relabels the probability axis).
    best_pixel, comparison_pixel, pixel_calibrator = fit_and_compare(
        data.oof_pixels["y_true"].to_numpy(),
        data.oof_pixels["p"].to_numpy(),
        seed=args.seed,
        method=calibrator,
    )
    calibrated_pixels = pixel_calibrator.predict(probabilities)
    pixel_threshold_calibrated = float(pixel_calibrator.predict(np.asarray([pixel_threshold]))[0])
    run_logger.info(
        "Pixel calibrator %s: F1-max threshold %.4f (raw) -> %.4f (calibrated).",
        best_pixel,
        pixel_threshold,
        pixel_threshold_calibrated,
    )

    raw_grid = np.full(ref.shape, NODATA, dtype=np.float32)
    raw_grid.reshape(-1)[pixel_ids] = probabilities
    prob_grid = np.full(ref.shape, NODATA, dtype=np.float32)
    prob_grid.reshape(-1)[pixel_ids] = calibrated_pixels
    # Binary decision taken directly on the calibrated surface (``prob_grid``) at the calibrated
    # threshold, so the published binary raster is exactly
    # ``ogf_probability_3035_10m >= threshold``.
    binary_grid = np.zeros(ref.shape, dtype=np.uint8)
    binary_grid.reshape(-1)[pixel_ids] = (calibrated_pixels >= pixel_threshold_calibrated).astype(
        np.uint8
    )
    write_geotiff(raw_path, raw_grid, ref, nodata=NODATA, logger=run_logger)
    write_geotiff(prob_path, prob_grid, ref, nodata=NODATA, logger=run_logger)
    write_geotiff(
        run_dir / _BINARY_TIF, binary_grid, ref, dtype="uint8", nodata=255, logger=run_logger
    )
    area_ha = ogf_area_ha(calibrated_pixels >= pixel_threshold_calibrated)

    # Parcel-level calibration (independent of the pixel calibration): the parcel probability in the
    # product is the raw parcel mean recalibrated by the parcel-fitted calibrator; the verdict stays
    # on the raw mean and raw threshold (a monotonic recalibration selects the same parcels).
    parcel_probs = aggregate_to_parcels(pixel_ids, probabilities, parcel_grid)
    best_method, comparison, calibrated, parcel_calibrator = calibrate_parcels(
        data.oof_parcels, parcel_probs["p_mean"].to_numpy(), seed=args.seed, method=calibrator
    )
    parcel_threshold = _pooled_threshold(data.source_run, "parcel")
    parcel_threshold_calibrated = float(
        parcel_calibrator.predict(np.asarray([parcel_threshold]))[0]
    )
    run_logger.info(
        "Parcel calibrator %s: F1-max threshold %.4f (raw) -> %.4f (calibrated).",
        best_method,
        parcel_threshold,
        parcel_threshold_calibrated,
    )
    # The pixel-threshold area (above) is the per-pixel operating point; this is the
    # parcel-consistent one (pixels in parcels whose mean probability clears the parcel
    # threshold), matching the parcel product. The two differ because the thresholds do. It is
    # computed on the raw mean at the raw threshold, which selects the identical parcel set as the
    # calibrated verdict (monotonic), so the area is unchanged either way.
    parcel_area_ha = parcel_consistent_area_ha(
        parcel_grid.reshape(-1)[pixel_ids], probabilities, threshold=parcel_threshold
    )
    comparison["selected"] = comparison["method"] == best_method
    write_csv(
        comparison, run_dir / "calibration_comparison.csv", CALIBRATION_SCHEMA, allow_extra=True
    )
    comparison_pixel["selected"] = comparison_pixel["method"] == best_pixel
    write_csv(
        comparison_pixel,
        run_dir / "calibration_comparison_pixel.csv",
        CALIBRATION_SCHEMA,
        allow_extra=True,
    )

    mapped = build_mapped_frame(
        template, labels, parcel_probs, calibrated, threshold_calibrated=parcel_threshold_calibrated
    )
    mapped_path = paths.repo_root / _MAPPED_GPKG
    mapped.to_file(mapped_path, driver="GPKG")
    run_logger.info(
        "Selected calibrator %s; predicted old-growth area %.0f ha; wrote %s.",
        best_method,
        area_ha,
        mapped_path.name,
    )

    manifest_inputs = [
        *(data.source_run / _PARCEL_PRED_TEMPLATE.format(fold=fold) for fold in FOLD_IDS),
        *(data.source_run / _PIXEL_PRED_TEMPLATE.format(fold=fold) for fold in FOLD_IDS),
        data.source_run / "pooled_pixel_metrics.csv",
        data.source_run / "pooled_parcel_metrics.csv",
        paths.repo_root / _PARCEL_TEMPLATE,
        paths.repo_root / _LABELS,
    ]
    write_input_manifest(
        manifest_inputs, run_dir / "input_manifest.json", repo_root=paths.repo_root
    )
    write_environment_snapshot(run_dir / "env_snapshot.txt")
    return {
        "created_at": datetime.now(UTC).isoformat(),
        "script": "scripts/run_final_inference.py",
        "feature_set": _FEATURE_SET,
        "architecture": _ARCHITECTURE,
        "source_run": str(data.source_run.relative_to(paths.repo_root)),
        "cv_tuned_pooled_parcel_pr_auc": best_value,
        "selected_calibrator": best_method,
        "selected_calibrator_pixel": best_pixel,
        "n_mapped_pixels": int(pixel_ids.size),
        "pixel_threshold": float(pixel_threshold),
        "pixel_threshold_calibrated": pixel_threshold_calibrated,
        "parcel_threshold": float(parcel_threshold),
        "parcel_threshold_calibrated": parcel_threshold_calibrated,
        "predicted_ogf_area_ha": float(area_ha),
        "predicted_ogf_area_parcel_ha": float(parcel_area_ha),
        "device": args.device,
        "seed": args.seed,
        "outputs": {
            "mapped_gpkg": str(_MAPPED_GPKG),
            "probability_tif": _PROB_TIF,
            "probability_raw_tif": _RAW_PROB_TIF,
            "binary_tif": _BINARY_TIF,
        },
    }


def main(argv: Sequence[str] | None = None) -> int:
    """Run the final inference pipeline and write its artefacts.

    Args:
        argv: Argument list, or ``None`` to read from ``sys.argv``.

    Returns:
        Process exit code (0 on success).
    """
    args = _parse_args(argv)
    paths = get_project_paths()
    ensure_writable_dirs(paths)
    if args.resume:
        run_id = args.resume
        run_dir = paths.results / _RESULTS_ROLE / run_id
        if not run_dir.is_dir():
            raise FileNotFoundError(f"--resume run directory not found: {run_dir}")
    else:
        run_id = f"{datetime.now(UTC):%Y%m%d_%H%M%S}__{_ARCHITECTURE}__{_FEATURE_SET}"
        run_dir = paths.results / _RESULTS_ROLE / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    summary = _run(args, paths, run_dir)
    write_json(summary, run_dir / "run_metadata.json")
    logger.info("Final inference complete: %s.", run_id)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
