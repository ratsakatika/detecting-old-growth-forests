"""Five-product parcel comparison: performance, stratification and agreement (Stage 9).

The four existing products' aligned masks (``build_existing_product_masks.py``) and
this study's calibrated map (``run_final_inference.py``) are aggregated to the shared
parcel template by their mean pixel value, alongside the rasterised reference label.
The result is one parcel table with each product's old-growth verdict and this study's
calibrated probability, plus ``total_ogf`` (how many of the five products call the
parcel old-growth).

From that table the script writes: the overall performance against the reference
(PR-AUC, ROC-AUC, F1, precision, recall) for each product, with this study reported
twice (the honest out-of-fold parcels and the in-sample trained-and-tested fit, kept
separate because this study alone is tuned on the study area); the same metrics
stratified by forest type, species, altitude and slope; and the cross-study agreement
statistics. Outputs go to ``results/comparison/<run_id>/``.

The performance, stratified and sweep tables (and the notebook figures that read
``product_eval_parcels.parquet``) are computed on a single common evaluation set: the
parcels this study scored out-of-fold, with every existing product re-aggregated over
the *same eroded strict-interior pixel geometry* those out-of-fold scores use (no refit
of this study's predictions). So all five maps share one parcel set and one erosion
rule; they differ only on the validity axis (a product's NoData is a real 0 kept in the
denominator, whereas this study drops predictor-NoData pixels as a no-claim). The
wall-to-wall cross-study agreement keeps the centroid-in verdicts, since it spans every
in-domain parcel rather than the labelled set; this study is therefore represented there by
its deployed verdict, the only one defined off the labelled parcels.
"""

import argparse
import logging
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Final

import geopandas as gpd
import numpy as np
import numpy.typing as npt
import pandas as pd
import rasterio
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from scripts.run_final_inference import (
    _GRID_MEMMAP,
    _MAPPED_GPKG,
    _PARCEL_TEMPLATE,
    _parcel_id_grid,
    _pooled_threshold,
)
from scripts.run_sampling_sensitivity import _latest_run
from utils.comparison import (
    agreement_by_reference,
    cohen_kappa_matrix,
    consensus_breakdown,
    fleiss_kappa,
    pairwise_agreement,
    per_study_agreement_counts,
    total_ogf_votes,
)
from utils.env_capture import write_environment_snapshot
from utils.io import ColumnSpec, Schema, write_csv, write_json
from utils.logging import make_run_logger
from utils.manifest import write_input_manifest
from utils.metrics import metrics_at_threshold, parcel_aggregations
from utils.paths import ProjectPaths, ensure_writable_dirs, get_project_paths
from utils.raster_io import open_reference_grid
from utils.terminology import COMPARISON_STUDIES, FOLD_IDS, NODATA

logger = logging.getLogger(__name__)

_LOGGER_NAME: Final[str] = "run_product_comparison"
_RESULTS_ROLE: Final[str] = "comparison"
_SOURCE_ROLE: Final[str] = "main_nested_cv"
_PIXEL_INDEX: Final[str] = "pixel_index.parquet"  # this study's eroded OOF pixel geometry
_FEATURE_SET: Final[str] = "baseline_tessera"
_PRODUCT_MASKS: Final[Path] = Path("data/processed/rasters/existing_products_10m")
_MINE: Final[str] = "ratsakatika"
_STUDIES: Final[tuple[str, ...]] = (*COMPARISON_STUDIES, _MINE)
_STRATA: Final[tuple[str, ...]] = ("corine_forest_type", "dominant_english", "altitude_band")
_ALTITUDE_TERTILES: Final[tuple[float, ...]] = (0.0, 1 / 3, 2 / 3, 1.0)
_METRICS: Final[tuple[str, ...]] = (
    "pr_auc",
    "roc_auc",
    "f1",
    "precision",
    "recall",
    "false_positive_rate",
)
# Decision thresholds for the supplementary sensitivity sweep; 0.5 is the primary majority rule.
_SWEEP_THRESHOLDS: Final[tuple[float, ...]] = tuple(round(0.1 * i, 2) for i in range(1, 10))

PERFORMANCE_SCHEMA: Final[Schema] = {
    "product": ColumnSpec("object"),
    "kind": ColumnSpec("object"),
    "n": ColumnSpec("int64"),
    **{metric: ColumnSpec("float64") for metric in _METRICS},
    "f1_inner": ColumnSpec("float64", nullable=True),
    "precision_inner": ColumnSpec("float64", nullable=True),
    "recall_inner": ColumnSpec("float64", nullable=True),
}


def aggregate_product_to_parcels(
    mask: npt.NDArray[np.uint8], parcel_grid: npt.NDArray[np.int64]
) -> pd.DataFrame:
    """Mean product value (the old-growth fraction) per parcel."""
    flat_mask = mask.reshape(-1).astype(np.float64)
    flat_parcel = parcel_grid.reshape(-1)
    inside = flat_parcel >= 0
    return parcel_aggregations(flat_parcel[inside], flat_mask[inside]).rename(
        columns={"p_mean": "frac"}
    )


def build_comparison_frame(
    template: gpd.GeoDataFrame,
    mapped: gpd.GeoDataFrame,
    product_fracs: dict[str, pd.DataFrame],
    *,
    parcel_threshold: float,
) -> gpd.GeoDataFrame:
    """Assemble the five-product parcel table with verdicts, probability and total_ogf.

    This study's verdict is the deployed one from the map (``ogf_ratsakatika`` = raw parcel mean
    >= parcel threshold) when the map carries it, so the comparison matches the product and the
    wall-to-wall area; ``parcel_threshold`` is only a fallback for inputs that carry the
    probability without the precomputed verdict. (Thresholding the calibrated probability at the
    uncalibrated threshold would select a different, inconsistent set of parcels.)
    """
    frame = template[["parcel_id", "geometry"]].copy()
    has_verdict = "ogf_ratsakatika" in mapped.columns
    keep = [
        "parcel_id",
        "reference_label",
        "ogf_ratsakatika_probability",
        *(["ogf_ratsakatika"] if has_verdict else []),
        *(c for c in _STRATA if c in mapped),
    ]
    frame = frame.merge(mapped[keep], on="parcel_id", how="left")
    if not has_verdict:
        frame["ogf_ratsakatika"] = frame["ogf_ratsakatika_probability"] >= parcel_threshold
    for study in COMPARISON_STUDIES:
        fracs = product_fracs[study].rename(columns={"frac": f"ogf_{study}_frac"})
        frame = frame.merge(fracs[["parcel_id", f"ogf_{study}_frac"]], on="parcel_id", how="left")
        frame[f"ogf_{study}_frac"] = frame[f"ogf_{study}_frac"].fillna(0.0)
        frame[f"ogf_{study}"] = frame[f"ogf_{study}_frac"] >= 0.5
    verdict_cols = [f"ogf_{study}" for study in _STUDIES]
    frame["total_ogf"] = total_ogf_votes(frame, verdict_cols).to_numpy()
    return gpd.GeoDataFrame(frame, geometry="geometry", crs=template.crs)


def _binary_metrics(
    reference: np.ndarray, score: np.ndarray, verdict: np.ndarray
) -> dict[str, float]:
    """PR-AUC/ROC-AUC from the continuous parcel old-growth fraction; the rest from the verdict.

    The existing products are binary per pixel, but aggregated to parcels they have a graded
    old-growth area fraction, which serves as the ranking score for the area-under-curve
    metrics. The false-positive rate enables separation (recall minus FPR) downstream.
    """
    negatives = reference == 0
    return {
        "pr_auc": float(average_precision_score(reference, score)),
        "roc_auc": float(roc_auc_score(reference, score)),
        "f1": float(f1_score(reference, verdict, zero_division=0.0)),
        "precision": float(precision_score(reference, verdict, zero_division=0.0)),
        "recall": float(recall_score(reference, verdict, zero_division=0.0)),
        "false_positive_rate": float(verdict[negatives].mean())
        if negatives.any()
        else float("nan"),
    }


def product_performance(
    comparison: gpd.GeoDataFrame, oof_parcels: pd.DataFrame, *, inner_threshold: float
) -> pd.DataFrame:
    """Overall performance per product, with this study's out-of-fold and in-sample rows."""
    labelled = comparison[
        comparison["reference_label"].notna() & comparison["ogf_ratsakatika_probability"].notna()
    ].copy()
    reference = labelled["reference_label"].to_numpy().astype(np.int64)
    rows: list[dict[str, object]] = []
    for study in COMPARISON_STUDIES:
        metrics = _binary_metrics(
            reference, labelled[f"ogf_{study}_frac"].to_numpy(), labelled[f"ogf_{study}"].to_numpy()
        )
        rows.append({"product": study, "kind": "existing", "n": len(labelled), **metrics})
    # This study, in-sample (trained and tested on the same labels).
    in_sample = _binary_metrics(
        reference,
        labelled["ogf_ratsakatika_probability"].to_numpy(),
        labelled["ogf_ratsakatika"].to_numpy(),
    )
    rows.append({"product": _MINE, "kind": "in_sample", "n": len(labelled), **in_sample})
    # This study, honest out-of-fold parcels, with both operating points.
    oof = oof_parcels.dropna(subset=["y_true"])
    y = oof["y_true"].to_numpy().astype(np.int64)
    p = oof["p_mean"].to_numpy()
    outer = _binary_metrics(y, p, p >= _max_f1_threshold(y, p))
    inner = metrics_at_threshold(y, p, inner_threshold)
    rows.append(
        {
            "product": _MINE,
            "kind": "out_of_fold",
            "n": len(oof),
            **outer,
            "f1_inner": inner["f1"],
            "precision_inner": inner["precision"],
            "recall_inner": inner["recall"],
        }
    )
    return pd.DataFrame(rows).reindex(columns=list(PERFORMANCE_SCHEMA))


def _max_f1_threshold(y: np.ndarray, p: np.ndarray) -> float:
    from utils.metrics import best_f1_threshold

    return best_f1_threshold(y, p).threshold


def stratified_performance(comparison: gpd.GeoDataFrame, *, strata: Sequence[str]) -> pd.DataFrame:
    """Per-product F1, recall and false-positive rate within each stratum.

    F1 is computed from each product's binary old-growth verdict against the reference label
    within the stratum; recall and the false-positive rate are kept alongside it.
    """
    labelled = comparison[comparison["reference_label"].notna()]
    rows: list[dict[str, object]] = []
    for column in strata:
        if column not in labelled:
            continue
        for value, group in labelled.groupby(column):
            ref = group["reference_label"].to_numpy().astype(bool)
            if ref.sum() == 0 or (~ref).sum() == 0:
                continue
            for study in _STUDIES:
                pred = group[f"ogf_{study}"].to_numpy().astype(bool)
                rows.append(
                    {
                        "stratum": column,
                        "value": str(value),
                        "product": study,
                        "n": len(group),
                        "f1": float(f1_score(ref, pred, zero_division=0.0)),
                        "recall": float(pred[ref].mean()),
                        "false_positive_rate": float(pred[~ref].mean()),
                    }
                )
    return pd.DataFrame(rows)


def threshold_sensitivity(honest: gpd.GeoDataFrame, *, thresholds: Sequence[float]) -> pd.DataFrame:
    """F1, precision and recall for every product across a sweep of decision thresholds.

    The 0.5 area-fraction majority rule is the primary operating point (see
    :func:`product_performance`); this supplementary sweep shows how each product's threshold
    metrics move with the cut-off. Every product is thresholded on the same uniform parcel score
    -- the existing products on their old-growth area fraction, this study on its out-of-fold
    probability -- against the same labelled reference, so the aggregation is identical across
    products. A NaN score (no coverage) is treated as below every threshold, i.e. non-old-growth.
    """
    labelled = honest[honest["reference_label"].notna()].copy()
    reference = labelled["reference_label"].to_numpy().astype(bool)
    score_col = {
        **{study: f"ogf_{study}_frac" for study in COMPARISON_STUDIES},
        _MINE: "ogf_ratsakatika_probability",
    }
    rows: list[dict[str, object]] = []
    for study in _STUDIES:
        score = labelled[score_col[study]].to_numpy(dtype=float)
        for threshold in thresholds:
            verdict = score >= threshold  # NaN >= t is False, so no-coverage counts as non-OGF
            rows.append(
                {
                    "product": study,
                    "threshold": float(threshold),
                    "n": len(labelled),
                    "f1": float(f1_score(reference, verdict, zero_division=0.0)),
                    "precision": float(precision_score(reference, verdict, zero_division=0.0)),
                    "recall": float(recall_score(reference, verdict, zero_division=0.0)),
                    "false_positive_rate": float(verdict[~reference].mean())
                    if (~reference).any()
                    else float("nan"),
                }
            )
    return pd.DataFrame(rows)


def agreement_tables(comparison: gpd.GeoDataFrame) -> dict[str, pd.DataFrame]:
    """Consensus, pairwise agreement, Cohen's kappa and agreement-by-reference.

    Pass the wall-to-wall in-domain frame (every parcel this study predicts, deployed verdicts):
    the consensus ``all`` scope, the pairwise-agreement matrix and the kappa matrix are computed
    over all of it, and only the reference-split rows subset to the labelled parcels.
    """
    verdict_cols = [f"ogf_{study}" for study in _STUDIES]
    labelled = comparison[comparison["reference_label"].notna()].copy()
    labelled["reference_label"] = labelled["reference_label"].astype(bool)
    consensus = pd.DataFrame(
        [
            {
                "scope": "all",
                **consensus_breakdown(comparison, verdict_cols),
                "fleiss_kappa": fleiss_kappa(comparison, verdict_cols),
            },
            {
                "scope": "reference_ogf",
                **consensus_breakdown(labelled[labelled["reference_label"]], verdict_cols),
            },
            {
                "scope": "reference_non_ogf",
                **consensus_breakdown(labelled[~labelled["reference_label"]], verdict_cols),
            },
        ]
    )
    return {
        "consensus": consensus,
        "pairwise_agreement": pairwise_agreement(comparison, verdict_cols).reset_index(
            names="product"
        ),
        "cohen_kappa": cohen_kappa_matrix(comparison, verdict_cols).reset_index(names="product"),
        "agreement_by_reference": agreement_by_reference(labelled, verdict_cols, "reference_label"),
    }


def _product_fracs(
    masks_dir: Path, parcel_grid: npt.NDArray[np.int64], run_logger: logging.Logger
) -> dict[str, pd.DataFrame]:
    """Aggregate every existing-product mask to parcel old-growth fractions."""
    fracs: dict[str, pd.DataFrame] = {}
    for study in COMPARISON_STUDIES:
        with rasterio.open(masks_dir / f"{study}_ogf_3035_10m.tif") as src:
            mask = src.read(1).astype(np.uint8)
        fracs[study] = aggregate_product_to_parcels(mask, parcel_grid)
        run_logger.info("%s: aggregated %d parcels.", study, len(fracs[study]))
    return fracs


def parcel_mean_elevation(
    grid_band: npt.NDArray[np.float32], parcel_grid: npt.NDArray[np.int64]
) -> pd.Series:
    """Mean elevation (metres) per parcel from the grid's elevation band."""
    elevation = grid_band.reshape(-1).astype(np.float64)
    parcels = parcel_grid.reshape(-1)
    inside = (parcels >= 0) & (elevation != NODATA)
    return (
        pd.DataFrame({"parcel_id": parcels[inside], "elevation": elevation[inside]})
        .groupby("parcel_id")["elevation"]
        .mean()
    )


def altitude_band(elevation: pd.Series, *, reference: pd.Series | None = None) -> pd.Series:
    """Bin per-parcel mean elevation into three equal-count altitude tertiles.

    The tertile edges come from ``reference`` (the labelled parcels) when given, so each band
    holds roughly the same number of labelled parcels; every parcel is then binned by those
    edges. The labels report the elevation range of each tertile in metres.
    """
    basis = (elevation if reference is None else reference).dropna()
    quantiles = basis.quantile(list(_ALTITUDE_TERTILES)).to_numpy()
    labels = [f"{quantiles[i]:.0f}-{quantiles[i + 1]:.0f} m" for i in range(3)]
    edges = [-np.inf, float(quantiles[1]), float(quantiles[2]), np.inf]
    return pd.cut(elevation, bins=edges, labels=labels, include_lowest=True).astype("object")


def honest_eval_frame(
    evaluable: gpd.GeoDataFrame, oof_parcels: pd.DataFrame, *, threshold: float
) -> gpd.GeoDataFrame:
    """Labelled parcels with this study's verdict replaced by its out-of-fold prediction.

    The mapping model trained on every labelled parcel, so its in-sample verdict there is
    optimistic; for a fair stratified and agreement comparison this study uses its honest
    out-of-fold probability instead (the existing products are already independent).
    """
    labelled = evaluable[evaluable["reference_label"].notna()].copy()
    oof_prob = labelled["parcel_id"].map(oof_parcels.set_index("parcel_id")["p_mean"])
    labelled["ogf_ratsakatika_probability"] = oof_prob
    labelled["ogf_ratsakatika"] = oof_prob >= threshold
    labelled["total_ogf"] = total_ogf_votes(
        labelled, [f"ogf_{study}" for study in _STUDIES]
    ).to_numpy()
    return labelled


def eroded_product_fracs(
    masks_dir: Path, pixel_index: pd.DataFrame, *, run_logger: logging.Logger
) -> pd.DataFrame:
    """Old-growth area fraction per parcel over this study's eroded out-of-fold pixel geometry.

    Aligns the existing products to this study's OOF pixel set: each product's binary mask is
    sampled at the exact eroded strict-interior pixels of the shared pixel index (the same
    geometry the OOF parcel scores use), then averaged per parcel. A product's NoData is already
    0 -- a real non-old-growth claim -- and stays in the denominator. The result therefore shares
    one geometry (erosion) with the OOF score; the two differ only on the validity axis (this
    study additionally drops predictor-NoData pixels), never on erosion. No refit of this study's
    predictions is involved: only the products are re-aggregated onto the OOF pixel set.
    """
    rows = pixel_index["row"].to_numpy()
    cols = pixel_index["col"].to_numpy()
    parcel_id = pixel_index["parcel_id"].to_numpy(dtype=np.int64)
    out = pd.DataFrame({"parcel_id": np.unique(parcel_id)})
    for study in COMPARISON_STUDIES:
        with rasterio.open(masks_dir / f"{study}_ogf_3035_10m.tif") as src:
            mask = src.read(1)
        values = mask[rows, cols].astype(np.float64)  # 0/1 over the eroded pixels (NoData is 0)
        frac = parcel_aggregations(parcel_id, values).set_index("parcel_id")["p_mean"]
        out[f"ogf_{study}_frac"] = out["parcel_id"].map(frac)
        run_logger.info(
            "Eroded-geometry old-growth fraction for %s over %d parcels.",
            study,
            int(out[f"ogf_{study}_frac"].notna().sum()),
        )
    return out


def build_eval_frame(
    evaluable: gpd.GeoDataFrame, eroded_fracs: pd.DataFrame, common_ids: Sequence[int]
) -> gpd.GeoDataFrame:
    """Restrict the labelled set to the common OOF parcels and swap in the eroded product fracs.

    The performance, stratified and sweep tables are computed on this frame, so every existing
    product is scored on the same eroded pixel geometry as this study's OOF and every column
    shares one parcel set (the parcels this study scored out-of-fold). The wall-to-wall
    agreement and the map GeoPackage keep the centroid-in verdicts, which are a separate analysis.
    """
    frame = evaluable[evaluable["parcel_id"].isin(common_ids)].copy()
    eroded = eroded_fracs.set_index("parcel_id")
    for study in COMPARISON_STUDIES:
        frame[f"ogf_{study}_frac"] = frame["parcel_id"].map(eroded[f"ogf_{study}_frac"])
        frame[f"ogf_{study}"] = frame[f"ogf_{study}_frac"] >= 0.5
    return frame


def _write_eval_parcels(
    run_dir: Path, eval_frame: gpd.GeoDataFrame, oof_parcels: pd.DataFrame
) -> None:
    """Write the common-set parcel table the notebook figures read (eroded fracs + OOF score)."""
    out = pd.DataFrame({"parcel_id": eval_frame["parcel_id"].to_numpy()})
    out["reference_label"] = eval_frame["reference_label"].to_numpy()
    out["ratsakatika_oof"] = out["parcel_id"].map(oof_parcels.set_index("parcel_id")["p_mean"])
    for study in COMPARISON_STUDIES:
        out[f"ogf_{study}_frac"] = eval_frame[f"ogf_{study}_frac"].to_numpy()
    out.to_parquet(run_dir / "product_eval_parcels.parquet", index=False)


def _write_evaluation_diagnostics(
    run_dir: Path,
    comparison: gpd.GeoDataFrame,
    eval_frame: gpd.GeoDataFrame,
    eroded: pd.DataFrame,
    pixel_index: pd.DataFrame,
    oof_parcels: pd.DataFrame,
    common_ids: pd.Index,
    run_logger: logging.Logger,
) -> None:
    """Write the parcel-drop and denominator-reconciliation diagnostics.

    Records which labelled parcels are excluded from the common evaluation set and why (they
    have no strict-interior pixel, so *no* map can score them -- the drop is symmetric across all
    five maps and never triggered by a product's own NoData), plus the geometry / NoData rule per
    map and the residual valid-pixel gap (now validity-only, since erosion is shared).
    """
    labelled = comparison[comparison["reference_label"].notna()]
    dropped = labelled[~labelled["parcel_id"].isin(common_ids)].copy()
    # A dropped parcel with a wall-to-wall (centroid-in) prediction lost its pixels only to
    # erosion; one without any prediction is genuinely sub-pixel.
    dropped["reason"] = np.where(
        dropped["ogf_ratsakatika_probability"].notna(), "erosion_only", "sub_pixel"
    )
    dropped[["parcel_id", "reference_label", "reason"]].to_csv(
        run_dir / "evaluation_drops.csv", index=False
    )

    # Symmetry: with a shared eroded geometry every product is scorable exactly where this study
    # is, so no parcel is dropped for one map but kept for another.
    mine_ok = set(common_ids)
    asymmetric = {
        study: int(len(set(eroded.loc[eroded[f"ogf_{study}_frac"].notna(), "parcel_id"]) ^ mine_ok))
        for study in COMPARISON_STUDIES
    }

    # Residual valid-pixel gap: products count every eroded pixel (NoData as 0); this study drops
    # predictor-NoData pixels. Erosion no longer contributes, so the gap is validity-only.
    eroded_counts = pd.Series(1, index=pixel_index["parcel_id"]).groupby(level=0).size()
    mine_counts = oof_parcels.set_index("parcel_id")["n_pixels"]
    gap = (eroded_counts.reindex(common_ids) - mine_counts.reindex(common_ids)).dropna()

    recon = pd.DataFrame(
        [
            {
                "map": COMPARISON_STUDIES[s].display,
                "geometry_rule": "eroded_strict_interior",
                "nodata_handling": "0 (real non-old-growth claim; kept in denominator)",
            }
            for s in COMPARISON_STUDIES
        ]
        + [
            {
                "map": "This study",
                "geometry_rule": "eroded_strict_interior",
                "nodata_handling": "excluded (no-claim; dropped from numerator and denominator)",
            }
        ]
    )
    recon.to_csv(run_dir / "denominator_reconciliation.csv", index=False)

    n_non_ogf = int((dropped["reference_label"].astype(float) == 0).sum())
    run_logger.info(
        "Evaluation set: %d labelled -> %d common (dropped %d: %d erosion-only, %d sub-pixel; "
        "%d reference non-old-growth). Asymmetric drops per product: %s. "
        "Residual valid-pixel gap (products - this study): median %d, max %d.",
        len(labelled),
        len(eval_frame),
        len(dropped),
        int((dropped["reason"] == "erosion_only").sum()),
        int((dropped["reason"] == "sub_pixel").sum()),
        n_non_ogf,
        asymmetric,
        int(gap.median()) if len(gap) else 0,
        int(gap.max()) if len(gap) else 0,
    )


def _parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--masks-dir", default=None)
    return parser.parse_args(argv)


def _run(args: argparse.Namespace, paths: ProjectPaths, run_dir: Path) -> dict:
    """Build the comparison table and write the performance and agreement artefacts."""
    run_logger = make_run_logger(_LOGGER_NAME, run_dir)
    ref = open_reference_grid()
    template = gpd.read_file(paths.repo_root / _PARCEL_TEMPLATE)[["parcel_id", "geometry"]]
    template["parcel_id"] = template["parcel_id"].astype("int64")  # zero-padded string in the gpkg
    mapped = gpd.read_file(paths.repo_root / _MAPPED_GPKG)
    mapped["parcel_id"] = mapped["parcel_id"].astype("int64")
    masks_dir = Path(args.masks_dir) if args.masks_dir else paths.repo_root / _PRODUCT_MASKS
    source_run = _latest_run(paths.results / _SOURCE_ROLE, _FEATURE_SET)
    parcel_threshold = _pooled_threshold(source_run, "parcel")
    oof = pd.concat(
        [
            pd.read_parquet(source_run / f"parcel_predictions_fold{fold}.parquet")
            for fold in FOLD_IDS
        ],
        ignore_index=True,
    )

    parcel_grid = _parcel_id_grid(template, ref)
    fracs = _product_fracs(masks_dir, parcel_grid, run_logger)
    comparison = build_comparison_frame(template, mapped, fracs, parcel_threshold=parcel_threshold)
    grid = np.load(paths.repo_root / _GRID_MEMMAP, mmap_mode="r")
    elevation = parcel_mean_elevation(np.asarray(grid[0]), parcel_grid)
    labelled_elevation = elevation.reindex(
        comparison.loc[comparison["reference_label"].notna(), "parcel_id"]
    )
    comparison["altitude_band"] = comparison["parcel_id"].map(
        altitude_band(elevation, reference=labelled_elevation)
    )
    comparison.to_file(run_dir / "product_parcel_comparison.gpkg", driver="GPKG")

    # My study has no wall-to-wall prediction for parcels outside the inference domain
    # (their probability is NaN); the GeoPackage keeps all parcels for the map.
    evaluable = comparison[comparison["ogf_ratsakatika_probability"].notna()].copy()

    # Align the existing products to this study's out-of-fold pixel geometry: re-aggregate every
    # product over the eroded strict-interior pixels of the shared pixel index (the same pixels
    # the OOF scores use), and evaluate on the common set of parcels this study scored
    # out-of-fold. This makes one parcel set shared by every column of the performance,
    # stratified and sweep tables, with the same erosion for all five maps.
    pixel_index = pd.read_parquet(paths.results / _SOURCE_ROLE / _PIXEL_INDEX)
    eroded = eroded_product_fracs(masks_dir, pixel_index, run_logger=run_logger)
    common_ids = pd.Index(np.sort(oof["parcel_id"].unique()))
    eval_frame = build_eval_frame(evaluable, eroded, common_ids)
    _write_evaluation_diagnostics(
        run_dir, comparison, eval_frame, eroded, pixel_index, oof, common_ids, run_logger
    )

    # The overall table reports this study both in-sample and out-of-fold; the stratified and
    # sweep tables use the out-of-fold prediction so it is not optimistic on its own training
    # parcels. All are computed on the eroded, common evaluation set.
    performance = product_performance(eval_frame, oof, inner_threshold=parcel_threshold)
    write_csv(performance, run_dir / "overall_performance.csv", PERFORMANCE_SCHEMA)

    honest = honest_eval_frame(eval_frame, oof, threshold=parcel_threshold)
    stratified_performance(honest, strata=_STRATA).to_csv(
        run_dir / "stratified_performance.csv", index=False
    )
    threshold_sensitivity(honest, thresholds=_SWEEP_THRESHOLDS).to_csv(
        run_dir / "threshold_sensitivity.csv", index=False
    )
    # The parcel table the notebook figures read: the eroded product fractions, this study's OOF
    # score and the reference label, on the single common evaluation set.
    _write_eval_parcels(run_dir, eval_frame, oof)

    # Cross-study agreement is a wall-to-wall analysis (every in-domain parcel, deployed verdicts),
    # so it keeps the centroid-in verdicts rather than the eroded evaluation set. It must not go
    # through honest_eval_frame: that restricts to the labelled parcels and swaps in the
    # out-of-fold score, which exists only there, and would silently shrink this analysis to the
    # labelled subset. The reference-split rows inside agreement_tables subset to the labelled
    # parcels themselves.
    for name, table in agreement_tables(evaluable).items():
        table.to_csv(run_dir / f"agreement_{name}.csv", index=False)
    per_study_agreement_counts(evaluable, [f"ogf_{study}" for study in _STUDIES]).to_csv(
        run_dir / "agreement_per_study.csv", index=False
    )
    run_logger.info(
        "Comparison complete: %d parcels (%d common eval set, prevalence %.4f).",
        len(comparison),
        len(eval_frame),
        float(eval_frame["reference_label"].astype(float).mean()),
    )

    manifest_inputs = [
        paths.repo_root / _MAPPED_GPKG,
        *(masks_dir / f"{study}_ogf_3035_10m.tif" for study in COMPARISON_STUDIES),
        *(source_run / f"parcel_predictions_fold{fold}.parquet" for fold in FOLD_IDS),
    ]
    write_input_manifest(
        manifest_inputs, run_dir / "input_manifest.json", repo_root=paths.repo_root
    )
    write_environment_snapshot(run_dir / "env_snapshot.txt")
    return {
        "created_at": datetime.now(UTC).isoformat(),
        "script": "scripts/run_product_comparison.py",
        "n_parcels": int(len(comparison)),
        "n_labelled": int(comparison["reference_label"].notna().sum()),
        "studies": list(_STUDIES),
        "source_run": str(source_run.relative_to(paths.repo_root)),
    }


def main(argv: Sequence[str] | None = None) -> int:
    """Build the five-product comparison and write its artefacts.

    Args:
        argv: Argument list, or ``None`` to read from ``sys.argv``.

    Returns:
        Process exit code (0 on success).
    """
    args = _parse_args(argv)
    paths = get_project_paths()
    ensure_writable_dirs(paths)
    run_id = f"{datetime.now(UTC):%Y%m%d_%H%M%S}__product_comparison"
    run_dir = paths.results / _RESULTS_ROLE / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    summary = _run(args, paths, run_dir)
    write_json(summary, run_dir / "run_metadata.json")
    logger.info("Product comparison complete: %s.", run_id)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
