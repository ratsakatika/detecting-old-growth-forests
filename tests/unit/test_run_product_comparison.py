"""Functional smoke test for scripts/run_product_comparison.py.

A small synthetic parcel table exercises the product-to-parcel aggregation, the
five-product comparison frame, the performance rows (existing, in-sample and
out-of-fold) and the agreement tables.
"""

import logging

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
from affine import Affine
from rasterio.crs import CRS
from shapely.geometry import box

from scripts import run_product_comparison as rpc
from utils.io import validate_schema
from utils.terminology import COMPARISON_STUDIES


def _template(n: int = 8) -> gpd.GeoDataFrame:
    return gpd.GeoDataFrame(
        {"parcel_id": np.arange(n)},
        geometry=[box(i, 0, i + 1, 1) for i in range(n)],
        crs="EPSG:3035",
    )


def _mapped(n: int = 8) -> gpd.GeoDataFrame:
    reference = np.array([1, 1, 1, 1, 0, 0, 0, 0])
    prob = np.array([0.9, 0.8, 0.7, 0.55, 0.4, 0.3, 0.2, 0.1])
    return gpd.GeoDataFrame(
        {
            "parcel_id": np.arange(n),
            "reference_label": reference.astype(float),
            "ogf_ratsakatika_probability": prob,
            "corine_forest_type": ["broadleaf", "coniferous"] * (n // 2),
        },
        geometry=[box(i, 0, i + 1, 1) for i in range(n)],
        crs="EPSG:3035",
    )


def _fracs(n: int = 8) -> dict[str, pd.DataFrame]:
    rng = np.random.default_rng(0)
    out: dict[str, pd.DataFrame] = {}
    for k, study in enumerate(COMPARISON_STUDIES):
        # Each product partly tracks the reference, with some noise.
        base = np.array([0.8, 0.7, 0.6, 0.55, 0.45, 0.3, 0.2, 0.1])
        frac = np.clip(base + 0.1 * rng.standard_normal(n) + 0.02 * k, 0.0, 1.0)
        out[study] = pd.DataFrame({"parcel_id": np.arange(n), "frac": frac, "n_pixels": 4})
    return out


def test_aggregate_product_to_parcels() -> None:
    mask = np.array([[1, 1, 0], [1, 0, 0]], dtype=np.uint8)
    parcel_grid = np.array([[10, 10, -1], [10, 20, 20]], dtype=np.int64)
    fracs = rpc.aggregate_product_to_parcels(mask, parcel_grid)
    assert set(fracs["parcel_id"]) == {10, 20}
    assert (
        fracs.loc[fracs["parcel_id"] == 10, "frac"].iloc[0] == (1 + 1 + 1) / 3
    )  # three pixels, all ogf


def test_build_comparison_and_total_ogf() -> None:
    comparison = rpc.build_comparison_frame(_template(), _mapped(), _fracs(), parcel_threshold=0.5)
    for study in (*COMPARISON_STUDIES, "ratsakatika"):
        assert f"ogf_{study}" in comparison.columns
    assert comparison["total_ogf"].between(0, 5).all()
    assert "reference_label" in comparison.columns
    assert comparison["ogf_ratsakatika"].iloc[0]  # prob 0.9 >= 0.5


def test_product_performance_rows_and_schema() -> None:
    comparison = rpc.build_comparison_frame(_template(), _mapped(), _fracs(), parcel_threshold=0.5)
    oof = pd.DataFrame(
        {
            "parcel_id": np.arange(8),
            "y_true": [1, 1, 1, 1, 0, 0, 0, 0],
            "p_mean": [0.85, 0.75, 0.65, 0.5, 0.45, 0.35, 0.25, 0.15],
        }
    )
    performance = rpc.product_performance(comparison, oof, inner_threshold=0.4)
    validate_schema(performance, rpc.PERFORMANCE_SCHEMA)
    kinds = set(performance["kind"])
    assert {"existing", "in_sample", "out_of_fold"} <= kinds
    oof_row = performance[performance["kind"] == "out_of_fold"].iloc[0]
    assert np.isfinite(oof_row["f1_inner"])  # the inner operating point is filled for my study
    existing_row = performance[performance["kind"] == "existing"].iloc[0]
    assert pd.isna(existing_row["f1_inner"])  # external products have no inner point


def test_agreement_tables() -> None:
    comparison = rpc.build_comparison_frame(_template(), _mapped(), _fracs(), parcel_threshold=0.5)
    tables = rpc.agreement_tables(comparison)
    assert {"consensus", "pairwise_agreement", "cohen_kappa", "agreement_by_reference"} <= set(
        tables
    )
    assert "fleiss_kappa" in tables["consensus"].columns
    assert len(tables["pairwise_agreement"]) == 5  # four existing studies plus this study


def test_agreement_tables_span_unlabelled_parcels() -> None:
    """Agreement is wall-to-wall: unlabelled in-domain parcels stay in the denominator.

    Routing this through ``honest_eval_frame`` would silently drop them and rescale every
    agreement value onto the labelled subset alone.
    """
    mapped = _mapped()
    mapped.loc[[2, 3, 5, 6], "reference_label"] = np.nan  # predicted but unlabelled
    comparison = rpc.build_comparison_frame(_template(), mapped, _fracs(), parcel_threshold=0.5)
    scopes = ["all_ogf", "all_non_ogf", "mixed"]
    consensus = rpc.agreement_tables(comparison)["consensus"].set_index("scope")
    assert consensus.loc["all", scopes].sum() == len(comparison)  # all 8, not the 4 labelled
    counts = rpc.per_study_agreement_counts(comparison, [f"ogf_{s}" for s in rpc._STUDIES])
    assert (counts.groupby("product")["n_parcels"].sum() == len(comparison)).all()
    # Only the reference-split rows subset to the labelled parcels.
    labelled = int(comparison["reference_label"].notna().sum())
    by_ref = consensus.loc[["reference_ogf", "reference_non_ogf"], scopes]
    assert by_ref.to_numpy().sum() == labelled


def test_stratified_performance() -> None:
    comparison = rpc.build_comparison_frame(_template(), _mapped(), _fracs(), parcel_threshold=0.5)
    table = rpc.stratified_performance(comparison, strata=["corine_forest_type"])
    assert {"stratum", "value", "product", "recall", "false_positive_rate"} <= set(table.columns)
    assert (table["product"].isin([*COMPARISON_STUDIES, "ratsakatika"])).all()


def test_product_performance_drops_nan_probability_parcels() -> None:
    """A labelled parcel with no prediction (NaN probability) is excluded, not a crash."""
    mapped = _mapped()
    mapped.loc[0, "ogf_ratsakatika_probability"] = np.nan  # parcel 0 has no wall-to-wall pixel
    comparison = rpc.build_comparison_frame(_template(), mapped, _fracs(), parcel_threshold=0.5)
    oof = pd.DataFrame(
        {
            "parcel_id": np.arange(8),
            "y_true": [1, 1, 1, 1, 0, 0, 0, 0],
            "p_mean": [0.85, 0.75, 0.65, 0.5, 0.45, 0.35, 0.25, 0.15],
        }
    )
    performance = rpc.product_performance(comparison, oof, inner_threshold=0.4)
    in_sample = performance[performance["kind"] == "in_sample"].iloc[0]
    assert in_sample["n"] == 7  # the NaN-probability parcel is dropped
    assert np.isfinite(in_sample["pr_auc"])


def test_eroded_product_fracs_over_pixel_geometry(tmp_path) -> None:
    """Each product is aggregated over the shared eroded pixel index, matching the OOF geometry."""
    transform = Affine(10.0, 0.0, 0.0, 0.0, -10.0, 20.0)
    profile = {
        "driver": "GTiff",
        "height": 2,
        "width": 3,
        "count": 1,
        "dtype": "uint8",
        "crs": CRS.from_epsg(3035),
        "transform": transform,
        "nodata": 255,
    }
    masks = {
        "sabatini": np.array([[1, 1, 0], [0, 0, 0]], np.uint8),  # parcel 10 -> 1.0, parcel 20 -> 0
        "munteanu": np.array([[1, 0, 0], [0, 0, 0]], np.uint8),  # parcel 10 -> 0.5
        "kathmann": np.array([[0, 0, 0], [1, 0, 0]], np.uint8),  # parcel 20 -> 1.0
        "schickhofer": np.array([[1, 1, 1], [1, 0, 0]], np.uint8),  # parcel 10 -> 1.0, 20 -> 1.0
    }
    for study, mask in masks.items():
        with rasterio.open(tmp_path / f"{study}_ogf_3035_10m.tif", "w", **profile) as dst:
            dst.write(mask, 1)
    # parcel 10 = pixels (0,0),(0,1); parcel 20 = pixel (1,0)
    pixel_index = pd.DataFrame({"row": [0, 0, 1], "col": [0, 1, 0], "parcel_id": [10, 10, 20]})
    out = rpc.eroded_product_fracs(
        tmp_path, pixel_index, run_logger=logging.getLogger("test")
    ).set_index("parcel_id")
    assert set(out.index) == {10, 20}  # exactly the pixel-index parcels -> same set as the OOF
    assert out.loc[10, "ogf_sabatini_frac"] == 1.0
    assert out.loc[20, "ogf_sabatini_frac"] == 0.0
    assert out.loc[10, "ogf_munteanu_frac"] == 0.5
    assert out.loc[20, "ogf_kathmann_frac"] == 1.0


def test_build_eval_frame_common_set_and_eroded_swap() -> None:
    comparison = rpc.build_comparison_frame(_template(), _mapped(), _fracs(), parcel_threshold=0.5)
    evaluable = comparison[comparison["ogf_ratsakatika_probability"].notna()].copy()
    ramp = np.linspace(0.0, 1.0, 8)
    eroded = pd.DataFrame(
        {"parcel_id": np.arange(8), **{f"ogf_{s}_frac": ramp for s in COMPARISON_STUDIES}}
    )
    common = [0, 1, 2, 3, 4, 5]  # drop parcels 6 and 7
    frame = rpc.build_eval_frame(evaluable, eroded, common)
    assert set(frame["parcel_id"]) == set(common)  # restricted to the common set (drop-for-all)
    for study in COMPARISON_STUDIES:
        assert np.allclose(frame[f"ogf_{study}_frac"].to_numpy(), ramp[common])  # eroded swapped in
        assert (frame[f"ogf_{study}"].to_numpy() == (ramp[common] >= 0.5)).all()  # 0.5 verdict


def test_threshold_sensitivity_sweeps_all_products() -> None:
    comparison = rpc.build_comparison_frame(_template(), _mapped(), _fracs(), parcel_threshold=0.5)
    oof = pd.DataFrame({"parcel_id": np.arange(8), "p_mean": np.linspace(0.9, 0.1, 8)})
    honest = rpc.honest_eval_frame(comparison, oof, threshold=0.5)
    sweep = rpc.threshold_sensitivity(honest, thresholds=(0.25, 0.5, 0.75))
    assert set(sweep["product"]) == set(rpc._STUDIES)  # every product, this study included
    assert set(sweep["threshold"]) == {0.25, 0.5, 0.75}
    assert {"f1", "precision", "recall", "false_positive_rate", "n"} <= set(sweep.columns)
    # Recall is non-increasing as the threshold rises (a higher bar selects fewer positives).
    for _, group in sweep.groupby("product"):
        recall = group.sort_values("threshold")["recall"].to_numpy()
        assert (recall[1:] <= recall[:-1] + 1e-9).all()


def test_altitude_band_equal_count_tertiles() -> None:
    elevation = pd.Series(np.arange(300.0, 2400.0, 7.0))
    bands = rpc.altitude_band(elevation)
    counts = bands.value_counts()
    assert len(counts) == 3  # three tertiles
    assert counts.max() - counts.min() <= 2  # roughly equal n per band
    assert all(b.endswith(" m") and "-" in b for b in counts.index)


def test_altitude_band_uses_reference_edges() -> None:
    # Tertile edges come from the reference (1000 and 1400); every parcel is binned by them.
    elevation = pd.Series([700.0, 1100.0, 1600.0])  # one per tertile
    reference = pd.Series([600.0, 900.0, 1200.0, 1500.0, 1800.0])
    bands = rpc.altitude_band(elevation, reference=reference)
    assert bands.iloc[0] == "600-1000 m"  # below the first tertile edge
    assert bands.iloc[2] == "1400-1800 m"  # above the second tertile edge


def test_parcel_mean_elevation_excludes_nodata() -> None:
    band = np.array([[500.0, 600.0], [1000.0, -9999.0]], dtype=np.float32)
    parcel_grid = np.array([[10, 10], [20, 20]], dtype=np.int64)
    elevation = rpc.parcel_mean_elevation(band, parcel_grid)
    assert elevation[10] == 550.0
    assert elevation[20] == 1000.0  # the NoData cell is excluded


def test_honest_eval_frame_replaces_with_oof() -> None:
    comparison = rpc.build_comparison_frame(_template(), _mapped(), _fracs(), parcel_threshold=0.5)
    oof = pd.DataFrame({"parcel_id": np.arange(8), "p_mean": np.full(8, 0.1)})  # all non-OGF
    honest = rpc.honest_eval_frame(comparison, oof, threshold=0.5)
    assert not honest["ogf_ratsakatika"].any()  # OOF (0.1) overrides the in-sample verdict
    assert (honest["ogf_ratsakatika_probability"] == 0.1).all()
    assert honest["total_ogf"].between(0, 5).all()


def test_build_comparison_uses_deployed_verdict() -> None:
    mapped = _mapped()
    # Deployed verdict (>= 0.6) disagrees with the 0.5 fallback on parcel 3 (prob 0.55).
    mapped["ogf_ratsakatika"] = mapped["ogf_ratsakatika_probability"] >= 0.6
    comparison = rpc.build_comparison_frame(_template(), mapped, _fracs(), parcel_threshold=0.5)
    verdict = comparison.set_index("parcel_id")["ogf_ratsakatika"]
    assert not bool(verdict.loc[3])  # uses the deployed verdict (False), not prob >= 0.5 (True)
    assert bool(verdict.loc[2])  # prob 0.7 >= 0.6
