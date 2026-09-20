"""Unit tests for the canonical names and constants in utils.terminology."""

import re
from dataclasses import FrozenInstanceError

import pytest

from utils.terminology import (
    _ALPHAEARTH_BANDS,
    _BASELINE_BANDS,
    _HRVPP_BANDS,
    _TESSERA_BANDS,
    _WORLDCOVER_BANDS,
    ARCHITECTURES,
    BAND_LABELS,
    BOOTSTRAP_N_UNITS,
    CATEGORY_DISPLAY,
    COMPARISON_STUDIES,
    CRS,
    DISPLAY_CRS,
    DISPLAY_CRS_NAME,
    FEATURE_SET_BANDS,
    FEATURE_SETS,
    FOLD_IDS,
    N_FOLDS,
    N_HP_TRIALS,
    N_HP_TRIALS_XGBOOST,
    PALETTE_CATEGORICAL,
    PALETTE_CATEGORICAL_ORDER,
    SEED,
    ComparisonStudy,
)

_HEX_RE = re.compile(r"^#[0-9A-Fa-f]{6}$")
_VALID_FOLDS = {1, 2, 3, 4, 5, 6}

# Union of every band name across all feature sets; the basis for label coverage.
_ALL_BANDS = {band for bands in FEATURE_SET_BANDS.values() for band in bands}


def test_seed_is_42() -> None:
    assert SEED == 42


def test_analytical_crs_is_equal_area_laea() -> None:
    assert CRS == "EPSG:3035"


def test_bootstrap_unit_count_is_pinned() -> None:
    assert isinstance(BOOTSTRAP_N_UNITS, int)
    assert BOOTSTRAP_N_UNITS == 20


def test_hp_trial_budget_is_pinned() -> None:
    # The shared XGBoost/CNN inner-search budget per outer fold.
    assert isinstance(N_HP_TRIALS, int)
    assert N_HP_TRIALS == 30
    assert N_HP_TRIALS_XGBOOST == 50  # main XGBoost nested CV (manuscript Section 2.7)


def test_display_crs_constants_exist() -> None:
    # DISPLAY_CRS is for rendering maps upright only; all computation stays on CRS.
    assert DISPLAY_CRS == "EPSG:32635"
    assert isinstance(DISPLAY_CRS_NAME, str) and DISPLAY_CRS_NAME.strip()
    assert DISPLAY_CRS != CRS


def test_fold_ids_are_six_values_in_one_to_six() -> None:
    assert len(FOLD_IDS) == 6
    assert all(fold in _VALID_FOLDS for fold in FOLD_IDS)
    assert set(FOLD_IDS) == _VALID_FOLDS
    assert N_FOLDS == len(FOLD_IDS)


def test_feature_sets_have_expected_keys() -> None:
    assert set(FEATURE_SETS) == {
        "baseline",
        "baseline_conventional_eo",
        "baseline_tessera",
        "baseline_alphaearth",
    }


def test_feature_set_bands_keys_match_feature_sets() -> None:
    # Guards the rename and prevents drift between the two feature-set constants.
    assert set(FEATURE_SET_BANDS) == set(FEATURE_SETS)


def test_feature_set_bands_have_expected_lengths() -> None:
    assert len(FEATURE_SET_BANDS["baseline"]) == 6
    assert len(FEATURE_SET_BANDS["baseline_conventional_eo"]) == 28
    assert len(FEATURE_SET_BANDS["baseline_tessera"]) == 134
    assert len(FEATURE_SET_BANDS["baseline_alphaearth"]) == 70


def test_feature_set_bands_are_unique_within_each_set() -> None:
    for name, bands in FEATURE_SET_BANDS.items():
        assert len(set(bands)) == len(bands), name


def test_feature_set_bands_start_with_baseline_in_order() -> None:
    for name, bands in FEATURE_SET_BANDS.items():
        assert bands[:6] == _BASELINE_BANDS, name


def test_feature_set_bands_compose_as_documented() -> None:
    assert FEATURE_SET_BANDS["baseline"] == _BASELINE_BANDS
    assert (
        FEATURE_SET_BANDS["baseline_conventional_eo"]
        == _BASELINE_BANDS + _WORLDCOVER_BANDS + _HRVPP_BANDS
    )
    assert FEATURE_SET_BANDS["baseline_tessera"] == _BASELINE_BANDS + _TESSERA_BANDS
    assert FEATURE_SET_BANDS["baseline_alphaearth"] == _BASELINE_BANDS + _ALPHAEARTH_BANDS


def test_feature_set_bands_are_immutable() -> None:
    # MappingProxyType rejects item assignment, and the tuple values reject it too.
    with pytest.raises(TypeError):
        FEATURE_SET_BANDS["baseline"] = ()  # type: ignore[index]
    with pytest.raises(TypeError):
        FEATURE_SET_BANDS["baseline"][0] = "mutated"  # type: ignore[index]


def test_band_labels_keys_equal_feature_set_bands() -> None:
    # Every band has exactly one label and there are no orphan labels.
    assert set(BAND_LABELS) == _ALL_BANDS


def test_band_labels_count() -> None:
    assert len(BAND_LABELS) == 6 + 12 + 10 + 128 + 64  # 220 unique bands


def test_band_labels_are_non_empty_strings() -> None:
    for name, label in BAND_LABELS.items():
        assert isinstance(label, str) and label.strip(), name


def test_band_labels_match_hrvpp_product_definitions() -> None:
    # Lock the HR-VPP labels whose meaning is easy to transpose (Copernicus
    # HR-VPP Product User Manual, issue 2.5): lslope is the rising green-up limb,
    # rslope the falling green-down limb; sprod is the seasonal (small) integral.
    assert BAND_LABELS["vpp_lslope"] == "Green-up slope"
    assert BAND_LABELS["vpp_rslope"] == "Green-down slope"
    assert BAND_LABELS["vpp_sosd"] == "Start-of-season date"
    assert BAND_LABELS["vpp_eosd"] == "End-of-season date"
    assert BAND_LABELS["vpp_sprod"] == "Seasonal productivity"
    assert BAND_LABELS["vpp_ampl"] == "Seasonal amplitude"


def test_band_labels_for_embeddings_follow_index_pattern() -> None:
    assert BAND_LABELS["tessera_t000"] == "TESSERA dimension 0"
    assert BAND_LABELS["tessera_t127"] == "TESSERA dimension 127"
    assert BAND_LABELS["alphaearth_a00"] == "AlphaEarth dimension 0"
    assert BAND_LABELS["alphaearth_a63"] == "AlphaEarth dimension 63"


def test_band_labels_are_immutable() -> None:
    with pytest.raises(TypeError):
        BAND_LABELS["elevation_m"] = "mutated"  # type: ignore[index]


def test_architectures_have_expected_keys() -> None:
    assert set(ARCHITECTURES) == {"xgboost", "cnn_3x3", "cnn_5x5", "cnn_7x7"}


def test_comparison_studies_have_expected_keys() -> None:
    assert set(COMPARISON_STUDIES) == {"sabatini", "munteanu", "kathmann", "schickhofer"}


def test_comparison_study_categories_resolve_to_display_names() -> None:
    for study in COMPARISON_STUDIES.values():
        assert study.category in CATEGORY_DISPLAY


def test_palette_categorical_has_six_named_hex_strings() -> None:
    assert set(PALETTE_CATEGORICAL) == {
        "blue",
        "teal",
        "light_green",
        "yellow",
        "orange",
        "magenta",
    }
    for colour in PALETTE_CATEGORICAL.values():
        assert _HEX_RE.match(colour) is not None, colour


def test_palette_categorical_order_matches_dict_values() -> None:
    assert PALETTE_CATEGORICAL_ORDER == tuple(PALETTE_CATEGORICAL.values())
    assert len(PALETTE_CATEGORICAL_ORDER) == 6


def test_mapping_constants_are_read_only() -> None:
    for mapping, key, value in (
        (FEATURE_SETS, "baseline", "mutated"),
        (ARCHITECTURES, "xgboost", "mutated"),
        (CATEGORY_DISPLAY, "predictive", "mutated"),
        (COMPARISON_STUDIES, "sabatini", ComparisonStudy("mutated", "predictive")),
        (PALETTE_CATEGORICAL, "blue", "#000000"),
    ):
        with pytest.raises(TypeError):
            mapping[key] = value


def test_comparison_study_records_are_frozen() -> None:
    study = COMPARISON_STUDIES["sabatini"]
    with pytest.raises(FrozenInstanceError):
        study.display = "mutated"


def test_cnn_training_budgets_are_pinned() -> None:
    from utils.terminology import (
        CNN_EARLY_STOP_MIN_DELTA,
        CNN_MAX_EPOCHS,
        CNN_PATIENCE,
        CNN_WARMUP_EPOCHS,
    )

    assert CNN_MAX_EPOCHS == 40
    assert CNN_PATIENCE == 7
    assert CNN_WARMUP_EPOCHS == 5
    assert CNN_EARLY_STOP_MIN_DELTA == pytest.approx(1e-4)


def test_cnn_patch_sizes_derive_from_architectures() -> None:
    from utils.terminology import ARCHITECTURES, CNN_PATCH_SIZES

    assert CNN_PATCH_SIZES == (3, 5, 7)
    cnn_archs = [name for name in ARCHITECTURES if name.startswith("cnn_")]
    assert len(cnn_archs) == len(CNN_PATCH_SIZES)
    for name, size in zip(cnn_archs, CNN_PATCH_SIZES, strict=False):
        assert name == f"cnn_{size}x{size}"
