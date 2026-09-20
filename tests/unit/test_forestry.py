import math

import pytest

from utils.forestry import (
    CompositionMetrics,
    classify_composition,
    classify_forest_type,
    composition_metrics,
    fix_composition_anomalies,
    normalise_stand_age,
    standardise_composition,
    to_delimited_pairs,
)


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("10MO", "10MO"),
        ("6FA4MO", "6FA 4MO"),
        ("6MO1BR1FA1ME1SAC", "6MO 1BR 1FA 1ME 1SAC"),
        ("5MO 3BR 2FA", "5MO 3BR 2FA"),
        ("CLEARCUT", "CLEARCUT"),
        ("Taiere rasa", "CLEARCUT"),
        ("10M0", "10MO"),
        ("9NO", "9MO"),
        ("9NO 1FA", "9MO 1FA"),
        ("9MO-1JN", None),
        ("9NO1FA", None),
        ("100", None),
        ("", None),
        ("nan", None),
        ("7MO3", None),
    ],
)
def test_standardise_composition(raw, expected):
    assert standardise_composition(raw) == expected


_VALID = frozenset({"MO", "FA", "BR", "GI"})


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("9MO1FA", "9MO 1FA"),
        ("6FA4MO", "6FA 4MO"),
        ("10MO", "10MO"),
        ("CLEARCUT", "CLEARCUT"),
        ("Taiere rasa", "CLEARCUT"),
        ("8MO2JN", None),
        ("4IA6FA", None),
        ("5MOFA", None),
        ("100", None),
        ("", None),
    ],
)
def test_standardise_composition_validates(raw, expected):
    assert standardise_composition(raw, valid_codes=_VALID) == expected


@pytest.mark.parametrize(
    ("raw", "expected"),
    [(0, None), ("0", None), (-5, None), (85, 85.0), ("120", 120.0), (None, None), ("", None)],
)
def test_normalise_stand_age(raw, expected):
    assert normalise_stand_age(raw) == expected


def test_normalise_stand_age_nan():
    assert normalise_stand_age(math.nan) is None


@pytest.mark.parametrize(
    ("composition", "expected"),
    [
        ("10MO", CompositionMetrics(100.0, 0.0, "MO")),
        ("6FA 4MO", CompositionMetrics(40.0, 60.0, "FA")),
        ("6MO 4FA", CompositionMetrics(60.0, 40.0, "MO")),
        ("6MO 1BR 1FA 1ME 1SAC", CompositionMetrics(70.0, 30.0, "MO")),
        ("5FA 5MO", CompositionMetrics(50.0, 50.0, "FA")),
        ("CLEARCUT", CompositionMetrics(None, None, None)),
        (None, CompositionMetrics(None, None, None)),
        ("", CompositionMetrics(None, None, None)),
    ],
)
def test_composition_metrics(composition, expected):
    assert composition_metrics(composition) == expected


def test_classify_forest_type_pure_and_boundary() -> None:
    assert classify_forest_type(80.0, 20.0) == "broadleaf"
    assert classify_forest_type(20.0, 80.0) == "coniferous"
    assert classify_forest_type(70.0, 30.0) == "broadleaf"
    assert classify_forest_type(30.0, 70.0) == "coniferous"


def test_classify_forest_type_mixed() -> None:
    assert classify_forest_type(60.0, 40.0) == "mixed"
    assert classify_forest_type(50.0, 50.0) == "mixed"


def test_classify_forest_type_missing_returns_none() -> None:
    assert classify_forest_type(None, 50.0) is None
    assert classify_forest_type(50.0, None) is None
    assert classify_forest_type(float("nan"), 50.0) is None
    assert classify_forest_type(50.0, float("nan")) is None


def test_classify_forest_type_respects_custom_threshold() -> None:
    assert classify_forest_type(60.0, 40.0, dominance_pct=50) == "broadleaf"
    assert classify_forest_type(80.0, 20.0, dominance_pct=90) == "mixed"


def test_fix_composition_anomalies_passes_through_non_strings() -> None:
    assert fix_composition_anomalies(123) == 123  # type: ignore[arg-type]


def test_classify_composition_returns_unknown_for_unparseable() -> None:
    assert classify_composition("ABCDEF") == "UNKNOWN"


def test_to_delimited_pairs_returns_empty_without_pairs() -> None:
    assert to_delimited_pairs("no pairs") == ""


def test_composition_metrics_zero_total_returns_none() -> None:
    assert composition_metrics("0MO") == CompositionMetrics(None, None, None)
