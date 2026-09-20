"""Cleaning of Romanian forestry composition strings and stand ages."""

from __future__ import annotations

import math
import re
from typing import NamedTuple

from utils.terminology import FOREST_TYPE_DOMINANCE_PCT, SPECIES, SPECIES_GROUP

_VALID_SPECIES_CODES = frozenset(SPECIES)
_PAIR = re.compile(r"(\d{1,2})([A-Z]{2,4})")
_SPECIES_CORRECTIONS = {"OMO": "MO", "MP": "MO", "NO": "MO", "GA": "GI"}


def clean_composition(value: str) -> str:
    """Uppercase a raw composition string and normalise its spacing and dashes."""
    value = str(value).upper().strip()
    value = re.sub(r"\s+", " ", value)
    value = re.sub(r"(\d)\s+([A-Z])", r"\1\2", value)
    value = re.sub(r"([A-Z])\s+(\d)", r"\1 \2", value)
    value = re.sub(r"[\u2013\u2014\u2212]", "-", value)
    value = re.sub(r"[-\s]+", "-", value).strip("-")
    return value


def fix_composition_anomalies(value: str) -> str:
    """Correct known typos: a digit-zero misread for O, and miscoded species at a token boundary."""
    if not isinstance(value, str):
        return value
    value = value.strip().upper()
    if value in {"NONE", "NAN", ""}:
        return ""
    if value == "10M0":
        return "10MO"
    for wrong, correct in _SPECIES_CORRECTIONS.items():
        value = re.sub(rf"(\d{{1,2}}){wrong}\b", rf"\g<1>{correct}", value)
    return value


def classify_composition(value: str) -> str:
    """Categorise a cleaned composition string by its structural pattern."""
    if value.strip() == "":
        return "EMPTY"
    if re.fullmatch(r"\d{1,3}", value):
        return "NUMERIC_ONLY"
    if value == "CLEARCUT" or re.search(r"TAIE|RASE", value):
        return "CLEARCUT"
    if re.fullmatch(r"\d{1,2}[A-Z]{2,4}", value):
        return "SINGLE_PAIR"
    if re.fullmatch(r"(\d{1,2}[A-Z]{2,4}){2,}", value):
        return "CONCATENATED_PAIRS"
    if re.fullmatch(r"(\d{1,2}[A-Z]{2,4}[\s\-]*){2,}", value):
        return "DELIMITED_PAIRS"
    if re.search(r"\d+[A-Z]{2,4}", value):
        return "PARTIAL_PAIRING"
    return "UNKNOWN"


def to_delimited_pairs(value: str) -> str:
    """Re-emit proportion-species pairs as space-delimited tokens, largest share first."""
    pairs = _PAIR.findall(value)
    if not pairs:
        return ""
    pairs = sorted(pairs, key=lambda pair: (-int(pair[0]), pair[1]))
    return " ".join(f"{share}{code}" for share, code in pairs)


def standardise_composition(
    value: str, *, valid_codes: frozenset[str] = _VALID_SPECIES_CODES
) -> str | None:
    """Clean a raw composition string to a valid canonical form, ``"CLEARCUT"``, or ``None``.

    Returns ``"CLEARCUT"`` for a clear-felled record. Returns a canonical
    ``"<share><CODE>"`` string (shares ordered descending) only when every code is
    present in ``valid_codes``; if any code is unrecognised, or the value cannot be
    parsed into proportion-species pairs, returns ``None`` so the field is blanked
    while the record is kept.
    """
    cleaned = fix_composition_anomalies(clean_composition(value))
    pattern = classify_composition(cleaned)
    if pattern == "CLEARCUT":
        return "CLEARCUT"
    if pattern in {"SINGLE_PAIR", "CONCATENATED_PAIRS", "DELIMITED_PAIRS"}:
        codes = [code for _, code in _PAIR.findall(cleaned)]
        if not codes or any(code not in valid_codes for code in codes):
            return None
        return to_delimited_pairs(cleaned)
    return None


class CompositionMetrics(NamedTuple):
    """Coniferous and broadleaf percentages and the dominant species code of a stand."""

    pct_coniferous: float | None
    pct_broadleaf: float | None
    dominant_code: str | None


def composition_metrics(composition: str | None) -> CompositionMetrics:
    """Summarise a canonical composition string by species group and dominant species.

    Parses a ``"<share><CODE>"`` string (as produced by :func:`standardise_composition`)
    and sums the shares of coniferous and broadleaf species using
    :data:`utils.terminology.SPECIES_GROUP`. Percentages are normalised by the total
    share, so a tenths-based composition such as ``"6FA 4MO"`` yields 60 and 40. The
    dominant code is the species holding the greatest summed share.

    Args:
        composition: A canonical composition string, ``"CLEARCUT"``, ``None`` or any
            value without recognised proportion-species pairs.

    Returns:
        A :class:`CompositionMetrics`. The percentages and dominant code are all
        ``None`` when the input carries no recognised proportion-species pairs.
    """
    if not isinstance(composition, str):
        return CompositionMetrics(None, None, None)
    pairs = _PAIR.findall(composition)
    if not pairs:
        return CompositionMetrics(None, None, None)
    shares: dict[str, int] = {}
    coniferous = broadleaf = total = 0
    for share_str, code in pairs:
        share = int(share_str)
        shares[code] = shares.get(code, 0) + share
        group = SPECIES_GROUP.get(code)
        if group == "coniferous":
            coniferous += share
        elif group == "broadleaf":
            broadleaf += share
        total += share
    if total == 0:
        return CompositionMetrics(None, None, None)
    return CompositionMetrics(
        round(100 * coniferous / total, 2),
        round(100 * broadleaf / total, 2),
        max(shares, key=lambda code: shares[code]),
    )


def normalise_stand_age(value: object) -> float | None:
    """Return a positive stand age as a float, treating zero, blanks and non-numbers as missing."""
    try:
        age = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None
    if math.isnan(age) or age <= 0:
        return None
    return age


def classify_forest_type(
    pct_broadleaf: float | None,
    pct_coniferous: float | None,
    *,
    dominance_pct: float = FOREST_TYPE_DOMINANCE_PCT,
) -> str | None:
    """Classify a parcel as broadleaf, coniferous or mixed from its group shares.

    The parcel is ``"broadleaf"`` or ``"coniferous"`` when that group's percentage
    reaches ``dominance_pct`` and ``"mixed"`` otherwise. Returns ``None`` when either
    share is missing, leaving parcels without species composition unclassified.

    Args:
        pct_broadleaf: Broadleaf share of the stand in percent (0-100).
        pct_coniferous: Coniferous share of the stand in percent (0-100).
        dominance_pct: Minimum percentage for a pure class; defaults to
            :data:`utils.terminology.FOREST_TYPE_DOMINANCE_PCT`.

    Returns:
        ``"broadleaf"``, ``"coniferous"``, ``"mixed"`` or ``None``.
    """
    if pct_broadleaf is None or pct_coniferous is None:
        return None
    if math.isnan(pct_broadleaf) or math.isnan(pct_coniferous):
        return None
    if pct_broadleaf >= dominance_pct:
        return "broadleaf"
    if pct_coniferous >= dominance_pct:
        return "coniferous"
    return "mixed"
