"""Unit tests for utils.comparison (cross-study agreement statistics)."""

import numpy as np
import pandas as pd
import pytest

from utils.comparison import (
    agreement_by_reference,
    cohen_kappa_matrix,
    consensus_breakdown,
    fleiss_kappa,
    pairwise_agreement,
    total_ogf_votes,
)

_STUDIES = ["study_a", "study_b", "study_c"]


def _frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "study_a": [1, 0, 1, 1],
            "study_b": [1, 0, 0, 1],
            "study_c": [1, 0, 1, 0],
            "reference": [1, 0, 1, 0],
        }
    ).astype(bool)


def test_total_ogf_votes() -> None:
    votes = total_ogf_votes(_frame(), _STUDIES)
    assert list(votes) == [3, 0, 2, 2]
    assert votes.name == "total_ogf"


def test_consensus_breakdown() -> None:
    assert consensus_breakdown(_frame(), _STUDIES) == {"all_ogf": 1, "all_non_ogf": 1, "mixed": 2}


def test_pairwise_agreement() -> None:
    agree = pairwise_agreement(_frame(), _STUDIES)
    assert agree.loc["study_a", "study_a"] == 1.0
    assert agree.loc["study_a", "study_b"] == pytest.approx(0.75)
    assert agree.loc["study_a", "study_b"] == agree.loc["study_b", "study_a"]


def test_cohen_kappa_matrix_diagonal_and_symmetry() -> None:
    kappa = cohen_kappa_matrix(_frame(), _STUDIES)
    assert np.allclose(np.diag(kappa.to_numpy()), 1.0)
    assert np.allclose(kappa.to_numpy(), kappa.to_numpy().T)


def test_fleiss_kappa_perfect_agreement() -> None:
    same = pd.DataFrame({c: [1, 0, 1, 0, 1] for c in _STUDIES})
    assert fleiss_kappa(same, _STUDIES) == pytest.approx(1.0)
    # A mixed frame yields a finite kappa in range.
    value = fleiss_kappa(_frame(), _STUDIES)
    assert -1.0 <= value <= 1.0


def test_agreement_by_reference() -> None:
    table = agreement_by_reference(_frame(), _STUDIES, "reference")
    row_a = table[table["product"] == "study_a"].iloc[0]
    assert row_a["frac_ogf_ref_ogf"] == pytest.approx(1.0)  # predicts ogf on both ref-ogf parcels
    assert row_a["frac_ogf_ref_non"] == pytest.approx(0.5)  # one of two ref-non parcels
    assert row_a["separation"] == pytest.approx(0.5)
    assert table.attrs["n_ref_ogf"] == 2
    assert table.attrs["n_ref_non"] == 2


def test_constant_column_kappa_is_handled() -> None:
    frame = pd.DataFrame({"study_a": [1, 1, 1, 1], "study_b": [1, 1, 1, 1]})
    kappa = cohen_kappa_matrix(frame, ["study_a", "study_b"])
    assert kappa.loc["study_a", "study_b"] == 1.0  # identical constant columns agree


def test_fleiss_kappa_all_one_class() -> None:
    # Every product calls every parcel old-growth: agreement is total but chance is too.
    frame = pd.DataFrame({c: [1, 1, 1] for c in _STUDIES})
    assert fleiss_kappa(frame, _STUDIES) == pytest.approx(1.0)


def test_agreement_by_reference_missing_column() -> None:
    with pytest.raises(ValueError, match="missing reference column"):
        agreement_by_reference(_frame(), _STUDIES, "absent")


def test_empty_frame_rejected() -> None:
    empty = pd.DataFrame({c: pd.Series(dtype=bool) for c in _STUDIES})
    with pytest.raises(ValueError, match="empty"):
        total_ogf_votes(empty, _STUDIES)


@pytest.mark.parametrize(
    "cols, match",
    [(["study_a"], "at least two"), (["study_a", "missing"], "missing study columns")],
)
def test_validation(cols: list[str], match: str) -> None:
    with pytest.raises(ValueError, match=match):
        total_ogf_votes(_frame(), cols)


def test_per_study_agreement_counts() -> None:
    from utils.comparison import per_study_agreement_counts

    table = per_study_agreement_counts(_frame(), _STUDIES)
    # study_a: p1,p2 agree with both others (2); p3,p4 agree with one (1); none with zero.
    a = table[table["product"] == "study_a"].set_index("n_agree")["n_parcels"]
    assert a.loc[0] == 0 and a.loc[1] == 2 and a.loc[2] == 2
    assert set(table["n_agree"]) == {0, 1, 2}  # n-1 = 2 others for 3 studies
    assert int(table["n_parcels"].sum()) == len(_frame()) * len(_STUDIES)
