"""Cross-study agreement statistics for the old-growth product comparison.

Given a parcel table whose columns are the binary old-growth verdicts of several
products (the four existing studies and this study), these pure helpers quantify
where the products agree, where they disagree, and how that splits by the
reference label. They feed notebook 011: the consensus breakdown, the pairwise
agreement and Cohen's kappa matrices, the overall Fleiss' kappa, the per-parcel
vote count (``total_ogf``), and the agreement-by-reference table that shows where
each product over- or under-calls old-growth.

Importing this module binds names only; the functions are pure and perform no
input/output or logging.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd
from sklearn.metrics import cohen_kappa_score


def _binary_matrix(frame: pd.DataFrame, study_cols: Sequence[str]) -> np.ndarray:
    """Coerce the named columns to a ``(n_parcels, n_studies)`` 0/1 matrix."""
    if len(study_cols) < 2:
        raise ValueError("at least two study columns are required.")
    missing = [c for c in study_cols if c not in frame.columns]
    if missing:
        raise ValueError(f"missing study columns: {missing}.")
    matrix = frame[list(study_cols)].to_numpy()
    if matrix.shape[0] == 0:
        raise ValueError("frame is empty.")
    return matrix.astype(bool).astype(np.int64)


def total_ogf_votes(frame: pd.DataFrame, study_cols: Sequence[str]) -> pd.Series:
    """Per-parcel count of products calling old-growth (0 to ``len(study_cols)``)."""
    return pd.Series(
        _binary_matrix(frame, study_cols).sum(axis=1), index=frame.index, name="total_ogf"
    )


def consensus_breakdown(frame: pd.DataFrame, study_cols: Sequence[str]) -> dict[str, int]:
    """Counts of parcels where all products agree old-growth, all non, or mixed.

    Returns:
        ``all_ogf`` (every product positive), ``all_non_ogf`` (every product
        negative) and ``mixed`` (the products disagree).
    """
    votes = _binary_matrix(frame, study_cols).sum(axis=1)
    n_studies = len(study_cols)
    return {
        "all_ogf": int(np.sum(votes == n_studies)),
        "all_non_ogf": int(np.sum(votes == 0)),
        "mixed": int(np.sum((votes > 0) & (votes < n_studies))),
    }


def pairwise_agreement(frame: pd.DataFrame, study_cols: Sequence[str]) -> pd.DataFrame:
    """Fraction of parcels on which each pair of products agrees (square matrix)."""
    matrix = _binary_matrix(frame, study_cols)
    n = len(study_cols)
    out = np.eye(n, dtype=np.float64)
    for i in range(n):
        for j in range(i + 1, n):
            agree = float(np.mean(matrix[:, i] == matrix[:, j]))
            out[i, j] = out[j, i] = agree
    return pd.DataFrame(out, index=list(study_cols), columns=list(study_cols))


def cohen_kappa_matrix(frame: pd.DataFrame, study_cols: Sequence[str]) -> pd.DataFrame:
    """Pairwise Cohen's kappa between products (square matrix, 1.0 on the diagonal)."""
    matrix = _binary_matrix(frame, study_cols)
    n = len(study_cols)
    out = np.eye(n, dtype=np.float64)
    for i in range(n):
        for j in range(i + 1, n):
            if matrix[:, i].std() == 0 and matrix[:, j].std() == 0:
                kappa = 1.0 if np.array_equal(matrix[:, i], matrix[:, j]) else 0.0
            else:
                kappa = float(cohen_kappa_score(matrix[:, i], matrix[:, j]))
            out[i, j] = out[j, i] = kappa
    return pd.DataFrame(out, index=list(study_cols), columns=list(study_cols))


def fleiss_kappa(frame: pd.DataFrame, study_cols: Sequence[str]) -> float:
    """Fleiss' kappa for the binary verdicts across all products (chance-corrected).

    Treats the products as ``len(study_cols)`` raters of each parcel into
    old-growth / not, per Fleiss (1971).
    """
    matrix = _binary_matrix(frame, study_cols)
    n_raters = matrix.shape[1]
    positives = matrix.sum(axis=1)
    counts = np.column_stack([n_raters - positives, positives]).astype(np.float64)
    agreement = (counts**2).sum(axis=1) - n_raters
    p_i = agreement / (n_raters * (n_raters - 1))
    p_bar = float(p_i.mean())
    p_class = counts.sum(axis=0) / counts.sum()
    p_e = float((p_class**2).sum())
    if np.isclose(p_e, 1.0):
        return 1.0
    return (p_bar - p_e) / (1.0 - p_e)


def agreement_by_reference(
    frame: pd.DataFrame, study_cols: Sequence[str], reference_col: str
) -> pd.DataFrame:
    """Per-product positive rate split by the reference label.

    For each product, the fraction of parcels it calls old-growth among the
    reference old-growth parcels (its recall) and among the reference non-old-
    growth parcels (its false-positive rate); the gap between the two is its
    discriminating power. ``n`` rows record the reference counts.

    Args:
        frame: Parcel table with the study columns and the reference column.
        study_cols: The binary product columns.
        reference_col: The binary reference-label column (1 = old-growth).

    Returns:
        One row per product with ``frac_ogf_ref_ogf``, ``frac_ogf_ref_non`` and
        ``separation`` (their difference), plus a leading ``n`` row.
    """
    if reference_col not in frame.columns:
        raise ValueError(f"missing reference column {reference_col!r}.")
    _binary_matrix(frame, study_cols)  # validation
    ref = frame[reference_col].to_numpy().astype(bool)
    rows: list[dict[str, float | str]] = []
    for col in study_cols:
        pred = frame[col].to_numpy().astype(bool)
        frac_ogf = float(pred[ref].mean()) if ref.any() else float("nan")
        frac_non = float(pred[~ref].mean()) if (~ref).any() else float("nan")
        rows.append(
            {
                "product": col,
                "frac_ogf_ref_ogf": frac_ogf,
                "frac_ogf_ref_non": frac_non,
                "separation": frac_ogf - frac_non,
            }
        )
    table = pd.DataFrame(rows)
    table.attrs["n_ref_ogf"] = int(ref.sum())
    table.attrs["n_ref_non"] = int((~ref).sum())
    return table


def per_study_agreement_counts(frame: pd.DataFrame, study_cols: Sequence[str]) -> pd.DataFrame:
    """How many of the other products agree with each product, per parcel.

    For each product and parcel, count the other products that share its old-growth
    verdict, then tabulate how many parcels fall at each agreement count.

    Args:
        frame: Parcel table with the binary product columns.
        study_cols: The binary product columns.

    Returns:
        One row per ``(product, n_agree)`` with ``n_parcels``; ``n_agree`` ranges from 0
        to ``len(study_cols) - 1`` (the number of other products sharing the verdict).
    """
    matrix = _binary_matrix(frame, study_cols)
    n_studies = len(study_cols)
    rows: list[dict[str, object]] = []
    for column, product in enumerate(study_cols):
        agree = (matrix == matrix[:, [column]]).sum(axis=1) - 1
        counts = np.bincount(agree, minlength=n_studies)
        rows.extend(
            {"product": product, "n_agree": k, "n_parcels": int(counts[k])}
            for k in range(n_studies)
        )
    return pd.DataFrame(rows)


__all__ = [
    "agreement_by_reference",
    "cohen_kappa_matrix",
    "consensus_breakdown",
    "fleiss_kappa",
    "pairwise_agreement",
    "per_study_agreement_counts",
    "total_ogf_votes",
]
