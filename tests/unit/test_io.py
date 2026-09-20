"""Unit tests for the schema-validated I/O helpers in utils.io."""

import importlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import utils.io
from utils.io import (
    FEATURE_IMPORTANCE_SCHEMA,
    HP_BOUNDARY_SCHEMA,
    HP_TRIALS_SCHEMA,
    PARCEL_PREDICTIONS_SCHEMA,
    PER_FOLD_METRICS_SCHEMA,
    PIXEL_INDEX_SCHEMA,
    PIXEL_PREDICTIONS_SCHEMA,
    PIXEL_VALID_SCHEMA,
    POOLED_METRICS_SCHEMA,
    SUMMARY_METRICS_SCHEMA,
    ColumnSpec,
    Schema,
    validate_schema,
    write_csv,
    write_json,
    write_parquet,
)


def _valid_frame() -> pd.DataFrame:
    """Build a frame that matches :func:`_valid_schema`."""
    return pd.DataFrame(
        {
            "id": pd.Series([1, 2, 3], dtype="int64"),
            "value": pd.Series([0.1, 0.2, 0.3], dtype="float64"),
            "flag": pd.Series([True, False, True], dtype="bool"),
            "label": pd.Series(["a", "b", "c"], dtype="object"),
        }
    )


def _valid_schema() -> dict[str, ColumnSpec]:
    """Schema matching :func:`_valid_frame`."""
    return {
        "id": ColumnSpec("int64", nullable=False),
        "value": ColumnSpec("float64"),
        "flag": ColumnSpec("bool"),
        "label": ColumnSpec(("object", "string")),
    }


def test_importing_io_creates_nothing(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    # Reloading re-executes the module body in an empty working directory; if any
    # file or folder were created at import time it would appear here.
    monkeypatch.chdir(tmp_path)
    module = importlib.reload(utils.io)

    assert hasattr(module, "validate_schema")
    assert list(tmp_path.iterdir()) == []


def test_validate_schema_passes_for_matching_frame() -> None:
    # Should not raise.
    validate_schema(_valid_frame(), _valid_schema())


def test_validate_schema_accepts_tuple_of_dtypes() -> None:
    df = pd.DataFrame({"label": pd.Series(["a", "b"], dtype="string")})
    schema = {"label": ColumnSpec(("object", "string"))}

    validate_schema(df, schema)


def test_validate_schema_reports_all_problems_together() -> None:
    df = pd.DataFrame(
        {
            # Wrong dtype (object, expected int64) and a missing value while the
            # column is declared non-nullable.
            "id": pd.Series([1, None, 3], dtype="object"),
            "value": pd.Series([0.1, 0.2, 0.3], dtype="float64"),
            # Not in the schema -> unexpected column.
            "extra": pd.Series([1, 2, 3], dtype="int64"),
        }
    )
    schema = {
        "id": ColumnSpec("int64", nullable=False),
        "value": ColumnSpec("float64"),
        # Absent from df -> missing column.
        "label": ColumnSpec("string"),
    }

    with pytest.raises(ValueError) as excinfo:
        validate_schema(df, schema)

    message = str(excinfo.value)
    assert "missing required columns" in message
    assert "label" in message
    assert "unexpected columns" in message
    assert "extra" in message
    assert "dtype" in message
    assert "int64" in message
    assert "non-nullable" in message


def test_validate_schema_rejects_extra_columns_by_default() -> None:
    df = _valid_frame()
    df["surplus"] = pd.Series([1, 2, 3], dtype="int64")

    with pytest.raises(ValueError, match="unexpected columns"):
        validate_schema(df, _valid_schema())


def test_validate_schema_allow_extra_permits_extra_columns() -> None:
    df = _valid_frame()
    df["surplus"] = pd.Series([1, 2, 3], dtype="int64")

    # Should not raise when extra columns are explicitly allowed.
    validate_schema(df, _valid_schema(), allow_extra=True)


def test_write_csv_validates_and_writes(tmp_path: Path) -> None:
    target = tmp_path / "nested" / "out.csv"

    returned = write_csv(_valid_frame(), target, _valid_schema())

    assert returned == target
    assert target.is_file()
    read_back = pd.read_csv(target)
    assert list(read_back.columns) == ["id", "value", "flag", "label"]
    assert len(read_back) == 3


def test_write_csv_rejects_invalid_frame_without_writing(tmp_path: Path) -> None:
    target = tmp_path / "nested" / "bad.csv"
    bad = _valid_frame().drop(columns=["label"])

    with pytest.raises(ValueError, match="missing required columns"):
        write_csv(bad, target, _valid_schema())

    assert not target.exists()


def test_write_json_writes_sorted_indented_utf8(tmp_path: Path) -> None:
    target = tmp_path / "meta" / "run_metadata.json"
    data = {"beta": 1, "alpha": {"delta": 4, "charlie": 3}, "note": "façade"}

    returned = write_json(data, target)

    assert returned == target
    text = target.read_text(encoding="utf-8")
    # Round-trips to the same object.
    assert json.loads(text) == data
    # Indented by two spaces.
    assert "\n  " in text
    # Keys are sorted at both levels.
    assert text.index('"alpha"') < text.index('"beta"') < text.index('"note"')
    assert text.index('"charlie"') < text.index('"delta"')
    # Non-ASCII written as real UTF-8, not escaped.
    assert "façade" in text


def test_write_json_is_deterministic(tmp_path: Path) -> None:
    first = write_json({"b": 2, "a": 1}, tmp_path / "a.json")
    second = write_json({"a": 1, "b": 2}, tmp_path / "b.json")

    assert first.read_text(encoding="utf-8") == second.read_text(encoding="utf-8")


def test_write_parquet_rejects_invalid_frame_without_writing(tmp_path: Path) -> None:
    # Validation runs before any parquet engine is needed, so this holds even
    # where no engine is installed.
    target = tmp_path / "nested" / "bad.parquet"
    bad = _valid_frame().drop(columns=["label"])

    with pytest.raises(ValueError, match="missing required columns"):
        write_parquet(bad, target, _valid_schema())

    assert not target.exists()


def test_write_parquet_validates_and_writes(tmp_path: Path) -> None:
    pytest.importorskip("pyarrow")
    target = tmp_path / "nested" / "out.parquet"

    returned = write_parquet(_valid_frame(), target, _valid_schema())

    assert returned == target
    assert target.is_file()
    read_back = pd.read_parquet(target)
    pd.testing.assert_frame_equal(read_back, _valid_frame())


# --------------------------------------------------------------------------- #
# Nested cross-validation pipeline schemas.
# --------------------------------------------------------------------------- #

_NESTED_CV_SCHEMAS = {
    "PIXEL_INDEX_SCHEMA": PIXEL_INDEX_SCHEMA,
    "PIXEL_VALID_SCHEMA": PIXEL_VALID_SCHEMA,
    "PIXEL_PREDICTIONS_SCHEMA": PIXEL_PREDICTIONS_SCHEMA,
    "PARCEL_PREDICTIONS_SCHEMA": PARCEL_PREDICTIONS_SCHEMA,
    "HP_TRIALS_SCHEMA": HP_TRIALS_SCHEMA,
    "HP_BOUNDARY_SCHEMA": HP_BOUNDARY_SCHEMA,
    "FEATURE_IMPORTANCE_SCHEMA": FEATURE_IMPORTANCE_SCHEMA,
    "PER_FOLD_METRICS_SCHEMA": PER_FOLD_METRICS_SCHEMA,
    "SUMMARY_METRICS_SCHEMA": SUMMARY_METRICS_SCHEMA,
    "POOLED_METRICS_SCHEMA": POOLED_METRICS_SCHEMA,
}


def test_nested_cv_schemas_are_well_formed_and_exported() -> None:
    for name, schema in _NESTED_CV_SCHEMAS.items():
        assert isinstance(schema, dict) and schema, name
        assert all(isinstance(column, str) for column in schema), name
        assert all(isinstance(spec, ColumnSpec) for spec in schema.values()), name
        assert name in utils.io.__all__, name


def test_pixel_index_schema_validates_matching_frame() -> None:
    frame = pd.DataFrame(
        {
            "pixel_id": pd.Series([10, 11], dtype="int64"),
            "row": pd.Series([0, 0], dtype="int32"),
            "col": pd.Series([10, 11], dtype="int32"),
            "x": pd.Series([1.0, 2.0], dtype="float64"),
            "y": pd.Series([3.0, 4.0], dtype="float64"),
            "parcel_id": pd.Series([1, 1], dtype="int64"),
            "fold_id": pd.Series([1, 1], dtype="int8"),
            "ogf": pd.Series([0, 1], dtype="int8"),
            "forest_type_corine": pd.Series(["broadleaf", "mixed"], dtype="object"),
            "forest_type_inventory": pd.Series(["broadleaf", "coniferous"], dtype="object"),
        }
    )
    validate_schema(frame, PIXEL_INDEX_SCHEMA)


def test_pixel_index_schema_rejects_wrong_dtype() -> None:
    frame = pd.DataFrame(
        {
            "pixel_id": pd.Series([10], dtype="int64"),
            "row": pd.Series([0], dtype="int64"),  # expected int32
            "col": pd.Series([10], dtype="int32"),
            "x": pd.Series([1.0], dtype="float64"),
            "y": pd.Series([3.0], dtype="float64"),
            "parcel_id": pd.Series([1], dtype="int64"),
            "fold_id": pd.Series([1], dtype="int8"),
            "ogf": pd.Series([0], dtype="int8"),
            "forest_type_corine": pd.Series(["broadleaf"], dtype="object"),
            "forest_type_inventory": pd.Series(["broadleaf"], dtype="object"),
        }
    )
    with pytest.raises(ValueError, match="row"):
        validate_schema(frame, PIXEL_INDEX_SCHEMA)


def test_prediction_schemas_validate_matching_frames() -> None:
    pixel = pd.DataFrame(
        {
            "pixel_id": pd.Series([1, 2], dtype="int64"),
            "outer_fold": pd.Series([1, 1], dtype="int8"),
            "y_true": pd.Series([0, 1], dtype="int8"),
            "p": pd.Series([0.2, 0.8], dtype="float64"),
        }
    )
    parcel = pd.DataFrame(
        {
            "parcel_id": pd.Series([1, 2], dtype="int64"),
            "outer_fold": pd.Series([1, 1], dtype="int8"),
            "y_true": pd.Series([0, 1], dtype="int8"),
            "p_mean": pd.Series([0.3, 0.7], dtype="float64"),
            "n_pixels": pd.Series([5, 9], dtype="int64"),
        }
    )
    validate_schema(pixel, PIXEL_PREDICTIONS_SCHEMA)
    validate_schema(parcel, PARCEL_PREDICTIONS_SCHEMA)


def test_hp_boundary_schema_matches_diagnostics_columns() -> None:
    # The schema must accept exactly what hp_boundary_diagnostics produces.
    from utils.hp_search import hp_boundary_diagnostics

    best = {
        "n_estimators": 300,
        "learning_rate": 0.05,
        "max_depth": 5,
        "min_child_weight": 5.0,
        "subsample": 0.8,
        "colsample_bytree": 0.7,
        "reg_lambda": 2.0,
        "reg_alpha": 0.1,
        "gamma": 0.1,
        "max_delta_step": 2,
    }
    frame = hp_boundary_diagnostics(best)
    validate_schema(frame, HP_BOUNDARY_SCHEMA)


def test_hp_trials_schema_covers_every_tunable_param() -> None:
    from utils.hp_search import CNN_SEARCH_SPACE, XGB_SEARCH_SPACE

    for name in XGB_SEARCH_SPACE | CNN_SEARCH_SPACE:
        assert name in HP_TRIALS_SCHEMA, name
    assert "cnn_epochs" in HP_TRIALS_SCHEMA


def test_feature_importance_schema_matches_producer_columns() -> None:
    from xgboost import XGBClassifier

    from utils.models.xgboost import feature_importance

    rng = np.random.default_rng(0)
    model = XGBClassifier(n_estimators=5, max_depth=2, tree_method="hist", verbosity=0)
    model.fit(rng.normal(size=(40, 3)), rng.integers(0, 2, size=40))
    frame = feature_importance(model, ["a", "b", "c"])
    validate_schema(frame, FEATURE_IMPORTANCE_SCHEMA)


def test_per_fold_metrics_schema_covers_metric_keys() -> None:
    from utils.metrics import pixel_metrics

    rng = np.random.default_rng(1)
    y_true = np.array([0, 0, 1, 1, 0, 1])
    y_prob = rng.uniform(size=6)
    keys = set(pixel_metrics(y_true, y_prob))
    # Every metric key is representable as a column in the per-fold table.
    assert keys <= set(PER_FOLD_METRICS_SCHEMA)


def test_summary_and_pooled_schemas_validate_matching_frames() -> None:
    summary = pd.DataFrame(
        {
            "level": pd.Series(["pixel"], dtype="object"),
            "metric": pd.Series(["pr_auc"], dtype="object"),
            "mean": pd.Series([0.8], dtype="float64"),
            "sd": pd.Series([0.05], dtype="float64"),
            "min": pd.Series([0.7], dtype="float64"),
            "max": pd.Series([0.9], dtype="float64"),
        }
    )
    pooled = pd.DataFrame(
        {
            "metric": pd.Series(["pr_auc", "roc_auc"], dtype="object"),
            "value": pd.Series([0.81, 0.92], dtype="float64"),
        }
    )
    validate_schema(summary, SUMMARY_METRICS_SCHEMA)
    validate_schema(pooled, POOLED_METRICS_SCHEMA)


def test_schema_type_alias_is_importable() -> None:
    # The Schema alias is part of the public API used by the pipeline modules.
    assert Schema is not None
