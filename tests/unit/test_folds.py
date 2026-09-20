"""Unit tests for utils.folds.

All tests are offline: they build small synthetic GeoDataFrames of unit squares
on EPSG:3035 with known centroids and masses, so the spatial partition outcome
is predictable.
"""

from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import pytest
import shapely
from shapely.geometry import box

import utils.folds as folds_module
from utils.folds import (
    assign_folds,
    assign_spatial_partition,
    connected_components,
    cross_unit_distances,
    decode_pixel_id,
    encode_pixel_id,
    load_fold_assignment,
)
from utils.terminology import FOLD_IDS

CRS = "EPSG:3035"


def _square(x: float, y: float, size: float = 1.0) -> box:
    return box(x, y, x + size, y + size)


def _gdf(boxes: list, index: list | None = None) -> gpd.GeoDataFrame:
    return gpd.GeoDataFrame(
        {"parcel_id": list(range(len(boxes)))},
        geometry=boxes,
        crs=CRS,
        index=index,
    )


# --------------------------------------------------------------------------- #
# encode_pixel_id / decode_pixel_id
# --------------------------------------------------------------------------- #


def test_pixel_id_scalar_round_trip() -> None:
    pid = encode_pixel_id(3, 5, 10)
    assert isinstance(pid, int)
    assert pid == 35
    assert decode_pixel_id(35, 10) == (3, 5)


def test_pixel_id_array_round_trip() -> None:
    rows = np.array([0, 3, 7])
    cols = np.array([0, 5, 9])
    pid = encode_pixel_id(rows, cols, 10)
    assert np.array_equal(pid, np.array([0, 35, 79]))

    out_rows, out_cols = decode_pixel_id(pid, 10)
    assert np.array_equal(out_rows, rows)
    assert np.array_equal(out_cols, cols)


def test_pixel_id_rejects_nonpositive_width() -> None:
    with pytest.raises(ValueError, match="width must be positive"):
        encode_pixel_id(1, 1, 0)
    with pytest.raises(ValueError, match="width must be positive"):
        decode_pixel_id(10, -3)


# --------------------------------------------------------------------------- #
# assign_spatial_partition
# --------------------------------------------------------------------------- #


def test_partition_groups_touching_components() -> None:
    # Two clusters of touching squares, far apart: each cluster is one component.
    gdf = _gdf([_square(0, 0), _square(1, 0), _square(100, 0), _square(101, 0)])
    units = assign_spatial_partition(gdf, 2, grid_shape=(2, 1), group_connected=True)

    assert set(units) == {1, 2}
    assert units.iloc[0] == units.iloc[1]  # cluster A kept together
    assert units.iloc[2] == units.iloc[3]  # cluster B kept together
    assert units.iloc[0] != units.iloc[2]
    assert list(units.index) == list(gdf.index)


def test_partition_individual_splits_a_connected_run() -> None:
    # Four touching squares form one component end to end.
    gdf = _gdf([_square(i, 0) for i in range(4)])

    individual = assign_spatial_partition(gdf, 2, grid_shape=(2, 1), group_connected=False)
    assert list(individual) == [1, 1, 2, 2]  # split by centroid x into balanced halves

    with pytest.raises(ValueError, match="only 1 unit"):
        assign_spatial_partition(gdf, 2, grid_shape=(2, 1), group_connected=True)


def test_partition_buffer_merges_near_parcels() -> None:
    gdf = _gdf([_square(0, 0), _square(1.5, 0), _square(100, 0)])  # 0.5 gap between first two

    no_buffer = assign_spatial_partition(gdf, 2, grid_shape=(2, 1), buffer_m=0.0, tolerance=0.6)
    with_buffer = assign_spatial_partition(gdf, 2, grid_shape=(2, 1), buffer_m=0.6, tolerance=0.6)

    assert no_buffer.iloc[0] != no_buffer.iloc[1]  # separate components without a buffer
    assert with_buffer.iloc[0] == with_buffer.iloc[1]  # merged into one component by the buffer


def test_partition_infers_grid_for_composite_and_rejects_prime() -> None:
    units = assign_spatial_partition(_gdf([_square(10 * i, 0) for i in range(6)]), 6)
    assert set(units) == set(range(1, 7))  # (3, 2) grid inferred

    with pytest.raises(ValueError, match="prime"):
        assign_spatial_partition(_gdf([_square(10 * i, 0) for i in range(5)]), 5)


def test_partition_single_unit() -> None:
    units = assign_spatial_partition(_gdf([_square(0, 0), _square(10, 0)]), 1)
    assert set(units) == {1}


def test_partition_balances_secondary_mass() -> None:
    gdf = _gdf([_square(0, 0), _square(10, 0), _square(20, 0), _square(30, 0)])
    units = assign_spatial_partition(
        gdf,
        2,
        grid_shape=(2, 1),
        secondary_mass=np.array([1.0, 0.0, 1.0, 0.0]),
        group_connected=False,
    )
    assert set(units) == {1, 2}


def test_partition_raises_on_primary_imbalance() -> None:
    gdf = _gdf([_square(10 * i, 0) for i in range(4)])
    with pytest.raises(ValueError, match="mass imbalance"):
        assign_spatial_partition(
            gdf,
            2,
            grid_shape=(2, 1),
            mass=np.array([1.0, 1.0, 1.0, 10.0]),
            tolerance=0.1,
            group_connected=False,
        )


def test_partition_raises_on_secondary_imbalance() -> None:
    gdf = _gdf([_square(10 * i, 0) for i in range(4)])
    with pytest.raises(ValueError, match="secondary_mass imbalance"):
        # Mass splits evenly into two columns, but the secondary mass does not.
        assign_spatial_partition(
            gdf,
            2,
            grid_shape=(2, 1),
            secondary_mass=np.array([0.0, 1.0, 3.0, 0.0]),
            group_connected=False,
        )


def test_partition_validates_inputs() -> None:
    gdf = _gdf([_square(0, 0), _square(10, 0)])

    empty = gpd.GeoDataFrame(geometry=gpd.GeoSeries([], crs=CRS))
    with pytest.raises(ValueError, match="empty"):
        assign_spatial_partition(empty, 1)

    no_crs = gpd.GeoDataFrame({"parcel_id": [0]}, geometry=[_square(0, 0)])
    with pytest.raises(ValueError, match="CRS"):
        assign_spatial_partition(no_crs, 1)

    with pytest.raises(ValueError, match="n_units must be positive"):
        assign_spatial_partition(gdf, 0)

    with pytest.raises(ValueError, match="inconsistent"):
        assign_spatial_partition(gdf, 5, grid_shape=(2, 2))

    with pytest.raises(ValueError, match="mass must have length"):
        assign_spatial_partition(gdf, 1, mass=np.array([1.0]))
    with pytest.raises(ValueError, match="mass must be non-negative"):
        assign_spatial_partition(gdf, 1, mass=np.array([-1.0, 1.0]))
    with pytest.raises(ValueError, match="mass must be non-negative"):
        assign_spatial_partition(gdf, 1, mass=np.array([0.0, 0.0]))

    with pytest.raises(ValueError, match="secondary_mass must have length"):
        assign_spatial_partition(gdf, 1, secondary_mass=np.array([1.0]))
    with pytest.raises(ValueError, match="secondary_mass must be non-negative"):
        assign_spatial_partition(gdf, 1, secondary_mass=np.array([0.0, 0.0]))


# --------------------------------------------------------------------------- #
# assign_folds
# --------------------------------------------------------------------------- #


def test_assign_folds_six_balanced_folds() -> None:
    boxes: list = []
    ogf: list = []
    for col in range(3):
        for row in range(2):
            x, y = col * 1000.0, row * 1000.0
            boxes += [_square(x, y), _square(x + 1, y)]  # one touching pair per fold cell
            ogf += [True, False]
    gdf = gpd.GeoDataFrame({"parcel_id": range(len(boxes))}, geometry=boxes, crs=CRS)

    folds = assign_folds(gdf, np.array(ogf))

    assert folds.name == "fold_id"
    assert set(folds) <= set(FOLD_IDS)
    assert set(folds) == set(range(1, 7))
    values = folds.to_numpy()
    for i in range(0, len(boxes), 2):  # each touching pair shares its fold
        assert values[i] == values[i + 1]


def test_assign_folds_accepts_series_and_other_counts() -> None:
    gdf = _gdf([_square(1000 * i, 0) for i in range(4)], index=[10, 11, 12, 13])
    ogf = pd.Series([True, True, True, True], index=[10, 11, 12, 13])

    folds = assign_folds(gdf, ogf, n_folds=4)  # (2, 2) grid inferred

    assert set(folds) == {1, 2, 3, 4}
    assert list(folds.index) == [10, 11, 12, 13]


def test_assign_folds_large_buffer_collapses_components() -> None:
    gdf = _gdf([_square(1000 * i, 0) for i in range(8)])
    with pytest.raises(ValueError, match="only 1 unit"):
        assign_folds(gdf, np.ones(8, dtype=bool), buffer_m=1_000_000.0)


# --------------------------------------------------------------------------- #
# load_fold_assignment
# --------------------------------------------------------------------------- #


def test_load_fold_assignment_reads_supported_formats(tmp_path: Path) -> None:
    frame = pd.DataFrame({"parcel_id": [1, 2, 3], "fold_id": [1, 2, 6], "extra": [0, 0, 0]})

    csv_path = tmp_path / "folds.csv"
    frame.to_csv(csv_path, index=False)
    from_csv = load_fold_assignment(csv_path)
    assert list(from_csv.columns) == ["parcel_id", "fold_id"]
    assert from_csv["fold_id"].tolist() == [1, 2, 6]

    parquet_path = tmp_path / "folds.parquet"
    frame.to_parquet(parquet_path)
    assert load_fold_assignment(parquet_path)["fold_id"].tolist() == [1, 2, 6]

    gpkg_path = tmp_path / "folds.gpkg"
    gpd.GeoDataFrame(frame, geometry=[_square(i, 0) for i in range(3)], crs=CRS).to_file(gpkg_path)
    assert load_fold_assignment(gpkg_path)["fold_id"].tolist() == [1, 2, 6]


def test_load_fold_assignment_errors(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="unsupported"):
        load_fold_assignment(tmp_path / "folds.txt")

    missing = tmp_path / "missing.csv"
    pd.DataFrame({"parcel_id": [1]}).to_csv(missing, index=False)
    with pytest.raises(ValueError, match="missing column"):
        load_fold_assignment(missing)

    invalid = tmp_path / "invalid.csv"
    pd.DataFrame({"parcel_id": [1, 2], "fold_id": [1, 0]}).to_csv(invalid, index=False)
    with pytest.raises(ValueError, match="outside FOLD_IDS"):
        load_fold_assignment(invalid)


# --------------------------------------------------------------------------- #
# secondary objective: share vs ratio (by count / by area)
# --------------------------------------------------------------------------- #


def _spreads(labels: np.ndarray, area: np.ndarray, is_ogf: np.ndarray) -> dict[str, float]:
    a = np.array([area[labels == lbl].sum() for lbl in (1, 2)])
    ogf_a = np.array([area[(labels == lbl) & is_ogf].sum() for lbl in (1, 2)])
    cnt = np.array([(labels == lbl).sum() for lbl in (1, 2)], dtype=float)
    ogf_c = np.array([((labels == lbl) & is_ogf).sum() for lbl in (1, 2)], dtype=float)
    return {
        "ogf_area": float(ogf_a.max() - ogf_a.min()),
        "area_prev": float((ogf_a / a).max() - (ogf_a / a).min()),
        "count_prev": float((ogf_c / cnt).max() - (ogf_c / cnt).min()),
    }


def test_secondary_objectives_each_tighten_their_own_metric() -> None:
    areas = np.array([1.0, 1.0, 1.0, 1.0, 5.0, 5.0, 5.0, 5.0])
    is_ogf = np.array([True, True, False, False, True, False, False, False])
    boxes = [_square(10.0 * i, 0.0, float(np.sqrt(areas[i]))) for i in range(len(areas))]
    gdf = gpd.GeoDataFrame({"parcel_id": range(len(boxes))}, geometry=boxes, crs=CRS)
    # primary_first with a loose tolerance makes every cut primary-feasible, so each
    # objective minimises its own secondary metric over the same set of cuts.
    common = {
        "grid_shape": (2, 1),
        "group_connected": False,
        "rotations": (0.0,),
        "tolerance": 1.0,
        "priority": "primary_first",
        "mass": areas,
    }
    share = assign_spatial_partition(
        gdf, 2, secondary_mass=np.where(is_ogf, areas, 0.0), secondary_objective="share", **common
    ).to_numpy()
    by_count = assign_spatial_partition(
        gdf,
        2,
        secondary_mass=np.where(is_ogf, 1.0, 0.0),
        secondary_denominator=np.ones(len(areas)),
        secondary_objective="ratio",
        **common,
    ).to_numpy()
    by_area = assign_spatial_partition(
        gdf,
        2,
        secondary_mass=np.where(is_ogf, areas, 0.0),
        secondary_denominator=areas,
        secondary_objective="ratio",
        **common,
    ).to_numpy()

    s_share = _spreads(share, areas, is_ogf)
    s_count = _spreads(by_count, areas, is_ogf)
    s_area = _spreads(by_area, areas, is_ogf)

    assert s_share["ogf_area"] <= min(s_count["ogf_area"], s_area["ogf_area"]) + 1e-9
    assert s_count["count_prev"] <= min(s_share["count_prev"], s_area["count_prev"]) + 1e-9
    assert s_area["area_prev"] <= min(s_share["area_prev"], s_count["area_prev"]) + 1e-9


# --------------------------------------------------------------------------- #
# priority: joint vs primary_first
# --------------------------------------------------------------------------- #


def test_primary_first_keeps_primary_within_tolerance() -> None:
    gdf = _gdf([_square(10 * i, 0) for i in range(6)])
    area = shapely.area(gdf.geometry.to_numpy())
    secondary = np.array([0.0, 1.0, 2.0, 2.0, 1.0, 0.0])
    common = {"grid_shape": (2, 1), "mass": area, "secondary_mass": secondary, "rotations": (0.0,)}
    joint = assign_spatial_partition(gdf, 2, priority="joint", tolerance=1.0, **common).to_numpy()
    primary_first = assign_spatial_partition(
        gdf, 2, priority="primary_first", tolerance=0.2, **common
    ).to_numpy()

    for lbl in (1, 2):
        share = area[primary_first == lbl].sum() / area.sum()
        assert abs(share - 0.5) <= 0.2 + 1e-9

    def secondary_spread(labels: np.ndarray) -> float:
        totals = np.array([secondary[labels == lbl].sum() for lbl in (1, 2)])
        return float(totals.max() - totals.min())

    assert secondary_spread(primary_first) <= secondary_spread(joint) + 1e-9


def test_primary_first_infeasible_falls_back_and_raises() -> None:
    gdf = _gdf([_square(10 * i, 0) for i in range(4)])
    with pytest.raises(ValueError, match="mass imbalance"):
        assign_spatial_partition(
            gdf,
            2,
            grid_shape=(2, 1),
            mass=np.array([1.0, 1.0, 1.0, 10.0]),
            secondary_mass=np.array([1.0, 1.0, 1.0, 1.0]),
            secondary_objective="ratio",
            priority="primary_first",
            tolerance=0.1,
            group_connected=False,
            rotations=(0.0,),
        )


def test_partition_raises_when_a_column_is_too_thin_for_rows() -> None:
    # A dominant-mass parcel takes a whole column alone, leaving it too thin for two rows.
    gdf = _gdf([_square(10 * i, 0) for i in range(6)])
    with pytest.raises(ValueError, match="imbalance"):
        assign_spatial_partition(
            gdf,
            6,
            grid_shape=(3, 2),
            mass=np.array([1.0, 1.0, 1.0, 1.0, 1.0, 20.0]),
            group_connected=False,
            rotations=(0.0,),
            tolerance=0.25,
        )


# --------------------------------------------------------------------------- #
# rotation search
# --------------------------------------------------------------------------- #


def test_rotation_balances_a_diagonal_layout_better() -> None:
    # Old-growth fills a diagonal band; an axis-aligned 2x2 grid cannot split it
    # evenly, but a rotated grid can.
    boxes: list = []
    is_ogf: list = []
    for i in range(8):
        for j in range(8):
            boxes.append(_square(10.0 * i, 10.0 * j))
            is_ogf.append(5 <= i + j <= 9)  # a thick anti-diagonal band of old-growth
    gdf = gpd.GeoDataFrame({"parcel_id": range(len(boxes))}, geometry=boxes, crs=CRS)
    ogf = np.array(is_ogf)
    area = shapely.area(gdf.geometry.to_numpy())

    def ogf_area_spread(rotations: tuple[float, ...]) -> float:
        labels = assign_spatial_partition(
            gdf,
            4,
            grid_shape=(2, 2),
            group_connected=False,
            secondary_mass=np.where(ogf, area, 0.0),
            rotations=rotations,
            tolerance=1.0,
        ).to_numpy()
        totals = np.array([area[(labels == lbl) & ogf].sum() for lbl in (1, 2, 3, 4)])
        return float(totals.max() - totals.min())

    assert ogf_area_spread(tuple(float(d) for d in range(0, 180, 5))) < ogf_area_spread((0.0,))


# --------------------------------------------------------------------------- #
# connected_components
# --------------------------------------------------------------------------- #


def test_connected_components_grouping_and_buffer() -> None:
    gdf = _gdf([_square(0, 0), _square(1, 0), _square(5, 0)], index=[7, 8, 9])
    components = connected_components(gdf)
    assert components.loc[7] == components.loc[8]  # touching squares share a component
    assert components.loc[7] != components.loc[9]
    assert list(components.index) == [7, 8, 9]

    merged = connected_components(gdf, buffer_m=5.0)  # buffer bridges the gap to the third
    assert merged.nunique() == 1


def test_assign_spatial_partition_calls_connected_components(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[float] = []
    real = folds_module.connected_components

    def spy(parcels: gpd.GeoDataFrame, *, buffer_m: float = 0.0) -> pd.Series:
        calls.append(buffer_m)
        return real(parcels, buffer_m=buffer_m)

    monkeypatch.setattr(folds_module, "connected_components", spy)
    gdf = _gdf([_square(0, 0), _square(100, 0)])
    assign_spatial_partition(gdf, 2, grid_shape=(2, 1), group_connected=True, buffer_m=0.0)
    assert calls == [0.0]


# --------------------------------------------------------------------------- #
# cross_unit_distances
# --------------------------------------------------------------------------- #


def _two_unit_frame(extra_na: bool = False) -> gpd.GeoDataFrame:
    geoms = [box(0, 0, 1, 1), box(0, 3, 1, 4), box(10, 0, 11, 1), box(10, 3, 11, 4)]
    units: list = [1, 1, 2, 2]
    if extra_na:
        geoms.append(box(50, 50, 51, 51))
        units.append(np.nan)
    return gpd.GeoDataFrame({"unit": units}, geometry=geoms, crs=CRS)


def test_cross_unit_distances_nearest_and_pairwise() -> None:
    frame = _two_unit_frame()

    nearest = cross_unit_distances(frame, "unit", mode="nearest")
    assert list(nearest.columns) == ["n", "min", "q05", "q25", "q50", "q75", "q95"]
    assert nearest.index.name == "unit"
    assert int(nearest.loc[1, "n"]) == 2
    assert nearest.loc[1, "min"] == pytest.approx(9.0)
    assert nearest.loc[1, "q50"] == pytest.approx(9.0)

    pairwise, arrays = cross_unit_distances(
        frame, "unit", mode="pairwise", n_jobs=1, return_arrays=True
    )
    assert int(pairwise.loc[1, "n"]) == 4
    assert pairwise.loc[1, "min"] == pytest.approx(9.0)
    assert set(arrays) == {1, 2}
    assert arrays[1].min() == pytest.approx(9.0)
    assert arrays[1].max() == pytest.approx(np.hypot(9.0, 2.0))


def test_cross_unit_distances_drops_na_and_validates_mode() -> None:
    frame = _two_unit_frame(extra_na=True)
    nearest = cross_unit_distances(frame, "unit", mode="nearest")
    assert set(nearest.index) == {1, 2}  # the NA-unit parcel is dropped

    with pytest.raises(ValueError, match="mode must be"):
        cross_unit_distances(frame, "unit", mode="centroid")


# --------------------------------------------------------------------------- #
# assign_folds: prevalence basis and objective options
# --------------------------------------------------------------------------- #


def test_assign_folds_ratio_objective_options() -> None:
    gdf = _gdf([_square(1000 * i, 0) for i in range(8)])
    ogf = np.array([True, False] * 4)

    for basis in ("count", "area"):
        folds = assign_folds(
            gdf,
            ogf,
            n_folds=4,
            secondary_objective="ratio",
            priority="primary_first",
            prevalence_basis=basis,
            tolerance=1.0,
        )
        assert set(folds) == {1, 2, 3, 4}
        assert folds.name == "fold_id"

    with pytest.raises(ValueError, match="prevalence_basis"):
        assign_folds(gdf, ogf, n_folds=4, secondary_objective="ratio", prevalence_basis="nope")
