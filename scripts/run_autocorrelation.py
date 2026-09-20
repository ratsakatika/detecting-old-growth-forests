"""Compute spatial autocorrelation for the predictors and the reference labels.

Stage 5 of the refactor. For each predictor group and for the binary old-growth
reference label this script draws a single shared sample, then fits empirical and
theoretical variograms (range, sill, nugget) and a distance-band Moran's I
correlogram, writing four tidy CSV tables plus the run metadata. The output
motivates the spatial split (fold separation against the correlation range) and
the area block-bootstrap unit size.

The numerical primitives and the analysis constants live in
:mod:`utils.autocorrelation`; this script only orchestrates them: it resolves
inputs from :mod:`utils.paths`, slices bands by name from
:data:`utils.terminology.FEATURE_SET_BANDS`, reduces the foundation-model
embeddings with PCA, fans the per-variable work out across processes with joblib,
and writes the reproducibility floor (input manifest, environment snapshot and run
metadata).

This is a multi-minute run over roughly 230 variables (the 20 km Moran graph is
the heaviest part), which is why it is a script rather than a notebook cell. The
``--dry-run`` flag restricts the work to the label and the six baseline bands so
the wiring can be checked in seconds.
"""

import argparse
import logging
import os
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Final

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
from joblib import Parallel, delayed
from numpy.typing import NDArray
from rasterio.crs import CRS
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

from utils import terminology
from utils.autocorrelation import (
    MORAN_PERMUTATIONS,
    MORAN_THRESHOLDS_M,
    N_SAMPLE,
    VGRAM_ESTIMATOR,
    VGRAM_MAXLAG_M,
    VGRAM_MODELS,
    VGRAM_N_LAGS,
    VGRAM_USE_NUGGET,
    VariogramResult,
    fit_variograms,
    moran_correlogram,
    sample_raster_points,
)
from utils.env_capture import write_environment_snapshot
from utils.io import ColumnSpec, Schema, write_csv, write_json
from utils.logging import make_run_logger
from utils.manifest import write_input_manifest
from utils.paths import ProjectPaths, get_project_paths
from utils.terminology import FEATURE_SET_BANDS, SEED

# Logger name and the run directory role under results/ (see AGENTS.md paths).
_LOGGER_NAME: Final[str] = "run_autocorrelation"
_RESULTS_ROLE: Final[str] = "autocorrelation"

# Number of worker processes for the per-variable fan-out.
N_JOBS: Final[int] = min(32, os.cpu_count() or 1)

# Number of principal components retained for each embedding group.
_PCA_COMPONENTS: Final[int] = 5

# Distance-band threshold (m) whose Moran's I is echoed to the per-variable log.
_REPORT_THRESHOLD_M: Final[int] = 10_000

# Stage 1 naming conventions pinned by notebook 004 (feature-set construction):
# each stack is written as data/processed/rasters/stacks_10m/<feature_set>_3035_10m.tif.
_STACKS_SUBDIR: Final[Path] = Path("rasters") / "stacks_10m"
_STACK_SUFFIX: Final[str] = "_3035_10m.tif"

# Notebook 005 output: the base all-parcels reference labels. Autocorrelation
# precedes fold design, so the label variogram reads this rather than the
# partitioned (notebook 008) file; it needs only the old-growth binary and geometry.
_LABEL_FILENAME: Final[str] = "ogf_reference_labels.gpkg"
# Binary old-growth flag column in the labels layer (1 OGF, 0 Non-OGF, NaN unlabelled).
_OGF_COLUMN: Final[str] = "ogf"

# Group keys used in the output "group" column.
_LABEL_GROUP: Final[str] = "label"
_LABEL_VARIABLE: Final[str] = "ogf_label"


# --------------------------------------------------------------------------- #
# Output schemas. Concrete result schemas are defined where they are needed
# (utils.io owns only the validation machinery); every CSV write is validated
# against one of these (hard rule 6).
# --------------------------------------------------------------------------- #

_FLOAT = ColumnSpec("float64")
_STR = ColumnSpec("object")
_INT = ColumnSpec("int64")
_BOOL = ColumnSpec("bool")

VARIOGRAM_FITS_SCHEMA: Final[Schema] = {
    "variable": _STR,
    "group": _STR,
    "is_pc": _BOOL,
    "model": _STR,
    "effective_range_m": _FLOAT,
    "sill": _FLOAT,
    "nugget": _FLOAT,
    "nugget_sill_ratio": _FLOAT,
    "rmse": _FLOAT,
    "range_hits_maxlag": _BOOL,
    "n": _INT,
    "maxlag_m": _FLOAT,
    "n_lags": _INT,
}

# Experimental semivariance at the true empirical bin centres (model-independent).
VARIOGRAM_EMPIRICAL_SCHEMA: Final[Schema] = {
    "variable": _STR,
    "group": _STR,
    "lag_m": _FLOAT,
    "semivariance": _FLOAT,
}

# Fitted theoretical-model curves on the fine plotting grid (one block per model).
VARIOGRAM_FITTED_SCHEMA: Final[Schema] = {
    "variable": _STR,
    "group": _STR,
    "model": _STR,
    "lag_m": _FLOAT,
    "semivariance": _FLOAT,
}

MORAN_SCHEMA: Final[Schema] = {
    "variable": _STR,
    "group": _STR,
    "threshold_m": _INT,
    "morans_i": _FLOAT,
    "expected_i": _FLOAT,
    "z_sim": _FLOAT,
    "p_sim": _FLOAT,
    "p_norm": _FLOAT,
    "n": _INT,
    "mean_neighbours": _FLOAT,
}


@dataclass(frozen=True, slots=True)
class _RasterGroup:
    """A raster predictor group sampled once and analysed band by band.

    Attributes:
        name: Group key written to the ``group`` column.
        feature_set: Canonical feature-set key naming the stack file.
        band_names: Bands to analyse, sliced by name from the stack.
        do_pca: Whether to additionally analyse the top principal components
            (used for the foundation-model embeddings).
    """

    name: str
    feature_set: str
    band_names: tuple[str, ...]
    do_pca: bool


@dataclass(frozen=True, slots=True)
class _Variable:
    """One analysis unit: an individual band, a principal component, or the label.

    Attributes:
        name: Variable name written to the ``variable`` column.
        group: Group key the variable belongs to.
        is_pc: Whether the variable is a principal component.
        coords: ``(m, 2)`` point coordinates in the project CRS.
        values: Length-``m`` values at ``coords``.
    """

    name: str
    group: str
    is_pc: bool
    coords: NDArray[np.float64]
    values: NDArray[np.float64]


def _build_raster_groups() -> dict[str, _RasterGroup]:
    """Build the raster group definitions from the canonical band order.

    The conventional-EO and embedding band lists are derived as each feature
    set's bands minus the shared baseline bands, so no band name is hard-coded.

    Returns:
        Mapping of group key to its :class:`_RasterGroup`.
    """
    baseline_names = FEATURE_SET_BANDS["baseline"]

    def non_baseline(feature_set: str) -> tuple[str, ...]:
        return tuple(name for name in FEATURE_SET_BANDS[feature_set] if name not in baseline_names)

    return {
        "baseline": _RasterGroup("baseline", "baseline", baseline_names, do_pca=False),
        "conventional_eo": _RasterGroup(
            "conventional_eo",
            "baseline_conventional_eo",
            non_baseline("baseline_conventional_eo"),
            do_pca=False,
        ),
        "alphaearth": _RasterGroup(
            "alphaearth", "baseline_alphaearth", non_baseline("baseline_alphaearth"), do_pca=True
        ),
        "tessera": _RasterGroup(
            "tessera", "baseline_tessera", non_baseline("baseline_tessera"), do_pca=True
        ),
    }


def _select_group_names(dry_run: bool) -> list[str]:
    """Return the raster group keys to process.

    Args:
        dry_run: When ``True``, only the baseline group is processed (the label
            is always processed regardless).

    Returns:
        The ordered group keys.
    """
    if dry_run:
        return ["baseline"]
    return ["baseline", "conventional_eo", "alphaearth", "tessera"]


def _stack_path(paths: ProjectPaths, feature_set: str) -> Path:
    """Resolve the on-disk path of a feature-set stack."""
    return paths.processed / _STACKS_SUBDIR / f"{feature_set}{_STACK_SUFFIX}"


def _band_indices(feature_set: str, band_names: Sequence[str]) -> list[int]:
    """Map band names to their one-based positions within a stack's band order."""
    order = FEATURE_SET_BANDS[feature_set]
    return [order.index(name) + 1 for name in band_names]


def _project_epsg() -> int | None:
    """Return the project CRS as an EPSG code."""
    return CRS.from_user_input(terminology.CRS).to_epsg()


def _check_raster_crs(path: Path, logger: logging.Logger) -> None:
    """Confirm a stack is on the project CRS, logging the check.

    Args:
        path: Stack path to check.
        logger: Run logger for the one INFO line.

    Raises:
        ValueError: If the stack has no CRS or a CRS other than the project CRS.
    """
    with rasterio.open(path) as src:
        crs = src.crs
    if crs is None or crs.to_epsg() != _project_epsg():
        found = crs.to_string() if crs is not None else "undefined"
        raise ValueError(
            f"Stack {path} has CRS {found}, expected {terminology.CRS}; reproject in "
            "pre-processing before running the autocorrelation analysis."
        )
    logger.info("CRS check passed for %s: %s.", path.name, terminology.CRS)


def _load_label_points(
    path: Path,
    logger: logging.Logger,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Load labelled-parcel representative points and their binary OGF flag.

    Every labelled parcel (old-growth or non-old-growth) contributes one
    representative interior point and its 1/0 old-growth flag; unlabelled parcels
    are dropped. The layer is reprojected to the project CRS if it differs.

    Args:
        path: Path to the partitioned labels GeoPackage.
        logger: Run logger for the load and reprojection lines.

    Returns:
        A ``(coords, values)`` pair: ``(m, 2)`` point coordinates and the
        length-``m`` binary old-growth flag.

    Raises:
        ValueError: If the layer has no CRS.
    """
    frame = gpd.read_file(path)
    if frame.crs is None:
        raise ValueError(f"Labels layer {path} has no CRS; cannot place points on the grid.")
    if frame.crs.to_epsg() != _project_epsg():
        logger.info(
            "Reprojecting labels: source CRS %s -> target %s, input=%s.",
            frame.crs.to_string(),
            terminology.CRS,
            path,
        )
        frame = frame.to_crs(terminology.CRS)

    labelled = frame[frame[_OGF_COLUMN].notna()]
    points = labelled.geometry.representative_point()
    coords = np.column_stack(
        (np.asarray(points.x, dtype=np.float64), np.asarray(points.y, dtype=np.float64))
    )
    values = np.asarray(labelled[_OGF_COLUMN].to_numpy(), dtype=np.float64)
    logger.info(
        "Loaded %d labelled parcels (%d old-growth) for the label variable.",
        coords.shape[0],
        int(values.sum()),
    )
    return coords, values


def _fit_pca(values: NDArray[np.float64]) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Z-score the sampled band values and project them onto the top components.

    Args:
        values: ``(m, n_bands)`` sampled values for one embedding group.

    Returns:
        A ``(scores, explained_variance_ratio)`` pair: ``(m, _PCA_COMPONENTS)``
        component scores and the per-component explained-variance ratio.
    """
    standardised = StandardScaler().fit_transform(values)
    pca = PCA(n_components=_PCA_COMPONENTS, random_state=SEED)
    scores = pca.fit_transform(standardised)
    return (
        np.asarray(scores, dtype=np.float64),
        np.asarray(pca.explained_variance_ratio_, dtype=np.float64),
    )


def _build_variables(
    paths: ProjectPaths,
    group_names: Sequence[str],
    logger: logging.Logger,
) -> tuple[list[_Variable], dict[str, dict[str, float]], list[Path]]:
    """Draw the shared per-group samples and assemble every analysis variable.

    Stage 1 draws one sample of valid pixels per raster group (reused across the
    variogram, Moran and PCA) and the labelled-parcel points. Stage 2 fits PCA on
    each embedding group's sample and appends the principal-component variables.

    Args:
        paths: Resolved project paths.
        group_names: Raster group keys to process.
        logger: Run logger.

    Returns:
        A ``(variables, pca_variance, inputs)`` triple: the analysis variables,
        the per-group per-PC explained-variance ratios, and the input files read
        (for the manifest).
    """
    groups = _build_raster_groups()
    variables: list[_Variable] = []
    pca_variance: dict[str, dict[str, float]] = {}
    inputs: list[Path] = []

    # Label variable: all labelled parcels, no sampling.
    label_path = paths.labels / _LABEL_FILENAME
    inputs.append(label_path)
    label_coords, label_values = _load_label_points(label_path, logger)
    variables.append(_Variable(_LABEL_VARIABLE, _LABEL_GROUP, False, label_coords, label_values))

    # Raster groups: one shared sample per group, analysed band by band.
    for group_name in group_names:
        group = groups[group_name]
        stack_path = _stack_path(paths, group.feature_set)
        inputs.append(stack_path)
        _check_raster_crs(stack_path, logger)
        indices = _band_indices(group.feature_set, group.band_names)
        logger.info(
            "Sampling %d bands of group %r from %s (seed=%d).",
            len(indices),
            group.name,
            stack_path.name,
            SEED,
        )
        coords, values = sample_raster_points(stack_path, bands=indices, n=N_SAMPLE, seed=SEED)
        logger.info("Group %r: %d valid pixels sampled.", group.name, coords.shape[0])

        for column, band_name in enumerate(group.band_names):
            variables.append(_Variable(band_name, group.name, False, coords, values[:, column]))

        if group.do_pca:
            scores, explained = _fit_pca(values)
            pca_variance[group.name] = {
                f"pc{i + 1}": float(explained[i]) for i in range(explained.size)
            }
            logger.info(
                "Group %r PCA: top-%d explained variance %.3f.",
                group.name,
                _PCA_COMPONENTS,
                float(explained.sum()),
            )
            for component in range(scores.shape[1]):
                variables.append(
                    _Variable(
                        f"{group.name}_pc{component + 1}",
                        group.name,
                        True,
                        coords,
                        scores[:, component],
                    )
                )

    return variables, pca_variance, inputs


def _empirical_frame(variable: _Variable, result: VariogramResult) -> pd.DataFrame:
    """Build the empirical-variogram table for one variable.

    One row per empirical lag class, carrying the true bin centre and the
    experimental semivariance there. The empirical variogram is
    model-independent, so there is no model column.

    Args:
        variable: The variable described.
        result: Its variogram result.

    Returns:
        The empirical table for the variable.
    """
    return pd.DataFrame(
        {
            "variable": variable.name,
            "group": variable.group,
            "lag_m": result.lags_m,
            "semivariance": result.semivariance,
        }
    )


def _fitted_frame(variable: _Variable, result: VariogramResult) -> pd.DataFrame:
    """Build the fitted-curve table for one variable.

    One block of fine-grid nodes per model, carrying the model's fitted
    semivariance for smooth plotting.

    Args:
        variable: The variable described.
        result: Its variogram result.

    Returns:
        The long fitted-curve table for the variable.
    """
    frames = [
        pd.DataFrame(
            {
                "variable": variable.name,
                "group": variable.group,
                "model": model,
                "lag_m": result.curve_lag_m,
                "semivariance": curve,
            }
        )
        for model, curve in result.fitted.items()
    ]
    return pd.concat(frames, ignore_index=True)


def _process_variable(
    variable: _Variable,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Fit the variogram and Moran correlogram for one variable.

    Runs inside a joblib worker; returns the variable's four table fragments.

    Args:
        variable: The variable to analyse.

    Returns:
        A ``(fits, empirical, fitted, moran)`` tuple of per-variable fragments.
    """
    result = fit_variograms(variable.coords, variable.values)

    fits = result.fits.copy()
    fits.insert(0, "variable", variable.name)
    fits.insert(1, "group", variable.group)
    fits.insert(2, "is_pc", variable.is_pc)
    fits["n"] = result.n
    fits["maxlag_m"] = result.maxlag_m
    fits["n_lags"] = result.n_lags

    empirical = _empirical_frame(variable, result)
    fitted = _fitted_frame(variable, result)

    moran = moran_correlogram(variable.coords, variable.values).reset_index()
    moran.insert(0, "variable", variable.name)
    moran.insert(1, "group", variable.group)

    return fits, empirical, fitted, moran


def _log_variable_summary(logger: logging.Logger, fits: pd.DataFrame, moran: pd.DataFrame) -> None:
    """Log the best-model effective range (km) and Moran's I at the report distance.

    Args:
        logger: Run logger.
        fits: The variable's fits fragment.
        moran: The variable's Moran fragment.
    """
    variable = str(fits["variable"].iloc[0])
    scored = fits[fits["rmse"].notna()]
    best = scored.loc[scored["rmse"].idxmin()] if not scored.empty else fits.iloc[0]
    range_km = float(best["effective_range_m"]) / 1_000.0
    at_report = moran.loc[moran["threshold_m"] == _REPORT_THRESHOLD_M, "morans_i"]
    moran_i = float(at_report.iloc[0]) if not at_report.empty else float("nan")
    logger.info(
        "Variable %s: best model %s, effective range %.2f km, Moran's I @%d km %.3f.",
        variable,
        str(best["model"]),
        range_km,
        _REPORT_THRESHOLD_M // 1_000,
        moran_i,
    )


def _assemble(
    results: Sequence[tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Concatenate the per-variable fragments and coerce the schema dtypes.

    Args:
        results: Per-variable ``(fits, empirical, fitted, moran)`` fragments.

    Returns:
        The four concatenated, schema-ordered tables.
    """
    fits = pd.concat([r[0] for r in results], ignore_index=True)
    empirical = pd.concat([r[1] for r in results], ignore_index=True)
    fitted = pd.concat([r[2] for r in results], ignore_index=True)
    # A correlogram is empty only if every threshold graph had no neighbours,
    # which does not happen for the real samples; drop empties defensively so an
    # all-NaN reindexed fragment cannot upcast the integer columns.
    moran_frames = [r[3] for r in results if not r[3].empty]
    moran = pd.concat(moran_frames, ignore_index=True)

    fits = fits.astype(
        {"is_pc": "bool", "range_hits_maxlag": "bool", "n": "int64", "n_lags": "int64"}
    )
    moran = moran.astype({"threshold_m": "int64", "n": "int64"})

    return (
        fits[list(VARIOGRAM_FITS_SCHEMA)],
        empirical[list(VARIOGRAM_EMPIRICAL_SCHEMA)],
        fitted[list(VARIOGRAM_FITTED_SCHEMA)],
        moran[list(MORAN_SCHEMA)],
    )


def _library_versions() -> dict[str, str]:
    """Return versions of the key analysis libraries for the run metadata."""
    libraries = (
        "numpy",
        "pandas",
        "scipy",
        "scikit-learn",
        "scikit-gstat",
        "esda",
        "libpysal",
        "rasterio",
        "geopandas",
        "joblib",
    )
    versions: dict[str, str] = {}
    for library in libraries:
        try:
            versions[library] = version(library)
        except PackageNotFoundError:
            versions[library] = "unknown"
    return versions


def _parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    """Parse the command-line arguments.

    Args:
        argv: Argument list, or ``None`` to read from ``sys.argv``.

    Returns:
        The parsed arguments.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Process only the label and the six baseline bands, to check the wiring fast.",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    """Run the autocorrelation analysis end to end.

    Args:
        argv: Argument list, or ``None`` to read from ``sys.argv``.

    Returns:
        Process exit code (0 on success).
    """
    args = _parse_args(argv)
    paths = get_project_paths()
    run_dir = paths.results / _RESULTS_ROLE
    run_dir.mkdir(parents=True, exist_ok=True)
    logger = make_run_logger(_LOGGER_NAME, run_dir)

    logger.info(
        "Starting autocorrelation analysis (dry_run=%s, n_jobs=%d, n_sample=%d, seed=%d).",
        args.dry_run,
        N_JOBS,
        N_SAMPLE,
        SEED,
    )

    group_names = _select_group_names(args.dry_run)

    # Stages 1-2: draw the shared samples and assemble every analysis variable.
    variables, pca_variance, inputs = _build_variables(paths, group_names, logger)
    logger.info("Assembled %d variables across groups %s.", len(variables), group_names)

    # Reproducibility floor: input manifest and environment snapshot.
    write_input_manifest(inputs, run_dir / "input_manifest.json", repo_root=paths.repo_root)
    write_environment_snapshot(run_dir / "env_snapshot.txt")

    # Stage 3: fit every variable's variogram and Moran correlogram in parallel.
    # joblib yields results as they complete so the tqdm bar tracks real
    # progress; each is logged on arrival in the main process (workers cannot
    # write to the run logger's file handler).
    generator = Parallel(n_jobs=N_JOBS, return_as="generator")(
        delayed(_process_variable)(variable) for variable in variables
    )
    results = []
    for fragment in tqdm(generator, total=len(variables), desc="autocorrelation"):
        _log_variable_summary(logger, fragment[0], fragment[3])
        results.append(fragment)
    fits, empirical, fitted, moran = _assemble(results)

    # Stage 4: write the four tidy CSV tables.
    fits_path = write_csv(fits, run_dir / "variogram_fits.csv", VARIOGRAM_FITS_SCHEMA)
    empirical_path = write_csv(
        empirical, run_dir / "variogram_empirical.csv", VARIOGRAM_EMPIRICAL_SCHEMA
    )
    fitted_path = write_csv(fitted, run_dir / "variogram_fitted.csv", VARIOGRAM_FITTED_SCHEMA)
    moran_path = write_csv(moran, run_dir / "moran_correlogram.csv", MORAN_SCHEMA)
    logger.info(
        "Summary: %d variables, %d variogram fits, %d empirical rows, %d fitted rows, "
        "%d Moran rows.",
        len(variables),
        len(fits),
        len(empirical),
        len(fitted),
        len(moran),
    )

    # Stage 5 hook (NOT implemented here): once a model exists (Stage 6), the
    # pooled out-of-fold residual becomes another variable analysed by exactly
    # the same path -- read residual (x, y) and value, append a _Variable with
    # group "residual", and re-run stages 3-4. It is deferred because it needs
    # the fitted model this script precedes.

    # Run metadata (doubles as the run config): the autocorrelation constants,
    # PCA explained variance, sample size, seed and N_JOBS, plus provenance.
    run_metadata = {
        "run_id": datetime.now(UTC).strftime("%Y%m%d_%H%M%S"),
        "created_at": datetime.now(UTC).isoformat(),
        "script": "scripts/run_autocorrelation.py",
        "dry_run": args.dry_run,
        "seed": SEED,
        "n_jobs": N_JOBS,
        "n_sample": N_SAMPLE,
        "groups": group_names,
        "n_variables": len(variables),
        "config": {
            "vgram_maxlag_m": VGRAM_MAXLAG_M,
            "vgram_n_lags": VGRAM_N_LAGS,
            "vgram_models": list(VGRAM_MODELS),
            "vgram_estimator": VGRAM_ESTIMATOR,
            "vgram_use_nugget": VGRAM_USE_NUGGET,
            "moran_thresholds_m": list(MORAN_THRESHOLDS_M),
            "moran_permutations": MORAN_PERMUTATIONS,
            "pca_components": _PCA_COMPONENTS,
        },
        "pca_explained_variance": pca_variance,
        "library_versions": _library_versions(),
        "outputs": {
            "variogram_fits": str(fits_path.relative_to(paths.repo_root)),
            "variogram_empirical": str(empirical_path.relative_to(paths.repo_root)),
            "variogram_fitted": str(fitted_path.relative_to(paths.repo_root)),
            "moran_correlogram": str(moran_path.relative_to(paths.repo_root)),
        },
    }
    write_json(run_metadata, run_dir / "run_metadata.json")
    logger.info("Wrote run metadata; autocorrelation analysis complete.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
