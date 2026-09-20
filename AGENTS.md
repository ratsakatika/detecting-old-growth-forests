# AGENTS.md

Conventions for coding agents making changes to this repository. The README covers setup, data and the reproduction order; this file covers how the
code is written. Run `make ci` before committing.

## Where things live

- Constants (seed, folds, CRS, NoData, sampling and bootstrap budgets, canonical
  names, palettes, figure sizes) are defined once in `utils/terminology.py`;
  result-table schemas in `utils/io.py`; paths in `utils/paths.py`; the log
  format in `utils/logging.py`. Import them, never copy the values. A constant
  changes in its module, with its tests, in the same commit.
- `utils/` modules have no side effects on import. `scripts/` are command-line
  steps named by verb (`download_`, `build_`, `run_`; `launch_*.sh` start
  several of them concurrently). Notebooks are numbered in pipeline order;
  001-005 and 008 download and pre-process data, the rest only read results
  and write figures and tables.

## Spatial conventions

- Everything processed is in the project CRS (EPSG:3035, `utils.terminology.CRS`)
  on the 10 m reference grid. On every load, check the CRS, reproject if it
  differs and log it. Raw files under `data/raw/` are never modified.
- Maps are displayed in `utils.terminology.DISPLAY_CRS` (UTM 35N) by
  reprojecting only the display geometry at plot time; every computation stays
  in EPSG:3035.
- NoData sentinels come from `utils.terminology`; `pixel_id = row * width + col`
  is computed once, at table build, never re-derived.
- Earth Engine requests take WGS84 geometry (a projected bounding box is
  silently sheared); request the export CRS separately.

## Modelling conventions

- Folds are numbered 1 to 6 (`utils.terminology.FOLD_IDS`) in every output.
- Scripts default to `--device cuda` (`run_nested_cv` to `auto`, which selects
  CUDA when it is available). The CPU path is for the unit tests and for
  smoke runs; it is orders of magnitude slower on the real data. Do not change a
  script's default device.
- Parcel scores are the unweighted mean of pixel probabilities (`p_mean`).
- Calibration is applied only to the final map and parcel product, fitted on the
  pooled out-of-fold predictions; ranking and threshold metrics use raw
  probabilities.
- One area-balanced block set (`utils.terminology.BOOTSTRAP_N_UNITS`) serves
  both the paired bootstrap contrasts and the mapped-area uncertainty.
- The four existing products are evaluated standalone against the reference
  labels, never ranked against this study's model.

## Outputs

- Runs go to `results/<group>/<run_id>/`, with
  `run_id = {YYYYMMDD_HHMMSS}__{architecture}__{feature_set}` and logs inside
  the run folder. Every long-running script writes `input_manifest.json`,
  `env_snapshot.txt` and `run_metadata.json` there.
- Every result table passes `utils.io.validate_schema` before it is written:
  Parquet for predictions, CSV for tables, JSON for metadata.
- Figures are saved only through `utils.style.save_figure`, as PDF, to
  `figures/<notebook>/<name>.pdf`, with the plotted data as CSV under `raw/`.
  Colours come from `utils.terminology`: teal is old-growth, orange is
  non-old-growth, and each feature set, fold and product keeps one colour in
  every figure.

## Code rules

1. Explicit imports only; no `from utils import *`.
2. No absolute home paths outside `utils/paths.py`.
3. `logger.info`, not `print`, in `utils/` and `scripts/`.
4. Raise exceptions for data validation; no `assert`.
5. Type annotations and British-English docstrings on every public function.
6. Tests accompany every change; `utils/` stays fully covered (`make test`).
7. The parcel labels are called "reference labels"; the pre-commit hook rejects
   the phrase "ground truth".
8. No file over 5 MB is committed.
9. Commit messages are a short imperative summary of the change.
