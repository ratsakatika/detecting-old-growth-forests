# Detecting old-growth forests from AlphaEarth and TESSERA geospatial foundation model embeddings and conventional Sentinel-1/2 features

This repository contains code required to reproduce the results presented in:

> Ratsakatika, T., Zotta, M., Keshav, S., Lines, E.R. (2026). Geospatial embeddings detect old-growth forests but buffered spatial validation narrows their advantage over Sentinel features. arXiv preprint. doi:TBC.

Use this repository to:

- **Download and pre-process the public prediction datasets:** Natura 2000 boundaries, road and footpath vectors, forest disturbances (1985–2023), coniferous/broadleaf/mixed forest cover, Digital Elevation Model with forests and buildings removed (FABDEM), Sentinel-1/2 annual composites and vegetation indices, AlphaEarth 64-dimension embeddings, TESSERA v2 128-dimension embeddings.
- **Construct old-growth and non-old-growth forest reference labels** from forest management plans if available, otherwise download from:

    > Ratsakatika, T., Zotta, M., Keshav, S., Lines, E.R. (2026). Old-growth forest reference labels and model predictions for the Făgăraș Mountains, Romania (2020, 10 m) (v1.0.0) [dataset]. Zenodo. <https://doi.org/10.5281/zenodo.22693148>

- **Run spatially blocked nested cross-validation** with incremental buffers from 0–30 km between the training and test labels.

- **Evaluate the absolute and relative performance of five feature sets:** baseline (topography and road/footpath access), baseline + conventional Earth observation (Sentinel-1/2 annual composites and vegetation indices), baseline + AlphaEarth embeddings, baseline + TESSERA v2 embeddings, and an x–y coordinate-only control.

- **Evaluate the absolute and relative performance of four model architectures:** single pixel-based XGBoost and patch-based CNN (3x3, 5x5 and 7x7 pixel receptive fields).

- **Evaluate four existing maps** against the old-growth and non-old-growth reference labels and each other.

- **Build a wall-to-wall map of old-growth forest probability at 10 m resolution**

- **Compute the Area of Applicability of the model across the entire Carpathian Mountain range**

## Repository structure

```
utils/       importable library: constants, schemas, I/O, modelling and plotting helpers
scripts/     command-line steps of the pipeline, run as `python -m scripts.<name>`
notebooks/   numbered notebooks in pipeline order; 001-005 and 008 download and pre-process data, the rest read results and write figures and tables
tests/       unit tests on synthetic data (pytest, no GPU)
assets/      the Charis SIL typeface used in the figures
data/        raw/ downloads and manual inputs, processed/ analysis-ready layers, cache/ memmaps
results/     one folder per analysis, one run folder per execution: results/<group>/<run_id>/
figures/     one folder per notebook (figures/<notebook>/), plotted data as CSV under raw/
```

`data/`, `results/` and `figures/` are created by the pipeline; only the
provenance records under `data/` are committed.

The script names are structured as follows:

- `download_*.py` fetch raw data (the Carpathian extension, and the TESSERA
  tiles that notebook 001 fetches through `download_tessera_tiles.py`; the other
  study-area downloads live in the notebook);
- `build_*.py` derive artefacts without modelling: caches, masks, aligned
  products, the packaged final outputs;
- `run_*.py` train, evaluate or analyse;
- `launch_*.sh` start several `run_*` or `download_*` processes concurrently and
  are run with `bash`.

## Requirements

The results were produced on an Intel Core i9-14900KS with 192 GB of RAM and an
RTX 4090 with 24 GB of VRAM. The launchers and the default thread, concurrency
and GPU-slot settings were chosen for that machine; adjust them to your own. The estimated minimum requirements are:

- Python 3.12.
- A CUDA 12.4 compatible GPU with at least 8 GB VRAM.
- At least 32 GB RAM, more for the CNN runs.
- About 100 GB of disk space for the downloaded, processed and result data, plus roughly 350 GB of memory-mapped caches under `data/cache/` if the CNN matrix is run.

Running on the CPU alone is possible but not practical. Every modelling script
accepts `--device cpu` but the XGBoost cross-validation, the buffered refits, the seed sweep, the 1,000-replicate area bootstrap and the Carpathian area of applicability would then take weeks rather than hours, and the CNN runs would be intractable on CPU.

## Setup

```bash
python3.12 -m venv .venv && source .venv/bin/activate
pip install -r requirements-lock.txt
pip install --no-deps -e .   # makes `utils` importable from the notebooks
pre-commit install
```

Copy `env.example` to `.env` and fill in the credentials for notebook 001:

- `EE_PROJECT`: requires a Google Earth Engine Cloud project ID to download AlphaEarth embeddings. Authenticate once beforehand by running `earthengine authenticate`.

- `CLMS_SERVICE_KEY_PATH`: requires a Copernicus Land Monitoring Service (CLMS) service-key JSON for CORINE Land Cover (see: https://land.copernicus.eu/en/how-to-guides/how-to-download-spatial-data/how-to-create-api-tokens).

- `WEKEO_USERNAME` and `WEKEO_PASSWORD`: requires WEkEO account credentials for Copernicus HR-VPP (see: registration at https://www.wekeo.eu).

## Data

Notebook 001 downloads every public input into `data/raw/` and writes a
`*.provenance.json` record beside each file: source, download time, size and
SHA256.

The provenance records for the manuscript are committed in this repo, so for a fresh repo clone, each download cell checks the file it downloaded against the provenance file and prints whether it matches the version used in the study. A mismatch means the provider has updated the dataset since the study, so results may differ.

### Inputs obtained manually

Some third-party data must be downloaded manually.

- The Sabatini (2020) primary forest map must be downloaded from the iDiv data portal, <https://idata.idiv.de/ddm/Data/ShowData/1841>, and placed in `data/raw/existing_products/sabatini/`.

- PRIMOFARO inventory (Schickhofer and Schwarz, 2019) must be requested from the authors directly and placed in `data/raw/existing_products/schickhofer/`, or omitted from the analysis.

- The forest parcel data, `ogf_labels_predictions.gpkg`, must be downloaded from <https://doi.org/10.5281/zenodo.22693148> and placed in `data/processed/vectors/labels/`. Notebook 003 rebuilds the parcel map and the reference labels from the Zenodo record.

Further details are provided in notebook 001.

## Reproducing the results

Run the notebooks top to bottom in Jupyter, and the scripts from the repository
root with the virtual environment active: `python -m scripts.<name>` for Python
scripts, `bash scripts/<name>.sh` for the launchers.

`--help` lists each script's options. These include:

- `--device cuda|cpu`
- `--n-jobs` (threads per fit; the launchers also set `--concurrency` and `--gpu-slots`)
- `--smoke` (a reduced end-to-end run)
- `--resume <run_id>` (continue an interrupted run).

Runs are written to `results/<group>/<run_id>/` and figures to `figures/<notebook>/`.

The steps below reproduce every figure, table and number in the manuscript and supplementary materials (shown in brackets). Running end-to-end will take approximately 1.5 weeks on the machine described above.

Data preparation (CPU)

1. `notebooks/001_public_data_download.ipynb`: download the public inputs
   (Tables 1 and 2).
2. `notebooks/002_raster_pre_processing.ipynb`: 10 m EPSG:3035 reference grid, terrain, distance to roads, HR-VPP, AlphaEarth and TESSERA layers, disturbance mask (Section 2.1; Table 2).
3. `notebooks/003_vector_pre_processing.ipynb`: CORINE forest classes, parcel map, management plans, virgin forests, ownership, or the rebuild from the published dataset (Section 2.2; Table 1).
4. `notebooks/004_feature_set_construction.ipynb`: the four feature-set stacks (Section 2.5; Table 2).
5. `notebooks/005_reference_label_construction.ipynb`: the old-growth and non-old-growth reference labels; skip if already reconstructed in notebook 003 from the Zenodo record (see Data section, above) (Section 2.2).
6. `notebooks/006_aoi_and_reference_label_statistics.ipynb`: study-area and reference-label statistics (Sections 2.1 and 2.2; Fig. S1).
7. `notebooks/008_fold_bootstrap_assignment.ipynb`: spatial folds and bootstrap blocks (Section 2.7; Fig. 1b, Figs S3 and S4).
8. `python -m scripts.run_autocorrelation`: semi-variograms and Moran's I of the labels and predictors (Section 2.6; Fig. 2, Table S2).

Nested cross-validation (GPU)

9. `python -m scripts.build_pixel_table`: pixel index and per-feature-set feature memmaps (Section 2.7).
10. `bash scripts/launch_nested_cv_xgboost.sh [N_JOBS] [GPU_SLOTS]`: XGBoost nested cross-validation, one process per feature set, four running concurrently; the defaults are 8 threads per process, 4 GPU slots and 50 Optuna trials per outer fold (30 for the CNNs in step 11) (Section 2.7 and Algorithm 1; Figs 4 and 6, Tables S4, S6, S7 and S11 to S13).
11. `bash scripts/launch_nested_cv_cnn.sh MAX_CONCURRENT PATCH_SIZE [PATCH_SIZE ...]`: the receptive-field CNNs, one process per feature set and patch size, with at most `MAX_CONCURRENT` running at once. The study ran `4 3 5` (the 3x3 and 5x5 patches, four processes at a time) and then `2 7` (the 7x7 patches, two at a time, because their memory-mapped caches total about 200 GB). (Sections 2.5 and 2.7 and Algorithm 1; Fig. 4, Tables S6, S7 and S10 to S13).
12. `python -m scripts.run_coordinate_xgboost`: the coordinate-only control for XGBoost (Section 2.5; Tables S8, S11 to S13 and S18).
13. `python -m scripts.run_coordinate_cnn`: the coordinate-only control for the three CNNs (Tables S11 to S13 and S18).

Spatial buffers, performance matrix and sensitivity analyses (GPU)

14. `python -m scripts.run_gap_analysis --feature-set <set>`, run once for each of `baseline`, `baseline_conventional_eo`, `baseline_alphaearth`, `baseline_tessera` and `xy_coords`: 0 to 30 km training-exclusion sweep with parcel-matched controls (Section 2.7; Fig. 5, Figs S6 and S7).
15. `python -m scripts.run_performance_matrix`: 0/10/20 km performance matrix, block bootstrap and pre-specified contrasts for all 20 configurations (Fig. 4; Tables S6 to S13).
16. `python -m scripts.run_coordinate_cnn_buffer --architecture <arch>`, run once for each of `cnn_3x3`, `cnn_5x5` and `cnn_7x7`: buffered arm of the CNN coordinate-only control (Tables S11 to S13).
17. `python -m scripts.run_sampling_sensitivity`: training-pixel sampling-mode sensitivity (Table S5).
18. `python -m scripts.run_block_size_sensitivity`: bootstrap block-count sensitivity, CPU only (Table S3).
19. `python -m scripts.run_seed_sweep`: training-seed sensitivity (Tables S16 and S17).
20. `notebooks/007_autocorrelation.ipynb`: autocorrelation analysis of the labels and predictors, and the out-of-fold residual autocorrelation (Fig. 2; Tables S2 and S18).
21. `notebooks/009_performance_matrix_and_buffers.ipynb`: the performance and buffer figures (Figs 4, 5 and 6; Figs S6 and S7).

Mapping and existing-product comparison (GPU)

22. `python -m scripts.run_final_inference`: final XGBoost + TESSERA model, wall-to-wall probability and binary rasters, calibrated parcel product (Section 2.8; Section 3.4, Table 3, Fig. 1e and 1f). Needs the `baseline_tessera` grid memmap, which step 11 builds; if step 11 was skipped, run `python -m scripts.build_grid_memmap --feature-sets baseline_tessera` first.
23. `python -m scripts.run_area_bootstrap`: 1,000-replicate retraining bootstrap of the mapped old-growth area (Section 2.8; Section 3.4, Fig. S8).
24. `python -m scripts.build_existing_product_masks`: align the four existing products to the reference grid (Section 2.4).
25. `python -m scripts.run_product_comparison`: parcel-level comparison of the four existing products and this study (Section 3.1; Table 3, Fig. 3, Tables S19 to S21).
26. `python -m scripts.build_continuous_product_parcels`: parcel means of the two continuous product surfaces, written into the comparison run of step 25 (Fig. S5).
27. `python -m scripts.build_final_outputs`: the published datasets in `results/final/` (Data and code availability; Fig. 1e and 1f).
28. `notebooks/010_mapping.ipynb`: calibration and mapped area (Section 3.4; Fig. S8).
29. `notebooks/011_existing_product_comparison.ipynb`: the existing product comparison figures (Section 3.1; Fig. 3, Figs S2, S5 and S9).
30. `notebooks/012_methods_figures.ipynb`: the map panels of Fig. 1 (the composite figure is assembled outside the repository).

Area of applicability (GPU)

31. `bash scripts/launch_carpathian_downloads.sh [GRID_RES]`: the Carpathian mountain range outline, CORINE, WorldCover, terrain and access layers and TESSERA v2 at the analysis resolution (100 m in the study), four downloads in parallel (Section 2.9).
32. `python -m scripts.run_area_of_applicability --grid-res 100`: area of applicability (AOA) analysis (Section 2.9; Section 3.5).
33. `notebooks/013_area_of_applicability.ipynb`: the AOA map and statistics (Section 3.5; Fig. 7).

Robustness checks and manuscript tables

34. `notebooks/014_robustness_checks.ipynb`: leave-one-fold-out, seed-sweep and unlabelled-parcel agreement tables (Section 3.4; Tables S14 to S17 and S21).
35. `notebooks/015_manuscript_tables.ipynb`: every numeric table of the manuscript as LaTeX under `figures/015_manuscript_tables/latex/` (Table 3; Tables S2 to S21).

## Tests

```bash
make test   # unit tests; no data or GPU needed
make ci     # pre-commit, tests with coverage, mypy
```

## Licence

This repository is released under the MIT License (see `LICENSE`).

For the old-growth forest reference labels and predictions, please refer to Zenodo record: https://doi.org/10.5281/zenodo.22693148

For third-party data (sources listed in notebook 001 and in the manuscript's data tables), please check each provider's own licence and attribution terms.

## Citation

Please cite both the paper and the dataset if using the old-growth forest reference labels, predictions or code.

Paper:

> Ratsakatika, T., Zotta, M., Keshav, S., Lines, E.R., (2026). Geospatial embeddings detect old-growth forests but buffered spatial validation narrows their advantage over Sentinel features, doi:TBC.

Data:
> Ratsakatika, T., Zotta, M., Keshav, S., Lines, E.R., 2026. Old-growth forest reference labels and model predictions for the Făgăraș Mountains, Romania (2020, 10 m) (v1.0.0) [dataset]. Zenodo. <https://doi.org/10.5281/zenodo.22693148>

Contact: trr26@cam.ac.uk

## Acknowledgements

T.R. was funded by the UKRI Centre for Doctoral Training in the Application of Artificial Intelligence to the Study of Environmental Risks (EP/S022961/1). S.K. was supported by the Robert Sansom Professorship in Computer Science. E.R.L. was funded by a UKRI Future Leaders Fellowship (MR/Y033981/1).
