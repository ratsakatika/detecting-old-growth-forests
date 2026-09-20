"""Single source of truth for canonical names and project-wide constants.

The wider codebase pins every fixed string and numeric constant here: the
random seed, the fold layout, sampling and hyper-parameter-search budgets,
spatial conventions, the categorical figure palette, and the canonical
feature-set, architecture and comparison-study identifiers used verbatim in
filenames, metadata, registry entries and figure labels.

Importing this module only binds names: it performs no input/output, configures
no logging and inspects no environment, in keeping with the repository rule that
``utils`` modules have no side effects on import.

Every value is intended to be immutable. The mapping constants are wrapped in
:class:`types.MappingProxyType` and the record types are frozen dataclasses, so
attempts to mutate them at runtime raise instead of silently corrupting shared
state. If a canonical constant must change, ``AGENTS.md`` is updated in the same
commit (hard rule 10).
"""

from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import Final

# --------------------------------------------------------------------------- #
# Project-wide constants (see AGENTS.md, "Project-wide constants").
# --------------------------------------------------------------------------- #

SEED: Final[int] = 42
N_FOLDS: Final[int] = 6
FOLD_IDS: Final[tuple[int, ...]] = (1, 2, 3, 4, 5, 6)
N_PER_PARCEL: Final[int] = 500
N_HP_TRIALS: Final[int] = 30  # inner-search budget per outer fold: the CNNs, and the default
# The main XGBoost nested cross-validation searched 50 trials per outer fold (manuscript
# Section 2.7: T = 50 for XGBoost, T = 30 for the CNNs); run_nested_cv applies it by architecture.
N_HP_TRIALS_XGBOOST: Final[int] = 50
MIN_PIXELS_PARCEL: Final[int] = 0  # no exclusion by default; sensitivity-tested.
# Confirm against the archive's value.

CRS: Final[str] = "EPSG:3035"  # ETRS89 / LAEA Europe (equal-area, EEA standard)
# All computation -- areas, masks, the reference grid, every analytical step --
# stays on CRS (EPSG:3035, equal-area). DISPLAY_CRS exists only to render maps
# upright: under the equal-area projection grid north differs from true north, so
# display geometry and hillshade are reprojected to UTM 35N for the figure alone.
DISPLAY_CRS: Final[str] = "EPSG:32635"  # UTM zone 35N, for upright map display only
DISPLAY_CRS_NAME: Final[str] = "UTM 35N"
REF_RESOLUTION_M: Final[float] = 10.0  # reference grid resolution, metres
NODATA: Final[float] = -9999.0  # float32 NoData sentinel for data/processed/

# Block count for the one labelled-parcel block bootstrap used for both performance
# and area. Contiguous, area-balanced spatial partition built by
# utils.folds.assign_spatial_partition, resampled with replacement. The predicted-area
# uncertainty is driven by the training-label sample (resample labelled blocks, retrain,
# predict over the fixed cadastre, sum area), the same resampling as the performance
# bootstrap, so one unit set serves both; resampling all parcels would wrongly treat the
# fully-observed prediction surface as a sample. Set to 20: only 1 of 20 blocks falls below
# the 6.68 km binding residual autocorrelation range (longest across feature sets), against
# 15 of 30 at the former count. 20 sits below the ~30 cluster-count floor (Cameron & Miller
# 2015, doi:10.3368/jhr.50.2.317), accepted because the min-width claim is cleaner and a
# block-size sensitivity analysis will show CI stability.
BOOTSTRAP_N_UNITS: Final[int] = 20
BOOTSTRAP_REPS_CROSS_CONFIG: Final[int] = 5_000
BOOTSTRAP_REPS_INFERENCE: Final[int] = 1_000

# --------------------------------------------------------------------------- #
# Forest-type categories (CORINE Land Cover 2018 forest classes).
# --------------------------------------------------------------------------- #

CORINE_FOREST_CLASSES: Final[Mapping[int, str]] = MappingProxyType(
    {
        311: "broadleaf",
        312: "coniferous",
        313: "mixed",
    }
)

# Full CORINE Land Cover 2018 Level-3 nomenclature (code -> official class name),
# used to label all land-cover classes within the AOI, not just the forest classes above.
CORINE_CLC_CLASSES: Final[Mapping[int, str]] = MappingProxyType(
    {
        111: "Continuous urban fabric",
        112: "Discontinuous urban fabric",
        121: "Industrial or commercial units",
        122: "Road and rail networks and associated land",
        123: "Port areas",
        124: "Airports",
        131: "Mineral extraction sites",
        132: "Dump sites",
        133: "Construction sites",
        141: "Green urban areas",
        142: "Sport and leisure facilities",
        211: "Non-irrigated arable land",
        212: "Permanently irrigated land",
        213: "Rice fields",
        221: "Vineyards",
        222: "Fruit trees and berry plantations",
        223: "Olive groves",
        231: "Pastures",
        241: "Annual crops associated with permanent crops",
        242: "Complex cultivation patterns",
        243: (
            "Land principally occupied by agriculture, "
            "with significant areas of natural vegetation"
        ),
        244: "Agro-forestry areas",
        311: "Broad-leaved forest",
        312: "Coniferous forest",
        313: "Mixed forest",
        321: "Natural grasslands",
        322: "Moors and heathland",
        323: "Sclerophyllous vegetation",
        324: "Transitional woodland-shrub",
        331: "Beaches, dunes and sands",
        332: "Bare rocks",
        333: "Sparsely vegetated areas",
        334: "Burnt areas",
        335: "Glaciers and perpetual snow",
        411: "Inland marshes",
        412: "Peat bogs",
        421: "Salt marshes",
        422: "Salines",
        423: "Intertidal flats",
        511: "Water courses",
        512: "Water bodies",
        521: "Coastal lagoons",
        522: "Estuaries",
        523: "Sea and ocean",
    }
)

# Minimum share (percent) of one forest group (broadleaf or coniferous) for a parcel
# to be classed as that pure type; otherwise the parcel is "mixed". Single source of
# truth, consumed by reference-label construction and statistics.
FOREST_TYPE_DOMINANCE_PCT: Final[int] = 70

# Half-width of the road and trail corridor removed from old-growth parcel
# geometries during reference-label construction (metres each side).
OGF_ROAD_BUFFER_M: Final[float] = 10.0

# --------------------------------------------------------------------------- #
# Feature sets (canonical name -> display label).
# --------------------------------------------------------------------------- #

FEATURE_SETS: Final[Mapping[str, str]] = MappingProxyType(
    {
        "baseline": "Baseline",
        "baseline_conventional_eo": "Baseline + EO",
        "baseline_tessera": "Baseline + TESSERA",
        "baseline_alphaearth": "Baseline + AlphaEarth",
    }
)

# Control "feature sets": diagnostics that run through the same pipeline but are not
# modelling stacks. They are deliberately kept OUT of FEATURE_SETS so that anything
# defaulting to "every feature set" (the performance matrix, the block bootstrap, the
# cache builders) is unchanged; a caller must name a control explicitly. Runners that
# should accept one validate against MODEL_INPUTS instead.
CONTROL_FEATURE_SETS: Final[Mapping[str, str]] = MappingProxyType(
    {"xy_coords": "Coordinate-only control (x, y)"}
)

# Every input a model can be fitted on: the modelling stacks plus the controls.
MODEL_INPUTS: Final[Mapping[str, str]] = MappingProxyType({**FEATURE_SETS, **CONTROL_FEATURE_SETS})

# --------------------------------------------------------------------------- #
# Feature-set band order (canonical name -> ordered band names).
#
# The component tuples are defined once and composed below so the band order is
# explicit and never duplicated. The "access" group (distance-to-roads) bands sit
# inside the baseline; the embedding sets deliberately exclude WorldCover and
# HR-VPP so that geospatial foundation-model embeddings are compared against the
# same baseline, not against conventional Earth-observation layers.
# --------------------------------------------------------------------------- #

_BASELINE_BANDS: Final[tuple[str, ...]] = (
    "elevation_m",
    "slope_deg",
    "heat_load_index",
    "dist_paved_road_m",
    "dist_unpaved_road_m",
    "dist_footpath_m",
)

_WORLDCOVER_BANDS: Final[tuple[str, ...]] = (
    "s2_red",
    "s2_green",
    "s2_blue",
    "s2_nir",
    "ndvi_p90",
    "ndvi_p50",
    "ndvi_p10",
    "s1_vv",
    "s1_vh",
    "s1_vh_vv_ratio",
    "swir_b11",
    "swir_b12",
)

_HRVPP_BANDS: Final[tuple[str, ...]] = (
    "vpp_ampl",
    "vpp_eosd",
    "vpp_eosv",
    "vpp_lslope",
    "vpp_maxv",
    "vpp_minv",
    "vpp_rslope",
    "vpp_sosd",
    "vpp_sosv",
    "vpp_sprod",
)

_TESSERA_BANDS: Final[tuple[str, ...]] = tuple(f"tessera_t{i:03d}" for i in range(128))

_ALPHAEARTH_BANDS: Final[tuple[str, ...]] = tuple(f"alphaearth_a{i:02d}" for i in range(64))

FEATURE_SET_BANDS: Final[Mapping[str, tuple[str, ...]]] = MappingProxyType(
    {
        "baseline": _BASELINE_BANDS,
        "baseline_conventional_eo": _BASELINE_BANDS + _WORLDCOVER_BANDS + _HRVPP_BANDS,
        "baseline_tessera": _BASELINE_BANDS + _TESSERA_BANDS,
        "baseline_alphaearth": _BASELINE_BANDS + _ALPHAEARTH_BANDS,
    }
)

# Bands of the control inputs. Kept out of FEATURE_SET_BANDS because the modelling
# stacks satisfy invariants a control does not (every stack extends the baseline, and
# every band carries a BAND_LABELS entry for figure axes). Consumers that must handle
# either kind -- the cache builders and the nested-CV runner -- index MODEL_INPUT_BANDS.
CONTROL_FEATURE_SET_BANDS: Final[Mapping[str, tuple[str, ...]]] = MappingProxyType(
    {"xy_coords": ("easting_m", "northing_m")}
)

MODEL_INPUT_BANDS: Final[Mapping[str, tuple[str, ...]]] = MappingProxyType(
    {**FEATURE_SET_BANDS, **CONTROL_FEATURE_SET_BANDS}
)

# --------------------------------------------------------------------------- #
# Plain-English band labels for figure axes (band name -> display label).
#
# Every band in FEATURE_SET_BANDS has exactly one label. The named bands
# (baseline, WorldCover, HR-VPP) are listed explicitly; the embedding labels are
# generated from the band tuples so they cannot drift. HR-VPP labels follow the
# Copernicus HR-VPP Product User Manual (issue 2.5): the green-up and green-down
# slopes are the rising and falling limbs of the seasonal Plant Phenology Index
# curve, seasonal productivity is the small seasonal integral (base level
# removed), and the value parameters are PPI values at the named season dates.
# --------------------------------------------------------------------------- #

BAND_LABELS: Final[Mapping[str, str]] = MappingProxyType(
    {
        # Baseline: terrain and access (distance to roads).
        "elevation_m": "Elevation (m)",
        "slope_deg": "Slope (degrees)",
        "heat_load_index": "Heat load index",
        "dist_paved_road_m": "Distance to paved road (m)",
        "dist_unpaved_road_m": "Distance to unpaved road (m)",
        "dist_footpath_m": "Distance to footpath (m)",
        # WorldCover composites: Sentinel-2 optical, NDVI percentiles, Sentinel-1 SAR, SWIR.
        "s2_red": "Sentinel-2 red",
        "s2_green": "Sentinel-2 green",
        "s2_blue": "Sentinel-2 blue",
        "s2_nir": "Sentinel-2 NIR",
        "ndvi_p90": "NDVI 90th percentile",
        "ndvi_p50": "NDVI median",
        "ndvi_p10": "NDVI 10th percentile",
        "s1_vv": "Sentinel-1 VV",
        "s1_vh": "Sentinel-1 VH",
        "s1_vh_vv_ratio": "Sentinel-1 VH/VV ratio",
        "swir_b11": "SWIR band 11",
        "swir_b12": "SWIR band 12",
        # HR-VPP phenology and productivity (Plant Phenology Index based).
        "vpp_ampl": "Seasonal amplitude",
        "vpp_eosd": "End-of-season date",
        "vpp_eosv": "End-of-season value",
        "vpp_lslope": "Green-up slope",
        "vpp_maxv": "Maximum-of-season value",
        "vpp_minv": "Minimum-of-season value",
        "vpp_rslope": "Green-down slope",
        "vpp_sosd": "Start-of-season date",
        "vpp_sosv": "Start-of-season value",
        "vpp_sprod": "Seasonal productivity",
        # Embedding dimensions: generated from the band tuples so they cannot drift.
        **{name: f"TESSERA dimension {i}" for i, name in enumerate(_TESSERA_BANDS)},
        **{name: f"AlphaEarth dimension {i}" for i, name in enumerate(_ALPHAEARTH_BANDS)},
    }
)

# --------------------------------------------------------------------------- #
# Architectures (canonical name -> display label). The model key combining an
# architecture with a feature set is "{architecture}__{feature_set}"; for CNNs
# the receptive-field patch size is encoded in the architecture name.
# --------------------------------------------------------------------------- #

ARCHITECTURES: Final[Mapping[str, str]] = MappingProxyType(
    {
        "xgboost": "XGBoost",
        "cnn_3x3": "CNN 3x3",
        "cnn_5x5": "CNN 5x5",
        "cnn_7x7": "CNN 7x7",
    }
)

# Receptive-field CNN patch sizes, derived from the cnn_<n>x<n> architecture
# names so ARCHITECTURES stays the single source of truth and the two cannot
# drift. Square patches only, per the naming convention.
CNN_PATCH_SIZES: Final[tuple[int, ...]] = tuple(
    int(name.removeprefix("cnn_").split("x")[0])
    for name in ARCHITECTURES
    if name.startswith("cnn_")
)

# --------------------------------------------------------------------------- #
# Comparison studies (standalone products evaluated against reference labels).
# --------------------------------------------------------------------------- #


@dataclass(frozen=True, slots=True)
class ComparisonStudy:
    """A published old-growth product evaluated against the reference labels.

    Attributes:
        display: Human-readable citation, used verbatim in figure labels and
            captions.
        category: Category key; one of the keys of :data:`CATEGORY_DISPLAY`.
    """

    display: str
    category: str


COMPARISON_STUDIES: Final[Mapping[str, ComparisonStudy]] = MappingProxyType(
    {
        "sabatini": ComparisonStudy("Sabatini (2020)", "predictive"),
        "munteanu": ComparisonStudy("Munteanu et al. (2022)", "predictive"),
        "kathmann": ComparisonStudy("Kathmann et al. (2017), Greenpeace", "rules_based"),
        "schickhofer": ComparisonStudy("Schickhofer and Schwarz (2019), PRIMOFARO", "rules_based"),
    }
)

# Comparison-study category display names (category key -> display string).
CATEGORY_DISPLAY: Final[Mapping[str, str]] = MappingProxyType(
    {
        "predictive": "Predictive mapping models",
        "rules_based": "Rules-based mapping",
    }
)

# --------------------------------------------------------------------------- #
# Figure palettes (see AGENTS.md, "Figures").
# --------------------------------------------------------------------------- #

# Six distinct hues for feature sets, models and studies. Key access lets
# callers refer to a colour by name; the derived tuple supports matplotlib
# colour cycles and any other ordered consumer.
PALETTE_CATEGORICAL: Final[Mapping[str, str]] = MappingProxyType(
    {
        "blue": "#2877B7",
        "teal": "#3BAF8F",
        "light_green": "#A1D86E",
        "yellow": "#FFD640",
        "orange": "#FF904E",
        "magenta": "#A62E5B",
    }
)

PALETTE_CATEGORICAL_ORDER: Final[tuple[str, ...]] = tuple(PALETTE_CATEGORICAL.values())

# Perceptually uniform sequential map for probability rasters and continuous metrics.
PALETTE_SEQUENTIAL: Final[str] = "viridis"

# Diverging map for residuals and disagreement maps.
PALETTE_DIVERGING: Final[str] = "RdBu_r"

# --------------------------------------------------------------------------- #
# Concept colour assignments.
#
# Rule: within a concept family, each member keeps its colour across every
# figure. The OGF / non-OGF pair is the one colour constant carried across the
# whole study. Folds necessarily use all six hues. Feature sets avoid the two
# label hues so they never clash with OGF context in performance figures.
# Colours may repeat across different families, which is acceptable because
# those families do not appear together in one figure.
#
# All assignments reference PALETTE_CATEGORICAL by key, so the hex codes have a
# single source of truth.
# --------------------------------------------------------------------------- #

# The two label classes: teal = old-growth forest, orange = non-old-growth.
SEMANTIC_COLOURS: Final[Mapping[str, str]] = MappingProxyType(
    {
        "ogf": PALETTE_CATEGORICAL["teal"],
        "non_ogf": PALETTE_CATEGORICAL["orange"],
    }
)

# Cross-validation folds 1..6 mapped to the six palette hues in order.
FOLD_COLOURS: Final[Mapping[int, str]] = MappingProxyType(
    dict(zip(FOLD_IDS, PALETTE_CATEGORICAL_ORDER, strict=True))
)

# Feature sets mapped to the four non-reserved palette hues.
FEATURE_SET_COLOURS: Final[Mapping[str, str]] = MappingProxyType(
    {
        "baseline": PALETTE_CATEGORICAL["blue"],
        "baseline_conventional_eo": PALETTE_CATEGORICAL["light_green"],
        "baseline_tessera": PALETTE_CATEGORICAL["yellow"],
        "baseline_alphaearth": PALETTE_CATEGORICAL["magenta"],
    }
)


@dataclass(frozen=True, slots=True)
class Species:
    """A tree species record: its broad group and its English, Latin and Romanian names."""

    group: str  # "coniferous" or "broadleaf"
    english: str
    latin: str | None  # None for the aggregate "diverse" codes
    romanian: str


CLEARCUT_TOKEN: Final[str] = "CLEARCUT"  # a stand state, not a species; forces non-OGF

SPECIES: Final[Mapping[str, Species]] = MappingProxyType(
    {
        # Coniferous
        "BR": Species("coniferous", "Silver fir", "Abies alba", "Brad"),
        "DR": Species("coniferous", "Mixed conifers", None, "Diverse rasinoase"),
        "DU": Species("coniferous", "Douglas fir", "Pseudotsuga menziesii", "Duglas"),
        "LA": Species("coniferous", "European larch", "Larix decidua", "Larice"),
        "MO": Species("coniferous", "Norway spruce", "Picea abies", "Molid"),
        "PI": Species("coniferous", "Scots pine", "Pinus sylvestris", "Pin silvestru"),
        "PIC": Species("coniferous", "Swiss stone pine", "Pinus cembra", "Pin cembru"),
        "PIN": Species("coniferous", "Black pine", "Pinus nigra", "Pin negru"),
        "PIS": Species("coniferous", "Eastern white pine", "Pinus strobus", "Pin strob"),
        # Broadleaf
        "AN": Species("broadleaf", "Grey alder", "Alnus incana", "Anin alb"),
        "ANN": Species("broadleaf", "Black alder", "Alnus glutinosa", "Anin negru"),
        "CA": Species("broadleaf", "Hornbeam", "Carpinus betulus", "Carpen"),
        "DM": Species("broadleaf", "Soft broadleaves", None, "Diverse moi"),
        "DT": Species("broadleaf", "Hard broadleaves", None, "Diverse tari"),
        "FA": Species("broadleaf", "European beech", "Fagus sylvatica", "Fag"),
        "FR": Species("broadleaf", "Common ash", "Fraxinus excelsior", "Frasin comun"),
        "GI": Species("broadleaf", "Hungarian oak", "Quercus frainetto", "Garnita"),
        "GO": Species("broadleaf", "Sessile oak", "Quercus petraea", "Gorun"),
        "ME": Species("broadleaf", "Silver birch", "Betula pendula", "Mesteacan"),
        "PA": Species("broadleaf", "Norway maple", "Acer platanoides", "Paltin de camp"),
        "PAM": Species("broadleaf", "Sycamore maple", "Acer pseudoplatanus", "Paltin de munte"),
        "PLT": Species("broadleaf", "Aspen", "Populus tremula", "Plop tremurator"),
        "SA": Species("broadleaf", "White willow", "Salix alba", "Salcie alba"),
        "SAC": Species("broadleaf", "Goat willow", "Salix caprea", "Salcie capreasca"),
        "SC": Species("broadleaf", "Black locust", "Robinia pseudoacacia", "Salcam"),
        "SR": Species("broadleaf", "Rowan", "Sorbus aucuparia", "Scorus"),
        "ST": Species("broadleaf", "Pedunculate oak", "Quercus robur", "Stejar pedunculat"),
        "TE": Species("broadleaf", "Silver linden", "Tilia tomentosa", "Tei argintiu"),
        "ULM": Species("broadleaf", "Wych elm", "Ulmus glabra", "Ulm de munte"),
    }
)
SPECIES_GROUP: Final[Mapping[str, str]] = MappingProxyType({c: s.group for c, s in SPECIES.items()})


# CNN training budgets (receptive-field CNN). Fixed, not tuned; provisional and
# to be confirmed from the Stage 6 CNN pilot and benchmark. Kept here, the single
# source of truth, so they cannot drift; the tuned CNN hyperparameters live in
# utils.hp_search (CNN_SEARCH_SPACE).
CNN_MAX_EPOCHS: Final[int] = 40  # upper bound on epochs; early stopping halts sooner
CNN_PATIENCE: Final[int] = 7  # epochs without inner-val-loss improvement before stopping
CNN_WARMUP_EPOCHS: Final[int] = 5  # linear warm-up length before cosine decay
CNN_EARLY_STOP_MIN_DELTA: Final[float] = 1e-4  # min val-loss decrease counted as improvement
