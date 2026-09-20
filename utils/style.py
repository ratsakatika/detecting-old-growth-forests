"""Figure style helpers for the old-growth-forests publication repository.

Centralises the publication figure conventions from ``AGENTS.md`` ("Figures"):
the publication typeface (Charis SIL, applied via :func:`use_publication_style`),
the categorical, sequential and diverging palettes, the standard journal column
widths and default aspect, and a single :func:`save_figure` entry point that
writes vector PDFs only (hard rule 5 forbids raw ``plt.savefig``).

Importing this module only binds names: it performs no input/output, mutates no
matplotlib ``rcParams`` and creates no directories. State changes happen only
when a function is called -- :func:`use_publication_style` and
:func:`register_fonts` mutate the global matplotlib configuration, and
:func:`save_figure` creates the figures directory.
"""

from __future__ import annotations

from collections.abc import Mapping
from types import MappingProxyType
from typing import TYPE_CHECKING, Final

from utils.paths import find_repo_root, get_project_paths
from utils.terminology import (
    PALETTE_CATEGORICAL_ORDER,
    PALETTE_DIVERGING,
    PALETTE_SEQUENTIAL,
)

if TYPE_CHECKING:
    from pathlib import Path as _Path

    from matplotlib.figure import Figure
    from matplotlib.transforms import Bbox
    from pandas import DataFrame

# Re-exported from the canonical source of truth (utils.terminology) so the
# style layer and the metadata/registry layer can never drift apart.
CATEGORICAL_PALETTE: Final[tuple[str, ...]] = PALETTE_CATEGORICAL_ORDER
SEQUENTIAL_CMAP: Final[str] = PALETTE_SEQUENTIAL
DIVERGING_CMAP: Final[str] = PALETTE_DIVERGING

# Publication typeface. The Charis SIL faces are bundled in-repo under
# ``assets/fonts`` (registered by :func:`register_fonts`) so figures render
# identically without relying on a system font install. The remaining families
# are metric-compatible serif fallbacks, used only if the bundled font is
# somehow unavailable.
PUBLICATION_FONT: Final[str] = "Charis SIL"
FONT_STACK: Final[tuple[str, ...]] = (
    PUBLICATION_FONT,
    "Times New Roman",
    "Liberation Serif",
    "Nimbus Roman",
    "DejaVu Serif",
)

# The bundled TrueType faces, relative to the repository root.
_FONTS_SUBDIR: Final[str] = "assets/fonts"

# Guard so the bundled faces are added to matplotlib at most once per session.
_fonts_registered: bool = False

DEFAULT_DPI: Final[int] = 600
DEFAULT_ASPECT: Final[float] = 0.62

# Matplotlib stores embedded rasters with lossless Flate, which is close to
# worst-case for the continuous-tone basemaps this project draws: the 018
# methods figures compressed to only 46% of raw, about 5 MB of image data each.
# :func:`save_figure` re-encodes them as JPEG at this quality, which halves a map
# figure at a worst-case error of 12/255 -- invisible in print. No chroma
# subsampling, since map overlays carry sharp saturated colour edges.
DEFAULT_JPEG_QUALITY: Final[int] = 98

# Rasters smaller than this stay lossless: they are colourbars and gradient
# legends, where JPEG's block structure reads as banding in a smooth ramp.
_MIN_JPEG_PIXELS: Final[int] = 100_000

MM_PER_INCH: Final[float] = 25.4

# Canonical column widths in millimetres, matching Elsevier
# figure specifications. Converted to inches inside get_figure_size,
# because matplotlib's figsize is always in inches.
FIGURE_WIDTHS_MM: Final[Mapping[str, float]] = MappingProxyType(
    {
        "single": 90.0,
        "one_half": 140.0,
        "double": 190.0,
    }
)

# Only vector PDF output is permitted (AGENTS.md, "Figures": "PDF only").
_ALLOWED_FORMAT: Final[str] = "pdf"


def register_fonts() -> None:
    """Register the bundled Charis SIL faces with matplotlib's font manager.

    Adds every ``CharisSIL-*.ttf`` in ``assets/fonts`` to matplotlib so the
    publication typeface resolves from within the repository, independent of any
    system font install. The lookup of the repository root and the matplotlib
    import both happen here rather than at module import, so importing
    :mod:`utils.style` remains side-effect free.

    Idempotent: the first call registers all faces and later calls are no-ops.
    """
    global _fonts_registered
    if _fonts_registered:
        return

    from matplotlib import font_manager as fm

    fonts_dir = find_repo_root() / _FONTS_SUBDIR
    for ttf in sorted(fonts_dir.glob("CharisSIL-*.ttf")):
        fm.fontManager.addfont(str(ttf))
    _fonts_registered = True


def use_publication_style() -> None:
    """Apply the publication typography to matplotlib's global ``rcParams``.

    Registers the bundled fonts (see :func:`register_fonts`) and sets the serif
    family to Charis SIL with metric-compatible fallbacks, plus a matching STIX
    math font. This is the single place the publication typeface is defined, so
    every figure notebook calls it once near the top and a future change of face
    is a one-line edit to :data:`FONT_STACK`.

    Only typography is set here; figure sizes, colours and other per-figure
    ``rcParams`` remain the caller's choice (see :func:`get_figure_size` and the
    palette constants). Unlike importing this module, calling this function
    mutates the global matplotlib configuration.
    """
    register_fonts()

    import matplotlib as mpl

    mpl.rcParams["font.family"] = "serif"
    mpl.rcParams["font.serif"] = list(FONT_STACK)
    mpl.rcParams["mathtext.fontset"] = "stix"


def get_figure_size(
    width: str | float = "single",
    aspect: float = DEFAULT_ASPECT,
    height_mm: float | None = None,
) -> tuple[float, float]:
    """Return a ``(width, height)`` figure size in inches.

    Args:
        width: Either a named column width (a key of :data:`FIGURE_WIDTHS_MM`)
            or a positive width in millimetres used directly.
        aspect: Height-to-width ratio, used only when ``height_mm`` is ``None``;
            height is ``width * aspect``.
        height_mm: Explicit height in millimetres. When given, it overrides
            ``aspect``.

    Returns:
        The ``(width, height)`` size in inches, for matplotlib ``figsize``.

    Raises:
        ValueError: If ``width`` is an unknown name, or if any resulting
            dimension is not positive.
    """
    if isinstance(width, str):
        if width not in FIGURE_WIDTHS_MM:
            allowed = ", ".join(sorted(FIGURE_WIDTHS_MM))
            raise ValueError(f"Unknown figure width {width!r}; choose one of: {allowed}.")
        width_mm = FIGURE_WIDTHS_MM[width]
    else:
        width_mm = float(width)

    if width_mm <= 0:
        raise ValueError(f"Figure width must be positive, got {width_mm} mm.")

    if height_mm is not None:
        height_value = float(height_mm)
        if height_value <= 0:
            raise ValueError(f"Figure height must be positive, got {height_value} mm.")
    else:
        if aspect <= 0:
            raise ValueError(f"Figure aspect must be positive, got {aspect}.")
        height_value = width_mm * aspect

    return (width_mm / MM_PER_INCH, height_value / MM_PER_INCH)


def _compress_rasters(path: _Path, quality: int) -> None:
    """Re-encode the large RGB rasters in a PDF as JPEG, in place.

    Only image XObjects are rewritten, so the page content stream, the embedded
    fonts and every path and glyph stay exactly as matplotlib wrote them. Left
    lossless: soft masks (the alpha channel is ``/DeviceGray``, and JPEG ringing
    on a mask edge shows as a halo), rasters under :data:`_MIN_JPEG_PIXELS`, and
    anything already JPEG -- so this is idempotent and never compounds loss.
    """
    import io

    import numpy as np
    import pikepdf
    from PIL import Image

    if not 1 <= quality <= 100:
        raise ValueError(f"JPEG quality must be in 1-100, got {quality}.")

    with pikepdf.open(path, allow_overwriting_input=True) as pdf:
        rewrote = False
        for page in pdf.pages:
            for img in page.get_images().values():
                if (
                    img.get("/Filter") == pikepdf.Name("/DCTDecode")
                    or img.get("/ColorSpace") != pikepdf.Name("/DeviceRGB")
                    or int(img.get("/BitsPerComponent", 0)) != 8
                    or int(img.Width) * int(img.Height) < _MIN_JPEG_PIXELS
                ):
                    continue
                samples = np.frombuffer(img.read_bytes(), dtype=np.uint8)
                buffer = io.BytesIO()
                try:
                    Image.fromarray(samples.reshape(int(img.Height), int(img.Width), 3)).save(
                        buffer, format="JPEG", quality=quality, subsampling=0, optimize=True
                    )
                except OSError:
                    # Pillow raises rather than returning an oversized buffer when
                    # JPEG expands the data, which it does on near-incompressible
                    # texture. Nothing to gain there anyway; keep the Flate stream.
                    continue
                if buffer.tell() >= len(img.read_raw_bytes()):
                    continue  # already smaller as Flate
                # filter= declares the bytes are *already* DCT-encoded, and clears
                # /DecodeParms, whose PNG predictor belongs to the Flate stream
                # being replaced and would corrupt the JPEG if left behind.
                img.write(buffer.getvalue(), filter=pikepdf.Name("/DCTDecode"))
                rewrote = True
        if rewrote:
            pdf.save(path)


# Operators whose operands are all coordinates in the current user space.
_PATH_OPERATORS: Final[frozenset[str]] = frozenset({"m", "l", "c", "v", "y", "re"})


def _round_path_coordinates(path: _Path, decimals: int) -> None:
    """Round the path coordinates in a PDF's content streams, in place.

    Matplotlib writes every vertex with six or more decimals of a point, far below
    print resolution: at two decimals the largest displacement is 0.005 pt
    (1.8 um), and a map of a few hundred thousand parcel vertices shrinks by about
    a third. Only the operands of the path-construction operators are rounded,
    and only while the current transformation matrix scales user space by at most
    one point per unit, so transformation matrices, text positioning, dash
    patterns and any path drawn in a magnified space keep their full precision.
    Page content streams and Form XObjects are both rewritten, Flate-compressed.
    """
    from decimal import Decimal

    import pikepdf

    if decimals < 0:
        raise ValueError(f"decimals must be non-negative, got {decimals}.")

    def rounded(value: object) -> object:
        if isinstance(value, Decimal):
            text = f"{value:.{decimals}f}".rstrip("0").rstrip(".") if decimals else f"{value:.0f}"
            return Decimal(text or "0")
        return value

    def rewrite(instructions: list) -> list:
        scale, stack, out = 1.0, [], []
        for operands, operator in instructions:
            name = str(operator)
            if name == "q":
                stack.append(scale)
            elif name == "Q":
                scale = stack.pop() if stack else 1.0
            elif name == "cm":
                a, b, c, d = (float(v) for v in operands[:4])
                scale *= max(abs(a), abs(b), abs(c), abs(d))
            elif name in _PATH_OPERATORS and scale <= 1.0:
                operands = [rounded(v) for v in operands]
            out.append((operands, operator))
        return out

    with pikepdf.open(path, allow_overwriting_input=True) as pdf:
        seen: set[tuple[int, int]] = set()

        def visit_forms(resources: pikepdf.Object | None) -> None:
            xobjects = resources.get("/XObject") if resources is not None else None
            if xobjects is None:
                return
            for key in xobjects.keys():
                xobject = xobjects[key]
                if xobject.get("/Subtype") != pikepdf.Name("/Form") or xobject.objgen in seen:
                    continue
                seen.add(xobject.objgen)
                xobject.write(
                    pikepdf.unparse_content_stream(rewrite(pikepdf.parse_content_stream(xobject)))
                )
                visit_forms(xobject.get("/Resources"))

        for page in pdf.pages:
            page.Contents = pdf.make_stream(
                pikepdf.unparse_content_stream(rewrite(pikepdf.parse_content_stream(page)))
            )
            visit_forms(page.get("/Resources"))
        pdf.save(path)


def _write_figure_data(data: DataFrame | Mapping[str, DataFrame], target: _Path) -> None:
    """Write a figure's underlying data as CSV beside the figure, in ``raw/``.

    The CSV is the exact data used to draw the figure. A single
    :class:`pandas.DataFrame` is written to ``<parent>/raw/<stem>.csv``; a mapping
    of suffix to :class:`~pandas.DataFrame` is written as one CSV per entry,
    ``<parent>/raw/<stem>_<suffix>.csv``. The index is not written.

    Args:
        data: The figure data: one DataFrame, or a mapping of suffix to DataFrame.
        target: The figure's resolved path without its suffix (its parent and
            name locate the CSVs).

    Raises:
        TypeError: If ``data`` is neither a DataFrame nor a mapping of DataFrames.
    """
    import pandas as pd

    raw_dir = target.parent / "raw"
    stem = target.name

    if isinstance(data, pd.DataFrame):
        raw_dir.mkdir(parents=True, exist_ok=True)
        data.to_csv(raw_dir / f"{stem}.csv", index=False)
    elif isinstance(data, Mapping):
        raw_dir.mkdir(parents=True, exist_ok=True)
        for suffix, frame in data.items():
            frame.to_csv(raw_dir / f"{stem}_{suffix}.csv", index=False)
    else:
        raise TypeError(
            "data must be a pandas DataFrame or a mapping of suffix to DataFrame, "
            f"got {type(data).__name__}."
        )


def save_figure(
    fig: Figure,
    name: str | _Path,
    *,
    data: DataFrame | Mapping[str, DataFrame] | None = None,
    output_dir: str | _Path | None = None,
    formats: tuple[str, ...] = ("pdf",),
    dpi: int = DEFAULT_DPI,
    raster_quality: int | None = DEFAULT_JPEG_QUALITY,
    path_decimals: int | None = None,
    bbox_inches: str | Bbox | None = None,
    pad_inches: float | None = None,
) -> list[_Path]:
    """Save ``fig`` as one or more vector PDFs and return the written paths.

    Callers organise figures into per-notebook subfolders by passing
    ``name="<notebook_stem>/<figure>"``; the subfolder is created automatically.
    When ``data`` is given, the exact data used to draw the figure is also
    persisted as CSV in a ``raw`` subfolder beside the figure (see
    :func:`_write_figure_data`); the PDF output is unaffected.

    Args:
        fig: The matplotlib figure to save.
        name: Figure name, optionally including a ``.pdf`` suffix and/or
            subfolders relative to ``output_dir``; other dots are kept.
        data: Optional figure data to persist as CSV. A single
            :class:`pandas.DataFrame` is written to ``raw/<name>.csv``; a mapping
            of suffix to DataFrame is written as ``raw/<name>_<suffix>.csv`` per
            entry. When ``None``, no CSV is written.
        output_dir: Destination directory. When ``None``, the project ``figures``
            directory (from :func:`utils.paths.get_project_paths`) is used.
        formats: Output formats. Only ``"pdf"`` is permitted.
        dpi: Resolution passed to ``fig.savefig`` (affects any rasterised
            elements embedded in the PDF).
        raster_quality: JPEG quality (1-100) for any rasters embedded in the PDF
            (see :func:`_compress_rasters` and :data:`DEFAULT_JPEG_QUALITY`).
            Vectors, text and fonts are never touched, so a purely vector figure
            is unaffected. Pass ``None`` to keep the rasters Flate-compressed.
        path_decimals: When given, round every path coordinate in the PDF to this
            many decimals of a point (see :func:`_round_path_coordinates`); ``2`` is
            visually lossless and shrinks vertex-heavy maps by about a third.
        bbox_inches: Optional crop passed to ``fig.savefig``: ``"tight"``, or a
            :class:`~matplotlib.transforms.Bbox` in figure inches (for example
            an axes rectangle, to write a PDF with no surrounding white space).
            Cropping here rather than re-saving over the returned path keeps the
            write inside this one entry point, so the raster compression is not
            silently discarded by the second write.
        pad_inches: Padding around ``bbox_inches``; ignored when it is ``None``.

    Returns:
        The PDF paths written, in the order of ``formats``.

    Raises:
        ValueError: If any requested format is not PDF, if ``raster_quality`` is
            outside 1-100, or if ``path_decimals`` is negative.
        TypeError: If ``data`` is neither a DataFrame nor a mapping of DataFrames.
    """
    from pathlib import Path

    for fmt in formats:
        if fmt.lower().lstrip(".") != _ALLOWED_FORMAT:
            raise ValueError(
                f"Only {_ALLOWED_FORMAT!r} output is permitted (AGENTS.md, "
                f"'Figures'); got {fmt!r}."
            )

    base_dir = Path(output_dir) if output_dir is not None else get_project_paths().figures
    base = Path(name)
    if base.suffix.lower() == f".{_ALLOWED_FORMAT}":
        base = base.with_suffix("")
    # Only a trailing ``.pdf`` is a suffix; any other dot is part of the name
    # (``fig_1_panel_c.3_tessera_embedding``), so outputs are built by appending.
    target = base if base.is_absolute() else base_dir / base

    written: list[Path] = []
    for fmt in formats:
        path = target.with_name(f"{target.name}.{fmt.lower().lstrip('.')}")
        path.parent.mkdir(parents=True, exist_ok=True)
        savefig_kwargs: dict[str, object] = {"dpi": dpi}
        if bbox_inches is not None:
            savefig_kwargs["bbox_inches"] = bbox_inches
        if pad_inches is not None:
            savefig_kwargs["pad_inches"] = pad_inches
        fig.savefig(path, **savefig_kwargs)  # type: ignore[arg-type]
        if raster_quality is not None:
            _compress_rasters(path, raster_quality)
        if path_decimals is not None:
            _round_path_coordinates(path, path_decimals)
        written.append(path)

    if data is not None:
        _write_figure_data(data, target)

    return written
