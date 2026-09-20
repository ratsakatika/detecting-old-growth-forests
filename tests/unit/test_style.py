"""Unit tests for the figure style helpers in utils.style."""

import importlib
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless backend; no display required for PDF output.

import matplotlib.pyplot as plt
import pandas as pd
import pytest
from matplotlib import font_manager as fm
from pandas.testing import assert_frame_equal

import utils.style
from utils.style import (
    CATEGORICAL_PALETTE,
    DEFAULT_ASPECT,
    DEFAULT_DPI,
    DEFAULT_JPEG_QUALITY,
    DIVERGING_CMAP,
    FIGURE_WIDTHS_MM,
    FONT_STACK,
    PUBLICATION_FONT,
    SEQUENTIAL_CMAP,
    get_figure_size,
    register_fonts,
    save_figure,
    use_publication_style,
)


def test_constants_match_agents_md() -> None:
    assert CATEGORICAL_PALETTE == (
        "#2877B7",
        "#3BAF8F",
        "#A1D86E",
        "#FFD640",
        "#FF904E",
        "#A62E5B",
    )
    assert SEQUENTIAL_CMAP == "viridis"
    assert DIVERGING_CMAP == "RdBu_r"
    assert DEFAULT_DPI == 600
    assert DEFAULT_ASPECT == 0.62
    assert dict(FIGURE_WIDTHS_MM) == {"single": 90.0, "one_half": 140.0, "double": 190.0}


def test_publication_font_is_charis_with_fallbacks() -> None:
    # Charis SIL is the primary face; metric-compatible serifs follow as fallbacks.
    assert PUBLICATION_FONT == "Charis SIL"
    assert FONT_STACK[0] == PUBLICATION_FONT
    assert "DejaVu Serif" in FONT_STACK


def test_register_fonts_adds_charis_and_is_idempotent() -> None:
    register_fonts()
    assert PUBLICATION_FONT in {f.name for f in fm.fontManager.ttflist}
    # A bundled face resolves without falling back to a default font.
    resolved = fm.findfont(fm.FontProperties(family=PUBLICATION_FONT), fallback_to_default=False)
    assert "CharisSIL" in Path(resolved).name
    register_fonts()  # second call is a no-op and must not raise.


def test_use_publication_style_sets_typography_rcparams() -> None:
    # rc_context restores rcParams on exit so the global state is left untouched.
    with matplotlib.rc_context():
        use_publication_style()
        assert matplotlib.rcParams["font.family"] == ["serif"]
        assert matplotlib.rcParams["font.serif"][0] == PUBLICATION_FONT
        assert matplotlib.rcParams["mathtext.fontset"] == "stix"


def test_importing_style_does_not_touch_rcparams(monkeypatch: pytest.MonkeyPatch) -> None:
    # Importing must not apply the publication style; only calling it should.
    with matplotlib.rc_context({"font.family": ["sans-serif"]}):
        importlib.reload(utils.style)
        assert matplotlib.rcParams["font.family"] == ["sans-serif"]


def test_get_figure_size_named_widths() -> None:
    assert get_figure_size("single") == (90.0 / 25.4, 90.0 / 25.4 * DEFAULT_ASPECT)
    assert get_figure_size("double", aspect=1.0) == (190.0 / 25.4, 190.0 / 25.4)


def test_get_figure_size_explicit_height() -> None:
    # height_mm overrides aspect.
    assert get_figure_size("single", height_mm=60.0) == (90.0 / 25.4, 60.0 / 25.4)


def test_get_figure_size_numeric_width_in_mm() -> None:
    assert get_figure_size(100.0, aspect=0.5) == (100.0 / 25.4, 50.0 / 25.4)


def test_figure_widths_mapping_is_read_only() -> None:
    with pytest.raises(TypeError):
        FIGURE_WIDTHS_MM["single"] = 1.0  # type: ignore[index]


def test_get_figure_size_rejects_unknown_name() -> None:
    with pytest.raises(ValueError, match="Unknown figure width"):
        get_figure_size("triple")


def test_get_figure_size_rejects_non_positive() -> None:
    with pytest.raises(ValueError):
        get_figure_size(0.0)
    with pytest.raises(ValueError):
        get_figure_size(-1.0)
    with pytest.raises(ValueError):
        get_figure_size("single", aspect=0.0)
    with pytest.raises(ValueError):
        get_figure_size("single", aspect=-0.5)


def test_importing_style_creates_no_figures_dir(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Reloading re-executes the module body in a markerless cwd. If the figures
    # directory were created (or get_project_paths called) at import, this would
    # leave a 'figures' folder behind or raise.
    monkeypatch.chdir(tmp_path)
    importlib.reload(utils.style)
    assert not (tmp_path / "figures").exists()


def test_save_figure_writes_pdf(tmp_path: Path) -> None:
    fig = plt.figure()
    try:
        written = save_figure(fig, "001_demo", output_dir=tmp_path)
    finally:
        plt.close(fig)

    assert written == [tmp_path / "001_demo.pdf"]
    assert written[0].is_file()
    assert written[0].suffix == ".pdf"


def test_save_figure_returns_written_paths_and_makes_subfolders(tmp_path: Path) -> None:
    fig = plt.figure()
    try:
        written = save_figure(fig, "sub/002_demo.pdf", output_dir=tmp_path)
    finally:
        plt.close(fig)

    assert written == [tmp_path / "sub" / "002_demo.pdf"]
    assert written[0].is_file()


def test_save_figure_rejects_png(tmp_path: Path) -> None:
    fig = plt.figure()
    try:
        with pytest.raises(ValueError, match="pdf"):
            save_figure(fig, "003_demo", output_dir=tmp_path, formats=("png",))
    finally:
        plt.close(fig)

    assert not (tmp_path / "003_demo.png").exists()


def test_get_figure_size_rejects_non_positive_height() -> None:
    with pytest.raises(ValueError, match="height must be positive"):
        get_figure_size("single", height_mm=0.0)
    with pytest.raises(ValueError, match="height must be positive"):
        get_figure_size("single", height_mm=-5.0)


def test_save_figure_writes_data_csv(tmp_path: Path) -> None:
    df = pd.DataFrame({"lag_m": [0.0, 1.0, 2.0], "semivariance": [0.1, 0.2, 0.3]})
    fig = plt.figure()
    try:
        written = save_figure(fig, "010_demo", output_dir=tmp_path, data=df)
    finally:
        plt.close(fig)

    # The PDF is written and is still the only returned path.
    assert written == [tmp_path / "010_demo.pdf"]
    csv_path = tmp_path / "raw" / "010_demo.csv"
    assert csv_path.is_file()
    assert_frame_equal(pd.read_csv(csv_path), df)


def test_save_figure_writes_data_csv_in_notebook_subfolder(tmp_path: Path) -> None:
    df = pd.DataFrame({"x": [1, 2], "y": [3, 4]})
    fig = plt.figure()
    try:
        written = save_figure(fig, "020_nb/panel", output_dir=tmp_path, data=df)
    finally:
        plt.close(fig)

    # The raw subfolder sits beside the figure inside the notebook subfolder.
    assert written == [tmp_path / "020_nb" / "panel.pdf"]
    assert (tmp_path / "020_nb" / "raw" / "panel.csv").is_file()


def test_save_figure_writes_mapping_of_data(tmp_path: Path) -> None:
    parts = {
        "left": pd.DataFrame({"a": [1, 2]}),
        "right": pd.DataFrame({"b": [3.0, 4.0]}),
    }
    fig = plt.figure()
    try:
        save_figure(fig, "030_demo", output_dir=tmp_path, data=parts)
    finally:
        plt.close(fig)

    for suffix, frame in parts.items():
        csv_path = tmp_path / "raw" / f"030_demo_{suffix}.csv"
        assert csv_path.is_file()
        assert_frame_equal(pd.read_csv(csv_path), frame)


def test_save_figure_without_data_writes_no_csv(tmp_path: Path) -> None:
    fig = plt.figure()
    try:
        save_figure(fig, "040_demo", output_dir=tmp_path)
    finally:
        plt.close(fig)

    assert (tmp_path / "040_demo.pdf").is_file()
    assert not (tmp_path / "raw").exists()


def test_save_figure_rejects_bad_data_type(tmp_path: Path) -> None:
    fig = plt.figure()
    try:
        with pytest.raises(TypeError, match="DataFrame"):
            save_figure(fig, "050_demo", output_dir=tmp_path, data=[1, 2, 3])  # type: ignore[arg-type]
    finally:
        plt.close(fig)


# --- Raster compression ---------------------------------------------------


def _raster_figure() -> plt.Figure:
    """A figure whose only content is one continuous-tone raster.

    Smoothed noise over a gradient, standing in for the orthophoto basemap.
    Neither extreme would work as a fixture: white noise is incompressible by
    any codec, and a clean gradient Flate compresses better than JPEG, so in
    both cases save_figure would rightly decline to re-encode it.
    """
    import numpy as np
    from scipy.ndimage import gaussian_filter

    rng = np.random.default_rng(0)
    field = gaussian_filter(rng.random((400, 400, 3)).astype("float32"), sigma=(1, 1, 0))
    ramp = np.linspace(0, 1, 400, dtype="float32")[:, None, None]
    image = ((0.5 * ramp + 0.5 * (field - field.min()) / np.ptp(field)) * 255).astype("uint8")

    fig = plt.figure(figsize=(4, 4))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.imshow(image, interpolation="nearest")
    ax.set_axis_off()
    return fig


# One device pixel per array pixel, as the real map figures render. At the 600
# dpi default matplotlib upsamples the fixture 6x, smoothing away the texture
# until Flate beats JPEG and nothing qualifies for compression.
_RASTER_DPI = 100


def _filters(pdf_path: Path) -> list[str]:
    """The /Filter of every image XObject on page 1."""
    import pikepdf

    with pikepdf.open(pdf_path) as pdf:
        return [str(img.get("/Filter")) for img in pdf.pages[0].get_images().values()]


def test_save_figure_compresses_rasters_by_default(tmp_path: Path) -> None:
    fig, raw_fig = _raster_figure(), _raster_figure()
    try:
        (small,) = save_figure(fig, "compressed", output_dir=tmp_path, dpi=_RASTER_DPI)
        (large,) = save_figure(
            raw_fig, "raw", output_dir=tmp_path, dpi=_RASTER_DPI, raster_quality=None
        )
    finally:
        plt.close(fig)
        plt.close(raw_fig)

    assert _filters(small) == ["/DCTDecode"]
    assert _filters(large) == ["/FlateDecode"]
    assert small.stat().st_size < large.stat().st_size


def test_save_figure_keeps_vector_content_identical(tmp_path: Path) -> None:
    """Only image streams may change: text and fonts must survive the rewrite."""
    import pikepdf

    fig = _raster_figure()
    fig.text(0.5, 0.5, "Old-growth")
    try:
        (compressed,) = save_figure(fig, "vectors", output_dir=tmp_path, dpi=_RASTER_DPI)
        (plain,) = save_figure(
            _raster_figure(), "plain", output_dir=tmp_path, dpi=_RASTER_DPI, raster_quality=None
        )
    finally:
        plt.close("all")

    with pikepdf.open(compressed) as a, pikepdf.open(plain) as b:
        assert "/Font" in a.pages[0].Resources
        assert a.pages[0].Resources.XObject.keys() == b.pages[0].Resources.XObject.keys()


def test_save_figure_compression_is_idempotent(tmp_path: Path) -> None:
    """Re-running must skip the JPEG stream rather than compound its loss."""
    fig = _raster_figure()
    try:
        (path,) = save_figure(fig, "once", output_dir=tmp_path, dpi=_RASTER_DPI)
    finally:
        plt.close(fig)
    before = path.read_bytes()

    utils.style._compress_rasters(path, DEFAULT_JPEG_QUALITY)

    assert path.read_bytes() == before


def test_save_figure_leaves_small_rasters_lossless(tmp_path: Path) -> None:
    """Colourbars and gradient legends would band under JPEG."""
    import numpy as np

    fig = plt.figure(figsize=(1, 1))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.imshow(np.linspace(0, 1, 64).reshape(-1, 1), aspect="auto")
    ax.set_axis_off()
    try:
        (path,) = save_figure(fig, "ramp", output_dir=tmp_path, dpi=_RASTER_DPI)
    finally:
        plt.close(fig)

    assert _filters(path) == ["/FlateDecode"]


def test_save_figure_rejects_bad_raster_quality(tmp_path: Path) -> None:
    fig = plt.figure()
    try:
        with pytest.raises(ValueError, match="quality must be in 1-100"):
            save_figure(fig, "bad", output_dir=tmp_path, raster_quality=200)
    finally:
        plt.close(fig)


def test_save_figure_leaves_incompressible_rasters_alone(tmp_path: Path) -> None:
    """White noise is incompressible by any codec, so JPEG would only grow it."""
    import numpy as np

    rng = np.random.default_rng(0)
    fig = plt.figure(figsize=(4, 4))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.imshow(rng.integers(0, 256, (400, 400, 3), dtype="uint8"), interpolation="nearest")
    ax.set_axis_off()
    try:
        (path,) = save_figure(fig, "noise", output_dir=tmp_path, dpi=_RASTER_DPI)
    finally:
        plt.close(fig)

    assert _filters(path) == ["/FlateDecode"]


def test_save_figure_crops_to_bbox(tmp_path: Path) -> None:
    """The location inset crops this way rather than re-saving over the output."""
    fig = _raster_figure()
    ax = fig.axes[0]
    try:
        fig.canvas.draw()
        bbox = ax.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
        (cropped,) = save_figure(
            fig,
            "cropped",
            output_dir=tmp_path,
            dpi=_RASTER_DPI,
            bbox_inches=bbox,
            pad_inches=0,
        )
        (full,) = save_figure(_raster_figure(), "full", output_dir=tmp_path, dpi=_RASTER_DPI)
    finally:
        plt.close("all")

    assert cropped.is_file() and full.is_file()
    assert _filters(cropped) == ["/DCTDecode"], "crop must not bypass compression"


def test_save_figure_keeps_flate_when_it_already_wins(tmp_path: Path) -> None:
    """A smooth ramp encodes fine as JPEG, but Flate compresses it far smaller.

    Vertically constant, so the PDF's PNG predictor zeroes every row after the
    first and Flate collapses the stream (~1.5 kB against JPEG's ~24 kB). Over
    256 distinct colours, which keeps matplotlib from writing it as an indexed
    palette and so exercises the DeviceRGB path.
    """
    import numpy as np

    x = np.linspace(0, 1, 400, dtype="float32")
    row = np.dstack([x * 0.9 + 0.05, x * 0.5 + 0.2, 1 - x * 0.7])[0]
    fig = plt.figure(figsize=(4, 4))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.imshow((np.tile(row, (400, 1, 1)) * 255).astype("uint8"), interpolation="nearest")
    ax.set_axis_off()
    try:
        (path,) = save_figure(fig, "ramp_large", output_dir=tmp_path, dpi=_RASTER_DPI)
    finally:
        plt.close(fig)

    assert _filters(path) == ["/FlateDecode"]


def _polygon_figure(n: int = 2000) -> plt.Figure:
    """A vector-only figure with many vertices at full floating-point precision."""
    import numpy as np

    rng = np.random.default_rng(0)
    fig, ax = plt.subplots(figsize=(3, 2))
    for _ in range(20):
        xy = rng.random((n, 2)) * 100
        ax.plot(xy[:, 0], xy[:, 1], linewidth=0.3)
    ax.fill(rng.random(50) * 100, rng.random(50) * 100, alpha=0.5)
    ax.text(50, 50, "label", fontsize=6)
    return fig


def test_save_figure_rounds_path_coordinates_only(tmp_path: Path) -> None:
    from decimal import Decimal

    import pikepdf

    fig = _polygon_figure()
    (full,) = save_figure(fig, "full", output_dir=tmp_path, raster_quality=None)
    (rounded,) = save_figure(
        fig, "rounded", output_dir=tmp_path, raster_quality=None, path_decimals=2
    )
    plt.close(fig)

    assert rounded.stat().st_size < 0.8 * full.stat().st_size

    def operands(path: Path) -> dict[str, list]:
        found: dict[str, list] = {}
        with pikepdf.open(path) as pdf:
            for ops, op in pikepdf.parse_content_stream(pdf.pages[0]):
                found.setdefault(str(op), []).append(
                    [v for v in ops if isinstance(v, Decimal | int)]
                )
        return found

    before, after = operands(full), operands(rounded)
    assert len(after["l"]) == len(before["l"]) > 1000
    for b, a in zip(before["l"], after["l"], strict=True):
        for x, y in zip(b, a, strict=True):
            assert abs(float(x) - float(y)) <= 0.005 + 1e-9
            assert "." not in str(y) or len(str(y).split(".")[1]) <= 2
    # Matrices keep their full precision (they carry the text scale and image placement).
    assert after.get("cm") == before.get("cm")
    with pytest.raises(ValueError, match="decimals"):
        save_figure(_polygon_figure(10), "bad", output_dir=tmp_path, path_decimals=-1)


def test_save_figure_keeps_dots_in_the_name(tmp_path: Path) -> None:
    fig, ax = plt.subplots()
    ax.plot([0, 1], [0, 1])
    (written,) = save_figure(
        fig, "012/fig_1_panel_c.3_embedding", data=pd.DataFrame({"x": [1]}), output_dir=tmp_path
    )
    plt.close(fig)
    assert written == tmp_path / "012" / "fig_1_panel_c.3_embedding.pdf"
    assert (tmp_path / "012" / "raw" / "fig_1_panel_c.3_embedding.csv").is_file()
    (again,) = save_figure(_polygon_figure(10), "plain.pdf", output_dir=tmp_path)
    assert again == tmp_path / "plain.pdf"
