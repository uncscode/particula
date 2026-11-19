"""Check physical output dimensions, scientific colors and native overrides."""

import io

import numpy as np
import pytest

mpl = pytest.importorskip("matplotlib")
from matplotlib import pyplot as plt  # noqa: E402
from matplotlib.colors import to_hex  # noqa: E402
from particula import plotting  # noqa: E402
from PIL import Image  # noqa: E402


@pytest.fixture(autouse=True)
def style():
    """Scope the package style and clean up figures."""
    with plt.style.context("particula.default"):
        yield
    plt.close("all")


def test_style():
    """Check native style use, style context and explicit artist overrides."""
    fig, ax = plt.subplots()
    lines = [ax.plot([0, 1])[0] for _ in range(4)]
    assert [to_hex(line.get_color()) for line in lines] == [
        color.lower() for color in plotting.COLORS.values()
    ]
    assert len({line.get_linestyle() for line in lines}) == 4
    assert ax.plot([0, 1], color="black", lw=5)[0].get_linewidth() == 5
    with plt.style.context({"axes.labelsize": 8}):
        assert plt.rcParams["axes.labelsize"] == 8
    assert plt.rcParams["axes.labelsize"] == 13
    assert fig.axes == [ax]


def luminance(cmap):
    """Compute relative luminance from linearized sRGB."""
    rgb = cmap(np.linspace(0, 1, 257))[:, :3]
    linear = np.where(
        rgb <= 0.04045, rgb / 12.92, ((rgb + 0.055) / 1.055) ** 2.4
    )
    return linear @ [0.2126, 0.7152, 0.0722]


def test_colormaps():
    """Monotone sequential lightness and a single neutral diverging peak."""
    assert np.all(np.diff(luminance(plotting.colormap("magnitude"))) < 0)
    values = luminance(plotting.colormap("difference"))
    assert np.all(np.diff(values[:129]) > 0)
    assert np.all(np.diff(values[128:]) < 0)
    assert to_hex(plotting.colormap("difference")(0.5)) == "#faf8f3"


@pytest.mark.parametrize(
    "size, inches", [("square", [3, 3]), ("wide", [5, 3.5])]
)
def test_export(size, inches):
    """Check the physical data area, full canvas and repeat PNG/PDF draws."""
    fig, axes = plotting.subplots(1, 2, size=size)
    line, field = axes
    for i in range(12):
        line.plot([0, 1], [0, i], label=f"Long comparison series {i}")
    entry = plotting.legend(line)
    line.set(xlabel="Time (min)", ylabel="Particle diameter (nm)")
    field.grid(False)
    image = field.imshow([[1, 2], [3, 4]], aspect="auto")
    bar = fig.colorbar(image, ax=field, label="Number concentration")
    plotting.fit(fig)
    canvas = fig.get_size_inches().copy()
    for ax in axes:
        np.testing.assert_allclose(
            ax.get_position().size * canvas, inches, atol=0.01
        )
    assert entry.get_window_extent().x0 >= line.bbox.x1
    assert entry.get_window_extent().x1 < field.bbox.x0
    assert bar.ax.bbox.x0 >= field.bbox.x1
    for ax in fig.axes:
        box = ax.get_tightbbox(fig.canvas.get_renderer())
        assert fig.bbox.contains(box.x0, box.y0)
        assert fig.bbox.contains(box.x1, box.y1)
    plotting.fit(fig)
    np.testing.assert_allclose(fig.get_size_inches(), canvas, atol=0.002)
    for fmt in ("png", "pdf"):
        buf = io.BytesIO()
        fig.savefig(buf, format=fmt, dpi=300)
        if fmt == "png":
            np.testing.assert_allclose(
                Image.open(buf).size, canvas * 300, atol=1
            )
        else:
            assert buf.getvalue().startswith(b"%PDF")
        np.testing.assert_allclose(
            line.get_position().size * canvas, inches, atol=0.01
        )


def test_placement():
    """Auto keeps short labels inside and explicit placement overrides it."""
    fig, ax = plotting.subplots()
    ax.plot([1, 2], label="A")
    for placement in ["auto", "right", "inside"]:
        entry = plotting.legend(ax, placement=placement)
        plotting.fit(fig)
        if placement == "right":
            assert entry.get_window_extent().x0 > ax.bbox.x1
        else:
            assert entry.get_window_extent().x1 < ax.bbox.x1
    with pytest.raises(ValueError, match="placement"):
        plotting.legend(ax, placement="left")
