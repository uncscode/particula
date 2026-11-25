"""Corner branding for graphical abstracts using the particula.toc style."""

from importlib.resources import files
from io import StringIO
from pathlib import Path
from xml.etree import ElementTree
from xml.sax.saxutils import escape

from matplotlib import rc_context
from matplotlib.patches import Circle


def add_toc_logo(fig, *, corner="bottom right"):
    """Place the Particula dots inside the figure margin.

    Args:
        fig: Matplotlib figure, normally three inches square.
        corner: "top left", "top right", "bottom left", or "bottom right".

    Returns:
        The logo axes, excluded from the scientific layout.
    """
    if corner not in ("top left", "top right", "bottom left", "bottom right"):
        raise ValueError("corner must specify top or bottom and left or right")

    asset = files(__package__).joinpath("brand", "dots.svg")
    # This resource ships inside the package.
    logo = ElementTree.fromstring(asset.read_bytes())  # noqa: S314
    left, top, logo_width, logo_height = map(
        float, logo.attrib["viewBox"].split()
    )
    width = 0.075
    height = (
        width
        * logo_height
        / logo_width
        * fig.get_figwidth()
        / fig.get_figheight()
    )
    pad = 0.045
    x = pad if corner.endswith("left") else 1 - pad - width
    y = pad if corner.startswith("bottom") else 1 - pad - height
    ax = fig.add_axes([x, y, width, height], zorder=10)
    ax.set_in_layout(False)
    ax.set_axis_off()
    ax.set(
        xlim=(left, left + logo_width),
        ylim=(top + logo_height, top),
        aspect="equal",
    )
    ax.set_gid("particula-dots")
    for dot in logo.findall("{http://www.w3.org/2000/svg}circle"):
        center = float(dot.attrib["cx"]), float(dot.attrib["cy"])
        ax.add_patch(
            Circle(
                center,
                float(dot.attrib["r"]),
                facecolor=logo.attrib["fill"],
                edgecolor="none",
            )
        )
    return ax


def save_toc_svg(fig, path, *, title, description):
    """Save the full TOC canvas as a reproducible SVG with accessible metadata.

    Args:
        fig: Completed Matplotlib figure.
        path: Destination SVG file.
        title: Short title for the standalone artwork.
        description: Accessible description of the scientific comparison.
    """
    path = Path(path)
    if path.suffix.lower() != ".svg":
        raise ValueError("TOC artwork requires an .svg destination")
    metadata = {"Title": title, "Description": description, "Date": None}
    stream = StringIO()
    with rc_context({"svg.hashsalt": path.stem, "savefig.bbox": None}):
        fig.savefig(stream, format="svg", metadata=metadata)
    # SVG viewers expose <desc> directly to assistive technology.
    drawing = stream.getvalue().replace(
        "</title>", f"</title>\n <desc>{escape(description)}</desc>", 1
    )
    # Matplotlib splits paths over lines and leaves trailing spaces.
    path.write_text(
        "\n".join(line.rstrip() for line in drawing.splitlines()) + "\n"
    )
