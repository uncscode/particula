"""Corner branding for graphical abstracts using the particula.toc style."""

from importlib.resources import files

import matplotlib.image as mpimg


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

    asset = files(__package__).joinpath("brand", "dots.png")
    with asset.open("rb") as stream:
        image = mpimg.imread(stream, format="png")
    width = 0.075
    height = (
        width
        * image.shape[0]
        / image.shape[1]
        * fig.get_figwidth()
        / fig.get_figheight()
    )
    pad = 0.045
    x = pad if corner.endswith("left") else 1 - pad - width
    y = pad if corner.startswith("bottom") else 1 - pad - height
    ax = fig.add_axes([x, y, width, height], zorder=10)
    ax.set_in_layout(False)
    ax.set_axis_off()
    ax.imshow(image, interpolation="antialiased")
    return ax
