"""Opt-in Matplotlib tools for paper-sized Particula figures.

Use `plt.style.context("particula.default")` around figure construction.
"""

from math import ceil

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.layout_engine import ConstrainedLayoutEngine

COLORS = {
    "teal": "#08786C",
    "rust": "#C2410C",
    "indigo": "#4338CA",
    "berry": "#BE185D",
}
_SIZES = {"square": (3.0, 3.0), "wide": (5.0, 3.5)}
_RAMPS = {
    "magnitude": ["#f0fdfa", "#9edfd3", "#45a89b", "#08786c", "#073d39"],
    "difference": ["#823216", "#db906b", "#faf8f3", "#6cbcaf", "#00534d"],
}


def colormap(name):
    """Return a fresh sequential magnitude or diverging difference colormap.

    Use with Matplotlib's own normalization, levels and colorbar functions.
    """
    return LinearSegmentedColormap.from_list(
        f"particula.{name}", _RAMPS[name], N=257
    )


def subplots(nrows=1, ncols=1, *, size="square", **kwargs):
    """Create an equal-panel grid. Finish with fit() after adding artists.

    Args:
        nrows: Number of rows.
        ncols: Number of columns.
        size: Physical size of each data axes in inches. Square is 3 by 3
            and wide is 5 by 3.5.
        **kwargs: Matplotlib subplot options, e.g. sharex or squeeze.

    Returns:
        A Matplotlib figure and its axes with ordinary subplots semantics.
    """
    width, height = _SIZES[size]
    engine = _PaperLayout(width, height, nrows, ncols)
    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(ncols * (width + 1), nrows * (height + 1)),
        layout=engine,
        **kwargs,
    )
    engine.axes = np.asarray(axes).ravel()
    return fig, axes


class _PaperLayout(ConstrainedLayoutEngine):
    """Keep the panel sizing contract with its native layout engine."""

    def __init__(self, width, height, nrows, ncols):
        super().__init__(wspace=0, hspace=0)
        self.panel = np.array([width, height])
        self.grid = np.array([ncols, nrows])
        self.finished = False
        self.outside = {}

    def execute(self, fig):
        """Preserve solved positions when redrawing for PNG, PDF or display."""
        if not self.finished:
            return super().execute(fig)


def legend(ax, *, placement="auto", **kwargs):
    """Place a measured legend inside the axes or in a right-hand gutter.

    Auto uses at most 40% of panel width and 35% of height inside. Right
    legends add columns if necessary. Pass other legend options by keyword.
    Call fit() after the final legend, axis label and colorbar are present.
    """
    if placement not in ("auto", "inside", "right"):
        raise ValueError("placement must be auto, inside, or right")
    fig = ax.get_figure()
    engine = fig.get_layout_engine()
    box = ax.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
    width, height = (
        engine.panel if isinstance(engine, _PaperLayout) else box.size
    )
    entry = ax.legend(**kwargs)
    if placement == "inside":
        return entry
    bounds = entry.get_window_extent().transformed(
        fig.dpi_scale_trans.inverted()
    )
    if placement == "auto" and (
        bounds.width <= 0.4 * width and bounds.height <= 0.35 * height
    ):
        return entry
    options = dict(kwargs)
    options.update(loc="center left", bbox_to_anchor=(1, 0.5))
    if "ncols" not in options and "ncol" not in options:
        options["ncols"] = min(
            max(1, len(entry.get_texts())), max(1, ceil(bounds.height / height))
        )
    entry = ax.legend(**options)
    if isinstance(engine, _PaperLayout):
        engine.outside[ax] = entry
    return entry


def _check_panel_aspect(ax, panel):
    """Reject incompatible data and box aspect constraints before resizing."""
    aspect = ax.get_aspect()
    ratio = ax.get_box_aspect()
    if aspect != "auto" and ax.get_adjustable() == "box":
        ratio = float(aspect) * ax.get_data_ratio()
    if ratio is not None and not np.isclose(
        ratio, panel[1] / panel[0], rtol=0.0001
    ):
        raise ValueError("Data aspect is incompatible with the panel size")


def fit(fig):
    """Size the canvas to labels, legends and colorbars at fixed panel size.

    Call after creating a figure with subplots() and adding all artists.
    Read the finished canvas dimensions in inches with fig.get_size_inches().
    The grid must have equally sized data axes. For images, use aspect='auto'
    unless the data aspect agrees with the requested physical panel shape.
    """
    engine = fig.get_layout_engine()
    if not isinstance(engine, _PaperLayout):
        raise ValueError("fit requires a figure created by plotting.subplots")
    axes = engine.axes
    for ax in axes:
        _check_panel_aspect(ax, engine.panel)
    # Give outside legends room before the first constrained-layout pass.
    gutters = np.zeros(engine.grid[0])
    for ax, entry in engine.outside.items():
        if ax.get_legend() is entry:
            col = ax.get_subplotspec().colspan.start
            legend_width = entry.get_window_extent().width / fig.dpi
            gutters[col] = max(gutters[col], legend_width + 0.2)
    minimum_width = (engine.panel[0] + 0.8) * engine.grid[0] + gutters.sum()
    if fig.get_figwidth() < minimum_width and not engine.finished:
        fig.set_figwidth(minimum_width)
    engine.finished = False
    for _ in range(16):
        fig.draw_without_rendering()
        canvas = fig.get_size_inches()
        sizes = np.array([ax.get_position().size * canvas for ax in axes])
        errors = engine.panel - sizes
        if np.max(np.abs(errors)) < 0.002:
            engine.finished = True
            return fig
        correction = errors.mean(axis=0) * engine.grid
        if np.max(np.abs(correction)) < 0.001:
            break
        fig.set_size_inches(canvas + correction)
    raise ValueError(
        "Panel sizes did not converge. Use an equal-cell grid and a "
        "data aspect compatible with the panel size."
    )
