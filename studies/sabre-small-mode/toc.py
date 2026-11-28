"""Draw the SABRE graphical abstract from the matched-window summaries."""

import json
from math import log10
from pathlib import Path
from zipfile import ZipFile

import pandas as pd
from matplotlib import pyplot as plt
from matplotlib.patches import Ellipse, Polygon
from matplotlib.transforms import Affine2D
from particula.plotting.toc import add_toc_logo, save_toc_svg

HERE = Path(__file__).resolve().parent
INK = "#243E43"
TEAL = "#135963"
RUST = "#A44D21"
PALE = "#CAD9D2"
SLATE = "#617679"


def aircraft(ax):
    """Draw a schematic twin-engine high-altitude research aircraft."""
    transform = (
        Affine2D().scale(0.72).rotate_deg(-19).translate(81, 12.2)
        + ax.transData
    )
    outline = [
        (15, 0),
        (13, -1),
        (5, -1.4),
        (1, -15),
        (-1.1, -15.5),
        (-1.2, -1.4),
        (-9, -0.9),
        (-11.5, -5),
        (-13, -5),
        (-12.3, 0),
        (-13, 5),
        (-11.5, 5),
        (-9, 0.9),
        (-1.2, 1.4),
        (-1.1, 15.5),
        (1, 15),
        (5, 1.4),
        (13, 1),
    ]
    ax.add_patch(Polygon(outline, closed=True, color=INK, transform=transform))
    for y in [-4.5, 4.5]:
        ax.add_patch(Ellipse((2.5, y), 8.5, 2, color=INK, transform=transform))


def contrast_axis(ax, reported, upper_bound):
    """Plot both maximum-to-minimum ratios on one logarithmic scale."""

    def position(ratio):
        return 7 + 85 * log10(ratio) / log10(12)

    for y, label, value, color, marker, fmt in [
        (60, "Area shifts alone, maximum", upper_bound, TEAL, "|", ".2f"),
        (77, "Preliminary field estimates", reported, RUST, "o", ".1f"),
    ]:
        ax.text(6, y - 3.8, label, va="bottom", fontsize=8.7)
        ax.text(
            94,
            y - 3.8,
            f"{value:{fmt}}×",
            fontsize=10,
            color=color,
            ha="right",
            va="bottom",
            weight="bold",
        )
        ax.plot(
            [position(1), position(12)],
            [y, y],
            color=PALE,
            lw=1.2,
            solid_capstyle="butt",
        )
        ax.plot(
            [position(1), position(value)],
            [y, y],
            color=color,
            lw=3,
            solid_capstyle="butt",
        )
        ax.plot(
            position(value),
            y,
            marker=marker,
            markersize=5,
            markeredgewidth=1.45,
            color=color,
        )
    for tick in (1, 2, 5, 10):
        x = position(tick)
        ax.plot([x, x], [78.5, 80], color=SLATE, lw=0.55)
        ax.text(x, 83, f"{tick}×", ha="center", va="center", fontsize=8)


def main():
    """Render the graphical abstract from the archived tracer-group weights."""
    archive = (
        HERE.parents[1]
        / "website/public/study-data/sabre-small-mode/figure-data.zip"
    )
    with ZipFile(archive) as data:
        result = pd.DataFrame(
            json.loads(data.read("figure-values.json"))["toc"]
        ).iloc[0]
    output = HERE.parents[1] / "docs/images/sabre-small-mode/toc.svg"
    area_range = (
        f"{100 * result.min_small_weight:.1f}–"
        f"{100 * result.max_small_weight:.1f}%"
    )
    with plt.style.context("particula.toc"):
        fig = plt.figure()
        ax = fig.add_axes([0, 0, 1, 1], xlim=(0, 100), ylim=(100, 0))
        ax.set_axis_off()
        ax.text(6, 8.7, "Arctic SABRE", fontsize=14, weight="bold", va="center")
        ax.text(6, 16.5, "Eight air-mass groups", fontsize=9.6, va="center")
        aircraft(ax)
        ax.plot([6, 94], [24, 24], color=SLATE, lw=0.55)
        ax.text(6, 30, "Small-mode wet-area share", fontsize=8.5, va="center")
        ax.text(
            94,
            30,
            area_range,
            ha="right",
            fontsize=9.1,
            weight="bold",
            color=TEAL,
            va="center",
        )
        ax.text(
            6,
            42,
            "N₂O₅ uptake contrast",
            fontsize=13,
            weight="bold",
            va="center",
        )
        contrast_axis(
            ax,
            result.field_max_over_min,
            result.two_population_max_over_min_bound,
        )
        ax.text(
            6,
            93,
            "Highest / lowest uptake probability",
            fontsize=8,
            va="center",
        )
        add_toc_logo(fig, corner="bottom right")
        save_toc_svg(
            fig,
            output,
            title="Surface fractions and the SABRE N₂O₅ uptake contrast",
            description=(
                f"The fitted small mode supplies {area_range} of modeled wet "
                "area across eight groups of SABRE air. At fixed reaction "
                "probabilities for the two particle populations, shifting "
                "their surface fractions permits at most a "
                f"{result.two_population_max_over_min_bound:.2f}-fold spread "
                "in effective uptake probability. Preliminary field-inferred "
                f"values span {result.field_max_over_min:.1f}-fold. "
                "The ratios share a log scale beginning at one. "
                "Matching is by flight date and nitrous oxide tracer group."
            ),
        )
        plt.close(fig)


if __name__ == "__main__":
    main()
