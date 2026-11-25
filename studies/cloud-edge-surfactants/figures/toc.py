"""Draw the measured dose comparison and the modeled cloud-edge response."""

import csv
from math import cos, radians, sin
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import Arc, Circle, PathPatch
from matplotlib.path import Path as DrawingPath
from matplotlib.transforms import Affine2D
from particula.plotting.toc import add_toc_logo

STUDY = Path(__file__).resolve().parents[1]
INK, TEAL, RUST = "#243E43", "#135963", "#A44D21"
PALE = "#CAD9D2"


def surfactant(ax, x, y, angle):
    """Draw a schematic sulfate head and alkyl tail pointing outwards."""
    transform = Affine2D().rotate_deg(angle).translate(x, y) + ax.transData
    ax.plot(
        [0, 0.012, 0.020, 0.030, 0.040],
        [0, 0.005, -0.004, 0.005, 0],
        color=INK,
        linewidth=1.25,
        solid_capstyle="round",
        transform=transform,
        zorder=4,
    )
    ax.add_patch(
        Circle(
            (0, 0),
            0.009,
            facecolor="#E6A159",
            edgecolor=INK,
            lw=0.8,
            transform=transform,
            zorder=5,
        )
    )


def render():
    """Draw three inflow-shift contrasts on one scale from a common zero."""
    with (STUDY / "plot_data/toc.csv").open() as stream:
        gains = {
            row["control"]: float(row["shift_gain_cm3"])
            for row in csv.DictReader(stream)
        }
    with plt.style.context("particula.toc"):
        fig = plt.figure()
        ax = fig.add_axes([0, 0, 1, 1], xlim=(0, 1), ylim=(0, 1))
        ax.set_axis_off()
        ax.text(0.06, 0.91, "Trapped acid droplets", size=11.7, weight="bold")
        ax.add_patch(Circle((0.164, 0.747), 0.095, fc=PALE, ec=TEAL, lw=1.3))
        ax.add_patch(
            Arc(
                (0.164, 0.747),
                0.14,
                0.14,
                theta1=93,
                theta2=154,
                color="white",
                lw=2.0,
                capstyle="round",
            )
        )
        # Sulfate heads sit inside the water boundary, tails extend into air.
        for angle in (12, 67, 122, 175, 224, 314):
            theta = radians(angle)
            surfactant(
                ax,
                0.164 + 0.087 * cos(theta),
                0.747 + 0.087 * sin(theta),
                angle,
            )
        ax.text(0.164, 0.60, "5–10 µm", ha="center", size=8.3)
        ax.text(
            0.31,
            0.775,
            "Lower surface tension",
            size=11.0,
            weight="bold",
            color=TEAL,
        )
        ax.text(
            0.31,
            0.719,
            "Dodecyl sulfate at 1/10\nthe octyl sulfate molar dose",
            size=9.3,
            va="top",
            linespacing=1.18,
        )

        # The scalloped boundary is a schematic cloud edge, separate from
        # the quantitative bars below it.
        edge = DrawingPath(
            [
                (0.0, 0.55),
                (0.08, 0.55),
                (0.10, 0.50),
                (0.18, 0.53),
                (0.24, 0.59),
                (0.30, 0.58),
                (0.35, 0.54),
                (0.43, 0.55),
                (0.46, 0.53),
                (0.49, 0.51),
                (0.58, 0.59),
                (0.66, 0.60),
                (0.73, 0.55),
                (0.82, 0.58),
                (0.87, 0.56),
                (0.91, 0.53),
                (0.95, 0.55),
                (0.98, 0.55),
                (1.0, 0.54),
                (1.0, 0.0),
                (0.0, 0.0),
                (0.0, 0.55),
            ],
            [DrawingPath.MOVETO]
            + [DrawingPath.CURVE4] * 18
            + [DrawingPath.LINETO] * 2
            + [DrawingPath.CLOSEPOLY],
        )
        ax.add_patch(PathPatch(edge, facecolor="#E1E9E2", edgecolor="none"))
        ax.text(
            0.06, 0.455, "Modeled smoke-cloud edge", size=11.6, weight="bold"
        )
        ax.text(
            0.06, 0.397, "Extra droplets/cm³ when dry inflow shifts", size=8.5
        )
        for y, control, label, color in (
            (0.303, "acid", "Citric acid alone", INK),
            (0.187, "finite", "Surfactants with finite supply", TEAL),
            (0.071, "fixed70", "Surface tension fixed at lab values", RUST),
        ):
            gain = gains[control]
            label_x = 0.06
            if control in ("finite", "fixed70"):
                surfactant(ax, 0.069, y + 0.037, 0)
                label_x = 0.127
            ax.text(label_x, y + 0.023, label, size=8.6)
            width = 0.75 * gain / 100
            ax.barh(y - 0.008, width, left=0.06, height=0.029, color=color)
            ax.text(
                0.077 + width,
                y - 0.008,
                f"+{gain:.1f}",
                va="center",
                size=10.8,
                weight="bold",
                color=color,
            )
        add_toc_logo(fig, corner="top right")
        output = STUDY.parents[1] / "docs/images/cloud-edge-surfactants"
        output.mkdir(parents=True, exist_ok=True)
        fig.savefig(output / "toc.png", dpi=300)
        plt.close(fig)


if __name__ == "__main__":
    render()
