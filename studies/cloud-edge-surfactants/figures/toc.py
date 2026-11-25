"""Draw the public cloud-edge graphical abstract from compact model counts."""

import csv
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch
from particula.plotting.toc import add_toc_logo

STUDY = Path(__file__).resolve().parents[1]
TEAL, RUST, INK, MUTED = "#076e70", "#a24527", "#263536", "#65737a"
PLACEMENTS = ("patterned", "opposed")


def render():
    """Draw the gallery thumbnail from six precomputed model counts."""
    with (STUDY / "plot_data/toc.csv").open() as stream:
        means = list(csv.DictReader(stream))
    table = {(r["placement"], r["control"]): int(r["total_cm3"]) for r in means}
    with plt.style.context("particula.toc"):
        fig = plt.figure()
        ax = fig.add_axes([0, 0, 1, 1], xlim=(0, 1), ylim=(0, 1), frameon=False)
        ax.set_axis_off()
        ax.text(0.065, 0.91, "Smoke cloud edge", weight="bold", size=12)
        # Context schematic: successive placements of the dry band, viewed
        # against stationary updraft paths. Geometry does not encode counts.
        for x in [0.14, 0.32, 0.50, 0.68, 0.86]:
            ax.add_patch(
                FancyArrowPatch(
                    (x - 0.05, 0.67),
                    (x + 0.02, 0.84),
                    connectionstyle="arc3,rad=.28",
                    arrowstyle="-|>",
                    mutation_scale=8,
                    lw=1.15,
                    color=MUTED,
                    zorder=1,
                )
            )
        ax.plot(
            [0.12, 0.38],
            [0.735, 0.735],
            lw=5,
            color=RUST,
            solid_capstyle="round",
        )
        ax.plot(
            [0.62, 0.88],
            [0.735, 0.735],
            lw=5,
            color=RUST,
            solid_capstyle="round",
        )
        ax.annotate(
            "",
            xy=(0.60, 0.735),
            xytext=(0.40, 0.735),
            arrowprops=dict(arrowstyle="->", color=INK, lw=1.3),
        )
        ax.text(0.50, 0.635, "Shift the dry inflow", ha="center", size=8.7)
        ax.plot([0.065, 0.935], [0.585, 0.585], color=INK, lw=0.65)
        ax.text(
            0.065, 0.515, "Droplets per cm³ at +100 m", weight="bold", size=9.2
        )
        for x, h in ((0.66, "Original"), (0.875, "Shifted")):
            ax.text(x, 0.425, h, ha="center", size=8.2)
        for y, control, label in (
            (0.324, "acid", "Citric acid"),
            (0.211, "finite", "Acid + surfactants"),
            (0.092, "fixed70", "70%-RH tensions fixed"),
        ):
            ax.text(0.065, y, label, va="center", size=8.3)
            for x, placement in zip((0.66, 0.875), PLACEMENTS, strict=True):
                ax.text(
                    x,
                    y,
                    f"{table[placement, control]:.0f}",
                    ha="center",
                    va="center",
                    size=12.3,
                    weight="bold",
                    color=RUST if control == "fixed70" else TEAL,
                )
        add_toc_logo(fig, corner="top right")
        output = STUDY.parents[1] / "docs/images/cloud-edge-surfactants"
        output.mkdir(parents=True, exist_ok=True)
        fig.savefig(output / "toc.png", dpi=300)
        plt.close(fig)


if __name__ == "__main__":
    render()
