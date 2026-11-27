"""Compare two fitted sulfur networks with measured wet-to-dry diameter ratios."""

import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
from matplotlib import pyplot as plt
from matplotlib.patches import FancyArrowPatch
from particula.plotting.toc import add_toc_logo, save_toc_svg

ROOT = Path(__file__).resolve().parents[2]
INK, TEAL, RUST = "#243E43", "#135963", "#A44D21"


def arrow(ax, start, end, color):
    ax.add_patch(
        FancyArrowPatch(
            start,
            end,
            arrowstyle="-|>",
            mutation_scale=9,
            linewidth=1.5,
            color=color,
            shrinkA=0,
            shrinkB=0,
        )
    )


def main():
    with (ROOT / "docs/data/xiong-sea-spray/growth-time.csv").open() as stream:
        (example,) = [
            row for row in csv.DictReader(stream) if float(row["time_h"]) == 3
        ]
    with plt.style.context("particula.toc"):
        fig = plt.figure()
        ax = fig.add_axes([0, 0, 1, 1], xlim=(0, 1), ylim=(0.080, 1))
        ax.set_axis_off()

        def label(
            x,
            y,
            text,
            size,
            color=INK,
            *,
            weight="normal",
            align="center",
            box=None,
        ):
            ax.text(
                x,
                y,
                text,
                fontsize=size,
                color=color,
                ha=align,
                va="center",
                weight=weight,
                bbox=box,
            )

        label(
            0.055,
            0.933,
            "Sea salt under SO₂ and UV",
            10.7,
            weight="bold",
            align="left",
        )
        ax.plot([0.553, 0.553], [0.115, 0.845], color="#BCC9BE", lw=0.8)

        # The H₂SO₄ route belongs to both fits. Only the parallel fit can
        # capture particle bisulfite before converting it to sulfate.
        label(0.275, 0.812, "SO₂", 16, weight="bold")
        arrow(ax, (0.245, 0.762), (0.136, 0.681), TEAL)
        arrow(ax, (0.312, 0.762), (0.418, 0.681), RUST)
        label(0.138, 0.640, "Bisulfite", 7.7, TEAL)
        label(0.425, 0.640, "Sulfuric acid", 7.5, RUST)
        label(
            0.14,
            0.549,
            "HSO₃⁻",
            12.3,
            TEAL,
            weight="bold",
            box=dict(
                boxstyle="round,pad=.22",
                facecolor="#D6E6DD",
                edgecolor=TEAL,
                linewidth=1,
            ),
        )
        label(0.14, 0.470, "Particle S(IV)", 7.6, TEAL)
        label(0.425, 0.549, "H₂SO₄", 12.3, RUST, weight="bold")
        arrow(ax, (0.142, 0.428), (0.231, 0.299), TEAL)
        arrow(ax, (0.415, 0.492), (0.318, 0.299), RUST)
        label(0.103, 0.352, "oxidation", 7.3, TEAL)
        label(0.448, 0.374, "uptake", 7.3, RUST)
        label(0.276, 0.241, "Sulfate", 14, weight="bold")
        label(0.276, 0.170, "S(VI)", 9.3)

        left = 0.594
        label(left, 0.813, "Growth factor", 10.0, weight="bold", align="left")
        label(left, 0.715, "Measured", 9.2, align="left")
        label(
            left,
            0.638,
            f"{float(example['gf']):.2f} ± {float(example['spread']):.2f}",
            12.1,
            weight="bold",
            align="left",
        )
        label(left, 0.522, "Direct model", 9.2, align="left")
        label(
            left,
            0.451,
            f"{float(example['retention_neutral']):.3f}",
            12.1,
            weight="bold",
            align="left",
        )
        label(left, 0.391, "H₂SO₄ uptake", 7.8, RUST, align="left")
        label(left, 0.286, "Parallel model", 9.2, align="left")
        label(
            left,
            0.215,
            f"{float(example['parallel_neutral']):.3f}",
            12.1,
            weight="bold",
            align="left",
        )
        label(left, 0.155, "H₂SO₄ + bisulfite", 7.8, align="left")

        add_toc_logo(fig, corner="top right")
        save_toc_svg(
            fig,
            ROOT / "docs/images/xiong-sea-spray/toc.svg",
            title="Two reaction pathways fit sea-salt water uptake",
            description=(
                "SO₂ reaches sulfate through sulfuric-acid uptake or through "
                "oxidation of particle bisulfite. Against measured growth of "
                "1.67 ± 0.02, the direct model gives 1.657 and the parallel model 1.654."
            ),
        )
        plt.close(fig)


if __name__ == "__main__":
    main()
