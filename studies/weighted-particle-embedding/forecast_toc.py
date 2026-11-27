"""Draw the graphical abstract from the matched native-vapor collision runs."""

import csv
from pathlib import Path
from zipfile import Path as ZipPath
from zipfile import ZipFile

import matplotlib.pyplot as plt
from matplotlib.patches import Circle, FancyBboxPatch
from particula.plotting.toc import add_toc_logo, save_toc_svg

STUDY = Path(__file__).resolve().parent
IMAGES = STUDY.parents[1] / "docs/images/studies/weighted-particle-embedding"
INK, TEAL, RUST = "#243E43", "#135963", "#A44D21"
COOL, WARM = "#CAD9D2", "#EFD1A5"


def render():
    """Compare both collision laws on the same sampled pairs and seed."""
    with ZipFile(STUDY / "measurements.zip") as archive:
        table = ZipPath(archive, "stage-3/native-vapor-sampled-collisions.csv")
        with table.open() as stream:
            rows = {
                row["sampled_kernel"]: row
                for row in csv.DictReader(stream)
                if row["n"] == "1000000" and row["charge"] == "neutral"
            }
    physical, learned = rows["physical"], rows["learned"]
    if (
        physical["median_accepted_mergers"]
        != learned["median_accepted_mergers"]
    ):
        raise ValueError("The two runs must have equal merger counts.")
    baseline = float(learned["native_full_s"]) / 60
    hybrid = float(learned["median_total_s"]) / 60
    control = float(physical["median_total_s"]) / 60

    with plt.style.context("particula.toc"):
        fig = plt.figure()
        ax = fig.add_axes((0, 0, 1, 1), aspect="equal")
        ax.set(xlim=(0, 100), ylim=(0, 100))
        ax.axis("off")
        ax.text(
            6,
            93,
            "One million particles",
            weight="bold",
            fontsize=12,
            va="center",
        )
        ax.text(6, 85, "32 vapors · 1 simulated second", fontsize=9)

        # Each link marks a candidate pair, independent of the rate law.
        ax.add_patch(
            FancyBboxPatch(
                (6, 59),
                88,
                20,
                boxstyle="round,pad=0.5,rounding_size=3",
                facecolor="#F3E7D3",
                edgecolor="#A38F76",
                linewidth=0.8,
            )
        )
        pairs = [
            ((14, 67, 2.7), (27, 72, 4.2)),
            ((37, 71, 2.3), (49, 65, 3.5)),
            ((57, 66, 3.5), (70, 73, 2.5)),
            ((78, 72, 2.7), (88, 66, 3.2)),
        ]
        for first, second in pairs:
            ax.plot(
                [first[0], second[0]],
                [first[1], second[1]],
                "--",
                color=INK,
                lw=1.3,
            )
            for (x, y, radius), shade in zip(
                (first, second), (COOL, WARM), strict=True
            ):
                ax.add_patch(
                    Circle(
                        (x, y),
                        radius,
                        facecolor=shade,
                        edgecolor=INK,
                        lw=1,
                        zorder=3,
                    )
                )
        for y, label, minutes, color in (
            (49, "Full native physics", baseline, INK),
            (28, "Sampled pairs + neural term", hybrid, TEAL),
        ):
            ax.text(7, y, label, fontsize=9, va="bottom", color=color)
            ax.barh(
                y - 5, 65 * minutes / baseline, height=3.2, left=7, color=color
            )
            ax.text(
                95,
                y - 5,
                f"{minutes:.1f} min",
                ha="right",
                va="center",
                fontsize=9,
                color=color,
            )
        ax.text(
            7,
            8,
            f"Physical rate on sampled pairs: {control:.2f} min",
            color=INK,
            fontsize=8,
            va="center",
        )
        add_toc_logo(fig, corner="top right")
        save_toc_svg(
            fig,
            IMAGES / "forecast-toc.svg",
            title="Native and hybrid forecasts of one million particles",
            description=(
                "Schematic lines join candidate particle pairs. Whole-forecast "
                f"time fell from {baseline:.1f} minutes with native physics "
                f"to {hybrid:.2f} minutes with a neural jump-distance term on "
                f"sampled pairs. The physical control took {control:.2f} "
                "minutes. All forecasts used native vapor transfer."
            ),
        )
        plt.close(fig)


if __name__ == "__main__":
    render()
