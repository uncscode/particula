"""Draw the TOC and pure-chloride figures from the archived joint fit."""

import csv
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import Circle, FancyBboxPatch, Rectangle
from particula.plotting.toc import add_toc_logo

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "website/public/study-data/straw-combustion/measurements.csv"
IMAGES = ROOT / "docs/images/straw-combustion"
INK = "#243E43"
TEAL = "#135963"
PAPER = "#FBF2E3"
SLATE = "#617679"


@plt.style.context("particula.toc")
def draw_toc(rows):
    """Compare measured counts and calibrated counts for the three SO2 feeds."""
    counts = {
        row["case"]: row
        for row in rows
        if row["dataset"] == "aerosol" and row["target"] == "count"
    }
    figure = plt.figure()
    diagram = figure.add_axes(
        (0, 0, 1, 1), xlim=(0, 100), ylim=(0, 100), aspect="equal"
    )
    diagram.axis("off")

    def label(x, y, text, size, align="center", weight="normal"):
        diagram.text(
            x,
            y,
            text,
            ha=align,
            va="center",
            fontsize=size,
            weight=weight,
            color=INK,
        )

    def flow(left, right):
        diagram.annotate(
            "",
            xy=(right, 83.8),
            xytext=(left, 83.8),
            arrowprops=dict(
                arrowstyle="-|>",
                color=SLATE,
                lw=1,
                mutation_scale=7,
                shrinkA=0,
                shrinkB=0,
            ),
        )

    label(14, 91, "Synthetic\nflue gas", 7.4)
    flow(7, 28)
    label(48, 94, "Cooling tube", 8, weight="bold")
    diagram.add_patch(
        FancyBboxPatch(
            (30, 80),
            36,
            7.5,
            boxstyle="round,pad=.2,rounding_size=2",
            fc="#E1E6DC",
            ec=INK,
            lw=1,
        )
    )
    for x, y, radius in [
        (40, 84, 0.48),
        (47, 82, 0.56),
        (52, 85, 0.67),
        (55, 82.6, 0.54),
        (60, 84, 0.7),
        (62.4, 82, 0.46),
    ]:
        diagram.add_patch(Circle((x, y), radius, fc=INK, lw=0))
    flow(67, 78)
    label(86, 94, "Counter", 8, weight="bold")
    diagram.add_patch(
        FancyBboxPatch(
            (79, 79.5),
            14,
            8.5,
            boxstyle="round,pad=.1,rounding_size=.7",
            fc=PAPER,
            ec=INK,
            lw=1,
        )
    )
    diagram.plot(
        [81, 83, 84, 85, 86.7, 88, 89, 91],
        [83, 83, 85.5, 81.7, 83, 83, 84, 84],
        color=TEAL,
        lw=0.7,
    )
    diagram.plot([6, 94], [75.8, 75.8], color="#CBCAB9", lw=0.65)
    label(6, 69, "Particles at the outlet", 10, "left", "bold")
    for x, series, face, edge, hatch in [
        (6, "Zeuthen measured", INK, INK, None),
        (57, "Particula fit", PAPER, TEAL, "////"),
    ]:
        diagram.add_patch(
            Rectangle(
                (x, 56.1),
                3.2,
                3.2,
                facecolor=face,
                edgecolor=edge,
                hatch=hatch,
                lw=0.7,
            )
        )
        label(x + 5, 57.7, series, 7.7, "left")

    chart = figure.add_axes((0.17, 0.17, 0.78, 0.355), facecolor=PAPER)
    chart.spines[["left", "top", "right"]].set_visible(False)
    chart.spines["bottom"].set_color(SLATE)
    for x, case in enumerate(("No SO2", "Reference", "High SO2")):
        record = counts[case]
        measured = float(record["measured"])
        predicted = float(record["predicted"])
        chart.bar(x - 0.17, measured, width=0.29, color=INK, zorder=2)
        chart.bar(
            x + 0.17,
            predicted,
            width=0.29,
            facecolor=PAPER,
            edgecolor=TEAL,
            hatch="////",
            linewidth=1,
            zorder=2,
        )
        chart.errorbar(
            x - 0.17,
            measured,
            yerr=float(record["scale"]),
            color=INK,
            fmt="none",
            capsize=2,
            lw=0.9,
            zorder=4,
        )
    chart.set(xlim=(-0.5, 2.5), ylim=(0, 36), yticks=[0, 10, 20, 30])
    chart.set_xticks([0, 1, 2], ["0", "50", "500"])
    chart.tick_params(axis="both", labelsize=7.2, length=0, pad=2, colors=INK)
    chart.set_ylabel("Millions per cm³", fontsize=7, labelpad=2, color=INK)
    chart.yaxis.grid(True, color="#DAD9CA", lw=0.6)
    chart.set_axisbelow(True)
    label(53, 6.4, "Sulfur dioxide in the feed (ppm)", 8, weight="bold")
    add_toc_logo(figure, corner="bottom left")
    figure.savefig(IMAGES / "toc.png")
    plt.close(figure)


def draw_pure_kcl(rows):
    """Compare number and mobility-derived mass across pure KCl feeds."""
    figure, axes = plt.subplots(1, 2, figsize=(8.2, 3.25), layout="constrained")
    for axis, target, units in zip(
        axes,
        ("count", "smps_mass"),
        ("Million particles per cm³", "Mobility-derived mass (mg/m³)"),
        strict=True,
    ):
        observations = sorted(
            (
                row
                for row in rows
                if row["dataset"] == "Fig14" and row["target"] == target
            ),
            key=lambda row: int(row["case"].split()[1]),
        )
        feed = [int(row["case"].split()[1]) for row in observations]
        axis.errorbar(
            feed,
            [float(row["measured"]) for row in observations],
            yerr=[float(row["scale"]) for row in observations],
            marker="o",
            ls="none",
            color=INK,
            capsize=3,
            label="Zeuthen ± working band",
        )
        axis.plot(
            feed,
            [float(row["predicted"]) for row in observations],
            "s--",
            color=TEAL,
            label="Particula fit",
        )
        axis.set(
            xlabel="Inlet KCl (ppm)", ylabel=units, xticks=feed, ylim=(0, None)
        )
        axis.spines[["right", "top"]].set_visible(False)
    axes[0].legend(frameon=False, fontsize=8)
    figure.savefig(IMAGES / "pure-kcl.png", dpi=180)
    plt.close(figure)


def selected_rows():
    """Load the archived joint fit for the study figures."""
    with DATA.open(encoding="utf-8", newline="") as stream:
        return [
            r
            for r in csv.DictReader(stream)
            if r["tag"] == "all"
            and r["model"] == "combined"
            and r["chemistry_free"] == "True"
        ]


if __name__ == "__main__":
    selected = selected_rows()
    draw_toc(selected)
    draw_pure_kcl(selected)
