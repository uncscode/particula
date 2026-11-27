"""Graphical abstract for the REACh study, drawn from the retained optical spectra."""

import re
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse, Rectangle
import pandas as pd
from zipfile import ZipFile

from particula.plotting.toc import add_toc_logo

HERE = Path(__file__).resolve().parent
INK, TEAL, RUST = "#243E43", "#135963", "#A44D21"
PAPER, COOL, SLATE = "#FBF2E3", "#CAD9D2", "#617679"


def draw_chamber(ax):
    ax.set(xlim=(0, 1), ylim=(0, 1))
    ax.axis("off")
    ax.text(0.01, 0.96, "REACh chamber · fans off", va="top", weight="bold", color=INK)
    left, right = 0.10, 0.44
    ax.add_patch(Ellipse((0.27, 0.19), 0.34, 0.13, facecolor=COOL, edgecolor=INK))
    ax.add_patch(Rectangle((left, 0.19), right - left, 0.50, facecolor=COOL, edgecolor="none"))
    for x in (left, right):
        ax.plot([x, x], [0.19, 0.69], color=INK, lw=1.1)
    ax.add_patch(Ellipse((0.27, 0.69), 0.34, 0.13, facecolor=PAPER, edgecolor=INK))
    for x, y, r in [
        (0.16, 0.60, 0.008),
        (0.24, 0.54, 0.007),
        (0.32, 0.61, 0.009),
        (0.37, 0.48, 0.007),
        (0.20, 0.29, 0.008),
        (0.29, 0.24, 0.007),
        (0.35, 0.32, 0.008),
    ]:
        ax.plot(x, y, "o", color=SLATE, ms=2.5 * r / 0.008, mew=0)
    for y, color, style in [(0.36, INK, "-"), (0.48, RUST, (0, (2.5, 2)))]:
        ax.plot([0.04, 0.49], [y, y], ls=style, lw=1.3, color=color)
    # Separate labels while retaining the vertical order of the two sampling paths.
    ax.plot([0.49, 0.52], [0.48, 0.54], color=RUST, lw=0.8)
    ax.text(0.54, 0.58, "Proposed view", color=RUST, fontsize=8, va="center")
    ax.text(0.54, 0.445, "5.5 cm higher", color=RUST, fontsize=7, va="center")
    ax.plot([0.49, 0.52], [0.36, 0.28], color=INK, lw=0.8)
    ax.text(0.54, 0.27, "Measured view", color=INK, fontsize=8, va="center")


def main():
    root = HERE.parents[1]
    with (
        ZipFile(root / "website/public/study-data/reach-droplets/figure-data.zip") as z,
        z.open("time-series.csv") as fp,
    ):
        table = pd.read_csv(fp)
    fan_off = table[table.run == "unforced"]
    with plt.style.context(
        ["particula.toc", {"svg.fonttype": "none", "svg.hashsalt": "reach-droplets"}]
    ):
        fig = plt.figure()
        draw_chamber(fig.add_axes([0.045, 0.56, 0.91, 0.425]))
        add_toc_logo(fig, corner="top right")
        ax = fig.add_axes([0.17, 0.17, 0.78, 0.325])
        ax.set_title("50–65 µm droplets (cm⁻³)", loc="left", fontsize=9, pad=4)
        ax.axvspan(5, 5.5, color="#EFD1A5", lw=0)
        for name, label, color, style in [
            ("Hologram", "Measured", INK, "-"),
            ("Spatial update", "Spatial fit", RUST, "-."),
            ("Forward physics", "Forward", TEAL, "--"),
        ]:
            series = fan_off[fan_off.series == name].sort_values("t_mid_s")
            edges = [*(series.t_mid_s - 0.125), series.t_mid_s.iloc[-1] + 0.125]
            ax.stairs(
                series.large_cm3, edges, baseline=None, color=color, ls=style, lw=1.6, label=label
            )
        ax.set(xlim=(3, 7), ylim=(0, 0.45), xticks=[3, 4, 5, 6, 7], yticks=[0, 0.2, 0.4])
        ax.set_xlabel("Time (s)", fontsize=8, labelpad=1)
        ax.tick_params(labelsize=7.5, pad=2)
        ax.spines[["right", "top"]].set_visible(False)
        handles, labels = ax.get_legend_handles_labels()
        fig.legend(
            handles,
            labels,
            ncol=3,
            loc="lower center",
            frameon=False,
            bbox_to_anchor=(0.5, 0.011),
            borderaxespad=0,
            handlelength=1.3,
            handletextpad=0.36,
            columnspacing=0.7,
            fontsize=7.3,
        )
        output = root / "docs/images/reach-droplets/toc.svg"
        fig.savefig(
            output,
            metadata={
                "Date": None,
                "Title": "The fan-off REACh droplet burst",
                "Description": "Original and proposed camera beams above measured, forward and fitted 50–65 µm droplet concentrations.",
            },
        )
        plt.close(fig)
        # Matplotlib wraps SVG path coordinates with trailing spaces.
        svg = re.sub(
            r' d="([^"]+)"', lambda m: ' d="' + " ".join(m[1].split()) + '"', output.read_text()
        )
        output.write_text(svg)


if __name__ == "__main__":
    main()
