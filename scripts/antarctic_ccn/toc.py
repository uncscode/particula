"""Render the three-inch graphical abstract from paired bias intervals."""

import pandas as pd
from matplotlib import pyplot as plt
from particula.plotting.toc import add_toc_logo
from scripts.antarctic_ccn.plots import IMAGES, INK, RUST, TEAL


def render():
    """Draw paired bias intervals against the zero-bias reference."""
    source = (
        IMAGES.parents[2]
        / "website/public/study-data/antarctic-ccn/paired-bias.csv"
    )
    df = (
        pd.read_csv(source)
        .query("contrast=='bias'")
        .set_index(["model", "metric"])
    )
    with plt.style.context("particula.toc"):
        fig = plt.figure()
        fig.text(0.07, 0.90, "King Sejong Station", weight="bold", fontsize=13)
        fig.text(0.07, 0.835, "Antarctica  ·  26 held-out dates", fontsize=8.6)
        add_toc_logo(fig, corner="top right")
        ax = fig.add_axes([0.09, 0.17, 0.86, 0.56])
        ax.set_title(
            "Difference from measured CCN",
            loc="left",
            pad=4,
            weight="bold",
            fontsize=9,
        )
        # End the zero guide below the directly labelled intervals.
        ax.vlines(0, -0.50, 1.54, ls=":", lw=0.85, color=INK)
        for i, (key, label) in enumerate(
            [
                ("C4", "Count at 0.4%"),
                ("rise24", "Rise, 0.2 → 0.4%"),
                ("rise46", "Rise, 0.4 → 0.6%"),
            ]
        ):
            y = 2 - i
            pair_center = (
                df.xs(key, level="metric")
                .loc[["salt", "maximum"], "mean"]
                .mean()
            )
            ax.annotate(
                label,
                (pair_center, y),
                xytext=(0, 6),
                textcoords="offset points",
                ha="center",
                va="bottom",
                fontsize=7.7,
                bbox={
                    "facecolor": fig.get_facecolor(),
                    "edgecolor": "none",
                    "pad": 0.3,
                },
            )
            for j, (mod, name, color, marker) in enumerate(
                [
                    ("salt", "Separate salts", TEAL, "o"),
                    ("maximum", "Full coating", RUST, "D"),
                ]
            ):
                r = df.loc[mod, key]
                pos = y - 0.26 * j
                ax.plot([r.lo95, r.hi95], [pos, pos], color=color, lw=1.15)
                ax.plot(
                    r["mean"],
                    pos,
                    marker=marker,
                    ms=4,
                    mfc=color if j else "#fbf2e3",
                    mec=color,
                )
                if i == 0:
                    ax.annotate(
                        name,
                        (r.hi95, pos),
                        xytext=(3, 0),
                        textcoords="offset points",
                        va="center",
                        color=color,
                        fontsize=7.8,
                    )
        ax.set(
            xlim=(-23, 13),
            ylim=(-0.50, 2.55),
            xticks=[-20, -10, 0, 10],
            yticks=[],
        )
        ax.grid(False)
        for side in ["left", "top", "right"]:
            ax.spines[side].set_visible(False)
        fig.text(
            0.08,
            0.055,
            "Lines show 95% paired date intervals cm⁻³",
            fontsize=7.3,
        )
        fig.savefig(IMAGES / "graphical-abstract.png", dpi=300)
        plt.close(fig)


if __name__ == "__main__":
    render()
