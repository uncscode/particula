"""Draw the Lund cloud TOC from the archived paired contrasts."""

from pathlib import Path
from zipfile import ZipFile

import matplotlib.pyplot as plt
import pandas as pd
from particula.plotting.toc import add_toc_logo, save_toc_svg

INK = "#202d39"
TEAL = "#007e77"
RUST = "#a54a24"


def toc(out, c):
    clouds = [
        ("reference", "Reference eddy"),
        ("isdac", "ISDAC profile"),
        ("mpace", "M-PACE profile"),
    ]
    with plt.style.context("particula.toc"):
        fig, ax = plt.subplots(figsize=(3, 3))
        fig.subplots_adjust(left=0.37, right=0.97, top=0.66, bottom=0.26)
        fig.text(0.05, 0.89, "Fresh dust inflow", fontsize=12, weight="bold")
        fig.text(0.05, 0.81, "Compared with a closed cloud top", fontsize=8.6)
        fig.text(0.05, 0.692, "Simulated case", fontsize=7.3, weight="bold")
        ax.text(
            -0.025,
            1.08,
            "Fewer freezes",
            ha="center",
            fontsize=7.2,
            color=RUST,
            transform=ax.get_xaxis_transform(),
        )
        ax.text(
            0.052,
            1.08,
            "More freezes",
            ha="center",
            fontsize=7.2,
            color=TEAL,
            transform=ax.get_xaxis_transform(),
        )
        values = []
        for sky, _ in clouds:
            rows = c[
                (c.sky == sky)
                & (c.grid == 10)
                & (c.abundance == "equal")
                & (c.source == "MineralDust")
                & (c.metric == "primary")
                & (c.comparison == "closed")
            ]
            values.append(rows.delta.item())
        ax.barh(
            range(3),
            values,
            color=[TEAL if v >= 0 else RUST for v in values],
            height=0.48,
        )
        ax.set_yticks(range(3), [label for _, label in clouds], fontsize=7.5)
        ax.tick_params(axis="y", length=0, pad=4)
        ax.tick_params(axis="x", labelsize=7.5)
        ax.set(
            ylim=(2.55, -0.55),
            xlim=(-0.057, 0.085),
            xticks=[-0.04, 0, 0.04, 0.08],
        )
        ax.axvline(0, lw=1.1, c=INK)
        ax.spines[["top", "right", "left"]].set_visible(False)
        ax.set_xlabel(
            "Effect of entrainment on primary ice formation", fontsize=8.1
        )
        ax.xaxis.set_label_coords(0.5, 0.14, transform=fig.transFigure)
        fig.text(
            0.5,
            0.055,
            "Six-hour change (freezing events L⁻¹)",
            ha="center",
            fontsize=7.8,
        )
        add_toc_logo(fig, corner="top right")
        save_toc_svg(
            fig,
            out / "cloud-toc.svg",
            title=(
                "Fresh aerosol can form ice while entrainment reduces "
                "freezing across the cloud"
            ),
            description=(
                "Six-hour change in primary freezing relative to a "
                "closed cloud top. Fresh dust-like inflow increased "
                "freezing in the reference eddy and reduced it in the "
                "rapid-exchange ISDAC and M-PACE simulations."
            ),
        )
        plt.close(fig)


if __name__ == "__main__":
    docs = Path(__file__).resolve().parents[3] / "docs"
    with ZipFile(docs / "data/lund-freezing/figure-tables.zip") as archive:
        with archive.open("paired.csv") as table:
            toc(docs / "images/lund-freezing", pd.read_csv(table))
