"""Recreate the Arctic mode-removal figure from the saved figure values."""

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

FULL, RETAINED, FILL, SOURCE = "#175776", "#9e460e", "#f4d9c1", "#343434"


def render(curves, spectra, annotations, output):
    """Plot the archived curves alongside the digitized spectra."""
    comparisons = annotations.set_index("n2o_lo_ppbv")
    fig, axes = plt.subplots(
        3, 2, sharex=True, sharey="row", figsize=(8.6, 8.3)
    )
    fig.subplots_adjust(
        top=0.94, bottom=0.13, left=0.15, right=0.975, wspace=0.12, hspace=0.23
    )
    for j, group in enumerate([270, 300]):
        c = curves[curves["n2o_lo_ppbv"] == group]
        s = spectra[spectra["n2o_lo_ppbv"] == group]
        summary = comparisons.loc[group]
        d = c["diameter_nm"]
        point_d = s["diameter_nm"]
        points = s["dndlog10d_cm3"]
        rows = [
            (c["fit_number"], c["fit_number"] - c["small_number"], None),
            (c["fit_surface"], c["retained_surface"], "area_fraction"),
            (
                c["full_extinction"],
                c["full_extinction"] - c["small_extinction"],
                "ext_fraction",
            ),
        ]
        for i, (full, retained, key) in enumerate(rows):
            ax = axes[i, j]
            ax.set_xscale("log")
            ax.set_xlim(3.7, 1000)
            ax.spines[["top", "right"]].set_visible(False)
            ax.grid(axis="y", alpha=0.25)
            ax.axvline(150, color="#6c777b", ls=":", lw=1, zorder=1)
            ax.set_axisbelow(True)
            ax.fill_between(d, retained, full, color=FILL)
            ax.plot(d, full, color=FULL, lw=2.2)
            ax.plot(d, retained, color=RETAINED, lw=1.8, ls="--")
            if i < 2:
                values = (
                    points if i == 0 else points * np.pi * (point_d / 1000) ** 2
                )
                ax.plot(
                    point_d,
                    values,
                    ls="none",
                    marker="o",
                    ms=3,
                    markerfacecolor="white",
                    markeredgecolor=SOURCE,
                    mew=0.8,
                )
            if key:
                ax.text(
                    0.025,
                    0.93,
                    f"{float(summary[key]):.0%} removed",
                    transform=ax.transAxes,
                    va="top",
                    fontsize=10,
                )
        axes[0, j].set_title(f"{group}–{group + 30} ppbv N₂O", weight="bold")
        axes[-1, j].set_xlabel("Particle diameter (nm)")
        axes[-1, j].set_xticks(
            [5, 10, 30, 100, 300, 900], ["5", "10", "30", "100", "300", "900"]
        )
    for ax, label in zip(
        axes[:, 0],
        [
            "Number per decade\n(cm⁻³)",
            "Surface area per decade\n(µm² cm⁻³)",
            "532 nm extinction per decade\n(Mm⁻¹)",
        ],
        strict=True,
    ):
        ax.set_ylabel(label)
        ax.set_ylim(bottom=0)
    handles = [
        Line2D(
            [],
            [],
            marker="o",
            color=SOURCE,
            markerfacecolor="white",
            ls="none",
            ms=4,
            label="AMP spectrum or derived area",
        ),
        Line2D([], [], color=FULL, lw=2.2, label="Full fit"),
        Line2D(
            [], [], color=RETAINED, lw=1.8, ls="--", label="Small mode removed"
        ),
        Patch(facecolor=FILL, label="Removed contribution"),
    ]
    fig.legend(
        handles=handles,
        loc="lower center",
        ncol=2,
        frameon=False,
        bbox_to_anchor=(0.54, 0.005),
        fontsize=9,
    )
    fig.savefig(
        output,
        dpi=165,
        pil_kwargs={"lossless": True, "method": 6, "quality": 100},
    )
    plt.close(fig)
