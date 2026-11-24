"""Replot the observed distributions and symmetric differences directly from saved counts."""

import sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import FixedLocator, FixedFormatter, NullLocator
from matplotlib.lines import Line2D
from budget import symmetric

ROOT = Path(__file__).resolve().parent
F, I, C = "#00698b", "#af4925", "#504d5b"
FIELD, MODEL = "#20282f", "#7863a8"


def style(ax):
    ticks = [6, 9, 13, 18, 25, 40]
    ax.set_xscale("log")
    ax.set_xlim(6, 40)
    ax.xaxis.set_major_locator(FixedLocator(ticks))
    ax.xaxis.set_major_formatter(FixedFormatter([str(x) for x in ticks]))
    ax.xaxis.set_minor_locator(NullLocator())
    ax.axvline(13, color="#a9aeb0", lw=0.8, ls=":")
    ax.grid(axis="y", color="#d7dee1", lw=0.5)
    ax.spines[["right", "top"]].set_visible(False)
    ax.set_axisbelow(True)


def run(out):
    frame = pd.read_csv(out / "distributions.csv")
    fig, axes = plt.subplots(
        3,
        2,
        figsize=(10, 8.8),
        sharex=True,
        sharey="row",
        gridspec_kw={"height_ratios": [1.15, 1, 1]},
    )
    fig.subplots_adjust(
        left=0.115,
        right=0.98,
        top=0.830,
        bottom=0.125,
        wspace=0.10,
        hspace=0.34,
    )
    fig.text(
        0.115,
        0.957,
        "The Pallas droplet size distributions",
        fontsize=21,
        weight="bold",
        color="#182b38",
    )
    fig.text(
        0.115,
        0.915,
        "Measured spectra, same-input optical simulations and the difference between probes",
        fontsize=11,
    )
    handles = [
        Line2D([], [], color=c, lw=2.2, ls=s, label=n)
        for n, c, s in [
            ("FSSP", F, "-"),
            ("ICEMET", I, "-"),
            ("CDA input", C, ":"),
            ("Measured difference", FIELD, "-"),
            ("Simulated difference", MODEL, "--"),
        ]
    ]
    fig.legend(
        handles=handles,
        loc="center left",
        bbox_to_anchor=(0.108, 0.877),
        ncol=5,
        frameon=False,
        fontsize=9.6,
        handlelength=2,
        columnspacing=1.3,
    )
    for j, (day, g) in enumerate(frame.groupby("day")):
        e = np.r_[g.low_um, g.high_um.iloc[-1]]
        width = np.diff(np.log10(e))
        a, b, c = axes[:, j]
        a.set_title(
            f"{day} September 2022  ·  {int(g.minutes.iloc[0]):,} paired minutes",
            fontsize=11.2,
            weight="bold",
            loc="left",
            pad=6,
        )
        for ax in [a, b, c]:
            style(ax)
        for ax in [a, b]:
            ax.set_ylim(bottom=0)
            ax.stairs(g.CDA / width, e, color=C, ls=":", lw=1.8)
        for name, color in [("FSSP", F), ("ICEMET", I)]:
            a.stairs(
                g[name + "_p975"] / width,
                e,
                baseline=g[name + "_p025"] / width,
                fill=True,
                color=color,
                alpha=0.13,
            )
            a.stairs(g[name] / width, e, color=color, lw=2)
            b.stairs(g["model_" + name] / width, e, color=color, lw=2)
        c.axhline(0, color="#69737b", lw=0.8)
        c.stairs(
            g.percent_p975,
            e,
            baseline=g.percent_p025,
            fill=True,
            color="#7b8f9e",
            alpha=0.29,
        )
        c.stairs(g.percent, e, color=FIELD, lw=2, baseline=None)
        c.stairs(
            g.model_percent, e, color=MODEL, ls="--", lw=1.9, baseline=None
        )
        c.set_ylim(-206, 206)
        c.set_yticks([-200, -100, 0, 100, 200])
        c.set_xlabel("Droplet diameter (µm)")
    for ax in axes[:2].flat:
        ax.set_ylim(
            0,
            max(
                frame[n].div(np.log10(frame.high_um / frame.low_um)).max()
                for n in [
                    "FSSP_p975",
                    "ICEMET_p975",
                    "CDA",
                    "model_ICEMET",
                    "model_FSSP",
                ]
            )
            * 1.045,
        )
    for ax, label in zip(
        axes[:, 0],
        [
            "Measured spectrum\ndN/dlog₁₀D (cm⁻³)",
            "Simulated spectrum\ndN/dlog₁₀D (cm⁻³)",
            "FSSP − ICEMET\n(% of their mean)",
        ],
    ):
        ax.set_ylabel(label)
    fig.text(
        0.115,
        0.046,
        "Shading shows pointwise central 95% ranges from the working instrument budget. The dotted vertical line marks 13 µm.",
        fontsize=9,
    )
    fig.text(
        0.115,
        0.022,
        "All curves use the same diameter bins. Very sparse tails can give large percent differences.",
        fontsize=9,
    )
    fig.savefig(out / "distribution-comparison.png", dpi=160)
    plt.close(fig)
    fig, axs = plt.subplots(1, 2, figsize=(9.6, 3.65), sharey=True)
    fig.subplots_adjust(
        left=0.10, right=0.98, top=0.73, bottom=0.23, wspace=0.13
    )
    fig.text(
        0.10,
        0.92,
        "Wavelength sensitivity within each simulated probe",
        fontsize=16,
        weight="bold",
    )
    fig.text(
        0.10,
        0.836,
        "660 nm minus 633 nm, as a percentage of their mean  •  Same CDA input and frozen calibration",
        fontsize=10,
    )
    for ax, (day, g) in zip(axs, frame.groupby("day")):
        style(ax)
        e = np.r_[g.low_um, g.high_um.iloc[-1]]
        ax.axhline(0, color="#69737b", lw=0.8)
        ax.stairs(
            symmetric(g.model_FSSP_swap, g.model_FSSP),
            e,
            color=F,
            label="FSSP",
            baseline=None,
            lw=2,
        )
        ax.stairs(
            symmetric(g.model_ICEMET, g.model_ICEMET_swap),
            e,
            color=I,
            label="ICEMET",
            baseline=None,
            lw=2,
        )
        ax.set_title(f"{day} September", fontsize=10, loc="left")
        ax.set_xlabel("Reported droplet diameter (µm)")
    axs[0].set_ylabel("Difference (%)")
    axs[-1].legend(frameon=False)
    fig.savefig(out / "wavelength-by-size.png", dpi=160)
    plt.close(fig)


if __name__ == "__main__":
    run(Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT)
