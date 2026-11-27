# %%
"""Produce observation-matched chamber data and redraw figures from saved CSV."""

from argparse import ArgumentParser
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from chamber import CYCLIC, DENSITY, HERE, REPORTED, SIZE, Scenario, predict
from published import wilson


def curves(out):
    fits = pd.read_csv(out / "fits.csv")

    def f(p, rh):
        g = fits[
            (fits.precursor == p) & (fits.rh == rh) & (fits.model == "primary")
        ]
        return g.sort_values("cstar_ug_m3").fitted_fraction.to_numpy()

    rows = []
    for name in DENSITY:
        for rh in ["dry", "wet"]:
            actual = pd.read_csv(HERE.parent / "sources" / "observations.csv")
            selected = actual[(actual.precursor == name) & (actual.rh == rh)]
            times = np.unique(
                np.r_[
                    np.linspace(0, 30, 31),
                    np.geomspace(31, selected.time_min.max(), 140),
                    selected.time_min,
                ]
            )
            own = predict(f(name, rh), name, rh, SIZE[name], times, Scenario())
            cy = predict(
                f(CYCLIC[name], rh), name, rh, SIZE[name], times, Scenario()
            )
            paper = wilson(REPORTED[name, rh], times, SIZE[name], DENSITY[name])
            for t, a, b, c in zip(times, own, cy, paper):
                rows.append((name, rh, t, a, b, c))
    pd.DataFrame(
        rows,
        columns="precursor rh time_min precursor_fit cyclic_transfer published_reconstruction".split(),
    ).to_csv(out / "curves.csv", index=False, float_format="%.10g")


def plot(out):
    plt.rcParams.update(
        {"font.size": 9, "axes.spines.top": False, "axes.spines.right": False}
    )
    fig, axes = plt.subplots(
        2,
        2,
        figsize=(9, 6.0),
        sharex=True,
        sharey=True,
        constrained_layout=True,
    )
    series = pd.read_csv(out / "series.csv")
    series = series[series.model == "primary"]
    smooth = pd.read_csv(out / "curves.csv")
    for ax, name, title in zip(
        axes.flat,
        ["pinene", "ocimene", "caryophyllene", "farnesene"],
        [
            "α-pinene · 165 nm",
            "β-ocimene · 220 nm",
            "β-caryophyllene · 165 nm",
            "β-farnesene · 140 nm",
        ],
    ):
        for rh, color in [("dry", "#20556c"), ("wet", "#a34d21")]:
            g = smooth[(smooth.rh == rh) & (smooth.precursor == name)]
            pts = series[(series.rh == rh) & (series.precursor == name)]
            ax.plot(
                g.time_min,
                g.published_reconstruction,
                color=color,
                ls=":",
                lw=1.3,
            )
            ax.plot(g.time_min, g.precursor_fit, color=color, lw=1.8)
            if name != CYCLIC[name]:
                ax.plot(
                    g.time_min, g.cyclic_transfer, color=color, ls="--", lw=1.4
                )
            ax.scatter(
                pts.time_min,
                pts.observed_vfr,
                color=color,
                s=23,
                marker="o" if rh == "dry" else "s",
                facecolor="white",
                zorder=5,
            )
        ax.set_xscale("symlog", linthresh=30)
        xt = [0, 30, 100, 300, 1000]
        ax.set_xticks(xt, labels=[str(i) for i in xt])
        ax.set_ylim(0.18, 1.03)
        ax.set_title(title, loc="left", fontweight="bold", fontsize=10)
        ax.grid(axis="y", alpha=0.15)
    for ax in axes[:, 0]:
        ax.set_ylabel("Dry organic volume fraction remaining")
    for ax in axes[-1]:
        ax.set_xlabel("Evaporation time (min)")
    fig.savefig(out / "chamber_comparison.png", dpi=170)
    plt.close(fig)
    df = pd.read_csv(out / "mechanisms.csv")
    df = df[
        (df.model == "primary")
        & (df.train_rh == df.test_rh)
        & (df.time_min > 900)
    ]
    fig, axs = plt.subplots(
        1, 2, figsize=(8.5, 3.8), sharey=True, constrained_layout=True
    )
    for ax, name in zip(axs, ["ocimene", "farnesene"]):
        for rh, offset, color in [
            ("dry", -0.08, "#20556c"),
            ("wet", 0.08, "#a34d21"),
        ]:
            g = df[(df.precursor == name) & (df.test_rh == rh)]
            for i, term in enumerate(
                [
                    "constant_rate",
                    "low_volatility_share",
                    "progressive_resistance",
                ]
            ):
                v = 100 * g[g.term == term].signed_error
                ax.plot([i + offset] * 2, [v.min(), v.max()], lw=2, color=color)
                ax.scatter(
                    i + offset,
                    v.mean(),
                    s=45,
                    marker="o" if rh == "dry" else "s",
                    color=color,
                    label=("<10% RH" if rh == "dry" else "75% RH")
                    if i == 0
                    else None,
                )
        ax.axhline(0, color="#333333", lw=1)
        ax.grid(axis="y", alpha=0.2)
        ax.set_xticks(
            [0, 1, 2],
            labels=[
                "Constant\nslowdown",
                "Low-volatility\nmaterial",
                "Progressive\nresistance",
            ],
        )
        ax.set_title("β-" + name, loc="left", fontweight="bold", fontsize=10)
        ax.set_xlim(-0.5, 2.5)
        ax.legend(frameon=False, fontsize=8)
    axs[0].set_ylabel(
        "Overnight predicted minus observed\nvolume remaining (percentage points)"
    )
    fig.savefig(out / "overnight.png", dpi=170)
    plt.close(fig)


if __name__ == "__main__":
    p = ArgumentParser()
    p.add_argument("--out", type=Path, default=HERE / "results")
    p.add_argument("--plot-only", action="store_true")
    a = p.parse_args()
    if not a.plot_only:
        curves(a.out)
    plot(a.out)
