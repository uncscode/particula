"""Render the three manuscript figures from the published numeric summaries."""

from pathlib import Path

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import ScalarFormatter
from particula import plotting

ROOT = Path(__file__).resolve().parents[3]
DATA = ROOT / "website/public/study-data/pallas-activation"
IMAGES = ROOT / "docs/images/pallas-activation"
RUST = plotting.COLORS["rust"]
TEAL = plotting.COLORS["teal"]


def save(fig, name):
    plotting.fit(fig)
    fig.savefig(IMAGES / name, dpi=200)
    plt.close(fig)


def sizes(ax, ticks):
    ax.set_xscale("log")
    ax.set_xticks(ticks)
    ax.xaxis.set_major_formatter(ScalarFormatter())
    ax.set_xlabel("Dry diameter (nm)")


def activation_and_salt(daily):
    curves = pd.read_csv(DATA / "activation.csv")
    fig, (left, right) = plotting.subplots(1, 2, size="wide")
    colors = [*plotting.COLORS.values(), "#39483F"]
    strokes = ["-", "--", "-.", ":", (0, (5, 1, 1, 1))]
    for (ss, scan), color, stroke in zip(
        curves.groupby("SS"), colors, strokes, strict=True
    ):
        left.plot(
            scan.D, scan.fitted, color=color, linestyle=stroke, label=f"{ss:g}%"
        )
        left.scatter(
            scan.D, scan.observed, s=17, edgecolor=color, facecolor="none"
        )
    sizes(left, [10, 30, 60, 100, 200, 300])
    left.set(title="Activation on 16 September", ylabel="Activated fraction")
    plotting.legend(
        left,
        placement="inside",
        loc="upper left",
        fontsize=9,
        title="Supersaturation",
    )
    knots = [30, 60, 100, 160, 300]
    diameter = np.unique(np.r_[np.geomspace(30, 300, 151), knots, 50, 120])
    profiles = (
        np.array(
            [
                np.interp(np.log(diameter), np.log(knots), row)
                for row in daily[[f"knot_{d}" for d in knots]].to_numpy()
            ]
        )
        * 100
    )
    right.fill_between(
        diameter,
        *np.quantile(profiles, [0.25, 0.75], axis=0),
        color=RUST,
        alpha=0.17,
        label="Middle half of dates",
    )
    right.plot(
        diameter, profiles.mean(axis=0), color=RUST, label="Mean of daily fits"
    )
    right.scatter(
        [50, 120], 100 * daily[["salt50", "salt120"]].mean(), color=RUST, s=30
    )
    sizes(right, [30, 50, 80, 120, 180, 300])
    right.set(
        title="Salt allocation across 27 dates",
        ylabel="Salt in dry volume (%)",
        ylim=(0, 102),
    )
    plotting.legend(right, placement="inside", loc="upper right", fontsize=10)
    save(fig, "activation-and-salt.png")


def daily_mixing(daily):
    fig, ax = plotting.subplots(size="wide")
    dates = pd.to_datetime(daily.date)
    for _, group in daily.groupby(dates.diff().dt.days.gt(4).cumsum()):
        ax.plot(pd.to_datetime(group.date), group.chi, color=TEAL, lw=1)
    ax.vlines(
        dates,
        daily.chi_lo,
        daily.chi_hi,
        color=TEAL,
        alpha=0.45,
        lw=2,
        label="Feasible range at fixed inventory",
    )
    ax.scatter(dates, daily.chi, color=TEAL, s=22, label="Best fit", zorder=3)
    ax.set(
        ylabel="Three-group dry-mass mixing index χ",
        xlabel="Sampling date in 2022",
        ylim=(-0.04, 1.06),
    )
    ax.xaxis.set_major_locator(mdates.MonthLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%d %b"))
    plotting.legend(ax, placement="inside", fontsize=9, loc="upper right")
    save(fig, "mixing-by-date.png")


def same_index():
    data = pd.read_csv(DATA / "between-sizes.csv")
    fig, ax = plotting.subplots(size="wide")
    ax.plot(data.D, 100 * data.original, color=RUST, label="Fitted allocation")
    ax.plot(
        data.D,
        100 * data.redistributed,
        color=RUST,
        linestyle="--",
        label="Redistributed salt",
    )
    markers = data[data.measured_size]
    ax.scatter(
        markers.D,
        100 * markers.original,
        s=35,
        zorder=3,
        color="#243E43",
        label="Measured activation sizes",
    )
    ax.set(xlabel="Dry diameter (nm)", ylabel="Salt in dry volume (%)")
    plotting.legend(ax, placement="inside", loc="lower center", fontsize=9)
    save(fig, "same-index-example.png")


if __name__ == "__main__":
    with plt.style.context("particula.default"):
        days = pd.read_csv(DATA / "daily.csv")
        activation_and_salt(days)
        daily_mixing(days)
        same_index()
