"""Draw the six Bai literature figures from their compact plot tables."""

import argparse
from calendar import month_abbr
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from particula import plotting
from toc import DATA, IMAGES
from toc import render as render_toc

CASES = [("clean", 0.25), ("clean", 2.0), ("rich", 0.25), ("rich", 2.0)]
SEASONS = {"DJF": "Winter", "MAM": "Spring", "JJA": "Summer", "SON": "Autumn"}
PALETTE = plotting.COLORS
CASE_COLORS = {
    case: PALETTE[c]
    for case, c in zip(CASES, ("teal", "indigo", "rust", "berry"), strict=True)
}
METRICS = ["delta_activated_cm3", "delta_median_um", "delta_up_site_wm2"]
YLABELS = ["Δ number\n(cm⁻³)", "Δ wet median\n(µm)", "Δ reflected SW\n(W m⁻²)"]


def curve_style(profile, speed):
    """Encode updraft with both line pattern and marker shape."""
    weak = speed == 0.25
    color = CASE_COLORS[profile, speed]
    stroke = "dashed" if weak else "solid"
    return dict(
        color=color,
        linestyle=(0, (5, 2.8)) if weak else "-",
        marker="s" if weak else "o",
        markersize=5.4,
        markerfacecolor="white" if weak else color,
        markeredgewidth=1.4,
        label=f"{profile.capitalize()}, {speed:g} m/s · {stroke}",
    )


def cloud_curves(ax, table, x, metric, hourly=False):
    """Plot matched sun-minus-dark responses for four environmental cases."""
    for j, (profile, speed) in enumerate(CASES):
        series = table[
            (table.profile == profile) & (table.w == speed)
        ].sort_values(x)
        ax.plot(
            series[x] + (0.5 if hourly else 0),
            series[metric],
            markevery=(j, 6) if hourly else 1,
            **curve_style(profile, speed),
        )
    ax.axhline(0, linewidth=0.8, color="#566661", linestyle="-", zorder=0)


def cloud_legend(fig, first, y):
    """Place one shared legend with long samples of each line pattern."""
    handles, labels = first.get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.53, y),
        ncol=2,
        handlelength=4.4,
        columnspacing=1.8,
        borderaxespad=0,
    )


def monthly(data):
    """Compare seven-day means for all months in the solar record."""
    table = pd.read_csv(data / "monthly.csv")
    fig, axes = plt.subplots(3, 1, figsize=(7.9, 7.5), sharex=True)
    for ax, metric, ylabel in zip(axes, METRICS, YLABELS, strict=True):
        cloud_curves(ax, table, "month", metric)
        ax.set_ylabel(ylabel)
    axes[-1].set_xticks(range(1, 13), list(month_abbr)[1:])
    cloud_legend(fig, axes[0], 0.987)
    fig.subplots_adjust(
        left=0.15, right=0.985, top=0.895, bottom=0.06, hspace=0.20
    )
    return fig


def diurnal(data):
    """Compare hourly seasonal cloud responses in local apparent solar time."""
    table = pd.read_csv(data / "diurnal.csv")
    fig, axes = plt.subplots(
        3, 4, figsize=(9.5, 6.4), sharex=True, sharey="row"
    )
    for j, season in enumerate(SEASONS):
        group = table[table.season == season]
        for i, metric in enumerate(METRICS):
            ax = axes[i, j]
            cloud_curves(ax, group, "solar_hour", metric, hourly=True)
            ax.set_xticks([0, 12, 24])
            ax.set_xlim(0, 24)
            if j == 0:
                ax.set_ylabel(YLABELS[i])
    cloud_legend(fig, axes[0, 0], 0.987)
    fig.supxlabel("Local apparent solar hour", y=0.015)
    fig.subplots_adjust(
        left=0.115,
        right=0.978,
        top=0.885,
        bottom=0.11,
        hspace=0.28,
        wspace=0.17,
    )
    return fig


def size_spectra(data):
    """Plot the change in number in each quarter-micrometre diameter bin."""
    table = pd.read_csv(data / "size-spectra.csv.gz")
    fig, axes = plt.subplots(2, 2, figsize=(8.0, 5.8))
    # Line patterns and names remain distinct in grayscale.
    seasonal_styles = [
        ("DJF", "indigo", ":"),
        ("MAM", "teal", "--"),
        ("JJA", "rust", "-"),
        ("SON", "berry", "-."),
    ]
    for ax, (profile, speed) in zip(axes.flat, CASES, strict=True):
        group = table[(table.profile == profile) & (table.w == speed)]
        peak = (
            group.groupby("lower_um")[["sun_number_cm3", "dark_number_cm3"]]
            .max()
            .max(axis=1)
        )
        visible = peak[peak > 0.004 * peak.max()].index
        for season, color, stroke in seasonal_styles:
            y = group[group.season == season].sort_values("lower_um")
            ax.stairs(
                y.delta_number_cm3,
                np.r_[y.lower_um, y.upper_um.iloc[-1]],
                baseline=None,
                color=PALETTE[color],
                linestyle=stroke,
                linewidth=2.1,
                label=SEASONS[season],
            )
        ax.axhline(0, color="#566661", lw=0.8, linestyle="-", zorder=0)
        ax.set_xlim(visible.min() - 0.25, visible.max() + 0.75)
    for ax in axes[:, 0]:
        ax.set_ylabel("Δ number per bin\n(cm⁻³)")
    fig.supxlabel("Wet diameter (µm)", y=0.014)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        ncol=4,
        loc="upper center",
        bbox_to_anchor=(0.53, 0.996),
        handlelength=3,
    )
    fig.subplots_adjust(
        left=0.15, right=0.981, top=0.915, bottom=0.10, hspace=0.26, wspace=0.25
    )
    return fig


def cohorts(data):
    """Separate tagged and background counts and common-cohort growth."""
    july = pd.read_csv(data / "july-cohorts.csv").sort_values(["profile", "w"])
    fig, axes = plt.subplots(2, 1, figsize=(7.7, 5.9))
    y = np.arange(len(july))
    labels = [
        f"{p.capitalize()}, {w:g} m/s"
        for p, w in zip(july.profile, july.w, strict=True)
    ]
    panels = [
        (
            axes[0],
            -july.lost_seed_cm3,
            july.gained_field_cm3,
            "Tagged seeds leaving",
            "Background entering",
            "Number change (cm⁻³)",
        ),
        (
            axes[1],
            july.shared_growth_um,
            july.exit_selection_um + july.entry_selection_um,
            "Growth of shared cohorts",
            "Entries and exits",
            "Change in median wet diameter (µm)",
        ),
    ]
    for ax, first, second, name1, name2, xlabel in panels:
        ax.grid(axis="y", visible=False)
        ax.grid(axis="x", visible=True)
        ax.barh(
            y - 0.19, first, height=0.32, color=PALETTE["teal"], label=name1
        )
        ax.barh(
            y + 0.19,
            second,
            height=0.32,
            color="white",
            edgecolor=PALETTE["rust"],
            hatch="////",
            linewidth=1,
            label=name2,
        )
        ax.set_yticks(y, labels)
        ax.invert_yaxis()
        ax.set_xlabel(xlabel)
        ax.axvline(0, linewidth=0.8, color="#566661", linestyle="-")
        ax.legend(
            loc="lower center",
            bbox_to_anchor=(0.5, 1.015),
            ncol=2,
            handlelength=1.5,
        )
    fig.subplots_adjust(
        left=0.23, right=0.98, top=0.925, bottom=0.105, hspace=0.60
    )
    return fig


def film(data):
    """Plot the three constant-humidity limonene film reconstructions."""
    table = pd.read_csv(data / "film-water.csv")
    fig, axes = plt.subplots(2, 1, figsize=(6.8, 5.65), sharex=True)
    for trial, marker, color in [
        ("S4", "o", "teal"),
        ("S5", "s", "rust"),
        ("S7", "^", "indigo"),
    ]:
        g = table[table.trial == trial].sort_values(
            "mass_fraction", ascending=False
        )
        args = dict(
            marker=marker,
            color=PALETTE[color],
            markersize=4.2,
            markerfacecolor="white",
            label=trial,
            ls="-",
            lw=1.4,
        )
        axes[0].errorbar(
            g.mass_fraction,
            g.kappa,
            yerr=g.kappa_bar,
            elinewidth=0.8,
            capsize=2,
            **args,
        )
        axes[1].plot(g.mass_fraction, g.water_ratio, **args)
    axes[0].set_ylabel("Hygroscopicity, κ")
    axes[1].set_ylabel("Film water / fresh water")
    axes[1].axhline(1, color="#566661", ls="--", lw=1)
    axes[1].set_xlabel("Remaining fraction of initial dry mass")
    axes[1].set_xlim(1.025, table.mass_fraction.min() - 0.018)
    for ax in axes:
        ax.axvline(0.3, ls=":", lw=1, color="#798782")
    handles, names = axes[1].get_legend_handles_labels()
    fig.legend(
        handles, names, ncol=3, loc="upper center", bbox_to_anchor=(0.54, 0.996)
    )
    fig.subplots_adjust(
        left=0.19, right=0.975, top=0.913, bottom=0.13, hspace=0.23
    )
    return fig


def main():
    """Render article and supporting figures into the requested directory."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=DATA)
    parser.add_argument("--output", type=Path, default=IMAGES)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    with plt.style.context("particula.default"):
        for name, draw in [
            ("monthly-response", monthly),
            ("seasonal-diurnal", diurnal),
            ("size-resolved-number", size_spectra),
            ("cohort-mechanism", cohorts),
            ("film-water", film),
        ]:
            fig = draw(args.data)
            fig.savefig(args.output / f"{name}.png", dpi=185)
            plt.close(fig)
    render_toc(args.data, args.output / "seasonal-toc.svg")


if __name__ == "__main__":
    main()
