"""Compact optical comparisons for the SABRE study narrative."""

import matplotlib.pyplot as plt
import numpy as np


def plot_spectrum(curves, spectra):
    """Plot the 300–330 ppbv campaign number spectrum and its small mode."""
    group = curves.loc[curves.n2o_lo_ppbv == 300].sort_values("diameter_nm")
    measured = spectra.loc[spectra.n2o_lo_ppbv == 300]
    fig, ax = plt.subplots(figsize=(6.8, 3.45), layout="constrained")
    ax.fill_between(
        group.diameter_nm,
        group.small_number,
        color="#b8d6df",
        label="Fitted small mode",
    )
    ax.plot(
        group.diameter_nm,
        group.fit_number,
        color="#175776",
        linewidth=2,
        label="Full size fit",
    )
    ax.plot(
        measured.diameter_nm,
        measured.dndlog10d_cm3,
        linestyle="none",
        marker="o",
        markersize=4,
        markerfacecolor="white",
        markeredgecolor="#343434",
        markeredgewidth=1,
        label="AMP spectrum",
    )
    ax.set_xscale("log")
    ax.set_xlim(3.5, 1050)
    ax.set_ylim(bottom=0)
    ticks = [5, 10, 30, 100, 300, 1000]
    ax.set_xticks(ticks, [str(value) for value in ticks])
    ax.set_xlabel("Dry particle diameter (nm)")
    ax.set_ylabel("Number per diameter decade (cm⁻³)")
    ax.set_axisbelow(True)
    ax.grid(axis="y", alpha=0.22)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(loc="upper right", frameon=False)
    return fig


def plot_flight_optics(flights):
    """Show paired medians of 532 nm extinction residuals from eight flights."""
    medians = flights.sort_values("flight")
    rows = np.arange(len(medians))
    fig, ax = plt.subplots(figsize=(6.8, 3.75), layout="constrained")
    ax.hlines(rows, medians.without_mode_ext, medians.full_ext, color="#9ca9ad")
    ax.scatter(
        medians.full_ext,
        rows,
        s=45,
        color="#175776",
        label="All particles",
        zorder=2,
    )
    ax.scatter(
        medians.without_mode_ext,
        rows,
        s=40,
        marker="D",
        color="#a84c22",
        label="Small mode removed",
        zorder=2,
    )
    ax.axvline(0, color="#4b5457", linestyle="--", linewidth=1.1)
    ax.set_yticks(rows, [f"Mar {int(str(d)[-2:])}" for d in medians.flight])
    ax.set_ylim(len(rows) - 0.35, -0.6)
    ax.set_xlim(-5.1, 0.25)
    ax.set_xticks(range(-5, 1))
    ax.set_xlabel("Modeled − measured extinction at 532 nm (Mm⁻¹)")
    ax.set_axisbelow(True)
    ax.grid(axis="x", alpha=0.22)
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.tick_params(axis="y", length=0)
    ax.legend(loc="lower left", bbox_to_anchor=(0, 1), ncol=2, frameon=False)
    return fig
