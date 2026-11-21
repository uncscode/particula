"""Plot observed timing independently from controlled condensation cases."""

import matplotlib.pyplot as plt
import numpy as np


def timing(report):
    """Show normalized instrument channels and their half-amplitude times."""
    data = np.asarray(report["traces"])
    fig, ax = plt.subplots(figsize=(7.4, 4), layout="constrained")
    for col, (label, color) in enumerate(
        [
            ("Formula-level MS", "#555555"),
            ("cis-Pinic acid", "#195c85"),
            ("Unidentified isomer", "#b0502d"),
        ],
        start=1,
    ):
        ax.plot(
            data[:, 0],
            data[:, col],
            ".-",
            ms=3,
            lw=1.5,
            color=color,
            label=label,
        )
        ax.plot(report["half_times"]["3"][col - 1], 0.5, "o", ms=7, color=color)
    ax.axhline(0.5, color="#777", ls=":", lw=1)
    ax.set(
        xlabel="Chamber time (min)",
        ylabel="Fraction of own late signal",
        ylim=(-0.13, 1.22),
    )
    ax.legend(frameon=False)
    return fig


def loading(report):
    """Compare equilibrium transfer across loadings for equal vapor shares."""
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.8), layout="constrained")
    seeds = np.asarray(report["seeds"])
    for case in report["cases"]:
        if case["pinic_fraction"] != 0.5:
            continue
        end = np.asarray(case["resolved"])[-1].sum(axis=1)
        lump = np.asarray(case["lumped"])[-1]
        ratio = case["ratio"]
        axes[1].plot(seeds, (lump - end), "o-", ms=3, label=f"{ratio:g}×")
        if ratio == 10:
            axes[0].plot(seeds, end, "o-", label="Two isomers", color="#195c85")
            axes[0].plot(
                seeds, lump, "s--", label="Single surrogate", color="#b0502d"
            )
    for ax in axes:
        ax.set_xscale("log")
        ax.axvline(10, color="#777", ls=":", lw=1)
    axes[0].set(
        ylabel="Condensed organic mass (µg/m³)", title="10× volatility contrast"
    )
    axes[1].set(
        ylabel="Surrogate minus resolved (µg/m³)",
        title="Volatility of unknown relative to pinic",
    )
    axes[0].legend(frameon=False, fontsize=9)
    axes[1].legend(frameon=False, fontsize=8, ncol=2)
    fig.supxlabel("Nonvolatile seed loading (µg/m³)")
    return fig
