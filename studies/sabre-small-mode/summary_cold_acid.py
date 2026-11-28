"""Compare equilibrium sulfuric-acid concentrations at SABRE humidities."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.figure import Figure
from matplotlib.ticker import NullLocator


def compile_cold_acid(study: Path) -> dict:
    """Collect paired flat-surface acid predictions and humidity coverage."""
    chemistry = study / "chemistry"
    pairs = pd.read_csv(chemistry / "cold-acid.csv").sort_values("rh_liquid")
    hydration = pd.read_csv(
        chemistry / "hydration.csv", usecols=["case", "rh_liquid"]
    )
    flat = hydration.loc[hydration["case"] == "no_kelvin", "rh_liquid"]
    difference = pairs["aiomfac_h2so4_wt_pct"] - pairs["tabazadeh_h2so4_wt_pct"]

    def vector(column):
        return [float(f"{v:.8g}") for v in column]

    return {
        "rh_liquid_pct": vector(100 * pairs["rh_liquid"]),
        "aiomfac_h2so4_wt_pct": vector(pairs["aiomfac_h2so4_wt_pct"]),
        "tabazadeh_h2so4_wt_pct": vector(pairs["tabazadeh_h2so4_wt_pct"]),
        "min_sampled_rh_pct": float(f"{100 * flat.min():.8g}"),
        "n_below_1pct_rh": int((flat < 0.01).sum()),
        "median_difference_pct_points": float(f"{difference.median():.8g}"),
    }


def plot_cold_acid(data: dict) -> Figure:
    """Show two model predictions at each eligible flight-window humidity."""
    rh = data["rh_liquid_pct"]
    aiomfac = data["aiomfac_h2so4_wt_pct"]
    reference = data["tabazadeh_h2so4_wt_pct"]
    fig, ax = plt.subplots(figsize=(7.0, 3.65), layout="constrained")
    ax.set_xscale("log")
    ax.axvspan(data["min_sampled_rh_pct"], 1, color="#edf0ef", zorder=0)
    ax.axvline(1, color="#7b8b8d", linestyle=":", linewidth=1)
    ax.vlines(rh, reference, aiomfac, color="#bdc9c7", linewidth=0.85)
    ax.scatter(
        rh,
        aiomfac,
        label="AIOMFAC",
        color="#1b536c",
        marker="o",
        s=20,
        zorder=3,
    )
    ax.scatter(
        rh,
        reference,
        label="Tabazadeh et al.",
        color="#ae4e21",
        marker="^",
        s=24,
        zorder=3,
    )
    ax.text(
        np.sqrt(data["min_sampled_rh_pct"]),
        0.47,
        f"{data['n_below_1pct_rh']} windows\nbelow range",
        transform=ax.get_xaxis_transform(),
        ha="center",
        va="center",
        color="#495a5b",
        fontsize=9.5,
    )
    ax.text(
        0.98,
        0.96,
        f"AIOMFAC higher by {data['median_difference_pct_points']:.2f} points\n"
        f"(median of {len(rh)} paired windows)",
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=10,
    )
    ax.legend(loc="lower left", bbox_to_anchor=(0.22, 0.02), frameon=False)
    ax.set_xlim(data["min_sampled_rh_pct"] * 0.98, max(rh) * 1.14)
    ax.set_xticks([0.5, 1, 2, 5, 10], ["0.5", "1", "2", "5", "10"])
    ax.xaxis.set_minor_locator(NullLocator())
    ax.set_xlabel("Relative humidity with respect to liquid water (%)")
    ax.set_ylabel("Sulfuric acid (% by mass)")
    ax.spines[["top", "right"]].set_visible(False)
    return fig
