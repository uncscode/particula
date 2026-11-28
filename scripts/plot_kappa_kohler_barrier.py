"""Regenerate the annotated Köhler curves on the equilibrium activation page."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from particula.particles import get_kappa_kohler_critical_dry_diameter
from scipy.optimize import minimize_scalar

FIGURE = (
    Path(__file__).resolve().parents[1]
    / "docs/images/thermodynamics/kappa-kohler-barrier.png"
)
KAPPA = 0.3
KELVIN_NM = 2.1
RATIO = 1.002


def supersaturation(wet_nm, dry_nm):
    """Evaluate Eq. 6 as percent supersaturation for D > D_d."""
    water_to_dry = np.expm1(3 * np.log(wet_nm / dry_nm))
    log_ratio = KELVIN_NM / wet_nm - np.log1p(KAPPA / water_to_dry)
    return 100 * np.expm1(log_ratio)


def main():
    """Draw the forward barrier for dry seeds bracketing the inverse cutoff."""
    cutoff = 1e9 * get_kappa_kohler_critical_dry_diameter(
        KAPPA, RATIO, KELVIN_NM * 1e-9
    )
    fig, ax = plt.subplots(figsize=(8.2, 4.8), layout="constrained")
    ink = "#243444"
    ax.set_xlim(200, 1800)
    ax.set_ylim(0, 0.385)
    ax.set(
        xlabel="Wet droplet diameter (nm)",
        ylabel="Equilibrium supersaturation (%)",
        xticks=np.arange(200, 1801, 200),
        yticks=np.arange(0, 0.351, 0.05),
    )
    ax.set_axisbelow(True)
    ax.grid(axis="y", color="#e4e8ec", linewidth=0.7)
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(colors=ink, labelsize=10)
    ax.xaxis.label.set_size(12)
    ax.yaxis.label.set_size(12)
    target = 100 * (RATIO - 1)
    ax.axhline(target, color=ink, linestyle=(0, (5, 4)), linewidth=1.3)
    ax.text(1780, target + 0.006, "Available 0.20%", ha="right", color=ink)
    wet = np.linspace(*ax.get_xlim(), 1601)
    for dry, color, style, marker, label_xy in (
        (80.0, "#a04716", "-.", "^", (570, 0.343)),
        (cutoff, "#145b84", "-", "o", (930, 0.265)),
        (130.0, "#227668", ":", "s", (1170, 0.060)),
    ):
        ax.plot(wet, supersaturation(wet, dry), color=color, ls=style, lw=2.4)
        peak = minimize_scalar(
            lambda size, seed_nm: -supersaturation(size, seed_nm),
            args=(dry,),
            bounds=(dry * 1.001, 3000.0),
            method="bounded",
        )
        peak_wet, peak_ss = peak.x, -peak.fun
        ax.plot(peak_wet, peak_ss, marker=marker, color=color, ms=6)
        if dry == cutoff:
            np.testing.assert_allclose(peak_ss, target, rtol=1e-10)
            label = f"{dry:.2f} nm dry seed\n{peak_wet:.0f} nm wet at threshold"
        else:
            label = f"{dry:.0f} nm dry seed\nPeak {peak_ss:.2f}%"
        ax.annotate(
            label,
            xy=(peak_wet, peak_ss),
            xytext=label_xy,
            color=color,
            ha="left",
            va="center",
            fontsize=10.5,
            bbox={"facecolor": "white", "edgecolor": "none", "pad": 1.5},
            arrowprops={"arrowstyle": "-", "color": color, "shrinkB": 5},
        )
    fig.savefig(FIGURE, dpi=190, facecolor="white")
    plt.close(fig)


if __name__ == "__main__":
    main()
