"""Render the SiMBA chamber figure from repository qualification records."""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

DATA = Path(__file__).resolve().parent
HERE = DATA.parents[1] / "docs" / "images" / "simba-discovery"


def main():
    """Plot retrieved vapor, particle mass and size-selective counts."""
    records = json.loads((DATA / "trajectory_qualification.json").read_text())
    cases = records["cuda:0"]
    colors = ["#83502d", "#126580"]
    times = [0, 0.8, 2]
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.7), layout="constrained")
    for case, color in zip(cases, colors, strict=True):
        calculated = np.asarray(case["simulation"])
        independent = np.asarray(case["reference"])
        loading = case["initial_vapor_mol_m3"] * 0.1 * 1e9
        label = f"Initial vapor {loading:g} µg/m³"
        for ax, channel in zip(axes[:2], (1, 2), strict=True):
            ax.plot(
                times,
                calculated[:, channel] * 1e9,
                color=color,
                label=label,
                linewidth=2,
            )
            ax.scatter(
                times,
                independent[:, channel] * 1e9,
                facecolors="white",
                edgecolors=color,
                zorder=3,
            )
    for ax, title in zip(
        axes[:2], ("Product vapor", "Suspended organics"), strict=True
    ):
        ax.set(title=title, xlabel="Time (s)", ylabel="Mass (µg/m³)")
        ax.set_xticks(times)
    axes[0].legend(fontsize=8, frameon=False)
    final = cases[1]
    spectrum = np.array(final["simulation"])[-1, 3:] / 1e6
    hard = np.array(final["hard_bin_counts_m3"])[-1] / 1e6
    x = np.arange(3)
    axes[2].bar(
        x - 0.19,
        spectrum,
        width=0.38,
        color=colors[1],
        label="Detector response",
    )
    axes[2].bar(x + 0.19, hard, width=0.38, color="#bfcad1", label="Hard bins")
    axes[2].set(
        title="Size spectrum at 2 s",
        ylabel="Number per bin (cm⁻³)",
        xlabel="Diameter channel (nm)",
    )
    axes[2].set_xticks(x, ["<80", "80–120", "≥120"])
    axes[2].legend(frameon=False, fontsize=8)
    for ax in axes:
        ax.spines[["top", "right"]].set_visible(False)
    fig.savefig(HERE / "coupled-trajectory.png", dpi=180)
    plt.close(fig)

    # Recenter AIC within each case so the competing ranks are comparable.
    rows = json.loads((DATA / "published_rankings.json").read_text())
    fig, axes = plt.subplots(1, 2, figsize=(8, 3.6), layout="constrained")
    for ax, case, title in zip(
        axes,
        ("aldol_condensation", "fruc_HMF2"),
        ("Aldol", "Fructose"),
        strict=True,
    ):
        fits = [
            r
            for r in rows
            if r["case"] == case and r["published_aic"] is not None
        ]
        for field, label, color, marker in (
            ("aic", "Archive refit", colors[1], "o"),
            ("published_aic", "Published", colors[0], "s"),
        ):
            scores = np.array([r[field] for r in fits])
            delta = scores - scores.min()
            ax.plot(
                [r["iteration"] for r in fits],
                delta,
                color=color,
                marker=marker,
                linewidth=1.8,
                label=label,
            )
        ax.axhspan(0, 2, color="grey", alpha=0.12)
        ax.set(
            title=title, xlabel="SiMBA iteration", ylabel="ΔAIC from best fit"
        )
        ax.set_xticks([r["iteration"] for r in fits])
        ax.set_yscale("symlog", linthresh=2)
        ax.spines[["top", "right"]].set_visible(False)
        ax.legend(frameon=False)
    fig.savefig(HERE / "mechanism-rankings.png", dpi=180)
    plt.close(fig)


def profile_figure():
    """Show which observable set constrains the product saturation level."""
    fitted = json.loads((DATA / "heldout_recovery.json").read_text())
    profile = fitted["volatility_profile"]
    x = np.asarray(profile["saturation_mol_m3"]) * 1e8
    fig, ax = plt.subplots(figsize=(6, 3.3), layout="constrained")
    for key, label, color in [
        ("joint", "Vapors, mass and size", "#126580"),
        ("precursor_only", "Precursor alone", "#83502d"),
    ]:
        ax.plot(
            x, profile["delta_loss"][key], marker="o", label=label, color=color
        )
    ax.set(
        xlabel="Product saturation concentration (10⁻⁸ mol/m³)",
        ylabel="Increase in negative log likelihood",
    )
    ax.legend(frameon=False)
    ax.spines[["top", "right"]].set_visible(False)
    fig.savefig(HERE / "volatility-profile.png", dpi=180)
    plt.close(fig)


if __name__ == "__main__":
    main()
    profile_figure()
