# %%
"""Plot the campaign fits at each measurement time without rerunning physics."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent


if __name__ == "__main__":
    curves = pd.read_csv(HERE / "results/campaign_curves.csv")
    doses = pd.read_csv(HERE / "results/campaign_dose_curves.csv")
    design = pd.read_csv(HERE / "data/campaign_peaks.csv").set_index("run")
    groups = list(curves.groupby("run"))
    fig, axs = plt.subplots(
        int(np.ceil(len(groups) / 4)),
        4,
        figsize=(12, 2.2 * np.ceil(len(groups) / 4)),
        sharex=True,
        layout="constrained",
    )
    for ax, (r, d) in zip(axs.flat, groups, strict=False):
        data = d[d.family == "additive"]
        ax.scatter(
            data.time_s / 3600, data.measured, color="0.35", s=6, label="AMS"
        )
        for family, label, color, linestyle in (
            ("additive", "Shared parent yields", "#b66038", "--"),
            (
                "parent_lags",
                "Mixture chemistry, nominal doses",
                "#066968",
                "-",
            ),
        ):
            y = d[d.family == family]
            ax.plot(
                y.time_s / 3600,
                y.modeled,
                color=color,
                ls=linestyle,
                label=label,
            )
        uncertain = doses[doses.run == r]
        ax.plot(
            uncertain.time_s / 3600,
            uncertain.modeled,
            color="#625078",
            lw=1.6,
            ls="-.",
            label="Mixture chemistry, fitted doses",
        )
        labels = [
            label
            for species, label in (
                ("alpha_pinene", "α-pinene"),
                ("o_cresol", "o-cresol"),
                ("isoprene", "isoprene"),
            )
            if design.loc[r, species + "_ppb"] > 0
        ]
        ax.set_title(f"{r} · " + ", ".join(labels), loc="left", fontsize=9)
        if data.measured.abs().max() == 0:
            ax.set_ylim(-0.05, 1.0)
        ax.grid(axis="y", alpha=0.13)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
    for ax in axs.flat[len(groups) :]:
        ax.set_visible(False)
    for ax in axs[-1, :]:
        if ax.get_visible():
            ax.set_xlabel("Hours after lights on", fontsize=9)
    fig.supylabel(
        "Suspended organic aerosol (µg m⁻³) · independent vertical scales"
    )
    handles, labels = axs.flat[0].get_legend_handles_labels()
    fig.legend(
        handles, labels, loc="outside lower center", ncol=2, frameon=False
    )
    fig.savefig(HERE / "results/campaign-curves.png", dpi=180)
    plt.close(fig)
