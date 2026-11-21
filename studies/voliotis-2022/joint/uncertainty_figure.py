# %%
"""Plot wall profiles and variance fractions from the crossed design."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

DEST = Path(__file__).resolve().parent / "results" / "uncertainty"


def main():
    """Render the primary results using only the archived summary tables."""
    profile = pd.read_csv(DEST / "wall_profile.csv")
    sensitivity = pd.read_csv(DEST / "sensitivity_by_run.csv")
    scores = pd.read_csv(DEST.parent / "wall/campaign_scores.csv")
    control = (
        scores.query("model == 'constant_brownian' and objective == 'joint'")[
            ["mass_nrmse", "count_nrmse"]
        ]
        .max()
        .max()
    )
    fig, (ax, right) = plt.subplots(
        1,
        2,
        figsize=(11, 4.7),
        layout="constrained",
        gridspec_kw={"width_ratios": [1, 1.05]},
    )
    blue, rust = "#17697a", "#a14424"
    ax.plot(
        profile.eddy_s_inv,
        100 * profile.frozen_chemistry,
        "--o",
        color=rust,
        label="Chemistry held fixed",
    )
    ax.plot(
        profile.eddy_s_inv,
        100 * profile.best_found,
        "-o",
        color=blue,
        label="Chemistry refitted",
    )
    ax.axhline(
        control * 100, color="#4b5563", linestyle=":", label="Constant-loss fit"
    )
    ax.annotate(
        "21.41%",
        (5.108, 21.405),
        xytext=(0, -18),
        textcoords="offset points",
        ha="center",
        color=blue,
    )
    ax.set(
        xlabel="Wall eddy coefficient E (s⁻¹)",
        ylabel="Worst mass or count trajectory error (%)",
        ylim=(18.5, 39),
        title="Can chemistry offset particle deposition?",
    )
    ax.legend(loc="upper left", frameon=False, fontsize=9)
    runs = list(sensitivity.run.unique())
    coord = np.arange(len(runs))
    for name, shift, marker, color, label in [
        ("peak_soa_ug_m3", -0.12, "o", rust, "Peak SOA mass"),
        (
            "five_hour_number_fraction",
            0.12,
            "s",
            blue,
            "Particle number at 5 h",
        ),
    ]:
        fr = sensitivity[sensitivity.outcome == name].set_index("run").loc[runs]
        right.scatter(
            coord + shift,
            100 * fr.wall_fraction,
            color=color,
            marker=marker,
            label=label,
            s=28,
        )
    right.set(
        title="How much of the spread comes from wall loss?",
        xlabel="Voliotis experiment",
        ylabel="Wall share of scenario variance (%)",
        ylim=(-4, 112),
        xticks=coord,
        xticklabels=runs,
    )
    right.legend(loc="center left", frameon=False, fontsize=9)
    for axis in (ax, right):
        axis.spines[["top", "right"]].set_visible(False)
        axis.grid(axis="y", alpha=0.18)
        axis.set_axisbelow(True)
    fig.savefig(DEST / "joint-uncertainty.png", dpi=170)
    plt.close(fig)


if __name__ == "__main__":
    main()
