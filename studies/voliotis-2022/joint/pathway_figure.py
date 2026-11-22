# %%
"""Compare shared pathway fits against the archived chamber trajectories."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from campaign import inputs

HERE = Path(__file__).resolve().parent
OUT = HERE / "results/pathways"
LABELS = {
    "reference": "Original shared chemistry",
    "prompt": "Prompt low-volatility product",
    "auto_competition": "Autoxidation + competition",
    "reactive_uptake": "Reactive uptake",
    "combined": "Autoxidation + uptake",
    "partner_yields": "Partner-specific yields",
    "volatile_diversion": "Diversion to volatile products",
    "diversion_uptake": "Diversion + uptake",
    "specified_uptake": "Diversion + prescribed uptake",
}


def main():
    """Plot matched objective scores and two runs that expose model limits."""
    scores = pd.read_csv(OUT / "comparison.csv")
    traces = pd.read_csv(OUT / "trajectories.csv.gz")
    mass = (
        scores.query("objective=='mass'").set_index("model").loc[list(LABELS)]
    )
    joint = scores.query("wall_free").set_index("model").loc[list(LABELS)]
    comparison = "volatile_diversion"
    fig = plt.figure(figsize=(12.3, 6.1), layout="constrained")
    grid = fig.add_gridspec(2, 2, width_ratios=[1.08, 1.3])
    ax = fig.add_subplot(grid[:, 0])
    y = np.arange(len(mass))
    ax.scatter(
        100 * mass.mass,
        y - 0.13,
        marker="s",
        color="#ab512e",
        label="Mass fit, fixed wall",
    )
    ax.scatter(
        100 * joint.joint,
        y + 0.13,
        marker="o",
        color="#13677a",
        label="Mass and number fit, wall refitted",
    )
    ax.axvline(10, color="0.55", ls=":", lw=1)
    ax.set(
        yticks=y,
        yticklabels=LABELS.values(),
        xlabel="Worst normalized trajectory error (%)",
        title="Shared fits across 15 experiments",
        ylim=(len(y) - 0.4, -1.2),
    )
    ax.legend(loc="upper left", fontsize=8, frameon=False)
    curve = traces.query("wall_free")
    counts = pd.read_csv(HERE / "data/wall_counts.csv")
    for i, (run, quantity, ylab) in enumerate(
        [
            (24, "soa_ug_m3", "Suspended SOA (µg m⁻³)"),
            (16, "number_fraction", "Fraction of initial number"),
        ]
    ):
        axis = fig.add_subplot(grid[i, 1])
        if run == 24:
            _, _, t, obs, _ = next(r for r in inputs() if r[0] == run)
        else:
            data = counts.query("run==16 and instrument=='WCPC'").sort_values(
                "time_s"
            )
            t = data.time_s.to_numpy()
            obs = data.count_cm3.to_numpy()
            obs = obs / obs[np.abs(t).argmin()]
        mask = (t >= 0) & (t <= 18000)
        axis.scatter(
            t[mask] / 3600,
            obs[mask],
            s=9,
            color="#353b3b",
            label="Measured",
            zorder=3,
        )
        for model, color, style in [
            ("reference", "#a46a44", "--"),
            (comparison, "#13677a", "-"),
        ]:
            data = curve[(curve.run == run) & (curve.model == model)]
            axis.plot(
                np.r_[0, data.time_min / 60],
                np.r_[float(run == 16), data[quantity]],
                style,
                color=color,
                label=LABELS[model],
                lw=2,
            )
        axis.set(
            ylabel=ylab,
            xlabel="Hours after lights on",
            title=f"Run {run} · "
            + ("ternary mixture" if run == 24 else "α-pinene and o-cresol"),
        )
        if i == 0:
            axis.legend(frameon=False, fontsize=8)
    for a in fig.axes:
        a.spines[["top", "right"]].set_visible(False)
        a.grid(axis="x", alpha=0.14)
    fig.savefig(OUT / "mechanism-comparison.png", dpi=170)
    plt.close(fig)


if __name__ == "__main__":
    main()
