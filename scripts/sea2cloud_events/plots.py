"""Published event readings and conditional marine cohort sensitivities."""

import matplotlib.pyplot as plt
import numpy as np


def windows(report):
    """Compare finite-window source capacity and approximate observed peaks."""
    fig, ax = plt.subplots(figsize=(7, 4), layout="constrained")
    rows = np.array(report["rows"])
    for i, ev in enumerate(report["events"]):
        hours = float(ev["end_hour_utc"]) - float(ev["start_hour_utc"])
        subset = rows[(rows[:, 0] == hours) & (rows[:, 4] == 2e7)]
        typical = subset[
            (subset[:, 1] == 1) & (subset[:, 2] == 150) & (subset[:, 3] == 100)
        ][0, 5]
        for offset, val, color, label in [
            (-0.25, subset[:, 5].max(), "#3b8795", "No-loss ceiling"),
            (0, typical, "#364779", "With scavenging¹"),
        ]:
            ax.bar(
                i + offset,
                val,
                width=0.22,
                color=color,
                label=label if i == 0 else None,
            )
        ax.errorbar(
            i + 0.23,
            float(ev["figure_peak_cm3"]),
            yerr=float(ev["reading_halfwidth_cm3"]),
            fmt="o",
            color="#af562a",
            capsize=5,
            label="Figure 5a peak" if i == 0 else None,
        )
    ax.axhline(500, color="#666", ls=":", label="Event criterion")
    ax.set(
        xticks=[0, 1],
        xticklabels=["19 March · 7 h", "22 March · 1 h"],
        ylabel="1–10 nm number (cm⁻³)",
        ylim=(0, 1850),
    )
    ax.legend(ncol=2, frameon=False, fontsize=9)
    fig.supxlabel(
        "¹ 150 background particles per cm³ at 100 nm, growth 1 nm/h",
        fontsize=9,
    )
    return fig


def sensitivity(report):
    """Show how growth changes scavenging and time spent below 10 nm."""
    rows = np.array(report["rows"])
    fig, axes = plt.subplots(1, 2, figsize=(8.5, 3.7), layout="constrained")
    for bg, color in zip(
        [0, 75, 150, 300],
        ["#383838", "#137785", "#ba672d", "#724678"],
        strict=True,
    ):
        cut = rows[
            (rows[:, 0] == 7)
            & (rows[:, 2] == bg)
            & (rows[:, 3] == 100)
            & (rows[:, 4] == 2e7)
        ]
        for ax, k in zip(axes, [5, 10], strict=True):
            ax.plot(
                cut[:, 1], cut[:, k], "o-", color=color, label=f"{bg:g} cm⁻³"
            )
    axes[0].legend(frameon=False, fontsize=9)
    axes[0].set_ylabel("Particles remaining (cm⁻³)")
    axes[1].set_ylabel("Background number elasticity")
    fig.supxlabel("Prescribed diameter growth (nm/h)")
    return fig
