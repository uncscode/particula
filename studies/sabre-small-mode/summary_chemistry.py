"""Compact chemistry comparisons for the main SABRE study page."""

import matplotlib.pyplot as plt
from chemistry.plot_field_fit import DUAL, INK, SHARED


def plot_predictions(rows):
    """Compare the two fitted predictions at the flight quartile medians."""
    fig, ax = plt.subplots(figsize=(7, 3.4), layout="constrained")
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(color="#879498", labelcolor=INK)
    ax.set_axisbelow(True)
    rows = rows.sort_values("median_small_wet_fraction")
    x = 100 * rows.median_small_wet_fraction
    for column, color, marker, line, label in (
        (
            "median_shared_effective_gamma",
            SHARED,
            "s",
            "--",
            "Common probability",
        ),
        (
            "median_separate_effective_gamma",
            DUAL,
            "o",
            "-",
            "Separate probabilities",
        ),
    ):
        ax.plot(
            x,
            rows[column],
            marker=marker,
            ls=line,
            color=color,
            lw=2.2,
            ms=7,
            label=label,
        )
    ax.set(
        xlim=(x.iloc[0] - 1.8, x.iloc[-1] + 1.8),
        ylim=(0.007, 0.0103),
        xlabel="Median small-mode share of wet area",
        ylabel="Predicted effective uptake probability",
    )
    ax.ticklabel_format(axis="y", useOffset=False, style="plain")
    ax.set_yticks([0.007, 0.008, 0.009, 0.010])
    ax.set_xticks(
        x,
        [
            f"{v:.1f}%\n{q} quarter"
            for v, q in zip(x, ("Lowest", "Highest"), strict=True)
        ],
    )
    ax.grid(axis="y", color="#e2e8e9")
    ax.legend(frameon=False, loc="lower left", bbox_to_anchor=(0, 1), ncol=2)
    return fig
