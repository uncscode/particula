"""Replot paired March 2023 flight residuals from saved diagnostic tables."""

import matplotlib.pyplot as plt
import numpy as np

CASES = [
    ("full_ext", "Full population", "#1b536c", "o", -0.16),
    ("without_below150_ext", "Below 150 nm omitted", "#328576", "s", 0.0),
    ("without_mode_ext", "Fitted mode omitted", "#ae4e21", "D", 0.16),
]


def render(medians, sensitivity, output):
    """Plot the archived per-flight biases and the nine sizing cases."""
    medians = medians.set_index("flight")
    counts = medians.pop("count")
    rows = np.arange(len(counts))
    fig, (left, right) = plt.subplots(
        1, 2, figsize=(10.4, 4.7), gridspec_kw={"width_ratios": [1.25, 1]}
    )
    fig.subplots_adjust(
        left=0.145, bottom=0.17, top=0.87, right=0.975, wspace=0.43
    )
    for key, label, color, symbol, offset in CASES:
        left.plot(
            medians[key],
            rows + offset,
            ls="none",
            marker=symbol,
            ms=5.4,
            color=color,
            label=label,
        )
    for i, (_, values) in enumerate(medians.iterrows()):
        left.plot(
            values,
            [i + case[-1] for case in CASES],
            color="#bac4c5",
            lw=0.9,
            zorder=0,
        )
    left.set_yticks(
        rows,
        [
            f"March {str(date)[-2:].lstrip('0')}   ({number})"
            for date, number in counts.items()
        ],
    )
    left.set_ylim(len(rows) - 0.35, -0.7)
    left.set_xlim(-5.15, 0.36)
    left.axvline(0, color="#777", lw=0.9, ls="--")
    left.set_xlabel("Median modeled − measured extinction (Mm⁻¹)")
    left.set_title(
        "Flight medians at nominal sizing", loc="left", weight="bold"
    )
    left.legend(
        loc="upper left",
        bbox_to_anchor=(0.00, -0.17),
        ncol=3,
        borderaxespad=0,
        handletextpad=0.3,
        columnspacing=1.0,
        frameon=False,
        fontsize=8.5,
    )

    for real, group in sensitivity.groupby("real_index"):
        color = {1.4: "#776699", 1.5: "#1b536c", 1.6: "#ad661c"}[real]
        group = group.sort_values("diameter_scale")
        right.plot(
            group.diameter_scale,
            group.median_full_bias_Mm1,
            marker="o",
            color=color,
            lw=1.8,
            label=f"n = {real:.1f}",
        )
    right.axhline(0, color="#777", lw=0.9, ls="--")
    right.set_xticks([0.9, 1.0, 1.1], ["0.90", "1.00", "1.10"])
    right.set_xlabel("Diameter multiplier")
    right.set_ylabel("Median modeled − measured\nextinction (Mm⁻¹)")
    right.set_title("Full population, all windows", loc="left", weight="bold")
    right.legend(
        loc="upper left",
        frameon=False,
        title="Real refractive index",
        fontsize=9,
        title_fontsize=9,
    )
    for ax in (left, right):
        ax.grid(axis="x" if ax is left else "y", alpha=0.23)
        ax.set_axisbelow(True)
        ax.spines[["top", "right"]].set_visible(False)
    fig.savefig(
        output,
        dpi=165,
        bbox_inches="tight",
        pil_kwargs={"lossless": True, "method": 6, "quality": 100},
    )
    plt.close(fig)
