# %% Replot from the compact saved daily concentrations.
import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from study import ROOT as root
table = pd.read_csv(root / "dates.csv")
upper = table[table.geometry == "upper"].set_index("day")
center = table[table.geometry == "center"].set_index("day")
fig, (a, b) = plt.subplots(
    1,
    2,
    figsize=(11.5, 5.7),
    gridspec_kw={"width_ratios": [1.28, 1]},
    layout="constrained",
)
blue = "#176789"
gold = "#cc8234"
grey = "#555961"
ys = np.arange(len(upper))
a.axvline(0, c="#43464c", lw=0.85)
for y, day in enumerate(upper.index):
    values = [
        upper.loc[day, "native"],
        upper.loc[day, "gap"],
        center.loc[day, "gap"],
    ]
    a.plot([min(values), max(values)], [y, y], color="#b5bdc0", lw=1.5)
a.scatter(
    upper.native,
    ys - 0.12,
    c=grey,
    s=48,
    marker="s",
    label="Archived native difference",
    zorder=3,
)
a.scatter(upper.gap, ys, c=blue, s=59, label="6–45 µm, upper edges", zorder=4)
a.scatter(
    center.gap,
    ys + 0.12,
    facecolors="white",
    edgecolors=gold,
    s=60,
    linewidths=1.8,
    marker="D",
    label="6–45 µm, centers",
    zorder=3,
)
a.axhline(0.5, c="#adb7bc", ls=":", lw=1)
a.set_yticks(
    ys, [f"Sep {day}   ({int(n):,})" for day, n in upper.minutes.items()]
)
a.invert_yaxis()
a.set(
    xlabel="FSSP minus ICEMET (droplets cm⁻³)",
    ylabel="Date (paired minutes)",
    title="Mean paired count difference",
)
a.legend(loc="lower left", bbox_to_anchor=(0, -0.30), frameon=False, fontsize=9)
indices = [0, 1, 2.65, 3.65]
bottom = np.zeros(4)
for band, color, label in [
    ("lower_common", "#73b4c5", "6–13.07 µm"),
    ("core", "#215876", "13.07–45 µm"),
]:
    heights = np.array(
        [
            upper.loc[day, f"{probe}_{band}"]
            for day in [15, 16]
            for probe in ["f", "i"]
        ]
    )
    b.bar(indices, heights, width=0.73, bottom=bottom, color=color, label=label)
    bottom += heights
for x, n in zip(indices, bottom):
    b.text(x, n + 4, f"{n:.0f}", ha="center", fontsize=10)
b.set_xticks(indices, ["FSSP", "ICEMET"] * 2)
b.text(0.5, -0.105, "Sep 15", transform=b.get_xaxis_transform(), ha="center")
b.text(3.15, -0.105, "Sep 16", transform=b.get_xaxis_transform(), ha="center")
b.set(
    ylabel="Droplets cm⁻³",
    title="Mean counts inside the shared window",
    ylim=(0, 238),
)
b.legend(loc="lower left", bbox_to_anchor=(0, -0.28), frameon=False, fontsize=9)
for ax in (a, b):
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="x" if ax is a else "y", alpha=0.15)
    ax.set_axisbelow(True)
fig.savefig(
    root / "comparison.png", dpi=180, bbox_inches="tight", facecolor="white"
)
plt.close(fig)
