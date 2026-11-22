"""Compact graphical abstract for the revised private packet."""

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

p = Path(__file__).resolve().parent
s = (
    pd.read_csv(p / "source-bounds.csv")
    .query("line==336")
    .set_index("topology")
)
fig = plt.figure(figsize=(3, 3), dpi=300, facecolor="#fff7dc")
fig.text(0.085, 0.9, "CERN CLOUD", weight="bold", size=11, color="#234149")
fig.text(0.085, 0.80, "248.3 K  ·  0.38 pptv NH₃", size=9)
fig.text(0.085, 0.735, "1.80 × 10⁷ H₂SO₄ molecules cm⁻³", size=8)
fig.text(0.085, 0.65, "Formation at 1.7 nm", size=10, weight="bold")
ax = fig.add_axes([0.11, 0.23, 0.79, 0.37], facecolor="none")
for y, t, color in [(2, "full", "#175671"), (1, "assisted", "#a44d25")]:
    r = s.loc[t]
    ax.hlines(y, r.p05, r.p95, lw=3.2, color=color)
    ax.scatter(r["median"], y, s=24, color=color, zorder=3)
    ax.text(
        0.012,
        y + 0.24,
        "Binary allowed" if t == "full" else "Ammonia required",
        size=7,
    )
ax.scatter(0.03, 0, color="#212529", marker="D", s=22)
ax.text(0.037, 0, "CLOUD rate", va="center", size=7)
ax.set(
    xscale="log",
    xlim=(0.009, 1.1),
    ylim=(-0.35, 2.62),
    yticks=[],
    xticks=[0.01, 0.1, 1],
    xticklabels=["0.01", "0.1", "1"],
)
ax.minorticks_off()
ax.tick_params(axis="x", labelsize=7)
for side in ("left", "right", "top"):
    ax.spines[side].set_visible(False)
fig.text(0.51, 0.10, "particles cm⁻³ s⁻¹", size=8, ha="center")
fig.savefig(p / "formation-overview.png", dpi=300)
