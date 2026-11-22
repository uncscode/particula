"""Render the cold CLOUD TOC from the published scenario quantiles."""

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from particula.plotting.toc import add_toc_logo

ROOT = Path(__file__).resolve().parents[3]
DATA = ROOT / "website/public/study-data/dunne-cloud/formation-overview.csv"
OUTPUT = ROOT / "docs/images/dunne-cloud/formation-toc.png"


@plt.style.context("particula.toc")
def render():
    """Show both central 90% scenario intervals and the archived measurement."""
    models = pd.read_csv(DATA).set_index("topology")
    fig = plt.figure()
    fig.text(0.06, 0.91, "CERN CLOUD", size=9, weight="bold")
    fig.text(0.06, 0.79, "Formation at 1.7 nm", size=14, weight="bold")
    fig.text(0.06, 0.70, "248.3 K  ·  0.38 pptv NH₃", size=8.5)
    fig.text(0.06, 0.63, "1.80 × 10⁷ H₂SO₄ molecules cm⁻³", size=8.5)
    add_toc_logo(fig, corner="top right")

    ax = fig.add_axes([0.10, 0.095, 0.83, 0.43])
    ax.set(
        xscale="log",
        xlim=(0.01, 1),
        ylim=(-0.35, 2.60),
        yticks=[],
        xticks=[0.01, 0.1, 1],
        xticklabels=["0.01", "0.1", "1"],
    )
    ax.minorticks_off()
    ax.spines["left"].set_visible(False)
    fig.text(0.06, 0.55, "Median and 90% range (particles cm⁻³ s⁻¹)", size=7.7)
    for y, (key, label, marker, color) in enumerate(
        [
            ("assisted", "Ammonia required", "s", "C1"),
            ("full", "Binary pathway allowed", "o", "C0"),
        ],
        start=1,
    ):
        row = models.loc[key]
        ax.text(0.0105, y + 0.21, label, size=8.2)
        ax.hlines(y, row.p05, row.p95, lw=2.5, color=color)
        ax.plot(row["median"], y, marker=marker, color=color)
    observed = models.loc["full", "observed"]
    ax.plot(observed, 0, marker="D", color=plt.rcParams["text.color"])
    ax.annotate(
        f"Measured {observed:.3f}",
        (observed, 0),
        (6, 0),
        textcoords="offset points",
        va="center",
        size=8,
    )
    fig.savefig(OUTPUT)
    plt.close(fig)


if __name__ == "__main__":
    render()
