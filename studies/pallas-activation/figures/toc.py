"""Pallas TOC: midpoint and full-curve salt-enrichment constraints."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from matplotlib.patches import Arc, Circle
from particula.plotting.toc import add_toc_logo

ROOT = Path(__file__).resolve().parents[3]
DATA = ROOT / "docs/data/pallas-activation"
IMAGES = ROOT / "docs/images/pallas-activation"
PAPER = "#FBF2E3"
INK = "#243E43"
TEAL = "#135963"
PALE = "#CAD9D2"
RUST = "#A44D21"


def particle(ax, x, diameter, fraction):
    """The silhouette encodes dry diameter; the arc outside it encodes mean salt."""
    y = 139
    radius = diameter * 0.39
    disk = Circle((x, y), radius, facecolor=PAPER, edgecolor=INK, lw=0.9)
    ax.add_patch(disk)
    # A regular stipple screen shades the silhouette without assigning chemical domains.
    step = 3.2
    for row, dy in enumerate(np.arange(-radius, radius, step)):
        dx = np.arange(-radius, radius, step) + (row % 2) * step / 2
        dx = dx[dx**2 + dy**2 < (radius - 1.6) ** 2]
        light = np.clip(0.45 + (dx - dy) / (2.8 * radius), 0.0, 1.0)
        ax.scatter(
            x + dx,
            np.full(len(dx), y + dy),
            s=(0.18 + 0.95 * light) ** 2,
            color="#617679",
            edgecolors="none",
            zorder=2,
        )
    gauge = 2 * (radius + 6)
    ax.add_patch(
        Arc(
            (x, y),
            gauge,
            gauge,
            theta1=90,
            theta2=450 - 360 * fraction,
            lw=3.6,
            capstyle="butt",
            color=TEAL,
        )
    )
    ax.add_patch(
        Arc(
            (x, y),
            gauge,
            gauge,
            theta1=90 - 360 * fraction,
            theta2=90,
            lw=3.6,
            capstyle="butt",
            color=RUST,
        )
    )
    ax.text(x, 72, f"{diameter} nm", ha="center", fontsize=10.4, color=INK)
    ax.text(
        x,
        52,
        f"{100 * fraction:.1f}% salts",
        ha="center",
        fontsize=11.4,
        fontweight="bold",
        color=RUST,
    )
    ax.text(
        x,
        36,
        f"{100 * (1 - fraction):.1f}% organics",
        ha="center",
        fontsize=10.5,
        fontweight="bold",
        color=TEAL,
    )


def draw():
    days = pd.read_csv(DATA / "daily.csv")
    scores = pd.read_csv(DATA / "midpoint-check.csv")
    assert set(scores.date) == set(days.date)
    with plt.style.context("particula.toc"):
        fig = plt.figure(figsize=(3, 3), facecolor=PAPER)
        ax = fig.add_axes(
            [0, 0, 1, 1], xlim=(0, 300), ylim=(0, 300), aspect="equal"
        )
        ax.axis("off")
        ax.text(
            17,
            274,
            "Pallas, Finland",
            fontsize=14,
            color=INK,
            fontweight="bold",
        )
        ax.text(
            17,
            252,
            "Salt enrichment required at 120 vs 50 nm",
            fontsize=8.05,
            color=INK,
        )
        for x, label, kind in [
            (76, "Five D₅₀ values", "midpoint"),
            (212, "Full activation curves", "curve"),
        ]:
            penalty = scores[f"{kind}_no_enrichment"] - scores[f"{kind}_rmse"]
            count = int((penalty > 0.01).sum())
            ax.text(x, 233, label, ha="center", fontsize=8.5, color=INK)
            ax.text(
                x,
                210,
                f"{count} of {len(scores)} days",
                ha="center",
                fontsize=12.3,
                fontweight="bold",
                color=INK,
            )
        ax.plot([16, 284], [198, 198], color=PALE, lw=0.7)
        particle(ax, 76, 50, days.salt50.mean())
        particle(ax, 212, 120, days.salt120.mean())
        ax.text(
            150,
            12,
            "Dry volume · mean of 27 full-curve fits",
            ha="center",
            fontsize=8.2,
            color=INK,
        )
        add_toc_logo(fig, corner="top right")
        fig.savefig(IMAGES / "salt-toc.png", dpi=300)
        plt.close(fig)


if __name__ == "__main__":
    draw()
