"""Pallas TOC: matched activation-fit errors and two diameter-scaled salt-fraction gauges."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from matplotlib.patches import Arc, Circle
from particula.plotting.toc import add_toc_logo

ROOT = Path(__file__).resolve().parents[3]
DATA = ROOT / "website/public/study-data/pallas-activation"
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
            251,
            "Activation-fraction RMSE (percentage points)",
            fontsize=8.4,
            color=INK,
        )
        for x, label, error in [
            (76, "Empirical κ(D) refit", days.empirical_rmse.mean()),
            (212, "Salt + two organics", days.rmse.mean()),
        ]:
            ax.text(x, 232, label, ha="center", fontsize=8.5, color=INK)
            ax.text(
                x,
                208,
                f"{100 * error:.2f}",
                ha="center",
                fontsize=17,
                fontweight="bold",
                color=INK,
            )
        ax.annotate(
            "",
            xy=(166, 216),
            xytext=(134, 216),
            arrowprops=dict(arrowstyle="->", lw=1, color=INK),
        )
        ax.plot([16, 284], [198, 198], color=PALE, lw=0.7)
        particle(ax, 76, 50, days.salt50.mean())
        particle(ax, 212, 120, days.salt120.mean())
        ax.text(
            150,
            12,
            "Dry volume · mean of 27 daily fits",
            ha="center",
            fontsize=8.2,
            color=INK,
        )
        add_toc_logo(fig, corner="top right")
        fig.savefig(IMAGES / "salt-toc.png", dpi=300)
        plt.close(fig)


if __name__ == "__main__":
    draw()
