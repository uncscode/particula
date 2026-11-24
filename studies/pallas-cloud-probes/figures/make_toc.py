"""Draw the Pallas study card from the common-wavelength sizing trials."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Arc, Circle, Rectangle
from particula.plotting.toc import add_toc_logo

DOCS = Path(__file__).resolve().parents[3] / "docs"
PAPER, INK, TEAL, RUST = "#FBF2E3", "#243E43", "#135963", "#A44D21"
SLATE, MIST = "#617679", "#CAD9D2"


def ridge(ax):
    """Set the comparison against a schematic cloud-covered ridge."""
    x = np.linspace(0, 300, 401)
    # Schematic rounded summit and two banks of low cloud.
    for base, peaks, color in [
        (99, [(43, 29, 43), (170, 19, 80), (255, 32, 53)], "#e3e8de"),
        (111, [(30, 24, 42), (215, 28, 57)], "#d8e0d5"),
        (120, [(65, 23, 48), (269, 29, 60)], MIST),
    ]:
        y = base - sum(h * np.exp(-(((x - c) / w) ** 2)) for c, h, w in peaks)
        ax.fill_between(x, y, 124, facecolor=color, lw=0)
    ax.plot([18, 73], [100, 100], color=PAPER, lw=1.35)
    ax.plot([32, 100], [105, 105], color=PAPER, lw=0.9)
    ax.plot([216, 282], [112, 112], color=PAPER, lw=1.05)


def sphere(ax, x, y, r):
    """Shade a spherical water droplet with a small specular highlight."""
    yy, xx = np.mgrid[-1:1:320j, -1:1:320j]
    z = np.sqrt(np.maximum(0, 1 - xx**2 - yy**2))
    light = np.clip(-0.42 * xx - 0.62 * yy + 0.64 * z, 0, 1)
    tint = (0.20 + 0.79 * light)[..., None]
    dark, bright = np.array([0.12, 0.37, 0.40]), np.array([0.86, 0.95, 0.92])
    rgba = np.ones((*xx.shape, 4))
    rgba[:, :, :3] = dark + (bright - dark) * tint
    rgba[:, :, 3] = xx**2 + yy**2 <= 1
    ax.imshow(
        rgba, extent=(x - r, x + r, y + r, y - r), origin="upper", zorder=4
    )
    ax.add_patch(Circle((x, y), r, fc="none", ec=TEAL, lw=0.95, zorder=5))
    ax.add_patch(
        Arc(
            (x, y),
            1.62 * r,
            1.62 * r,
            theta1=207,
            theta2=258,
            ec="white",
            lw=1.5,
            zorder=6,
        )
    )


def optics(ax, left, right):
    """Draw the forward-scattering cone and segmented image."""
    y = 161
    # Forward light scattered into a narrow cone; remaining beam shown faintly.
    ax.plot([left - 33, left - 5], [y, y], color=TEAL, lw=2.1)
    ax.plot([left + 5, left + 30], [y, y], color=SLATE, lw=0.6, ls=":")
    for sign in (-1, 1):
        ax.fill(
            [left, left + 29, left + 26],
            [y, y + sign * 5, y + sign * 15],
            color=TEAL,
            alpha=0.20,
            lw=0,
        )
        ax.plot([left + 3, left + 27], [y, y + sign * 11], color=TEAL, lw=1.1)
    ax.add_patch(Circle((left, y), 4, fc=PAPER, ec=TEAL, lw=1))
    ax.add_patch(Arc((left, y), 63, 63, theta1=-27, theta2=27, ec=TEAL, lw=2))
    # A segmented circular image in a detector frame.
    for row in range(-3, 4):
        for col in range(-3, 4):
            if row * row + col * col <= 9.5:
                ax.add_patch(
                    Rectangle(
                        (right + col * 3.65 - 1.65, y + row * 3.65 - 1.65),
                        3.3,
                        3.3,
                        facecolor=RUST,
                        lw=0,
                    )
                )
    for sx in [-1, 1]:
        for sy in [-1, 1]:
            ax.plot(
                [right + sx * 12, right + sx * 18, right + sx * 18],
                [y + sy * 16, y + sy * 16, y + sy * 10],
                color=RUST,
                lw=1,
            )


def draw():
    """Render the square card using the archived mean diameter estimates."""
    df = pd.read_csv(DOCS / "data/pallas-cloud-probes/method-sizing.csv")
    pair = df.query("true_um == 11.96875 and nm == 633").set_index("probe")
    true = pair.true_um.iloc[0]
    means = pair.mean_um
    if not (means.F < 13 < means.I and (pair.registered == 1).all()):
        raise ValueError("Recheck the illustrated cutoff against the trials.")
    with plt.style.context("particula.toc"):
        fig = plt.figure()
        ax = fig.add_axes(
            [0, 0, 1, 1], xlim=(0, 300), ylim=(300, 0), aspect="equal"
        )
        ax.axis("off")
        ridge(ax)
        ax.text(
            16, 23, "Pallas, Finland", weight="bold", ha="left", va="center"
        )
        sphere(ax, 112, 76, 24)
        ax.text(151, 70, "Simulated droplet", fontsize=8, va="center")
        ax.text(
            151, 88, f"{true:.2f} µm", fontsize=12, weight="bold", va="center"
        )

        # Schematic branches connect the common input to the two observations.
        def scale(diameter):
            return 26 + (diameter - 12) * 124

        left, right = scale(means.F), scale(means.I)
        for target in [left, right]:
            ax.plot(
                [112, 112, target, target],
                [101, 118, 137, 142],
                color=SLATE,
                lw=0.9,
                zorder=3,
            )
        optics(ax, left, right)
        for name, x, col, mark, value in [
            ("FSSP", left, TEAL, "o", means.F),
            ("ICEMET", right, RUST, "s", means.I),
        ]:
            ax.text(
                x,
                189,
                name,
                ha="center",
                va="center",
                color=col,
                weight="bold",
                fontsize=9,
            )
            ax.text(
                x,
                211,
                f"{value:.2f}",
                ha="center",
                va="center",
                color=col,
                fontsize=12,
                weight="bold",
            )
            ax.plot(x, 229, marker=mark, color=col, markersize=5.5, zorder=5)
        ax.plot([26, 274], [229, 229], color=INK, lw=1.1)
        for v in (12, 14):
            x = scale(v)
            ax.plot([x, x], [225, 233], color=INK, lw=0.8)
            ax.text(x, 244, str(v), ha="center", va="center", fontsize=8)
        ax.plot([scale(13)] * 2, [214, 236], color=INK, ls=":", lw=1.1)
        ax.text(
            scale(13), 245, "13 µm cutoff", ha="center", va="center", fontsize=8
        )
        ax.text(
            150,
            274,
            "Mean reported diameter (µm)",
            ha="center",
            va="center",
            fontsize=9,
        )
        add_toc_logo(fig, corner="top right")
        fig.savefig(DOCS / "images/pallas-cloud-probes/study-toc.png", dpi=300)
        plt.close(fig)


if __name__ == "__main__":
    draw()
