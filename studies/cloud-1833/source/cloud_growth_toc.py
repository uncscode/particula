"""Plot the original equal-flux growth-law comparison in CLOUD run 1833 from the frozen study packet."""

import os
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.patches import Circle, Ellipse, Rectangle
from particula.plotting.toc import add_toc_logo

root = Path(__file__).resolve().parents[1]
results = Path(os.environ.get("CLOUD_RESULTS", root / "full-transfer"))
figures = Path(
    os.environ.get("CLOUD_FIGURES", root.parents[1] / "docs/images/cloud-1833")
)
figures.mkdir(parents=True, exist_ok=True)
shape = (
    pd.read_csv(results / "stage-scores.csv")
    .query('cells == 288 and segment == "excimer_led" and stage == 999')
    .set_index("model")
    .shape_pct
)
trajectory = (
    pd.read_csv(results / "matched-series.csv.gz")
    .query(
        'cells == 288 and segment == "excimer_led" and model == "Attraction"'
    )
    .sort_values("paper_h")
)
observed_time = trajectory.loc[
    trajectory.observed_tail >= 5000, "paper_h"
].iloc[0]
modeled_time = trajectory.loc[trajectory.model_tail >= 5000, "paper_h"].iloc[0]
delay_minutes = round(60 * (modeled_time - observed_time))
ink, teal, rust, paper = "#243E43", "#135963", "#A44D21", "#FBF2E3"
with plt.style.context("particula.toc"):
    fig = plt.figure(figsize=(3, 3), dpi=300)
    ax = fig.add_axes(
        [0, 0, 1, 1], xlim=(0, 300), ylim=(300, 0), aspect="equal"
    )
    ax.axis("off")

    def text(x, y, label, size=8.7, weight="normal", color=ink, align="left"):
        return ax.text(
            x,
            y,
            label,
            ha=align,
            va="center",
            fontsize=size,
            fontweight=weight,
            color=color,
        )

    # Chamber icon is schematic.
    ax.plot([14, 23, 23], [26, 26, 30], color=ink, lw=1.15)
    ax.plot([51, 56, 56, 64], [35, 35, 28, 28], color=ink, lw=1.15)
    ax.add_patch(
        Rectangle((23, 15), 29, 24, facecolor=paper, edgecolor=ink, lw=1.15)
    )
    ax.add_patch(
        Ellipse((37.5, 39), 29, 8, facecolor=paper, edgecolor=ink, lw=1.15)
    )
    ax.add_patch(
        Rectangle((23.8, 32), 27.4, 6, facecolor=paper, edgecolor="none")
    )
    ax.add_patch(
        Ellipse((37.5, 15), 29, 8, facecolor=paper, edgecolor=ink, lw=1.15)
    )
    for x, y in [(29, 27), (35, 23), (41, 32), (46, 25), (34, 32)]:
        ax.add_patch(Circle((x, y), 1.25, color=teal))
    text(67, 26, "CERN CLOUD · Run 1833", 9.4, "bold")
    text(17, 62, "Same particle entry at 2.9 nm", 11.1, "bold")
    text(17, 82, "Size-distribution mismatch · stages 9–17", 8.7)
    labels = [
        ("Vapor transfer", "#58676A"),
        ("Hard sphere", rust),
        ("Attraction", teal),
    ]
    for y, (name, col) in zip([110, 135, 160], labels):
        pct = shape.loc[name]
        text(17, y, name, 8.8)
        ax.add_patch(
            Rectangle(
                (115, y - 4.5), pct * 4.55, 9, facecolor=col, edgecolor="none"
            )
        )
        text(121 + pct * 4.55, y, f"{pct:.0f}%", 9, "bold", col)
    ax.plot([17, 281], [183, 183], lw=0.6, color="#B1ABA1")
    text(17, 204, "When did larger particles build up?", 10.0, "bold")
    text(17, 222, "First 5,000/cm³ between 9.7 and 64.7 nm", 8.2)
    ax.plot([64, 228], [247, 247], color=ink, lw=0.85)
    ax.scatter(
        [64], [247], s=49, facecolor=paper, edgecolor=ink, lw=1.3, zorder=3
    )
    ax.scatter(
        [228], [247], s=49, facecolor=teal, edgecolor=teal, lw=1.3, zorder=3
    )
    text(
        146, 245, f"{delay_minutes} min later", 8.8, "bold", align="center"
    ).set_bbox(dict(facecolor=paper, edgecolor="none", pad=2))
    text(64, 270, "CLOUD measured", 8.5, color=ink, align="center")
    text(228, 270, "Particula · attraction", 8.5, color=teal, align="center")
    add_toc_logo(fig, corner="top right")
    fig.savefig(figures / "CLOUD-1833-growth-TOC.png", dpi=300)
    plt.close(fig)
