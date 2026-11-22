"""Draw the frozen yield comparison and cross-run kappa predictions."""

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.lines import Line2D

HERE = Path(__file__).resolve().parent
NAMES = {
    "AC": "α-pinene + o-cresol",
    "AI": "α-pinene + isoprene",
    "CI": "o-cresol + isoprene",
    "ACI": "All three precursors",
}
COLORS = {"mass": "#007c78", "joint": "#995022"}


def yields():
    """Compare measured yield points with both frozen fits."""
    points = pd.read_csv(HERE / "yield_points.csv")
    fig, axes = plt.subplots(2, 2, figsize=(9, 6), layout="constrained")
    for (code, name), ax in zip(NAMES.items(), axes.flat, strict=True):
        frame = points[points.mixture == code]
        measured = frame.drop_duplicates("consumed")
        ax.scatter(
            measured.consumed,
            measured.observed * 100,
            marker="D",
            c="black",
            s=28,
            zorder=4,
        )
        for fit, color in COLORS.items():
            for _, group in frame[frame.fit == fit].groupby("run"):
                group = group.sort_values("consumed")
                ax.plot(
                    group.consumed,
                    group.model * 100,
                    "o-",
                    color=color,
                    alpha=0.8,
                    ms=4,
                )
        ax.set_title(name, loc="left")
        ax.set_ylim(bottom=0)
        ax.grid(alpha=0.18)
    fig.supxlabel("Total precursor consumed (µg m⁻³)")
    fig.supylabel("Wall-corrected SOA mass yield (%)")
    fig.legend(
        handles=[
            Line2D(
                [],
                [],
                marker="D",
                ls="",
                color="black",
                label="Figure 5 points",
            ),
            *[
                Line2D([], [], color=c, label=f"{k.capitalize()} fit")
                for k, c in COLORS.items()
            ],
        ],
        loc="outside upper center",
        ncol=3,
        frameon=False,
    )
    fig.savefig(HERE / "mixture-yields.png", dpi=170)
    plt.close(fig)


def water():
    """Compare fitted and transferred hygroscopicity at sample times."""
    df = pd.read_csv(HERE / "water_predictions.csv.gz")
    df = df[df.fit == "mass"]
    fig, axes = plt.subplots(
        2, 2, figsize=(9, 6), layout="constrained", sharex=True, sharey=True
    )
    for i, (channel, label) in enumerate(
        [
            ("HTDMA_100nm", "100 nm humidified particles"),
            ("CCN", "Cloud-droplet activation"),
        ]
    ):
        for j, (run, name) in enumerate(
            [(6, "α-pinene"), (15, "o-cresol + isoprene")]
        ):
            ax = axes[i, j]
            data = df[(df.run == run) & (df.channel == channel)]
            obs = data.drop_duplicates("seconds")
            ax.scatter(
                obs.seconds / 3600,
                obs.observed,
                s=16,
                color="black",
                label="Measured",
            )
            for mode, train, color, ls, title in [
                (
                    "volume_mixing",
                    "pooled",
                    "#798791",
                    "--",
                    "Bulk volume mixing",
                ),
                ("enriched", "pooled", "#007c78", "-", "Two-run fit"),
                (
                    "enriched",
                    str(15 if run == 6 else 6),
                    "#995022",
                    "-.",
                    "Fit on the other run",
                ),
            ]:
                v = data[
                    (data["mode"] == mode) & (data.trained_on == train)
                ].sort_values("seconds")
                ax.plot(
                    v.seconds / 3600,
                    v.predicted,
                    color=color,
                    linestyle=ls,
                    label=title,
                )
            ax.set_title(f"{name} · run {run}" if i == 0 else "", loc="left")
            if j == 0:
                ax.set_ylabel(label + "\nEffective κ")
            ax.grid(alpha=0.18)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="outside upper center",
        ncol=4,
        frameon=False,
        fontsize=9,
    )
    fig.supxlabel("Hours after lights on")
    fig.savefig(HERE / "water-transfer.png", dpi=170)
    plt.close(fig)


if __name__ == "__main__":
    yields()
    water()
