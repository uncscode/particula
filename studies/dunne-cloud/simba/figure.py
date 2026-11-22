"""Draw the measured trajectory and the paired chemistry comparison."""

import csv
from pathlib import Path
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent


def load(name):
    with (HERE / name).open() as file:
        return list(csv.DictReader(file))


def show():
    rows = load("traces.csv")
    scores = load("sensitivity.csv")
    models = [
        ("original", "Original settings", "#8B959B", ":"),
        ("original_kernel_physics", "Physical refit", "#226c78", "-"),
        ("original_kernel_ozone", "Physical refit + ozone", "#b15534", "--"),
        (
            "original_kernel_peroxide",
            "Physical refit + peroxide",
            "#7766a3",
            "-.",
        ),
    ]
    fig, (a, b) = plt.subplots(
        1,
        2,
        figsize=(10.7, 4.3),
        gridspec_kw={"width_ratios": [1.4, 1]},
        layout="constrained",
    )
    for key, label, color, style in models:
        part = [
            r for r in rows if r["model"] == key and float(r["hour"]) < 6.001
        ]
        h = [float(r["hour"]) for r in part]
        if key == "original":
            a.plot(
                h,
                [float(r["obs_N10"]) / 1000 for r in part],
                color="#233139",
                lw=2.2,
                label="Measured",
            )
        a.plot(
            h,
            [float(r["pred_N10"]) / 1000 for r in part],
            c=color,
            ls=style,
            label=label,
            lw=1.7,
        )
    a.axvspan(3.8, 6.0, color="#c5dce6", alpha=0.23, zorder=0)
    a.text(3.9, 32, "Withheld", fontsize=9, color="#3b5266")
    a.set(
        xlim=(0, 6),
        ylim=(0, 34),
        xlabel="Hours from first spectrum",
        ylabel="Particles ≥9.7 nm physical (10³ cm⁻³)",
    )
    a.legend(frameon=False, loc="upper left", fontsize=8)
    a.set_title("CLOUD run 1833", loc="left", weight="bold")
    for shift, lock, marker, col, label in [
        (-0.1, "fixed", "s", "#82898e", "Original physical settings"),
        (0.1, "fitted", "o", "#226c78", "Refitted physical settings"),
    ]:
        ys = []
        for family in ["physics", "ozone", "peroxide", "both"]:
            r = next(
                r
                for r in scores
                if r["fit_window"] == "early"
                and r["physics"] == lock
                and r["family"] == family
            )
            ys.append(float(r["peak_block_rmse"]) / 1000)
        b.scatter(
            np.arange(4) + shift,
            ys,
            c=col,
            marker=marker,
            s=52,
            label=label,
            zorder=3,
        )
    b.set(
        xticks=range(4),
        xticklabels=["None", "Ozone", "Peroxide", "Both"],
        ylim=(0, 8.8),
        xlim=(-0.45, 3.45),
        xlabel="Added oxidant pathway",
        ylabel="Withheld RMSE (10³ cm⁻³)",
    )
    b.grid(axis="y", color="#e7e8e9")
    b.legend(frameon=False, fontsize=8, loc="upper center")
    b.set_title("Same 3.8–6 h test interval", loc="left", weight="bold")
    for ax in (a, b):
        ax.spines[["top", "right"]].set_visible(False)
    fig.savefig(HERE / "growth-gap.png", dpi=190)


if __name__ == "__main__":
    show()
