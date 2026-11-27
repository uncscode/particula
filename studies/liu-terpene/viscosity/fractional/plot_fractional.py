"""Redraw frozen-composition predictions from the saved simulations."""

from argparse import ArgumentParser
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


def render(out):
    c = pd.read_csv(out / "fractional_cases.csv")
    v = pd.read_csv(out / "fractional_curves.csv").merge(c)
    palette = [
        ("classical", "Classical", "#46677B", "--"),
        ("Evoy_fractional", "Evoy fractional", "#B2512F", "-"),
    ]
    fig, axs = plt.subplots(
        1,
        2,
        figsize=(8.5, 3.6),
        sharey=True,
        gridspec_kw={"width_ratios": [2.35, 1]},
        constrained_layout=True,
    )
    for ax in axs:
        for name, label, color, style in palette:
            g = v[v.law == name]
            e = g.groupby("time_min").predicted_vfr.agg(["min", "max"])
            nom = g[
                (g.viscosity_Pa_s == 2.1e8) & (g.tracer_radius_nm == 0.4)
            ].sort_values("time_min")
            ax.fill_between(
                e.index, e["min"], e["max"], alpha=0.14, facecolor=color
            )
            ax.plot(
                nom.time_min,
                nom.predicted_vfr,
                label=label,
                color=color,
                ls=style,
                lw=1.9,
            )
        actual = v.dropna(subset=["observed_vfr"]).drop_duplicates("time_min")
        ax.scatter(
            actual.time_min,
            actual.observed_vfr,
            facecolor="white",
            edgecolor="black",
            label="Figure 4b",
            zorder=5,
        )
        ax.grid(axis="y", alpha=0.18)
        ax.spines[["right", "top"]].set_visible(False)
        ax.set_xlabel("Evaporation time (min)")
    axs[0].set_xlim(-5, 240)
    axs[0].set_ylim(0.51, 1.02)
    axs[0].set_ylabel("Volume fraction remaining")
    axs[1].set_xlim(1160, 1215)
    axs[1].set_xticks([1160, 1180, 1200])
    axs[0].legend(frameon=False, fontsize=8, loc="lower left")
    fig.savefig(out / "fractional_comparison.png", dpi=180)
    plt.close(fig)


if __name__ == "__main__":
    p = ArgumentParser()
    p.add_argument(
        "--out", type=Path, default=Path(__file__).resolve().parent / "results"
    )
    render(p.parse_args().out)
