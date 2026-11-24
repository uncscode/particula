"""Regenerate the wall-loss recovery figure from final transfer and classification tables."""

from pathlib import Path
import sys
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from particula import plotting

plt.style.use("particula.default")


def plot(root, output):
    avg = pd.read_csv(root / "classification-summary.csv")
    fig, axes = plt.subplots(1, 2, figsize=(11.8, 4.8), layout="constrained")
    for kind, col, mark, label in [
        (
            "primary",
            plotting.COLORS["rust"],
            "o",
            "Stage offset + fluctuations",
        ),
        ("centered", plotting.COLORS["teal"], "s", "Fluctuations alone"),
        (
            "smooth",
            plotting.COLORS["indigo"],
            "^",
            "Stage drift + fluctuations",
        ),
    ]:
        group = (
            avg[avg.noise == kind]
            .groupby("precision")
            .correct_wall.mean()
            .reindex([1, 0.1, 0.05, 0.025, 0])
        )
        axes[0].plot(
            range(5),
            100 * group.values,
            color=col,
            marker=mark,
            label=label,
            lw=1.8,
        )
    axes[0].set(
        xticks=range(5),
        xticklabels=["None", "±10%", "±5%", "±2.5%", "Exact"],
        ylim=(40, 104),
        ylabel="Correct low/high wall class (%)",
        xlabel="Synthetic flow constraint",
        title="Recovery from simulated dark spectra",
    )
    series = pd.read_csv(root / "plot-series.csv")
    for mode, kind, clr, label in [
        ("absolute", "upper", plotting.COLORS["teal"], "Larger channels"),
        ("absolute", "joint", plotting.COLORS["rust"], "Both size groups"),
    ]:
        use = series[series.channels == kind]
        if kind == "upper":
            axes[1].plot(
                use.archive_h + 6.5,
                use.observed,
                ".",
                color=plt.rcParams["text.color"],
                label="Measured",
            )
        axes[1].plot(
            use.archive_h + 6.5, use.predicted, color=clr, lw=2, label=label
        )
    axes[1].axhline(5000, color="#888", ls="--", lw=1)
    axes[1].set(
        ylabel="Particles above 9.7 nm (cm⁻³)",
        xlabel="Time from run start (hours)",
        title="Dark fits during the illuminated rise",
    )
    for ax in axes:
        ax.legend(loc="upper left", bbox_to_anchor=(0, -0.20))
    fig.savefig(output)
    plt.close(fig)


if __name__ == "__main__":
    plot(Path(__file__).resolve().parent, sys.argv[1])
