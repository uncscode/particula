# %%
"""Plot the measured count histories and the campaign error trade-off."""

import matplotlib.pyplot as plt
import pandas as pd
from engine import HERE
from wall_fit import OUT, RUNS, WallObjective


def main():
    """Draw the count comparison and the independent calibration panel."""
    runs = WallObjective()
    trajectories = pd.read_csv(OUT / "trajectories.csv")
    curves = [
        ("constant_brownian", "Constant loss", "#b06a42", "--"),
        ("eddy_brownian", "Size-dependent loss", "#146d78", "-"),
    ]
    fig, axes = plt.subplots(2, 3, figsize=(11, 5.8), layout="constrained")
    for i, (r, ax) in enumerate(
        zip(RUNS, [axes[0, 0], axes[0, 1], axes[1, 0], axes[1, 1]], strict=True)
    ):
        _, _, observed, t = runs.counts[i]
        ax.scatter(
            t / 3600, observed, s=8, color="0.33", label="Measured count"
        )
        for name, label, color, ls in curves:
            q = trajectories[
                (trajectories.model == name)
                & (trajectories.run == r)
                & (trajectories.objective == "joint")
            ]
            ax.plot(
                q.time_s / 3600,
                q.number_fraction,
                color=color,
                ls=ls,
                lw=1.8,
                label=label,
            )
        ax.set(
            title=f"α-pinene · run {r}",
            xlabel="Hours after lights on",
            ylabel="Fraction of initial count",
            ylim=(0, 1.18),
        )
    ref = pd.read_csv(HERE / "wall_data/wall_rates.csv")
    ax = axes[0, -1]
    ax.loglog(
        ref.diameter_nm,
        ref["neutral_fit_s-1"] * 3600,
        color="#68618a",
        label="Neutral fit",
    )
    ax.scatter(
        ref.diameter_nm,
        ref["apparent_loss_s-1"] * 3600,
        s=14,
        color="0.33",
        label="Digitized means",
    )
    ax.set(
        title="Dark sulfate seed",
        xlabel="Diameter (nm)",
        ylabel="Apparent loss (h⁻¹)",
    )
    ax.legend(fontsize=8, frameon=False)
    ax = axes[1, -1]
    scores = pd.read_csv(OUT / "campaign_scores.csv")
    for model, _label, color, ls in curves:
        grouped = (
            scores[scores.model == model]
            .groupby("objective")[["mass_nrmse", "count_nrmse"]]
            .max()
        )
        ax.plot(
            100 * grouped.mass_nrmse,
            100 * grouped.count_nrmse,
            color=color,
            ls=ls,
        )
        for obj, marker in [("joint", "o"), ("mass", "x")]:
            row = grouped.loc[obj]
            ax.scatter(
                row.mass_nrmse * 100,
                row.count_nrmse * 100,
                c=color,
                marker=marker,
                s=40,
            )
    ax.set(
        title="15 runs · worst errors",
        xlabel="SOA error (% of peak)",
        ylabel="Count error (% of initial)",
    )
    for marker, label in [("o", "Joint fit"), ("x", "Mass-only fit")]:
        ax.scatter([], [], c="0.3", marker=marker, label=label)
    ax.legend(fontsize=8, frameon=False)
    for ax in axes.flat:
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(axis="y", alpha=0.12)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(
        handles, labels, loc="outside lower center", ncol=3, frameon=False
    )
    fig.savefig(OUT / "wall-loss.png", dpi=185)
    plt.close(fig)


if __name__ == "__main__":
    main()
