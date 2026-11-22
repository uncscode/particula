"""Summarize matched stochastic replays and plot perturbations against data."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

OUT = Path(__file__).resolve().parent / "results/perturbations"
METRICS = [
    "loss",
    "mass",
    "number",
    "spectrum_rmse",
    "mode_mae_nm",
    "surface",
    "water",
]


def summarize(scores):
    """Report empirical spread and paired changes in observable errors."""
    if scores.duplicated(["seed", "case", "draw"]).any():
        raise ValueError("Duplicate run identifiers in saved perturbations")
    summaries = []
    effects = []
    for seed, batch in scores.groupby("seed"):
        joint = batch[batch.case == "joint"].set_index("draw")[METRICS]
        center = batch[batch.case == "center"].iloc[0]
        frozen = batch[batch.case == "frozen"].iloc[0]
        for metric in METRICS:
            vals = joint[metric]
            summaries.append(
                dict(
                    seed=seed,
                    metric=metric,
                    n=len(vals),
                    frozen=frozen[metric],
                    center=center[metric],
                    p10=vals.quantile(0.1),
                    median=vals.median(),
                    p90=vals.quantile(0.9),
                    below_frozen=(vals < frozen[metric]).sum(),
                )
            )
        for group, pairs in batch.groupby("case"):
            if group in {"center", "frozen", "joint"}:
                continue
            fixed = pairs.set_index("draw")[METRICS]
            change = joint.loc[fixed.index] - fixed
            for metric in METRICS:
                effects.append(
                    dict(
                        group=group,
                        metric=metric,
                        n=len(change),
                        median_absolute=change[metric].abs().median(),
                        median_signed=change[metric].median(),
                    )
                )
    # Average within a draw before taking quantiles over physical parameters.
    coarse = scores[scores.seed.isin([311, 312])]
    means = (
        coarse[coarse.case.isin(["joint", "center", "frozen"])]
        .groupby(["case", "draw"])[METRICS]
        .mean()
    )
    seeds = coarse[coarse.case == "joint"].set_index(["seed", "draw"])
    seed_differences = (seeds.loc[311, METRICS] - seeds.loc[312, METRICS]).abs()
    combined = []
    for metric in METRICS:
        draws = means.loc["joint", metric]
        frozen = means.loc[("frozen", -1), metric]
        combined.append(
            dict(
                metric=metric,
                n=len(draws),
                frozen=frozen,
                center=means.loc[("center", -1), metric],
                p10=draws.quantile(0.1),
                median=draws.median(),
                p90=draws.quantile(0.9),
                below_frozen=(draws < frozen).sum(),
                median_seed_difference=seed_differences[metric].median(),
            )
        )
    pd.DataFrame(combined).to_csv(OUT / "paired_seed_summary.csv", index=False)

    blocks = ["mass", "number", "size", "mode", "surface", "water"]
    block_scores = coarse.groupby(["case", "draw"])[blocks].mean()
    draw_scores = block_scores.loc["joint"]
    comparisons = pd.DataFrame(index=draw_scores.index)
    for case in ["frozen", "center"]:
        improves = draw_scores.lt(block_scores.loc[(case, -1)])
        comparisons[f"blocks_better_than_{case}"] = improves.sum(axis=1)
    runs = pd.read_csv(OUT / "runs.csv.gz")
    for metric in ["mass", "number"]:
        errors = runs[runs.case == "joint"].pivot(
            index="draw", columns="run", values=metric
        )
        frozen_runs = runs[runs.case == "frozen"].set_index("run")[metric]
        comparisons[f"runs_better_than_frozen_{metric}_seed311"] = errors.lt(
            frozen_runs, axis=1
        ).sum(axis=1)
    comparisons.to_csv(OUT / "simultaneous_improvements.csv")

    pd.DataFrame(summaries).to_csv(OUT / "summary.csv", index=False)
    pd.DataFrame(effects).to_csv(OUT / "group_effects.csv", index=False)


def panel(ax, table, x, observed, predicted, factor=1, band=None):
    """Overlay archived observations, the center and the joint ensemble."""
    joint = table[table.case == "joint"].groupby(x)[predicted]
    span = joint.quantile([0.1, 0.9]).unstack() / factor
    center = table[table.case == "center"].sort_values(x)
    ax.fill_between(
        span.index,
        span[0.1],
        span[0.9],
        color="#9ccdbd",
        label="Joint draws, 10th–90th percentile",
    )
    ax.plot(
        center[x],
        center[predicted] / factor,
        color="#11624f",
        linewidth=2.3,
        label="Joint refit",
    )
    ax.errorbar(
        center[x],
        center[observed] / factor,
        yerr=center[band] / factor if band else None,
        fmt="o",
        ms=4,
        color="#2c3341",
        capsize=2,
        label="Manchester data",
    )
    ax.grid(axis="y", alpha=0.22)


def figure():
    """Compare the five-hour size spectrum and surface trajectory."""
    psd = pd.read_csv(OUT / "psd.csv.gz").query("time_s == 18000").copy()
    psd["diameter"] = np.sqrt(psd.low_nm * psd.high_nm)
    surface = pd.read_csv(OUT / "surface.csv.gz")
    surface["hour"] = surface.time_s / 3600
    joint_surface = surface[surface.case == "joint"].groupby("time_s")
    envelope = joint_surface.prediction.quantile([0.1, 0.5, 0.9]).unstack()
    envelope.columns = ["p10", "median", "p90"]
    envelope["observed"] = joint_surface.surface_nm2_cm3.first()
    envelope.to_csv(OUT / "surface_envelope_nm2_cm3.csv")
    # 1 nm²/cm³ = 1e-6 µm²/cm³.
    fig, axes = plt.subplots(1, 2, figsize=(9.3, 4.2))
    panel(
        axes[0],
        psd,
        "diameter",
        "dndlog10d_cm3",
        "simulated",
        factor=1000,
        band="reading_band_cm3",
    )
    axes[0].set(
        xscale="log",
        xlabel="Dry diameter (nm)",
        ylabel="Number density (10³ cm⁻³ per decade)",
        title="Run 24 · number spectrum at 5 h",
    )
    axes[0].set_ylim(bottom=0)
    axes[0].set_xticks(
        [30, 60, 100, 200, 400], labels=["30", "60", "100", "200", "400"]
    )
    panel(axes[1], surface, "hour", "surface_nm2_cm3", "prediction", factor=1e6)
    axes[1].set(
        xlabel="Time since lights on (h)",
        ylabel="Particle surface area (µm² cm⁻³)",
        title="Run 18 · measured surface area",
    )
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=3, frameon=False)
    fig.tight_layout(rect=(0, 0.10, 1, 1), w_pad=2)
    fig.savefig(OUT / "perturbation-comparison.png", dpi=180)
    plt.close(fig)


if __name__ == "__main__":
    summarize(pd.read_csv(OUT / "scores.csv"))
    figure()
