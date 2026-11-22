"""Compare paired five-hour Manchester errors with equal weight per run."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
SOURCE = HERE.parent / "milestone2/scorecard.csv"
FIGURE = HERE.parents[3] / "docs/images/manchester-soa/campaign-performance.png"


def compare(source):
    """Pair both frozen calibrations on each run and integration setting."""
    scores = pd.read_csv(source)
    errors = ["mass", "count", "joint"]
    scores["joint"] = scores[["mass", "count"]].max(axis=1, skipna=False)
    pairs = scores.pivot(
        index=["step", "run"], columns="network", values=errors
    )
    if (
        pairs.isna().any().any()
        or not pairs.groupby(level="step").size().eq(15).all()
    ):
        raise ValueError(
            "Both calibrations must cover the same 15 runs at every step"
        )
    rows, summaries = [], []
    for step, group in pairs.groupby(level="step", sort=False):
        for metric in errors:
            data = group[metric].droplevel("step") * 100
            before, after = data["incumbent"], data["selected"]
            change = after - before
            for run in data.index:
                rows.append(
                    dict(
                        step=step,
                        run=run,
                        metric=metric,
                        archived_pct=before[run],
                        selected_pct=after[run],
                        change_pp=change[run],
                        mean_change_without_run_pp=change.drop(run).mean(),
                    )
                )
            for statistic in ["mean", "median", "max"]:
                a, b = before.agg(statistic), after.agg(statistic)
                summaries.append(
                    dict(
                        step=step,
                        metric=metric,
                        statistic=statistic,
                        archived_pct=a,
                        selected_pct=b,
                        change_pp=b - a,
                    )
                )
    return pd.DataFrame(rows), pd.DataFrame(summaries)


def plot(rows):
    """Show the direction and magnitude of every paired joint change."""
    data = rows[
        (rows.step == "0.9375s") & (rows.metric == "joint")
    ].sort_values("change_pp")
    fig, ax = plt.subplots(figsize=(6.4, 6.4), layout="constrained")
    colors = np.where(data.change_pp < 0, "#116778", "#ae481a")
    ax.barh(data.run.astype(str), data.change_pp, color=colors, height=0.68)
    fig.set_facecolor("#fff8dc")
    ax.set_facecolor("#fff8dc")
    ax.axvline(0, color="#424242", lw=0.7)
    ax.invert_yaxis()
    lower = (data.change_pp < 0).sum()
    ax.set(
        xlabel="40-channel minus 38-channel error (percentage points)",
        ylabel="Manchester experiment",
        title=f"The larger error fell in {lower} of {len(data)} runs",
    )
    ax.spines[["top", "right"]].set_visible(False)
    fig.savefig(FIGURE, dpi=165)
    plt.close(fig)


if __name__ == "__main__":
    paired, campaign = compare(SOURCE)
    paired.to_csv(HERE / "paired_changes.csv", index=False)
    campaign.to_csv(HERE / "campaign_scores.csv", index=False)
    plot(paired)
