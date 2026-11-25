# %% [markdown]
# # Which Pallas dates require salt enrichment at 120 nm?
#
# Compare each day's best-fitting salt fractions at 120 and 50 nm. The
# [Pallas study](index.md) profiles the difference across alternative
# allocations that preserve the regional chemistry constraint. A range
# entirely above zero requires enrichment for that day under this model.
#
# %%
from pathlib import Path

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import pandas as pd
from particula import plotting


# %% [markdown]
# ## Compare each fit with its alternatives
#
# Each row describes one daily fit. Subtract the two best-fitting dry salt
# fractions, then plot the separately profiled minimum and maximum difference.
# The tolerance is one percentage point in activation RMSE. The vertical
# extent measures how differently salt can be allocated within that tolerance.
#
# %%
def plot_results(report):
    """Display the paired dry-volume difference and its feasible interval."""
    days = pd.DataFrame(report)
    dates = pd.to_datetime(days.date)
    with plt.style.context("particula.default"):
        fig, ax = plotting.subplots(size="wide")
        ax.axhline(0, color="#243E43", lw=1)
        ax.vlines(
            dates,
            100 * days.delta_lo,
            100 * days.delta_hi,
            color=plotting.COLORS["rust"],
            alpha=0.4,
            label="Feasible range",
        )
        ax.scatter(
            dates,
            100 * (days.salt120 - days.salt50),
            color=plotting.COLORS["rust"],
            s=22,
            label="Best fit",
        )
        ax.xaxis.set_major_locator(mdates.MonthLocator())
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%d %b"))
        ax.set(
            xlabel="Summit sampling date in 2022",
            ylabel="120 − 50 nm salt fraction (percentage points)",
        )
        plotting.legend(ax, placement="inside", loc="lower left", fontsize=10)
        plotting.fit(fig)
        plt.show()


def run_example():
    """Load only the quantities used to compare the two sizes."""
    path = Path("daily.csv")
    if not path.exists():
        path = (
            Path(__file__).resolve().parents[4]
            / "docs/data/pallas-activation/daily.csv"
        )
    fields = ["date", "salt120", "salt50", "delta_lo", "delta_hi"]
    report = pd.read_csv(path, usecols=fields).to_dict("records")
    plot_results(report)
    return report


if __name__ == "__main__":
    run_example()


# %% [markdown]
# *Figure S1. Each dot gives the fitted salt-volume fraction at 120 nm minus
# the fraction at 50 nm on one sampling date, in percentage points. Each bar
# spans the differences allowed by a one-point increase in activation RMSE.
# A bar entirely above zero requires more salt by dry volume at 120 nm.*
#
# All 27 best fits lie above zero. On 12 dates the full range also lies
# above zero. Those dates would be particularly informative for comparing
# the inferred enrichment with size-selected chemistry at the summit.
