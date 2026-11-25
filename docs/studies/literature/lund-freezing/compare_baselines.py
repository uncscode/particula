# %% [markdown]
# # Does the same fresh dust increase freezing against both controls?
#
# Use the six-hour results from the [Lund cloud study](index.md) to compare
# three prescribed clouds. Changing the incoming aerosol at an open top
# measures the effect of fresh activity. Opening a closed top also changes
# how much water and resident aerosol the cloud retains.

# %%
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from particula import plotting

# %% [markdown]
# ## Compare changes in the same units
#
# Subtract each control's event concentration from the fresh-inflow value.
# Plot the 95% interval calculated from the eight paired realizations. These
# intervals describe variability among particle draws at fixed cloud settings.


# %%
def plot_results(report):
    """Show the change in six-hour event concentration against each control."""
    data = pd.DataFrame(report)
    with plt.style.context("particula.default"):
        fig, ax = plotting.subplots(size="wide")
        for key, label, offset, color, marker in [
            ("open_survivor", "Previously exposed inflow", -0.12, "teal", "o"),
            ("closed", "Closed cloud top", 0.12, "rust", "s"),
        ]:
            prefix = "fresh_minus_" + key
            change = data.fresh - data[key]
            np.testing.assert_allclose(change, data[prefix + "_delta"])
            ax.errorbar(
                change,
                np.arange(len(data)) + offset,
                xerr=[
                    change - data[prefix + "_lo"],
                    data[prefix + "_hi"] - change,
                ],
                fmt=marker,
                capsize=3,
                color=plotting.COLORS[color],
                label=label,
            )
        ax.axvline(0, color="#243e43", lw=1)
        ax.set(
            yticks=np.arange(len(data)),
            yticklabels=data.cloud,
            xlabel="Fresh inflow minus control (primary freezing events per litre)",
        )
        ax.invert_yaxis()
        plotting.legend(ax, placement="right", fontsize=9)
        plotting.fit(fig)
        plt.show()


def run_example():
    """Read the six-hour paired comparisons for the three prescribed clouds."""
    path = Path("entrainment-baselines.csv")
    if not path.exists():
        path = Path(__file__).resolve().parents[3] / "data/lund-freezing" / path
    report = pd.read_csv(path).to_dict("records")
    plot_results(report)
    return report


if __name__ == "__main__":
    run_example()


# %% [markdown]
# *Figure 1. Positive values indicate that fresh dust produced more primary
# freezing events than the named control. All three clouds gain ice formation
# when fresh dust replaces previously exposed inflow. Relative to a closed
# top, the thin ISDAC layer loses freezing.*
#
# Try the subtraction with the closed and open columns interchanged. It
# answers a different question about opening the cloud even when the entering
# material has already experienced freezing conditions.
