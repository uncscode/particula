# %% [markdown]
# # How much less energy does the modeled summer cloud reflect in a day?
#
# The [limonene study](index.md) reports differences in upward shortwave flux
# above a modeled cloud and its local surface. A flux is a rate. Multiply its
# all-hour mean by a full day's duration to obtain the change in energy per
# square metre on an average sampled day. This example uses the same four
# background and updraft combinations as the study.
#
# ## Combine three sampled weeks
#
# Every month contributes seven complete days. The seasonal mean therefore
# weights its three monthly fluxes equally, including hours of darkness. A
# negative result means the sun-aged case reflected less energy than its
# equally old dark control.

# %%
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from particula import plotting

SEASONS = ["Winter", "Spring", "Summer", "Autumn"]


def daily_energy(monthly):
    """Convert sampled monthly mean W/m² to seasonal mean-day kJ/m²."""
    month_to_season = dict(
        zip(
            range(1, 13),
            ["Winter"] * 2
            + ["Spring"] * 3
            + ["Summer"] * 3
            + ["Autumn"] * 3
            + ["Winter"],
        )
    )
    data = monthly.assign(season=monthly.month.map(month_to_season))
    mean_flux = data.groupby(
        ["season", "profile", "w"]
    ).delta_up_site_wm2.mean()
    return (mean_flux * (24 * 3600 / 1000)).rename("delta_kj_m2").reset_index()


# %% [markdown]
# ## Compare the two backgrounds at each updraft
#
# Each panel holds the updraft fixed. The bars compare the change in energy
# over an average sampled day, with the same cloud illumination applied to both
# aging treatments. Select a different season in the output to examine how long
# daylight and precursor exposure affect the integrated change.


# %%
def plot_results(report):
    """Draw the energy difference for each modeled cloud regime."""
    daily = pd.DataFrame(report)
    with plt.style.context("particula.default"):
        fig, axs = plt.subplots(
            1, 2, figsize=(8, 3.7), sharey=True, layout="constrained"
        )
        for ax, speed in zip(axs, (0.25, 2.0)):
            for offset, profile, label, color, hatch in [
                (-0.19, "clean", "Clean", "teal", ""),
                (0.19, "rich", "Accumulation-rich", "rust", "//"),
            ]:
                subset = daily[(daily.profile == profile) & (daily.w == speed)]
                heights = subset.set_index("season").loc[SEASONS, "delta_kj_m2"]
                ax.bar(
                    np.arange(4) + offset,
                    heights,
                    0.36,
                    label=label,
                    color=plotting.COLORS[color],
                    hatch=hatch,
                    edgecolor="black",
                    linewidth=0.6,
                )
            ax.axhline(0, lw=0.7, color="#243e43")
            ax.set_xticks(np.arange(4), SEASONS, rotation=25)
            ax.set_title(f"{speed:g} m/s ascent")
        axs[0].set_ylabel("Change in reflected energy\n(kJ/m² per sampled day)")
        axs[0].legend(loc="lower left", frameon=False, fontsize=8)
        plt.show()


def run_example():
    """Read the published monthly plot data and compare seasonal energies."""
    data = Path("monthly.csv")
    if not data.exists():
        data = (
            Path(__file__).resolve().parents[4]
            / "website/public/study-data/bai-limonene/monthly.csv"
        )
    result = daily_energy(pd.read_csv(data)).to_dict("records")
    plot_results(result)
    return result


if __name__ == "__main__":
    run_example()


# %% [markdown]
# *Figure 1. Each bar is the sun-aged minus dark-aged difference above the
# modeled cloud and its local surface. Summer in the clean, slow-ascent case
# gives about 125 kJ/m² less reflected energy per sampled day. The
# accumulation-rich, slowly rising cloud reflects about 3 kJ/m² less.*
#
# A particular July day can differ from this seasonal mean. Integrating its
# hourly fluxes would preserve the observed sequence of bright and dim
# periods. Estimating energy over a region would also require cloud coverage,
# local emissions, and the occurrence of each modeled regime.
