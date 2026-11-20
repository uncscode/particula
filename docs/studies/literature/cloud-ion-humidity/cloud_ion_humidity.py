# %% [markdown]
# At the same ion production rate, the CERN CLOUD experiment recorded many more ions in humid air than in dry air. Can the published humidity dependence of ion recombination explain the difference? Reproduce three particle-resolved ensembles, then use two fitted explanations to design a discriminating measurement. The [study report](index.md) gives the observations, provenance and interpretation.
#
# The bundled `figure5d.csv` contains our digitization of [Franchin et al. (2015), Figure 5d](https://doi.org/10.5194/acp-15-7203-2015), used under [CC BY 3.0](https://creativecommons.org/licenses/by/3.0/). The bundled `two-inlet-design.csv` contains synthetic discrimination results using a noise reference from the [Mirme et al. counter archive](https://doi.org/10.5281/zenodo.17357061).
#
# ## Reproduce the three chamber conditions
#
# Each ion pair enters a well-mixed sampling volume. Opposite ions can recombine, and each ion can be lost at a wall. Relative humidity (RH) selects the measured recombination coefficient. The example requires a CUDA device and the complete downloadable bundle. The publication runner supplies verified numerical data for a CPU-only replot.
#
# Run the ensembles at 7% and 70% RH near 68 ion pairs per cubic centimetre per second, then repeat the humid case at 2.64. Compare their ten-minute averages with Figure 5d of Franchin et al. The printed uncertainty is the 95% Monte Carlo half-width of the simulated difference at fixed inputs.

# %%
import matplotlib.pyplot as plt
from scripts.cloud_ion_humidity.plots import (
    comparison,
    design_figure,
    recovery_figure,
)

if "saved_data" in globals():
    report = globals()["saved_data"]
else:
    from scripts.cloud_ion_humidity.study import simulate

    report = simulate()

print(
    f"Device: {report['device']}\n"
    f"Observed humidity increase: {report['observed_difference']:.0f} ions/cm³\n"
    f"Simulated increase: {report['simulated_difference']:.0f} "
    f"± {report['difference_95']:.0f} ions/cm³"
)

# %% [markdown]
# ## Humidity accounts for about half the measured rise in this reconstruction
#
# Compare the measurement and model at each humidity in the left panel. The right panel tests the humid low-source observation against the maximum possible well-mixed concentration at the weakest reported first-order loss. The panels use different concentration scales.

# %%
comparison(report)
plt.show()

# %% [markdown]
# Circles and bars are digitized observations and their published scatter. Diamonds show independent CUDA box means with 95% Monte Carlo intervals, some smaller than the symbols. The horizontal line uses zero recombination. Even the lower observed bar exceeds that ceiling by a factor of four.
#
# ## An independently verified zero separates two fitted explanations
#
# A constant readout offset and an additional physical ion source can reproduce the same ordinary beam-off trace if the loss rate is free. During a verified clearing step, physical ions are removed while the offset persists. The bands show predicted readings after a 120 s hold, across residual ions and detector lag. Symbols mark 30 s after release.

# %%
recovery_figure(report)
plt.show()

# %% [markdown]
# At 30 s, the offset case predicts 1,086–1,196 ions/cm³ and the extra-source case 82–286. These are conditional model ranges. A new chamber measurement would test them.
#
# ## A second inlet tests spatial explanations
#
# The blinded study compared a localized source with uniform wall loss, the same source with wall loss concentrated outside the beam, and the two uniform explanations above. For each gain and noise condition, the figure reads the smallest tested reduction of plume air at the second inlet that passed every pairwise comparison. Open symbols indicate that even the tested contrasts up to 100% failed the criterion.

# %%
design_figure(report)
plt.show()

# %% [markdown]
# With gains known to ±5% and reference noise, the second inlet must sample at most half as much plume air as the first among the tested designs. The noise came from a later CLOUD counter. Measure both humid channel gains and the exposure contrast before interpreting a spatial difference in a repeat experiment.
