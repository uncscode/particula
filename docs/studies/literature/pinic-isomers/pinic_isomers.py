# %% [markdown]
# ## Trace the two isomers in chamber aerosol
#
# Schaum et al. resolved cis-pinic acid and another molecule with the same
# elemental formula. The [report](index.md) compares their measured timing
# and tests a controlled condensation example at several seed loadings.

# %%
import matplotlib.pyplot as plt
from scripts.pinic_isomers.plots import loading, timing

if "saved_data" in globals():
    report = globals()["saved_data"]
else:
    from scripts.pinic_isomers.campaign import simulate

    report = simulate()

# %% [markdown]
# ## When does each channel reach half of its late signal?
#
# Each channel has its own baseline and scale. Points on the half-amplitude
# line show the first crossing of the centered three-point mean.

# %%
timing(report)
plt.show()

# %% [markdown]
# ## Transfer a formula-level surrogate to other seed loadings
#
# The dotted line marks the calibration loading. Positive differences mean
# the surrogate puts more organic material into the particles.

# %%
loading(report)
plt.show()
