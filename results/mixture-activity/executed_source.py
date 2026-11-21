# %% [markdown]
# ## Water activity of Mt. Hua ion mixtures
#
# The [study report](index.md) explains the experiment, reference observations,
# numerical controls and the limits of the comparison.

# %%
import matplotlib.pyplot as plt
from scripts.mt_hua_salts.activity_plots import plot

if "saved_data" in globals():
    report = globals()["saved_data"]
else:
    from scripts.mt_hua_salts.mixture_activity import simulate

    report = simulate()

# %% [markdown]
# ## Compare the observed or reference case with the model
#
# Replot the published numerical outputs using the physical units on each axis.

# %%
plot(report)
plt.show()
