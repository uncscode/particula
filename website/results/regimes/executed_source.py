# %% [markdown]
# ## What survives 24 hours of coagulation?
#
# Start each process pair from the same published spectrum. The saved run
# includes number and dry volume at 1, 6 and 24 h, plus independently
# sampled GPU boxes. The [study report](index.md) gives the size
# distributions and charge balance.

# %%
import matplotlib.pyplot as plt
from scripts.mam4xx_coagulation.regime_plots import plot

if "saved_data" in globals():
    report = globals()["saved_data"]
else:
    from scripts.mam4xx_coagulation.regimes import simulate

    report = simulate()

# %% [markdown]
# ## Which process changes ultrafine survival?
#
# Dots show the central change in sub-100 nm retention relative to the
# paired reference. Bars span material, atmospheric and numerical controls.
# Each panel has its own horizontal scale.

# %%
plot(report)
plt.show()
