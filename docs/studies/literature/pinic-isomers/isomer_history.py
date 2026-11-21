# %% [markdown]
# ## Fit the chamber signals before the late rise
#
# Read total aerosol mass from the SMPS and fit separate response factors
# for the two IMS channels. The final quarter of both signals is reserved
# for prediction. The [study report](index.md) describes the graph clock
# alignment.

# %%
import matplotlib.pyplot as plt
from scripts.pinic_isomers.history_plots import plot

if "saved_data" in globals():
    report = globals()["saved_data"]
else:
    from scripts.pinic_isomers.isomer_history import simulate

    report = simulate()

# %% [markdown]
# ## Does the fit predict the held-out interval?
#
# Shading marks the final quarter. The delayed production model follows the
# accelerating cis-pinic signal. Both curves use the same measured loading
# shown in the bottom panel.

# %%
plot(report)
plt.show()
