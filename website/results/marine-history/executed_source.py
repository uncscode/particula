# %% [markdown]
# ## Can ship-track nitric acid predict the two marine events?
#
# The [study report](index.md) explains the experiment, reference observations,
# numerical controls and the limits of the comparison.

# %%
import matplotlib.pyplot as plt
from scripts.sea2cloud_events.replay_plots import plot

if "saved_data" in globals():
    report = globals()["saved_data"]
else:
    from scripts.sea2cloud_events.marine_history import simulate

    report = simulate()

# %% [markdown]
# ## Compare the observed or reference case with the model
#
# Replot the published numerical outputs using the physical units on each axis.

# %%
plot(report)
plt.show()
