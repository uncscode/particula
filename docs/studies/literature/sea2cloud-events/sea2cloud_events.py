# %% [markdown]
# ## Compare two Sea2Cloud events with local particle production
#
# The [study](index.md) uses a finite exposure at the reported upper
# nitric-acid calibration limit. Start with an empty 1–10 nm size window.

# %%
import matplotlib.pyplot as plt
from scripts.sea2cloud_events.plots import sensitivity, windows

if "saved_data" in globals():
    report = globals()["saved_data"]
else:
    from scripts.sea2cloud_events.study import simulate

    report = simulate()

# %% [markdown]
# ## Event peaks and local production
#
# The points are Figure 5a readings with digitization bounds. Each bar uses
# the full detection window as the available formation time.

# %%
windows(report)
plt.show()

# %% [markdown]
# ## Where do newly formed particles go?
#
# The second panel shows the fractional change in number remaining for a
# fractional perturbation in background concentration, at fixed growth.

# %%
sensitivity(report)
plt.show()
