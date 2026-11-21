# %% [markdown]
# ## Follow a parcel from twelve hours before detection
#
# Prescribe the published ship-track nitric acid series and a background
# collision sink from the pre-event size spectra. The [study
# report](index.md) explains how ship position limits the parcel
# interpretation.

# %%
import matplotlib.pyplot as plt
from scripts.sea2cloud_events.replay_plots import plot

if "saved_data" in globals():
    report = globals()["saved_data"]
else:
    from scripts.sea2cloud_events.marine_history import simulate

    report = simulate()

# %% [markdown]
# ## Do modeled particles cross the detection threshold?
#
# Shading marks the reported events. Black bars span visible observed counts
# in each half-hour bin. The central parcel adds locally formed particles to
# survivors of the initial population. The dotted line marks 500 particles
# per cubic centimetre.

# %%
plot(report)
plt.show()
