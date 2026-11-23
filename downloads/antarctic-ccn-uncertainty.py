# %% [markdown]
# ## How uncertain is the bend in the Antarctic activation spectrum?
#
# Replay the measured dry-size and growth distributions for the same 373
# hours as Xu's counter. Each simulation uses the same hours. The film
# fraction is fitted only at 0.2%, 0.6%, 0.8% and 1.0% on the training dates.
# The [literature study](index.md) compares that prediction to a salt reference.
# %%
import matplotlib.pyplot as plt
from scripts.antarctic_ccn.plots import attribution, size_effects, spectrum

if "saved_data" in globals():
    report = globals()["saved_data"]
else:
    from scripts.antarctic_ccn.analysis import calculate

    report = calculate()

# %% [markdown]
# ### The measured and predicted rises
#
# Open circles show observed means. Colored lines show modeled means.
# The observation bars and model shading are pointwise 95% intervals
# from whole-date resampling and refitting. The lower right panel shows
# the 10th–90th percentile spread of the individual held-out hours.
# %%
spectrum(report)
plt.show()

# %% [markdown]
# ### Which dry diameters change the shape?
#
# Apply the film to one band at a time. Compare the two slopes on the same
# resampled hourly vectors. Negative changes in absolute bias mean better
# agreement. Dotted ranges use the retained instrument and growth settings.
# %%
size_effects(report)
plt.show()

# %% [markdown]
# ### Could the counter identify an organic film?
#
# Compare the count supplied by the film with the remaining salt deficit.
# The calibration ranges are spans across accepted settings. The synthetic
# panel uses Wilson intervals across independent records for each noise law.
# %%
attribution(report)
plt.show()
