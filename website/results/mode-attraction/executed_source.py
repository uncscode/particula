# %% [markdown]
# ## Does modal averaging preserve the attraction effect?
#
# Start with polluted air, then add coarse collection to the marine box.
# Match mode number and dry volume across representations. The study
# compares the archived-width closure with size sections and stochastic
# particle trajectories at 1, 6 and 24 hours. See the [study report](index.md)
# for the kernel domain and the paired setup.

# %%
import matplotlib.pyplot as plt
from scripts.mam4xx_coagulation.attraction_plots import effects, particles

if "saved_data" in globals():
    report = globals()["saved_data"]
else:
    from scripts.mam4xx_coagulation.attraction_campaign import run

    report = run()

# %% [markdown]
# ## How much does attraction change the final number below 100 nm?
#
# Each marker compares two otherwise matched 24-hour runs. Squares show
# modal integration, circles show sectional integration and triangles show
# the mean paired Monte Carlo change with its 95% interval. MAM and field
# refer to the initial geometric widths. Both preserve the field moments.

# %%
effects(report)
plt.show()

# %% [markdown]
# ## Which sizes account for the change?
#
# The top row shows the three-mode MAM-width input. Each
# solid trace gives the 24-hour ensemble mean. Shading spans the empirical
# 16th to 84th percentiles of the 16 refined independent trajectories in
# each diameter bin. The lower row magnifies this spread relative to the bin
# mean where at least ten particles per box are expected. The vertical line
# marks 100 nm. Independent histories at fixed inputs measure collision
# sampling variation. Field variability requires observations across air masses.

# %%
particles(report)
plt.show()
