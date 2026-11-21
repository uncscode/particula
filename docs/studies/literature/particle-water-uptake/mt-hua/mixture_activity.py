# %% [markdown]
# ## Solve equilibrium at the measured humidity
#
# Load 79 charge-balanced extracts and 15 pure-salt activity tables. Each
# uses the same dry volume and Kelvin correction at 90% RH. The [study
# report](index.md) documents the charge balance and missing sample.

# %%
import matplotlib.pyplot as plt
from scripts.mt_hua_salts.activity_plots import plot

if "saved_data" in globals():
    report = globals()["saved_data"]
else:
    from scripts.mt_hua_salts.mixture_activity import simulate

    report = simulate()

# %% [markdown]
# ## Which part of the bias comes from mixing?
#
# Thin lines connect predictions for one sample. The bold line is the group
# mean. The first step changes the pure-salt reference within ZSR. The
# second applies AIOMFAC to the full liquid mixture.

# %%
plot(report)
plt.show()
