# %% [markdown]
# ## Reconstruct the 100 nm particles measured at Mt. Hua
#
# Wang and colleagues humidified particles formed from extracted inorganic material.
# Use the 80 matched filter samples in the accompanying data file to solve for the
# diameter where evaporation and condensation balance at 90% relative humidity.
# The CUDA calculation evaluates two salt sets and optimizes the ion pairings.
# Read the [study report](index.md) for the reconstruction and measurement limits.

# %%
import matplotlib.pyplot as plt
from scripts.mt_hua_salts.plots import comparison, residuals

if "saved_data" in globals():
    report = globals()["saved_data"]
else:
    from pathlib import Path

    from scripts.mt_hua_salts import study

    root = Path(study.__file__).resolve().parents[2]
    report = study.simulate(root / "docs/data/mt-hua-salts/matched.csv")

# %% [markdown]
# ## Which measured particles can the salt mixtures reproduce?
#
# Each green dot is a proportional salt allocation. Its vertical line spans all
# neutral pairings of the balanced ion inventory at fixed pure-salt properties.
# Orange crosses show the reduced reconstruction using ammonium and sodium.
# Points above the diagonal grew more in the model than in the instrument.

# %%
comparison(report)
plt.show()

# %% [markdown]
# ## The remaining winter discrepancy depends on the reconstruction
#
# Compare the authors' expanded-salt biases (diamonds) with our grouping of the
# matched observations (circles). Lines show the range of mean bias allowed
# when each sample can choose its own most or least hygroscopic pairing.
# Growth factor is wet diameter divided by dry diameter and has no units.

# %%
residuals(report)
plt.show()

# %% [markdown]
# The neutral-salt allocation admits only measured counterions. Further comparison
# needs the dry volume assigned to unmatched ions and the authors' pure-component
# growth properties. Those inputs could distinguish the influence of missing
# counterions from the choice of activity model.
