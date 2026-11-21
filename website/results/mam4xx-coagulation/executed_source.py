# %% [markdown]
# ## Compare two descriptions of the same coagulating aerosol
#
# Fast et al. ported a four-mode aerosol module to GPUs. Here two dry modes
# collide for one hour in a closed box. The source bundle includes the frozen
# output of their one-subarea coagulation routine. Particula tracks individual
# mergers on CUDA. The [report](index.md) describes the independently matched
# number and dry-volume moments and the assumptions about mode widths.

# %%
import matplotlib.pyplot as plt
from scripts.mam4xx_coagulation.plots import curves, spectra

if "saved_data" in globals():
    report = globals()["saved_data"]
else:
    from scripts.mam4xx_coagulation.campaign import simulate
    from scripts.mam4xx_coagulation.compare import attach_modal
    report = attach_modal(simulate())

# %% [markdown]
# ## Follow particle loss as the modes grow
#
# Each panel names the starting accumulation-mode diameter and the
# Aitken-to-accumulation number ratio. Shading spans a 95% Student interval
# for the mean of the 12 independent high-resolution particle simulations.

# %%
curves(report)
plt.show()

# %% [markdown]
# ## Compare the shapes retained after one hour
#
# At equal starting mode counts, compare both final diameter distributions
# on identical logarithmic bins. The stored totals include underflow and
# overflow. The plot shows the interior bins where most particles remain.

# %%
spectra(report)
plt.show()

# %% [markdown]
# Holding dry material and aerosol number fixed isolates predictions of the
# two coagulation implementations. Both the collision-rate treatment and the
# imposed modal shape can contribute to the observed difference.
