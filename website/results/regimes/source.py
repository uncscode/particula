# %% [markdown]
# ## What survives 24 hours of coagulation?
#
# Start each process pair from the same published spectrum. The saved run
# includes number and dry volume at 1, 6 and 24 h, plus independently
# sampled GPU boxes. The [study report](index.md) gives the size
# distributions and charge balance.

# %%
import matplotlib.pyplot as plt
from scripts.mam4xx_coagulation.regime_plots import coefficients, plot, spectra

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

# %% [markdown]
# ## Which sizes contribute to the rankings?
#
# The upper panels show the published representative trimodal inputs. The
# middle panels compare the two-fine-mode sectional Brownian result with
# the native particle ensemble at 24 hours. Shading is one sample standard
# deviation among four refined collision histories, clipped at zero for
# display. In the third row, solid curves are the three-mode sectional
# spectra used for the ranking. Dashed curves and ribbons give the mean and
# pointwise 16th–84th percentiles of 16 collision histories using the same
# initial modes and kernel grid. The last row magnifies the percentile
# bounds relative to each sectional bin where at least ten particles per
# box are expected. Integrate bin density below 100 nm for ultrafine number.

# %%
spectra(report)
plt.show()

# %% [markdown]
# ## Which collision coefficients would change in MAM4xx?
#
# Solid lines are number-weighted enhancement factors on the evolving
# archived host states. Dashed lines weight the donor mode by its dry
# volume. The two weights give different answers even for the same pair.
# Coarse collection is the study extension to the archived fine-mode update.

# %%
coefficients(report)
plt.show()
