# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.17.3
#   kernelspec:
#     display_name: particula_dev312
#     language: python
#     name: python3
# ---

# %% [markdown]
# # One population, different size spectra
#
# Why can two plots of the same aerosol population have different heights and
# shapes? The ordinate may describe number per unit diameter, number in a bin,
# or number per logarithmic interval. Recognizing that distinction is essential
# when comparing an instrument spectrum with a model's initial conditions.
#
# Here we construct one- and two-component lognormal populations, compare a
# diameter density with bin-integrated numbers, and convert the density into
# Particula's native radius coordinate. The calculation uses CPU NumPy arrays;
# familiarity with [native particle construction](particle_data.py) helps with
# the last step. The populations are illustrative, not fitted measurements.

# %%
# In Colab uncomment the following command to install particula:
# Install from the repository root: `python -m pip install -e ".[extra]"`
import matplotlib.pyplot as plt
import numpy as np
import particula as par
from particula import plotting

plt.style.use("particula.default")

# %% [markdown]
# ## A density needs a coordinate
#
# For diameter $D>0$, a lognormal number density has the form
#
# $$
# \frac{dN}{dD}=\frac{N}{D\ln\sigma_g\sqrt{2\pi}}
# \exp\left[-\frac{\ln^2(D/D_g)}{2\ln^2\sigma_g}\right].
# $$
#
# Here $D_g$ is the geometric mean diameter and $\sigma_g>1$ the geometric
# standard deviation. Particula's argument is named `mode`, but the helper passes
# it to SciPy as `scale`: it is $D_g$, not the peak of $dN/dD$. The latter occurs
# at $D_g\exp[-\ln^2\sigma_g]$. See [SciPy's lognormal parameterization](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.lognorm.html).
#
# The helper normalizes each component numerically over the supplied grid to its
# requested total $N$. Thus a correct total alone does not prove that the grid
# captures the tails of the intended untruncated distribution.
#
# For the first plots, $D$ is in micrometres and the supplied totals are numbers,
# so the PDF ordinate is number per micrometre. Dividing by $N$ would give a
# probability density; the plotted quantity itself is population-scaled.

# %% [markdown]
# ### Choose the size range
#
# Logarithmic spacing resolves several size decades with a manageable grid.
# The domain below is 0.001–10 µm in diameter. Logarithmic spacing of the points
# does not itself turn a density per diameter into a density per log diameter.

# %%
# Define the x_values as a range of particle diameters
x_values = np.logspace(-3, 1, 2000)  # From 0.001 to 10 microns

# %% [markdown]
# ### Start with one component
#
# The first component has a geometric mean diameter of 0.02 µm, geometric
# standard deviation 1.4, and total number 1,000. Keeping the total separate
# from size and spread lets us distinguish abundance from distribution shape.

# %%
# %%
# Single mode distribution
single_mode_gsd = np.array([1.4])
single_mode = np.array([0.02])
single_mode_nparticles = np.array([1e3])

single_mode_distribution = par.particles.get_lognormal_pdf_distribution(
    x_values, single_mode, single_mode_gsd, single_mode_nparticles
)

# %% [markdown]
# ### Add a second component
#
# The second population retains the first component and adds a broader one
# centered at 1 µm, also with total number 1,000. Equal component totals do not
# imply equal peak heights in a density per unit diameter.

# %%
# %%
# Multi-mode distribution
multi_mode_gsd = np.array([1.4, 1.8])
multi_mode = np.array([0.02, 1.0])
multi_mode_nparticles = np.array([1e3, 1e3])

multi_mode_distribution = par.particles.get_lognormal_pdf_distribution(
    x_values, multi_mode, multi_mode_gsd, multi_mode_nparticles
)

# %% [markdown]
# ### Compare densities per diameter
#
# The next plot uses a logarithmic horizontal axis but retains $dN/dD$ on the
# vertical axis. Look for the narrow small-particle peak and the broader
# large-particle component. Their height difference reflects width and coordinate
# choice as well as abundance. Visual area on this log-axis figure is not the
# integral over linear diameter.

# %%
# %%
fig, ax = plotting.subplots(size="wide")
plt.plot(x_values, single_mode_distribution, label="Single Mode", linewidth=4)
plt.plot(x_values, multi_mode_distribution, label="Multi Mode")
plt.title("Lognormal Particle Size Distribution")
plt.xlabel("Particle Diameter (μm)")
plt.ylabel("Number per micrometre of diameter")
plt.xscale("log")
plotting.legend(plt.gca())
plt.grid(True)
plotting.fit(fig)
plt.show()

# %% [markdown]
# ### Recover the total with the matching integral
#
# The printed integrals use the diameter coordinates, rather than summing the
# PDF values. Compare them with the supplied totals: 1,000 for the single
# component and 2,000 for the mixture. This checks the normalization convention;
# the helper's explicit normalization means it is not an independent test of
# physical realism or tail coverage.

# %%
# %%
single_mode_total_concentration = np.trapezoid(
    single_mode_distribution, x_values
)
multi_mode_total_concentration = np.trapezoid(multi_mode_distribution, x_values)

print(f"Total Concentration for Single Mode: {single_mode_total_concentration}")
print(f"Total Concentration for Multi Mode: {multi_mode_total_concentration}")

# %% [markdown]
# ## Convert density to number in each bin
#
# A bin-integrated distribution stores a contribution $N_i$ for each interval;
# its total is $\sum_i N_i$. Particula calls this representation a PMF. Here
# "mass" in probability mass function does not mean particle material mass.
#
# The helper weights the PDF by bin widths and rescales the sum to the requested
# total. On a logarithmic grid, large-diameter bins are wider in linear diameter,
# so weighting changes the relative heights. Both plots still describe the same
# chosen population, within the numerical discretization.

# %%
single_pmf_distribution = par.particles.get_lognormal_pmf_distribution(
    x_values, single_mode, single_mode_gsd, single_mode_nparticles
)
multi_pmf_distribution = par.particles.get_lognormal_pmf_distribution(
    x_values, multi_mode, multi_mode_gsd, multi_mode_nparticles
)

# %% [markdown]
# ### Read the bin-integrated spectrum
#
# Compare this plot with the PDF above. Bin width is now included, so peak heights
# cannot be compared directly across the two ordinates. Refining the grid changes
# number per bin even when the underlying continuous population is unchanged.

# %%
fig, ax = plotting.subplots(size="wide")
plt.plot(x_values, single_pmf_distribution, label="Single Mode", linewidth=4)
plt.plot(x_values, multi_pmf_distribution, label="Multi Mode")
plt.title("Lognormal PMF Particle Size Distribution")
plt.xlabel("Particle Diameter (μm)")
plt.ylabel("Number of Particles")
plt.xscale("log")
plotting.legend(plt.gca())
plt.grid(True)
plotting.fit(fig)
plt.show()

# %% [markdown]
# ### Recover the total with a sum
#
# The PMF is already integrated over bins, so sum its entries without multiplying
# by bin widths again. The printed values can be compared with the earlier PDF
# integrals. Agreement checks consistent bookkeeping between representations.

# %%
# %%
single_mode_total_concentration = single_pmf_distribution.sum()
multi_mode_total_concentration = multi_pmf_distribution.sum()

print(f"Total Concentration for Single Mode: {single_mode_total_concentration}")
print(f"Total Concentration for Multi Mode: {multi_mode_total_concentration}")

# %% [markdown]
# ## Carry the measure into the model
#
# The useful distinction is the integration rule: integrate a density against
# its coordinate, or sum bin contributions. A change in plot shape can therefore
# come entirely from representation. Before attributing it to aerosol physics,
# check the coordinate, units, bin widths, and normalization.

# %% [markdown]
# ### Native radius-PDF storage
#
# Native `continuous_pdf` storage uses radius $r$ in metres and $dN/dr$.
# With $D$ in micrometres, $D=2\times10^6 r$, hence
# $dN/dr=(dN/dD)\,2\times10^6$. Changing the coordinate requires changing the
# ordinate too; rescaling the horizontal axis alone would change the integral.
#
# For this final model construction, interpret the supplied component totals as
# number concentrations in m⁻³. The same numerical spectrum then has units
# m⁻³ µm⁻¹ before conversion and m⁻⁴ afterward. This concentration convention is
# an input assumption, not a volume measurement. The radius grid accompanies
# the stored PDF.

# %%
radius_grid_m = x_values * 1e-6 / 2
radius_pdf = multi_mode_distribution * 2e6
native_pdf = par.particles.create_particle_data_from_radii(
    radii=radius_grid_m,
    concentration=radius_pdf,
    distribution_type="continuous_pdf",
    density=1000.0,
)
print("Native radius-PDF number density [m^-3]:", native_pdf.number_density)

# %% [markdown]
# ## What this lets you compare
#
# The final printed native number density should retain the mixture's supplied
# 2,000 m⁻³ under the stated concentration convention. The key result is that
# PDF, bin-integrated number, and native radius PDF require different ordinates
# while preserving the represented population.
#
# To adapt this to an instrument spectrum, first identify whether its ordinate
# is $dN/dD$, $dN/d\log_{10}D$, or number in a bin. For example,
# $dN/d\log_{10}D=D\ln(10)\,dN/dD$. Then compare integrated concentration before
# and after conversion, and test sensitivity to the grid and measurement range.
# These would be additional checks, not results established by this example.
# Continue to [population storage](distribution_tutorial.py) or use the same
# representation distinction in [coagulation comparisons](../../coagulation-populations/coagulation_4_compared.py).
