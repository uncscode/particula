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
# # Follow water–organic mixing from composition to water activity
#
# Can one water activity correspond to different liquid compositions? The Binary
# Activity Thermodynamics (BAT) model provides a reduced description of nonideal
# water–organic mixing. We first inspect composition-dependent activities, then
# ask for phase properties at prescribed water activities. This connects a
# composition calculation to the humidity-controlled inputs used in
# [organic partitioning](equilibria_part1.py).
#
# The model and its calibration are described by
# [Gorkowski, Preston, and Zuend (2019)](https://doi.org/10.5194/acp-19-13383-2019).
# These calculations illustrate its response to idealized molecular descriptors;
# they are not measurements of a named organic compound or a simulation of
# phase-separation kinetics.

# %%
# In Colab uncomment the following command to install particula:
# Install from the repository root: `python -m pip install -e ".[extra]"`
import matplotlib.pyplot as plt  # For plotting graphs
import numpy as np  # For numerical operations

# Specific functions from the particula package for activity calculations
from particula.activity import (
    activity_coefficients,
    phase_separation,
    water_activity,
)
from particula.particles.properties.organic_density_module import (
    get_organic_density_estimate,
)

# %% [markdown]
# ## Describe one generic organic component
#
# The organic has molar mass 100 g/mol and O:C=0.225. The water-to-organic
# molar-mass ratio is dimensionless; the density helper takes g/mol and returns
# kg/m³. Its default elemental assumptions are part of this illustrative density
# estimate, not independently measured composition.
#
# Sweep organic mole fraction from 0.001 to 1. The returned arrays contain water
# and organic activities, mass fractions, and activity coefficients. Keep those
# quantities distinct: $a_i=\gamma_i x_i$, so an activity coefficient above one
# is not by itself a phase-separation criterion.

# %%
# Define a range of organic mole fractions for the calculation
organic_mole_fraction = np.linspace(0.001, 1, 1000)

# Define other necessary parameters
oxygen2carbon = 0.225  # Oxygen to carbon ratio
molar_mass_ratio = 18.016 / 100  # Water to organic molecular weight ratio
density = get_organic_density_estimate(
    18.016 / molar_mass_ratio,
    oxygen2carbon,
)  # Estimate of organic compound density

# Calculate activity coefficients using the binary_activity function
(
    activity_water,
    activity_organic,
    mass_water,
    mass_organic,
    gamma_water,
    gamma_organic,
) = activity_coefficients.bat_activity_coefficients(
    molar_mass_ratio,
    organic_mole_fraction,
    oxygen2carbon,
    density,
    functional_group=None,
)

# %% [markdown]
# ## Read activities and coefficients separately
#
# The horizontal coordinate is **water** mole fraction, the complement of the
# input organic fraction. The first plot shows activities; the second shows
# activity coefficients. Departures from ideal molar behavior reflect the BAT
# model's nonideality for these descriptors.
#
# The separation helper inspects activity-curve monotonicity to select branches.
# Nonmonotonic curves can signal a region requiring a phase treatment in this
# model. Neither an arbitrary activity above one nor a large coefficient alone
# constitutes experimental proof of liquid–liquid separation.

# %%
fig, ax = plt.subplots()
ax.plot(
    1 - organic_mole_fraction,
    activity_water,
    label="water",
    linestyle="dashed",
)
ax.plot(
    1 - organic_mole_fraction,
    activity_organic,
    label="organic",
)
ax.set_ylim()
ax.set_xlabel("water mole fraction")
ax.set_ylabel("activity")
ax.legend()
plt.show()

fig, ax = plt.subplots()
ax.plot(
    1 - organic_mole_fraction, gamma_water, label="water", linestyle="dashed"
)
ax.plot(
    1 - organic_mole_fraction,
    gamma_organic,
    label="organic",
)
ax.set_ylim()
ax.set_xlabel("water mole fraction")
ax.set_ylabel("activity coefficient")
ax.legend()
plt.show()

# %% [markdown]
# ## Understand the phase-transition weight
#
# The helper extracts a separation activity and constructs a smooth $q^\alpha$
# transition. This dimensionless parameter weights the water-rich alpha branch
# relative to the organic-rich beta branch in the reduced model. It is not water
# mass fraction, a time coordinate, or a measured fraction of droplets.
#
# The plot shows how that parameter varies with the computed water activity;
# it does not evolve particles through a separation event.

# %%
# Finding phase separation points and calculating q_alpha
phase_sep_aw = phase_separation.find_phase_separation(
    activity_water, activity_organic
)
q_alpha = phase_separation.q_alpha(
    seperation_activity=phase_sep_aw["upper_seperation"],
    activities=activity_water,
)

# Plotting q_alpha
fig, ax = plt.subplots()
plt.plot(activity_water, q_alpha)
plt.xlabel("Water Activity")
plt.ylabel("$q^{\\alpha}$ [Organic Rich to Water Rich]")
plt.show()

# %% [markdown]
# ## Prescribe water activity instead of composition
#
# Now set O:C=0.25 and ask for properties over $a_w=0.5$ to 1. The density and
# molar mass remain fixed from the preceding setup, isolating the descriptor
# change rather than updating an entire material-property model. Under
# flat-interface equilibrium at a common temperature, $a_w=\mathrm{RH}/100$.
# Curvature and finite equilibration rates would change that connection.
#
# The first two returned objects hold phase properties; row 2 contains each
# phase's water mass fraction. The beta object may be `None` when only one branch
# is used. Plot water fractions alongside $q^\alpha$ to distinguish phase
# composition from phase weighting. The endpoint near unit activity is limited
# by the internal composition interpolation, not a pure-water measurement.
# In the low-O:C panel, the alpha water fraction sits near one while its
# weight is near zero over most of the plotted range. The helper clamps
# out-of-branch interpolation to endpoint compositions. Do not read that
# unweighted branch or the sharp final jump as a resolved physical transition.

# %%
# select the water activity desired
water_activity_desired = np.linspace(0.5, 1, 100)
oxygen2carbon = 0.25

# calculate the mass fraction of water in the alpha and beta phases
# for each water activity
alpha_raw, beta_raw, q_alpha_raw = water_activity.fixed_water_activity(
    water_activity=water_activity_desired,
    molar_mass_ratio=molar_mass_ratio,
    oxygen2carbon=oxygen2carbon,
    density=density,
)
alpha = np.atleast_1d(np.asarray(alpha_raw, dtype=float))
q_alpha = np.atleast_1d(np.asarray(q_alpha_raw, dtype=float))
beta = (
    np.atleast_1d(np.asarray(beta_raw, dtype=float))
    if beta_raw is not None
    else None
)

# plot the results vs water activity
fig, ax = plt.subplots()
ax.plot(
    water_activity_desired,
    alpha[2],
    label="alpha phase mass fraction water",
)
ax.plot(
    water_activity_desired,
    q_alpha,
    label="q_alpha",
)
if beta is not None:
    ax.plot(
        water_activity_desired,
        beta[2],
        label="beta phase mass fraction water",
    )
ax.set_ylim()
ax.set_xlabel("Water activity")
ax.set_ylabel("Water mass fraction or phase weight")
plt.legend()
plt.show()

# %% [markdown]
# ## Increase oxygenation while holding other descriptors fixed
#
# Repeat at O:C=0.6 with the same molar mass, density, and activity grid. In this
# parameter comparison the higher-oxygenation case uses a single branch, with
# $q^\alpha=1$ and no beta curve. Compare that with the lower-O:C panel rather
# than interpreting the second panel as a later stage of chemical aging.
#
# Real oxidation can change molar mass, density, volatility, and functionality
# together. This controlled descriptor change does not represent those coupled
# processes or establish a universal O:C threshold for every organic aerosol.

# %%
# select the water activity desired
water_activity_desired = np.linspace(0.5, 1, 100)
# select the oxygen to carbon ratio
oxygen2carbon = 0.6

# calculate the mass fraction of water in the alpha and beta phases
# for each water activity
alpha_raw, beta_raw, q_alpha_raw = water_activity.fixed_water_activity(
    water_activity=water_activity_desired,
    molar_mass_ratio=molar_mass_ratio,
    oxygen2carbon=oxygen2carbon,
    density=density,
)
alpha = np.atleast_1d(np.asarray(alpha_raw, dtype=float))
q_alpha = np.atleast_1d(np.asarray(q_alpha_raw, dtype=float))
beta = (
    np.atleast_1d(np.asarray(beta_raw, dtype=float))
    if beta_raw is not None
    else None
)

# plot the results vs water activity
fig, ax = plt.subplots()
ax.plot(
    water_activity_desired,
    alpha[2],
    label="alpha phase mass fraction water",
)
ax.plot(
    water_activity_desired,
    q_alpha,
    label="q_alpha",
)
if beta is not None:
    ax.plot(
        water_activity_desired,
        beta[2],
        label="beta phase mass fraction water",
    )
ax.set_ylim()
ax.set_xlabel("Water activity")
ax.set_ylabel("Water mass fraction or phase weight")
plt.legend()
plt.show()

# %% [markdown]
# ## Carry the phase information into a research question
#
# The model can associate prescribed water activity with different phase
# compositions and a transition weight. Those are distinct outputs: the amount
# of water within a phase is not the weight assigned to that phase. The comparison
# also shows why molecular descriptors matter when selecting a phase treatment.
#
# A next investigation could vary O:C and molar mass together, record the water
# activity at which the beta branch disappears, and compare that map with measured
# phase-separation humidity. Include density sensitivity and the model's
# calibration domain. To predict condensed organic mass rather than phase
# properties alone, continue with [liquid–vapor partitioning](equilibria_part1.py).
