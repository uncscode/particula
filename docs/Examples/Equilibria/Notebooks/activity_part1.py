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
# # Activity Example
#
# This notebook demonstrates the Binary Activity Theory (BAT) model application, crucial for calculating the activity of water and organic compounds in mixtures and understanding phase separation. This model, as detailed in Gorkowski, K., Preston, T. C., & Zuend, A. (2019), provides critical insights into aerosol particle behavior, essential in environmental and climate change research.
#
#  Reference: Gorkowski, K., Preston, T. C., & Zuend, A. (2019).
#  Relative-humidity-dependent organic aerosol thermodynamics Via an efficient
#  reduced-complexity model. Atmospheric Chemistry and Physics
#  https://doi.org/10.5194/acp-19-13383-2019

# %%
# In Colab uncomment the following command to install particula:
# #!pip install particula[extra] --quiet
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
# ## Activity Calculation
#
# Define the parameters required by the activity module to calculate the activity of water and organic compounds in a mixture, as well as phase separation. These parameters include organic mole fraction, density, molecular weight ratio [water/organic], and the density of the organic compound. Using these parameters helps in accurately modeling the behavior of aerosol particles in various environmental conditions.
#

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
# ## Plotting the Activity and Phase Separation
#
# Here we plot the activity of water and the organic compound as a function of the organic mole fraction. Visualizing these activities helps in identifying phase separation or miscibility gaps, crucial for understanding the behavior of aerosols under different environmental conditions. Phase separation is indicated by activities greater than 1.0 or non-monotonic behavior in the activity curve, as shown below.

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
# ## $ q^\alpha $
#
# The $q^\alpha$ parameter signifies the transition from an organic-rich phase to a water-rich phase. This transition is crucial for understanding the phase behavior of aerosol particles. It can be calculated using the `particula.activity.phase_separation` function. The plot below illustrates $q^\alpha$ based on the activity calculations performed earlier.
#

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
# ## Water Activity Focus
#
# In atmospheric aerosol modeling, water activity is often a more critical parameter than mole fraction. This is because water activity is typically a controllable or known variable in atmospheric conditions, unlike the exact mole fractions in a solution. To correlate water activity with the mole fraction required to achieve it, we utilize functions from the `particula.activity` module.

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
ax.set_xlabel("water activity (Relative Humidity/100)")
ax.set_ylabel("mass fraction of water")
plt.legend()
plt.show()

# %% [markdown]
# ## Higher Oxygen to Carbon Ratios
#
# Higher oxygen to carbon ratios in the mixture tend to inhibit phase separation. The following analysis demonstrates this effect. This observation is crucial in predicting the behavior of aerosol particles under varying chemical compositions (more or less 'aged').
#

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
ax.set_xlabel("water activity (Relative Humidity/100)")
ax.set_ylabel("mass fraction of water")
plt.legend()
plt.show()

# %% [markdown]
# ## Summary
#
# This notebook demonstrates how to use the activity module for calculating the activity of water and organic compounds in a mixture and assessing phase separation. The insights gained are vital for applications in aerosol thermodynamics, cloud condensation nuclei, and cloud microphysics.
#
# This is an implementation of the Binary Activity Theory (BAT) model
# developed in Gorkowski, K., Preston, T. C., & Zuend, A. (2019).
