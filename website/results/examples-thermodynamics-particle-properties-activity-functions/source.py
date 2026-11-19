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

"""Explore mixing rules and activity calculations.

This example walks through mass-based, molar-based, and volume-based
mixing rules, compares ideal activities, and demonstrates the
kappa value parameterization to model non-ideal solution behavior. Each
section builds the required data arrays, computes activities with the
public ``par.particles`` helpers, and plots the results.

Example:
    Run this example to generate activity plots for two- and
three-component systems under different mixing assumptions.
"""


# %% [markdown]
# # Compare activity rules along a composition path
#
# How much can an assumed mixing rule change the activity assigned to the same
# mixture? We follow binary and ternary composition paths, comparing mole-,
# volume-, and mass-fraction rules with a κ water-activity parameterization.
# This helps separate a change of composition coordinates from a change of
# thermodynamic model before coupling activity to condensation.
#
# Every vertical axis is **activity**, not activity coefficient. In the molar
# standard-state convention, $a_i=\gamma_i x_i$; ideal molar mixing has
# $\gamma_i=1$ but generally $a_i\ne1$. Substituting mass or volume fraction for
# activity is a different approximation, not another way to plot the same
# Raoult-law prediction. These generic components are not a calibrated material
# system. For construction interfaces, continue with the
# [strategy comparison](activity_tutorial.py).

# %%
# In Colab uncomment the following command to install particula:
# Install from the repository root: `python -m pip install -e ".[extra]"`
import matplotlib.pyplot as plt
import numpy as np
import particula as par
from particula import plotting

plt.style.use("particula.default")

# %% [markdown]
# ## Define the path before interpreting the curves
#
# Water mass increases from 0 to 100 relative units while component 2 remains
# at 30. In the ternary path, component 3 decreases from 150 to 10. The common
# mass scale cancels in these rules; total mass is not held constant. Densities
# are kg/m³ and molar masses are kg/mol.
#
# All panels use **water mole fraction** on the horizontal axis, computed once
# from these same masses. The ternary curves therefore trace one changing
# composition path, not a full ternary phase diagram. Component 3's relative
# abundance changes at the same time as water's.

# %%
masses_1 = np.linspace(0, 100, 1000)
masses_2 = np.full_like(masses_1, 30)
masses_3 = np.linspace(150, 10, 1000)

density_1 = 1000  # kg/m^3
density_2 = 2000  # kg/m^3
density_3 = 1500  # kg/m^3

molar_mass_1 = 18.01528e-3  # kg/mol
molar_mass_2 = 200e-3  # kg/mol
molar_mass_3 = 150e-3  # kg/mol

mass_2component = np.column_stack((masses_1, masses_2))
mass_3component = np.column_stack((masses_1, masses_2, masses_3))

density_2component = np.array([density_1, density_2])
density_3component = np.array([density_1, density_2, density_3])

# get mole fractions
mole_fractions_2component = par.particles.get_mole_fraction_from_mass(
    mass_2component, molar_masses=np.array([molar_mass_1, molar_mass_2])
)
mole_fractions_3component = par.particles.get_mole_fraction_from_mass(
    mass_3component,
    molar_masses=np.array([molar_mass_1, molar_mass_2, molar_mass_3]),
)

# %% [markdown]
# ## Use mole fractions as the reference
#
# Ideal molar activity equals mole fraction. Water's curve therefore lies on
# $a_w=x_w$ in both panels. In the binary panel, the solute follows $1-x_w$;
# in the ternary panel, the two solutes share that remainder according to the
# chosen path. This is a reference model, not evidence that the generic mixture
# is ideal.

# %%
activity_2component_molar = par.particles.get_ideal_activity_molar(
    mass_concentration=mass_2component,
    molar_mass=np.array([molar_mass_1, molar_mass_2]),
)


activity_3component_molar = par.particles.get_ideal_activity_molar(
    mass_concentration=mass_3component,
    molar_mass=np.array([molar_mass_1, molar_mass_2, molar_mass_3]),
)


# Create the figure and axis objects
fig, ax = plotting.subplots(2, 1)

# Plot each component in the 2-component system with separate colors
ax[0].plot(
    mole_fractions_2component[:, 0],
    activity_2component_molar[:, 0],
    color=plotting.COLORS["teal"],
    label="Water (1)",
)
ax[0].plot(
    mole_fractions_2component[:, 0],
    activity_2component_molar[:, 1],
    color=plotting.COLORS["indigo"],
    label="Component 2",
)
ax[0].set_xlabel("Water Mole Fraction")
ax[0].set_ylabel("Activity")
ax[0].set_title("Binary molar-based activity")
plotting.legend(ax[0])

# Plot the 3-component system without setting specific colors
ax[1].plot(
    mole_fractions_3component[:, 0],
    activity_3component_molar[:, 0],
    label="Water (1)",
    color=plotting.COLORS["teal"],
)
ax[1].plot(
    mole_fractions_3component[:, 0],
    activity_3component_molar[:, 1],
    label="Component 2",
    color=plotting.COLORS["indigo"],
)
ax[1].plot(
    mole_fractions_3component[:, 0],
    activity_3component_molar[:, 2],
    label="Component 3",
    color=plotting.COLORS["rust"],
)
ax[1].set_xlabel("Water Mole Fraction")
ax[1].set_ylabel("Activity")
ax[1].set_title("Ternary molar-based activity")
plotting.legend(ax[1])

# Display the plot
plotting.fit(fig)
plt.show()

# %% [markdown]
# ## Weight by liquid volume
#
# The volume rule normalizes $m_i/\rho_i$ and returns those fractions as
# activities. Its curves need not lie on the molar reference, even though the
# horizontal coordinate is unchanged. The assumed additive component volumes
# and the choice to equate volume fraction with activity are separate modeling
# assumptions; density conversion alone does not establish thermodynamic accuracy.

# %%
# Volume activity
activity_2component_volume = par.particles.get_ideal_activity_volume(
    mass_concentration=mass_2component,
    density=density_2component,
)

activity_3component_volume = par.particles.get_ideal_activity_volume(
    mass_concentration=mass_3component,
    density=density_3component,
)

# Create the figure and axis objects
fig, ax = plotting.subplots(2, 1)

# Plot each component in the 2-component system with separate colors
ax[0].plot(
    mole_fractions_2component[:, 0],
    activity_2component_volume[:, 0],
    color=plotting.COLORS["teal"],
    label="Water (1)",
)
ax[0].plot(
    mole_fractions_2component[:, 0],
    activity_2component_volume[:, 1],
    color=plotting.COLORS["indigo"],
    label="Component 2",
)
ax[0].set_xlabel("Water Mole Fraction")
ax[0].set_ylabel("Activity")
ax[0].set_title("Binary volume-based activity")
plotting.legend(ax[0])

# Plot the 3-component system without setting specific colors
ax[1].plot(
    mole_fractions_3component[:, 0],
    activity_3component_volume[:, 0],
    label="Water (1)",
    color=plotting.COLORS["teal"],
)
ax[1].plot(
    mole_fractions_3component[:, 0],
    activity_3component_volume[:, 1],
    label="Component 2",
    color=plotting.COLORS["indigo"],
)
ax[1].plot(
    mole_fractions_3component[:, 0],
    activity_3component_volume[:, 2],
    label="Component 3",
    color=plotting.COLORS["rust"],
)
ax[1].set_xlabel("Water Mole Fraction")
ax[1].set_ylabel("Activity")
ax[1].set_title("Ternary volume-based activity")
plotting.legend(ax[1])

# Display the plot
plotting.fit(fig)
plt.show()

# %% [markdown]
# ## Weight by mass
#
# The mass rule normalizes the input masses directly. Because water is much
# lighter per molecule than either solute, its mass fraction is below its mole
# fraction along these paths. The curved water trace against $x_w$ is therefore
# expected. It does not indicate a numerically inconsistent mass conversion.

# %%
# Mass activity
activity_2component_mass = par.particles.get_ideal_activity_mass(
    mass_concentration=mass_2component,
)
activity_3component_mass = par.particles.get_ideal_activity_mass(
    mass_concentration=mass_3component,
)

# Create the figure and axis objects
fig, ax = plotting.subplots(2, 1)

# Plot each component in the 2-component system with separate colors
ax[0].plot(
    mole_fractions_2component[:, 0],
    activity_2component_mass[:, 0],
    color=plotting.COLORS["teal"],
    label="Water (1)",
    linewidth=4,
    linestyle="--",
)
ax[0].plot(
    mole_fractions_2component[:, 0],
    activity_2component_mass[:, 1],
    color=plotting.COLORS["indigo"],
    label="Component 2",
)
ax[0].set_xlabel("Water Mole Fraction")
ax[0].set_ylabel("Activity")
ax[0].set_title("Binary mass-based activity")
plotting.legend(ax[0])

# Plot the 3-component system without setting specific colors
ax[1].plot(
    mole_fractions_3component[:, 0],
    activity_3component_mass[:, 0],
    label="Water (1)",
    color=plotting.COLORS["teal"],
    linewidth=4,
    linestyle="--",
)
ax[1].plot(
    mole_fractions_3component[:, 0],
    activity_3component_mass[:, 1],
    label="Component 2",
    color=plotting.COLORS["indigo"],
    linewidth=3,
)
ax[1].plot(
    mole_fractions_3component[:, 0],
    activity_3component_mass[:, 2],
    label="Component 3",
    color=plotting.COLORS["rust"],
)
ax[1].set_xlabel("Water Mole Fraction")
ax[1].set_ylabel("Activity")
ax[1].set_title("Ternary mass-based activity")
plotting.legend(ax[1])

# Display the plot
plotting.fit(fig)
plt.show()

# %% [markdown]
# ## Change water's rule with κ
#
# The illustrative solutes have κ=0.6 and 1.2; the water lane's κ is unused.
# Water activity follows $a_w=V_w/(V_w+\sum_i\kappa_i V_i)$ for the solute
# volumes $V_i$. Its dry endpoint is zero here. Along the ternary path, the
# relative contributions of the two solutes also change, so there is no single
# constant effective solute κ for the entire curve.
#
# The nonwater lanes retain ideal molar activities. Compare them with the first
# pair of panels to see that this model changes water's rule without computing
# nonideal activity coefficients for every component. The parameterization follows
# [Petters and Kreidenweis (2007)](https://acp.copernicus.org/articles/7/1961/2007/);
# the particular solute parameters here are illustrative.

# %%
# kappa activity

kappa_1 = 0.0  # kappa of water
kappa_2 = 0.6  # kappa of component 2
kappa_3 = 1.2  # kappa of component 3

water_index = 0

activity_2component_kappa = par.particles.get_kappa_activity(
    mass_concentration=mass_2component,
    kappa=np.array([kappa_1, kappa_2]),
    density=density_2component,
    molar_mass=np.array([molar_mass_1, molar_mass_2]),
    water_index=water_index,
)
activity_3component_kappa = par.particles.get_kappa_activity(
    mass_concentration=mass_3component,
    kappa=np.array([kappa_1, kappa_2, kappa_3]),
    density=density_3component,
    molar_mass=np.array([molar_mass_1, molar_mass_2, molar_mass_3]),
    water_index=water_index,
)

# Create the figure and axis objects
fig, ax = plotting.subplots(2, 1)

# Plot each component in the 2-component system with separate colors
ax[0].plot(
    mole_fractions_2component[:, 0],
    activity_2component_kappa[:, 0],
    color=plotting.COLORS["teal"],
    label="Water (1)",
)
ax[0].plot(
    mole_fractions_2component[:, 0],
    activity_2component_kappa[:, 1],
    color=plotting.COLORS["indigo"],
    label="Component 2",
)
ax[0].set_xlabel("Water Mole Fraction")
ax[0].set_ylabel("Activity")
ax[0].set_title("Binary κ activity")
plotting.legend(ax[0])

# Plot the 3-component system without setting specific colors
ax[1].plot(
    mole_fractions_3component[:, 0],
    activity_3component_kappa[:, 0],
    label="Water (1)",
    color=plotting.COLORS["teal"],
)
ax[1].plot(
    mole_fractions_3component[:, 0],
    activity_3component_kappa[:, 1],
    label="Component 2",
    color=plotting.COLORS["indigo"],
)
ax[1].plot(
    mole_fractions_3component[:, 0],
    activity_3component_kappa[:, 2],
    label="Component 3",
    color=plotting.COLORS["rust"],
)
ax[1].set_xlabel("Water Mole Fraction")
ax[1].set_ylabel("Activity")
ax[1].set_title("Ternary κ activity")
plotting.legend(ax[1])

# Display the plot
plotting.fit(fig)
plt.show()

# %% [markdown]
# ## What survives a change of mixing rule?
#
# The composition path and its mole-fraction coordinate stay fixed, while the
# predicted activities change. Thus interpreting a condensation calculation
# requires naming its activity approximation as well as its composition.
# Agreement at an endpoint or the fact that fractions sum to one does not choose
# the physically appropriate rule.
#
# For a research comparison, hold the dry solute ratio fixed, vary only water,
# and compare predicted $a_w$ with measured water activity at one temperature.
# Use residuals over the measured range to assess a fitted κ against ideal molar
# mixing. That controlled path would separate water uptake from the changing
# dry composition used in the ternary illustration here.
