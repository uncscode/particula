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
# # How does water uptake change organic aerosol partitioning?
#
# At fixed total organic loading, changing water activity can change both aerosol
# water and the amount of organic material in the condensed phase. Here a
# five-component volatility distribution is solved at equilibrium, first at
# $a_w=0.8$ and then across a water-activity sweep. The aim is to interpret that
# equilibrium sensitivity before adding time-dependent transport.
#
# Use the [BAT phase-property walkthrough](activity_part1.py) to understand the
# activity coefficients and phase weights. This functional calculation does not
# require an aerosol container and does not predict a condensation timescale.
# It uses the reduced thermodynamic framework of
# [Gorkowski, Preston, and Zuend (2019)](https://doi.org/10.5194/acp-19-13383-2019).

# %%
# In Colab uncomment the following command to install particula:
# Install from the repository root: `python -m pip install -e ".[extra]"`
# Importing necessary libraries
import matplotlib.pyplot as plt  # For creating plots and visualizations
import numpy as np  # For numerical operations
from particula import plotting

# For partitioning calculations in liquid-vapor equilibrium
from particula.equilibria import partitioning
from particula.particles.properties.organic_density_module import (
    get_organic_density_array,
)  # For calculating species density

plt.style.use("particula.default")

# %% [markdown]
# ## Span volatility while fixing the available organic mass
#
# The dry saturation concentrations $C_i^*$ range from $10^{-6}$ to $10^4$
# µg/m³. Total gas-plus-particle organic concentrations are [1, 5, 10, 15, 10]
# µg/m³, summing to 41 µg/m³. These are independent inputs: saturation
# concentration describes volatility, while total concentration bounds the
# available material. All five molar masses are 200 g/mol.
#
# O:C and estimated density supply the activity calculation. The density helper
# returns kg/m³ using its default elemental assumptions. The generic components
# are illustrative, not a measured speciation or fitted emission mixture.

# %%
# Defining system parameters
c_star_j_dry = np.array(
    [1e-6, 1e-4, 1e-1, 1e2, 1e4], dtype=float
)  # Volatility distribution in ug/m3
# Total concentration in ug/m3
concentration_organic_matter = np.array([1, 5, 10, 15, 10], dtype=float)
oxygen2carbon = np.array([0.2, 0.3, 0.5, 0.4, 0.4])  # Oxygen to carbon ratios

molar_mass = np.array([200, 200, 200, 200, 200])  # Molar mass in g/mol
water_activity_desired = np.array([0.8])  # Desired water activity
molar_mass_ratio = 18.015 / np.array(molar_mass)  # Molar mass ratio

# Calculate the density of organic compounds
density = get_organic_density_array(
    molar_mass,
    oxygen2carbon,
    hydrogen2carbon=None,
    nitrogen2carbon=None,
)

# %% [markdown]
# ## Connect phase properties to mass partitioning
#
# At $a_w=0.8$, first calculate organic activity coefficients, water fractions,
# and alpha/beta phase weights. The partitioning solver then uses them with
# volatility and total loading to allocate organic mass. The printed organic
# and water concentrations describe the combined aerosol phases.
#
# The saved calculation gives about 24.22 µg/m³ condensed organic material and
# 9.41 µg/m³ aerosol water. Organic mass is bounded by the 41 µg/m³ total;
# water comes from the imposed equilibrium activity rather than that organic
# inventory. These values depend on the chosen surrogate mixture.

# %%
# Calculate the properties needed for liquid-vapor partitioning
gamma_organic_ab, mass_fraction_water_ab, q_ab = (
    partitioning.get_properties_for_liquid_vapor_partitioning(
        water_activity_desired=water_activity_desired,
        molar_mass=molar_mass,
        oxygen2carbon=oxygen2carbon,
        density=density,
    )
)

# The optimization the partition coefficients, i.e. the partitioning calculation
alpha_opt, beta_opt, system_opt, fit_result = (
    partitioning.liquid_vapor_partitioning(
        c_star_j_dry=c_star_j_dry,
        concentration_organic_matter=concentration_organic_matter,
        molar_mass=molar_mass,
        gamma_organic_ab=gamma_organic_ab,
        mass_fraction_water_ab=mass_fraction_water_ab,
        q_ab=q_ab,
        partition_coefficient_guess=None,
    )
)

print(f"mass in organic aerosol [ug/m3]: {system_opt[0]}")
print(f"mass in water [ug/m3]: {system_opt[1]}")

# %% [markdown]
# ## Repeat equilibria across water activity
#
# Each of the 50 points from $a_w=0.01$ to 0.99 is a separate equilibrium
# calculation with unchanged organic loading. This is not a time trajectory or
# a closed water budget. Water activity equals RH/100 only under the assumed
# flat-interface equilibrium at a common temperature; curvature is omitted.
#
# The arrays retain **condensed mass concentrations**, not activity coefficients.
# Changing humidity influences those masses through both phase properties and
# partitioning. No parameter fitting to observations occurs in this sweep.

# %%
# Calculating activity coefficients across a range of RH values
# Range of water activity (RH/100)
water_activity_curve = np.linspace(0.01, 0.99, 50)
total_oa_concentration = np.empty([len(water_activity_curve), 1], dtype=float)
water_concentration = np.empty([len(water_activity_curve), 1], dtype=float)

for i, water_activity in enumerate(water_activity_curve):
    # Get properties for liquid-vapor partitioning at each RH value
    gamma_organic_ab, mass_fraction_water_ab, q_ab = (
        partitioning.get_properties_for_liquid_vapor_partitioning(
            water_activity_desired=water_activity,
            molar_mass=molar_mass,
            oxygen2carbon=oxygen2carbon,
            density=density,
        )
    )

    # Optimize the partition coefficients for each RH value
    alpha_opt, beta_opt, system_opt, fit_result = (
        partitioning.liquid_vapor_partitioning(
            c_star_j_dry=c_star_j_dry,
            concentration_organic_matter=concentration_organic_matter,
            molar_mass=molar_mass,
            gamma_organic_ab=gamma_organic_ab,
            mass_fraction_water_ab=mass_fraction_water_ab,
            q_ab=q_ab,
            partition_coefficient_guess=None,
        )
    )

    # Record the total organic and water concentration
    total_oa_concentration[i] = system_opt[0]
    water_concentration[i] = system_opt[1]

print("Calculation complete")

# %% [markdown]
# ## Interpret the two concentration scales
#
# The organic aerosol curve uses the left axis and the water curve uses
# the right axis for aerosol water, both in µg/m³. Follow each axis separately:
# a line crossing is not equality of concentrations. The high-activity water
# response illustrates how strongly water uptake can change while organic mass
# remains constrained by a fixed inventory.
#
# The plotted states are equilibrium predictions for these inputs, not evidence
# of how quickly an ambient aerosol responds or of validated multiphase kinetics.

# %%
fig, ax = plotting.subplots()
ax.plot(
    water_activity_curve,
    total_oa_concentration,
    label="total organic concentration",
    color=plotting.COLORS["indigo"],
)
aw = ax.twinx()
aw.plot(
    water_activity_curve,
    water_concentration,
    label="water concentration",
    color=plotting.COLORS["teal"],
)
plotting.legend(plt.gca(), loc="upper left")
ax.set_xlabel("Water activity")

handles, labels = ax.get_legend_handles_labels()
aw_handles, aw_labels = aw.get_legend_handles_labels()
handles.extend(aw_handles)
labels.extend(aw_labels)
plotting.legend(plt.gca(), loc="upper left", handles=handles, labels=labels)

ax.set_ylabel("organic aerosol concentration [ug/m3]")
aw.set_ylabel("aerosol water concentration [ug/m3]")
plotting.fit(fig)
plt.show()

# %% [markdown]
# ## Use the equilibrium curve as a bounded reference
#
# Water activity affects more than aerosol water: it also changes the equilibrium
# organic partitioning problem. The final checks require finite, nonnegative
# masses and organic concentration no greater than the available inventory.
# Those bounds are necessary checks, not proof of solver convergence at every
# point or agreement with measured aerosol composition.
#
# For a research extension, vary the total organic loading while preserving its
# species ratios, then compare the fraction condensed at each water activity.
# Record solver success and residuals as well as mass bounds; compare the result
# with a measured humidity-dependent partitioning curve. A time-dependent
# [condensation example](../../../tutorials/condensation/condensation_1_bin.py)
# would address the separate question of approaching those equilibria.

# %%
# Check finite, physically bounded condensed organic mass across the RH sweep.
assert np.all(np.isfinite(total_oa_concentration))
assert np.all(total_oa_concentration >= 0)
assert np.all(total_oa_concentration <= concentration_organic_matter.sum())
assert np.all(np.isfinite(water_concentration))
assert np.all(water_concentration >= 0)
