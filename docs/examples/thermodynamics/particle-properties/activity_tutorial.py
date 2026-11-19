"""Activity tutorial for mixing rules and water activity predictions.

This tutorial demonstrates mixing behavior in aerosol solutions, including
ideal and non-ideal mixing rules. Topics covered include mass-based and
molar-based activity calculations, and kappa value parameterization for
predicting water activity in non-ideal solutions.

Example:
    Run this tutorial to learn about different activity strategies::

        import particula as par
        activity_mass = par.particles.ActivityIdealMass()
        activities = activity_mass.activity(mass_concentration=[0.2, 0.8])
"""
# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.17.3
#   kernelspec:
#     display_name: .venv
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Which composition rule controls equilibrium pressure?
#
# A mixture containing 20% water by mass need not have water activity 0.2.
# Molecules, mass, and liquid volume weight the same composition differently.
# Here we compare three activity approximations for water and a sucrose-like
# solute, then see how each changes the flat-interface pressure $p_i=a_i p_i^0$.
#
# The calculation separates two choices: the physical activity rule and the
# construction interface (direct strategy, builder, or factory). Switching an
# interface does not itself change the physics. Read the
# [activity functions](activity_functions.py) for a wider composition comparison;
# [curvature](particle_surface_tutorial.py) is an additional effect omitted here.

# %%
# In Colab uncomment the following command to install particula:
# Install from the repository root: `python -m pip install -e ".[extra]"`
import matplotlib.pyplot as plt
import numpy as np
import particula as par
from particula import plotting

plt.style.use("particula.default")

# %% [markdown]
# ## Start with a mass-fraction approximation
#
# `ActivityIdealMass` sets each activity equal to its mass fraction. This is a
# mass-based approximation, distinct from the usual mole-fraction Raoult law.
# The [water, solute] mass ratio is [0.2, 0.8]; its common absolute scale cancels.
# The pressures [100, 10] Pa are deliberately illustrative, not measured water
# and sucrose vapor pressures. The returned [20, 8] Pa simply exposes the
# multiplication by activity.

# %%
activity_mass = par.particles.ActivityIdealMass()  # no parameters needed

# mixture
mass_mixture = np.array([0.2, 0.8])  # water, sucrose

activities = activity_mass.activity(mass_concentration=mass_mixture)
print(f"Activities: {activities}")

# partial pressures
pure_pressure = np.array([100, 10])  # water, sucrose
partial_pressures = activity_mass.partial_pressure(
    pure_vapor_pressure=pure_pressure, mass_concentration=mass_mixture
)

print(f"Partial pressures: {partial_pressures}")

# %% [markdown]
# ## Count molecules instead of mass
#
# The builder takes molar masses in kg/mol. Water's smaller molar mass means that
# 20% of the mixture's mass represents most of its molecules. Consequently the
# ideal molar water activity exceeds the mass-based value, while the solute
# activity is smaller. The pressure change follows from this different model;
# it is not caused by using a builder instead of direct construction.

# %%
activity_molar = (
    par.particles.ActivityIdealMolarBuilder()
    .set_molar_mass(
        np.array([18.01528, 342.29648]) * 1e-3, "kg/mol"
    )  # water, sucrose
    .build()
)

# mixture
activities = activity_molar.activity(mass_concentration=mass_mixture)
print(f"Activities: {activities}")

# partial pressures
pure_pressure = np.array([100, 10])  # water, sucrose
partial_pressures = activity_molar.partial_pressure(
    pure_vapor_pressure=pure_pressure, mass_concentration=mass_mixture
)

print(f"Partial pressures: {partial_pressures}")

# %% [markdown]
# ## Give water a hygroscopicity parameter
#
# The factory converts the supplied g/mol values and applies the κ water-activity
# rule using species densities in kg/m³. The solute κ=0.3 and density 1500 kg/m³
# are illustrative choices, not a fitted sucrose property set. Water occupies
# lane zero. Nonwater activities retain the ideal molar rule, so this is not a
# complete nonideal model for every species.

# %%
parameters_input = {
    "density": np.array(
        [1000, 1500]
    ),  # water, sucrose, kg/m^3 are the base SI default units
    "density_units": "kg/m^3",  # this tells the factory what the input units are so it can convert to kg/m^3
    "molar_mass": np.array([18.01528, 342.29648]),  # water, sucrose
    "molar_mass_units": "g/mol",  # this tells the factory what the input units are so it can convert to kg/mol
    "kappa": np.array([0, 0.3]),  # water, sucrose
    "water_index": 0,  # water is the first component
}

activity_kappa = par.particles.ActivityFactory().get_strategy(
    strategy_type="kappa", parameters=parameters_input
)

# mixture
activities = activity_kappa.activity(mass_concentration=mass_mixture)
print(f"Activities: {activities}")

# partial pressures
pure_pressure = np.array([100, 10])  # water, sucrose
partial_pressures = activity_kappa.partial_pressure(
    pure_vapor_pressure=pure_pressure, mass_concentration=mass_mixture
)

print(f"Partial pressures: {partial_pressures}")

# %% [markdown]
# ## Follow the full composition path
#
# Increase water mass fraction while keeping total relative mass equal to one.
# The solid curves compare water activity under the three assumptions. The dashed
# κ solute curve coincides with the molar solute curve by construction. Only
# activities are plotted; the preceding pressure calculations explain how these
# curves would scale a chosen pure-component vapor pressure.
#
# A separation between curves measures model sensitivity, not which model is
# accurate. An experimental water-activity relation at the relevant temperature
# and composition would be needed to decide that.

# %%
mass_water = np.linspace(0.001, 0.9999, 100)
mass_sucrose = 1 - mass_water
mass_mixture = np.array([mass_water, mass_sucrose]).T

activities_mass = np.zeros_like(mass_mixture)
activities_molar = np.zeros_like(mass_mixture)
activities_kappa = np.zeros_like(mass_mixture)


for i, mass in enumerate(mass_mixture):
    activities_mass[i] = activity_mass.activity(mass_concentration=mass)
    activities_molar[i] = activity_molar.activity(mass_concentration=mass)
    activities_kappa[i] = activity_kappa.activity(mass_concentration=mass)

fig, ax = plotting.subplots(size="wide")
ax.plot(mass_water, activities_mass[:, 0], label="Mass: Water")
ax.plot(
    mass_water, activities_mass[:, 1], label="Mass: Sucrose", linestyle="--"
)
ax.plot(mass_water, activities_molar[:, 0], label="Molar: Water")
ax.plot(
    mass_water,
    activities_molar[:, 1],
    label="Molar: Sucrose",
    linestyle="--",
    linewidth=3,
    alpha=0.85,
)
ax.plot(mass_water, activities_kappa[:, 0], label="Kappa: Water")
ax.plot(
    mass_water, activities_kappa[:, 1], label="Kappa: Sucrose", linestyle="--"
)
ax.set_xlabel("Mass fraction water")
ax.set_ylabel("Activity")
plotting.legend(ax)
plotting.fit(fig)
plt.show()

# %% [markdown]
# ## Turn a model choice into a testable question
#
# Composition alone does not determine the pressure reduction until an activity
# rule is specified. For a light solvent and heavy solute, replacing mass fraction
# with mole fraction can substantially change that reduction. κ adds a separate
# water-uptake assumption while leaving the nonwater molar rule intact.
#
# As a useful interface check, construct the *same* molar model directly, through
# a builder, and through the factory, then compare activities on this grid. For a
# scientific extension, fit κ to measured water activity and compare residuals
# against ideal molar predictions over the measured range. Neither exercise is
# performed here; fitted behavior outside that range would need new evidence.
