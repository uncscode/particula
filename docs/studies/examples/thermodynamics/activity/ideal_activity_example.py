# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.17.3
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Equal mass does not mean equal activity
#
# Why does a mixture containing half water by mass have water activity much
# closer to one than to one-half? This ideal-solution calculation converts mass
# concentrations to mole fractions and uses them to calculate equilibrium partial
# pressures. It provides a transparent reference before introducing
# [nonideal organic–water interactions](bat_activity_example.py).

# %%
import numpy as np
import particula as par

# %% [markdown]
# ## Specify the molecular bookkeeping
#
# The two components are water (18.015 g mol⁻¹) and an illustrative organic
# (200 g mol⁻¹), ordered consistently in every array. `ActivityIdealMolar` uses
# mole fraction as activity. An organic's molar mass alone does not establish that
# its real mixture with water is ideal.

# %%
# Molar masses for water and an organic compound
strategy = par.particles.ActivityIdealMolar(
    molar_mass=np.array([18.015e-3, 200.0e-3]),  # kg/mol: water, organic
)

print(f"Strategy: {type(strategy).__name__}")
print(f"Molar masses: {strategy.molar_mass * 1e3} g/mol")

# %% [markdown]
# ## Compare equal masses through their mole counts
#
# Each component contributes 0.5 µg m⁻³. Dividing by molar mass produces far more
# water molecules than organic molecules, so the calculated water activity is
# about 0.9174 and organic activity about 0.0826. Scaling both masses equally
# would preserve these ratios and activities.

# %%
# Equal mass of water and organic (50/50 by mass)
mass = np.array([0.5e-9, 0.5e-9])  # kg/m^3

# Compute activity (mole fraction based for ideal mixing)
activity = strategy.activity(mass_concentration=mass)

print("=== Ideal Activity (Raoult's Law) ===")
print(f"Mass concentrations: {mass * 1e9} ug/m^3")
print(f"Activity values: {activity}")

# %% [markdown]
# ## Translate activity into an equilibrium pressure
#
# The relation pᵢ = aᵢ pᵢ° gives the flat-interface equilibrium partial pressure
# for the specified mixture and pure-component reference pressures. The supplied
# 3169 Pa water and 10⁻³ Pa organic values are illustrative inputs near 298 K.
# This calculation does not specify the actual ambient gas pressure, include
# curvature, or compute a condensation rate.

# %%
# Pure vapor pressures at 298 K (example values)
pure_pressure = np.array([3169.0, 1e-3])  # Pa: water, organic
partial_pressure = strategy.partial_pressure(
    pure_vapor_pressure=pure_pressure,
    mass_concentration=mass,
)

print(f"Pure vapor pressures: {pure_pressure} Pa")
print(f"Partial pressures: {partial_pressure} Pa")

# %% [markdown]
# ## Verify the chosen model independently
#
# The direct mole-fraction calculation checks the activity strategy, and a second
# assertion checks multiplication by the pure-component pressure. These are
# algebraic consistency checks under ideal mixing, not evidence that a particular
# organic–water solution obeys that model.

# %%
# Calculate mole fractions directly for comparison
moles = mass / strategy.molar_mass
mole_fractions = moles / np.sum(moles)
print(f"Mole fractions: {mole_fractions}")
print(
    f"Activity = mole fraction (ideal): {np.allclose(activity, mole_fractions)}"
)

np.testing.assert_allclose(activity, mole_fractions, rtol=1e-12, atol=0)
np.testing.assert_allclose(
    partial_pressure, mole_fractions * pure_pressure, rtol=1e-12, atol=0
)

# %% [markdown]
# ## Carry the reference into a research question
#
# Mass fraction and mole fraction answer different questions even for an ideal
# solution. The result explains why equal mass does not imply equal thermodynamic
# activity; it does not demonstrate water uptake from an unspecified gas reservoir.
#
# To test whether ideality is adequate for your system, compare the organic
# activity with a justified nonideal model at the same composition and molecular
# properties. Then compare predicted equilibrium pressure with observations or an
# independent thermodynamic reference. For small particles, include
# [surface curvature](../particle-properties/particle_surface_tutorial.py) before
# interpreting the pressure difference as a growth threshold.
