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
# # Ideal Activity Example
#
# Demonstrates using `ActivityIdealMolar` to compute activity from mass
# concentrations using Raoult's Law.
#
# ## What This Example Shows
#
# - Creating an ideal molar activity strategy
# - Computing activity from mass concentrations
# - Computing partial pressures from pure vapor pressures
# - Verifying that activity equals mole fraction for ideal mixing

# %%
import numpy as np
import particula as par

# %% [markdown]
# ## 1. Create Ideal Activity Strategy
#
# Define molar masses for a two-component system: water and an organic compound.

# %%
# Molar masses for water and an organic compound
strategy = par.particles.ActivityIdealMolar(
    molar_mass=np.array([18.015e-3, 200.0e-3]),  # kg/mol: water, organic
)

print(f"Strategy: {type(strategy).__name__}")
print(f"Molar masses: {strategy.molar_mass * 1e3} g/mol")

# %% [markdown]
# ## 2. Define Mass Concentrations and Compute Activity
#
# Use equal mass of water and organic (50/50 by mass) and compute activity.
# For ideal mixing, activity equals mole fraction.

# %%
# Equal mass of water and organic (50/50 by mass)
mass = np.array([0.5e-9, 0.5e-9])  # kg/m^3

# Compute activity (mole fraction based for ideal mixing)
activity = strategy.activity(mass_concentration=mass)

print("=== Ideal Activity (Raoult's Law) ===")
print(f"Mass concentrations: {mass * 1e9} ng/m^3")
print(f"Activity values: {activity}")

# %% [markdown]
# ## 3. Compute Partial Pressures
#
# Partial pressure is calculated as activity times pure vapor pressure:
# $p_i = a_i \cdot p_i^{\circ}$

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
# ## 4. Verify Activity Equals Mole Fraction
#
# For ideal mixing (Raoult's Law), activity equals mole fraction.
# Let's calculate mole fractions directly to verify.

# %%
# Calculate mole fractions directly for comparison
moles = mass / strategy.molar_mass
mole_fractions = moles / np.sum(moles)
print(f"Mole fractions: {mole_fractions}")
print(
    f"Activity = mole fraction (ideal): {np.allclose(activity, mole_fractions)}"
)
