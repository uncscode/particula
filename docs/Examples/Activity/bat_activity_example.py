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
# # BAT Model Activity Example
#
# Demonstrates using `ActivityNonIdealBinary` for organic-water mixtures,
# comparing ideal vs non-ideal activity coefficients.
#
# ## What is the BAT Model?
#
# The Binary Activity Thermodynamic (BAT) model accounts for non-ideal
# interactions between water and organic species based on the organic's
# oxygen-to-carbon ratio.
#
# ## What This Example Shows
#
# - Creating a non-ideal activity strategy via builder
# - Comparing ideal vs non-ideal activity at various compositions
# - Understanding the effect of O:C ratio on non-ideality

# %%
import numpy as np
import particula as par

# %% [markdown]
# ## 1. Define Organic Properties
#
# Set up properties for an organic species with a moderate oxygen-to-carbon
# ratio.

# %%
molar_mass_organic = 200.0e-3  # kg/mol
o2c_ratio = 0.5  # oxygen to carbon ratio
density_organic = 1200.0  # kg/m^3

print("=== Organic Properties ===")
print(f"Molar mass: {molar_mass_organic * 1e3:.1f} g/mol")
print(f"O:C ratio: {o2c_ratio}")
print(f"Density: {density_organic} kg/m^3")

# %% [markdown]
# ## 2. Create Non-Ideal (BAT) Activity Strategy
#
# Use the builder pattern to create the BAT activity strategy.

# %%
non_ideal = (
    par.particles.ActivityNonIdealBinaryBuilder()
    .set_molar_mass(molar_mass_organic, "kg/mol")
    .set_oxygen2carbon(o2c_ratio)
    .set_density(density_organic, "kg/m^3")
    .build()
)

print(f"Strategy: {type(non_ideal).__name__}")

# %% [markdown]
# ## 3. Create Ideal Activity Strategy for Comparison
#
# Create an ideal strategy with the same molar masses to compare results.

# %%
ideal = par.particles.ActivityIdealMolar(
    molar_mass=np.array([18.015e-3, molar_mass_organic]),
)

print(f"Ideal strategy: {type(ideal).__name__}")

# %% [markdown]
# ## 4. Compare Ideal vs Non-Ideal Activity
#
# Test at various organic mass fractions and compare the activity values
# and activity coefficients.
#
# **Note**: The BAT model returns only the organic activity (scalar), not both
# water and organic activities like the ideal model.

# %%
mass_fractions = np.array([0.2, 0.5, 0.8])
print("=== Ideal vs Non-Ideal Activity Comparison ===")
print(f"Organic O:C ratio: {o2c_ratio}")
print(f"Organic molar mass: {molar_mass_organic * 1e3:.1f} g/mol")
print("Note: BAT model returns organic activity only (scalar)\n")

for org_frac in mass_fractions:
    # Mass concentrations: [water, organic]
    mass = np.array([1.0 - org_frac, org_frac]) * 1e-9  # kg/m^3

    ideal_activity = ideal.activity(mass_concentration=mass)
    # BAT model returns only organic activity (scalar)
    bat_organic_activity = float(non_ideal.activity(mass_concentration=mass))

    print(f"Organic mass fraction: {org_frac:.1f}")
    print(
        f"  Mass conc: water={mass[0] * 1e9:.1f}, "
        f"org={mass[1] * 1e9:.1f} ng/m^3"
    )
    print(
        f"  Ideal activity:     water={ideal_activity[0]:.4f}, "
        f"org={ideal_activity[1]:.4f}"
    )
    print(f"  Non-ideal (BAT):    org={bat_organic_activity:.4f}")

    # Activity coefficient = activity / mole fraction
    moles = mass / np.array([18.015e-3, molar_mass_organic])
    mole_frac = moles / np.sum(moles)
    gamma_organic = bat_organic_activity / mole_frac[1]
    print(f"  Organic activity coeff (gamma): {gamma_organic:.4f}\n")
