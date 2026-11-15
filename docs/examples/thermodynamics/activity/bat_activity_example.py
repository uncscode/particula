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
# # Compare ideal and nonideal organic activity
#
# Does a mole fraction alone describe an organic component's tendency to leave
# a water mixture? This example compares ideal organic activity with the
# Binary Activity Thermodynamics (BAT) approximation at three compositions.
# The diagnostic γ = a/x distinguishes the activity coefficient from activity
# itself; they are not interchangeable.
#
# BAT uses reduced molecular descriptors to represent organic–water interactions.
# Its scientific basis and limitations are described by
# [Gorkowski et al. (2019)](https://acp.copernicus.org/articles/19/13383/2019/).
# This example compares model predictions, not a measured compound.

# %%
import numpy as np
import particula as par

# %% [markdown]
# ## Fix the organic descriptors
#
# The illustrative organic has molar mass 200 g mol⁻¹, O:C = 0.5, and density
# 1200 kg m⁻³. O:C is held fixed in this page; a claim about its effect would
# require another sweep. The default functional-group assumption is retained.

# %%
molar_mass_organic = 200.0e-3  # kg/mol
o2c_ratio = 0.5  # oxygen to carbon ratio
density_organic = 1200.0  # kg/m^3

print("=== Organic Properties ===")
print(f"Molar mass: {molar_mass_organic * 1e3:.1f} g/mol")
print(f"O:C ratio: {o2c_ratio}")
print(f"Density: {density_organic} kg/m^3")

# %% [markdown]
# ## Construct the binary approximation
#
# The builder records the descriptors with explicit units. `ActivityNonIdealBinary`
# expects the input order [water, organic] and returns the organic activity only.
# The underlying BAT helper can provide other quantities, but this strategy's
# scalar output must not be interpreted as a two-species activity vector.

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
# ## Use a matching ideal reference
#
# The ideal strategy uses the same organic molar mass and returns both species'
# mole fractions. Compare its organic entry with the BAT scalar. The water molar
# mass used by the BAT implementation has slightly more precision than the rounded
# 18.015 g mol⁻¹ reference here; the displayed comparison is correspondingly rounded.

# %%
ideal = par.particles.ActivityIdealMolar(
    molar_mass=np.array([18.015e-3, molar_mass_organic]),
)

print(f"Ideal strategy: {type(ideal).__name__}")

# %% [markdown]
# ## Interpret activity and γ at the same composition
#
# The organic mass fractions are 0.2, 0.5, and 0.8 at total mass 1 µg m⁻³.
# The saved γ values are approximately 1.34, 0.824, and 0.752: departure from the
# ideal reference changes with composition and need not always increase activity.
# A value γ > 1 does not by itself imply a > 1, because activity is γ times mole
# fraction. No phase-equilibrium or mass-transfer calculation is performed here.

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
        f"org={mass[1] * 1e9:.1f} ug/m^3"
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

# %% [markdown]
# ## Ask whether nonideality changes your conclusion
#
# For these descriptors, ideality changes the organic activity in a
# composition-dependent way. The comparison identifies a model sensitivity,
# not which prediction is correct for a specific compound.
#
# To extend it, vary O:C while keeping molar mass, functional-group assumptions,
# and composition fixed, then compare predicted equilibrium partial pressures.
# Check the chosen descriptor range against the model's domain and independent
# data. The [activity comparison](activity_tutorial.py) includes such a sweep;
# [partitioning](equilibria_example.py) connects thermodynamics to an equilibrium
# inventory rather than just a component activity.
