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
# # Connect water content to hygroscopicity
#
# At the same water content, how does hygroscopicity change the equilibrium
# water activity? This example first varies the water mass fraction of a
# water–ammonium-sulfate parameterization, then varies κ in a separate generic
# solute system. It lets you distinguish composition sensitivity from parameter
# sensitivity before coupling activity to particle growth.
#
# The κ relation used here is a_w = V_w / (V_w + κ V_s). This is the composition
# contribution to equilibrium; curvature and ambient vapor are not included in
# these activity-only calls.

# %%
import numpy as np
import particula as par

# %% [markdown]
# ## State the chosen salt approximation
#
# The first system uses κ = 0.61, density 1770 kg m⁻³, and molar mass 132.14 g mol⁻¹
# for ammonium sulfate. The chosen κ corresponds to a CCN-derived value summarized
# by [Petters and Kreidenweis (2007)](https://acp.copernicus.org/articles/7/1961/2007/);
# it is not a universal constant across concentration and phase state.
# The water lane's κ is unused in forming the dry-solute κ; setting it to zero
# does not mean that pure water has zero activity.

# %%
# Define species properties for water and ammonium sulfate
# kappa=0 for water (not hygroscopic itself)
# kappa=0.61 for ammonium sulfate (highly hygroscopic)
kappa_values = np.array([0.0, 0.61])
densities = np.array([1000.0, 1770.0])  # kg/m^3
molar_masses = np.array([18.015e-3, 132.14e-3])  # kg/mol

print("=== Species Properties ===")
print("Species: water, ammonium sulfate")
print(f"Kappa values: {kappa_values}")
print(f"Densities: {densities} kg/m^3")
print(f"Molar masses: {molar_masses * 1e3} g/mol")

# %% [markdown]
# ## Identify the water component explicitly
#
# `water_index=0` selects the component whose activity is replaced by the κ model.
# In this implementation, the other component's returned activity remains its
# mole fraction. The printed “salt activity” is therefore not a full electrolyte
# activity or dissociation calculation.

# %%
strategy = par.particles.ActivityKappaParameter(
    kappa=kappa_values,
    density=densities,
    molar_mass=molar_masses,
    water_index=0,
)

print(f"Strategy: {type(strategy).__name__}")

# %% [markdown]
# ## Dilute the solute with more water
#
# The total mass concentration remains 1 µg m⁻³ while water mass fraction rises
# from 0.3 to 0.9. Densities convert these mass fractions to volumes. The saved
# water activity rises from about 0.5543 to 0.9631 as the solute is diluted.
# These are prescribed compositions; no humidity-controlled growth is solved.

# %%
print("=== Kappa Parameter Activity ===")
print(f"Kappa values: {kappa_values}")
print("Species: water (kappa=0), ammonium sulfate (kappa=0.61)\n")

water_fractions = [0.3, 0.5, 0.7, 0.9]
for water_frac in water_fractions:
    # Mass concentrations (water fraction, salt fraction)
    mass = np.array([water_frac, 1.0 - water_frac]) * 1e-9  # kg/m^3
    activity = strategy.activity(mass_concentration=mass)
    print(f"Water mass fraction: {water_frac:.1f}")
    print(f"  Water activity: {activity[0]:.4f}")
    print(f"  Salt activity:  {activity[1]:.4f}\n")

# %% [markdown]
# ## Change κ while holding a different composition model fixed
#
# This second sweep uses a generic solute density of 1500 kg m⁻³ and molar mass
# 100 g mol⁻¹, with 50% water by mass throughout. It is not the same salt setup
# as above. Increasing κ from zero to one lowers water activity from 1 to 0.6
# for this fixed volume ratio. Compare points within this sweep to isolate κ;
# do not attribute cross-sweep differences solely to κ.

# %%
print("=== Effect of Kappa on Water Activity ===")
print("(at 50% water by mass)\n")

# Test different kappa values
kappa_test_values = [0.0, 0.1, 0.3, 0.61, 1.0]
for kappa in kappa_test_values:
    test_strategy = par.particles.ActivityKappaParameter(
        kappa=np.array([0.0, kappa]),
        density=np.array([1000.0, 1500.0]),
        molar_mass=np.array([18.015e-3, 100.0e-3]),
        water_index=0,
    )
    mass = np.array([0.5, 0.5]) * 1e-9
    activity = test_strategy.activity(mass_concentration=mass)
    print(f"kappa={kappa:.2f}: water activity = {activity[0]:.4f}")

# %% [markdown]
# ## Use the relation to pose a growth question
#
# A more hygroscopic solute lowers water activity at fixed water-to-solute volume
# ratio. Under a maintained ambient humidity, that can support greater uptake
# before equilibrium is reached, but this page has not simulated that uptake.
#
# To study a material, choose κ and density supported for its phase and conditions,
# solve for water content at specified humidity, and include curvature when needed.
# Compare the resulting growth factor with data or another activity model. The
# [finite-vapor condensation example](../../processes/condensation/condensation_3_massresolved.py)
# adds competition for a limited reservoir, where initial humidity is not maintained.
