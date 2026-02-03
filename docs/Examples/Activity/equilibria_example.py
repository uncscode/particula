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
# # Equilibria Partitioning Example
#
# Demonstrates using `LiquidVaporPartitioningStrategy` for computing
# gas-particle equilibrium concentrations.
#
# ## What This Example Shows
#
# - Creating a liquid-vapor partitioning strategy
# - Solving for equilibrium with organic species
# - Interpreting the equilibrium results
# - Observing the effect of relative humidity on partitioning

# %%
import numpy as np
import particula as par

# %% [markdown]
# ## 1. Define Organic Species Properties
#
# Set up three organic species with different volatilities (C* values).
# Lower C* means less volatile and more tendency to partition to the
# particle phase.

# %%
# Three species with different volatilities (C* values)
c_star_j_dry = np.array([1e-6, 1e-4, 1e-2])  # saturation concentrations
concentration_organic = np.array([1.0, 5.0, 10.0])  # ug/m^3
molar_mass = np.array([200.0, 200.0, 200.0])  # g/mol
o2c_ratio = np.array([0.2, 0.3, 0.5])  # O:C ratios
density = np.array([1200.0, 1200.0, 1200.0])  # kg/m^3

print("=== Organic Species Properties ===")
print(f"C* (dry): {c_star_j_dry}")
print(f"Total concentration: {concentration_organic} ug/m^3")
print(f"O:C ratios: {o2c_ratio}")

# %% [markdown]
# ## 2. Create Partitioning Strategy at 75% RH
#
# Water activity corresponds to relative humidity for aerosol systems.

# %%
strategy = par.equilibria.LiquidVaporPartitioningStrategy(
    water_activity=0.75,  # 75% relative humidity
)

print(f"Strategy: {type(strategy).__name__}")
print(f"Water activity (RH): {strategy.water_activity * 100:.0f}%")

# %% [markdown]
# ## 3. Solve for Equilibrium

# %%
result = strategy.solve(
    c_star_j_dry=c_star_j_dry,
    concentration_organic_matter=concentration_organic,
    molar_mass=molar_mass,
    oxygen2carbon=o2c_ratio,
    density=density,
)

print("Equilibrium solved successfully!")

# %% [markdown]
# ## 4. Print Equilibrium Results
#
# The result contains partition coefficients (fraction in condensed phase),
# phase concentrations, and water content.

# %%
print("=== Liquid-Vapor Partitioning ===")
print(f"Water activity (RH): {strategy.water_activity * 100:.0f}%\n")

print("Input organic species:")
print(f"  C* (dry): {c_star_j_dry}")
print(f"  Total concentration: {concentration_organic} ug/m^3")
print(f"  O:C ratios: {o2c_ratio}")

print("\n=== Equilibrium Results ===")
print(f"Partition coefficients: {result.partition_coefficients}")
print("  (fraction in condensed phase)")

print("\nAlpha phase (water-rich):")
print(f"  Species concentrations: {result.alpha_phase.species_concentrations}")
print(
    "  Water concentration: "
    f"{result.alpha_phase.water_concentration:.2f} ug/m^3"
)
print(
    "  Total concentration: "
    f"{result.alpha_phase.total_concentration:.2f} ug/m^3"
)

if result.beta_phase is not None:
    print("\nBeta phase (organic-rich):")
    print(
        f"  Species concentrations: {result.beta_phase.species_concentrations}"
    )
    print(
        "  Water concentration: "
        f"{result.beta_phase.water_concentration:.2f} ug/m^3"
    )

print(
    f"\nWater content: alpha={result.water_content[0]:.2f}, "
    f"beta={result.water_content[1]:.2f} ug/m^3"
)
print(f"Optimization error: {result.error:.2e}")

# %% [markdown]
# ## 5. Effect of RH on Partitioning
#
# As relative humidity increases, more water is absorbed and partitioning
# changes.

# %%
print("=== Effect of RH on Partitioning ===\n")
rh_values = [0.3, 0.5, 0.75, 0.9]
for rh in rh_values:
    strat = par.equilibria.LiquidVaporPartitioningStrategy(
        water_activity=rh,
    )
    res = strat.solve(
        c_star_j_dry=c_star_j_dry,
        concentration_organic_matter=concentration_organic,
        molar_mass=molar_mass,
        oxygen2carbon=o2c_ratio,
        density=density,
    )
    mean_partition = np.mean(res.partition_coefficients)
    print(
        f"RH={rh * 100:.0f}%: mean partition coeff = {mean_partition:.4f}, "
        f"water = {res.alpha_phase.water_concentration:.2f} ug/m^3"
    )
