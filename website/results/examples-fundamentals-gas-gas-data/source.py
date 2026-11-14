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
# # Native Gas Species Tutorial
#
# `GasData` stores ordered names, molar masses, concentrations and partitioning
# flags. Vapor-pressure strategies are separate scientific configuration.
# This example retains the water, butanol and styrene temperature sweeps.
# The organic coefficients are illustrative inputs retained from this tutorial;
# validate their temperature convention and applicability before research use.

# %%
import matplotlib.pyplot as plt
import numpy as np
import particula as par

# %% [markdown]
# ## Define vapor-pressure strategies
#
# Organics use Antoine; water uses Buck. These strategies calculate pressures
# independently of storage and can later be supplied to a process.

# %%
# Define the coefficients for Butanol using the Antoine equation.
# 'a', 'b', and 'c' are coefficients specific to the Antoine equation used to calculate vapor pressure.
butanol_coefficients = {"a": 7.838, "b": 1558.19, "c": 196.881}
# Create a vapor pressure strategy for Butanol using the Antoine equation.
butanol_antione = par.gas.VaporPressureFactory().get_strategy(
    strategy_type="antoine", parameters=butanol_coefficients
)

# Define the coefficients for Styrene, similar to Butanol, using the
# Antoine equation.
styrene_coefficients = {"a": 6.924, "b": 1420, "c": 226}
# Create a vapor pressure strategy for Styrene using the Antoine equation.
styrene_antione = par.gas.VaporPressureFactory().get_strategy(
    strategy_type="antoine", parameters=styrene_coefficients
)

# Water uses a different model for vapor pressure calculation called the Buck equation.
# The Buck equation is particularly suited for water vapor calculations.
# No additional parameters are required to be passed for the Buck equation
# in this instance.
water_buck = par.gas.VaporPressureFactory().get_strategy(
    strategy_type="water_buck"
)

# %% [markdown]
# ## Construct native gas data
#
# The water builder converts g/mol and micrograms/m^3 into SI. Concentration
# has shape (boxes, species); molar mass and flags have one entry per species.

# %%
water_species = (
    par.gas.GasDataBuilder()
    .set_names(["Water"])
    .set_molar_mass([18.01528], "g/mol")
    .set_partitioning([True])
    .set_concentration([1e2], "ug/m^3")
    .build()
)
organic_names = ["butanol", "styrene"]
organic_vapor_pressure = [butanol_antione, styrene_antione]
organic_species = (
    par.gas.GasDataBuilder()
    .set_names(organic_names)
    .set_molar_mass([0.074121, 104.15e-3], "kg/mol")
    .set_partitioning([True, True])
    .set_concentration([2e-6, 1e-9], "kg/m^3")
    .build()
)
print(water_species)
print(organic_species)

# %% [markdown]
# ## Pure vapor pressures
#
# Evaluate each strategy in the same explicit order as the gas data.

# %%
temperature_range = np.linspace(273.15, 373.15, 100)
organic_pure_vapor_pressure = np.stack(
    [
        strategy.pure_vapor_pressure(temperature_range)
        for strategy in organic_vapor_pressure
    ]
)
water_pure_vapor_pressure = water_buck.pure_vapor_pressure(temperature_range)

# Plotting the vapor pressure curves for the organic species.
fig, ax = plt.subplots(figsize=(8, 6))
for i in range(len(organic_names)):
    ax.plot(
        temperature_range,
        organic_pure_vapor_pressure[i],
        label=organic_names[i],
    )
ax.plot(temperature_range, water_pure_vapor_pressure, label="Water")
ax.set_xlabel("Temperature (K)")
ax.set_ylabel("Vapor Pressure (Pa)")
ax.set_yscale("log")
ax.legend()
plt.show()

# %% [markdown]
# ## Saturation ratios
#
# Saturation ratio is partial pressure divided by pure vapor pressure.
# Values above one indicate supersaturation relative to the pure substance;
# particle composition and curvature also affect actual condensation.

# %%
organic_saturation_ratio = np.stack(
    [
        strategy.saturation_ratio(
            organic_species.concentration[0, lane],
            organic_species.molar_mass[lane],
            temperature_range,
        )
        for lane, strategy in enumerate(organic_vapor_pressure)
    ]
)
water_saturation_ratio = water_buck.saturation_ratio(
    water_species.concentration[0, 0],
    water_species.molar_mass[0],
    temperature_range,
)

# Plotting the saturation ratio curves for the organic species.
fig, ax = plt.subplots(figsize=(8, 6))
for i in range(len(organic_names)):
    ax.plot(
        temperature_range, organic_saturation_ratio[i], label=organic_names[i]
    )
ax.plot(temperature_range, water_saturation_ratio, label="Water")
ax.set_ylim(0, 5)
ax.set_xlabel("Temperature (K)")
ax.set_ylabel("Saturation Ratio")
ax.legend()
plt.show()

# %% [markdown]
# Gas data owns storage; strategies own thermodynamics. The next tutorial
# combines the ordered gas lanes with a separately held environment.
