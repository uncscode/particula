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
# # Gas Mixture and Environment Tutorial
#
# Native `GasData` holds all species in one explicit order. `EnvironmentData`
# holds temperature, pressure, and saturation ratios in that same gas order.
# Vapor-pressure strategies remain separate scientific configuration. See the
# [gas tutorial](Gas_Species.ipynb) for the water and organic properties.

# %%
import matplotlib.pyplot as plt
import numpy as np
import particula as par

# %%
water_buck = par.gas.VaporPressureFactory().get_strategy("water_buck")
butanol_antione = par.gas.VaporPressureFactory().get_strategy(
    "antoine", {"a": 7.838, "b": 1558.19, "c": 196.881}
)
styrene_antione = par.gas.VaporPressureFactory().get_strategy(
    "antoine", {"a": 6.924, "b": 1420, "c": 226}
)
vapor_pressure_strategies = [water_buck, butanol_antione, styrene_antione]
gas_mixture = (
    par.gas.GasDataBuilder()
    .set_names(["water", "butanol", "styrene"])
    .set_molar_mass([18.01528e-3, 0.074121, 104.15e-3], "kg/mol")
    .set_partitioning([True, True, True])
    .set_concentration([1e-3, 2e-6, 1e-9], "kg/m^3")
    .build()
)
environment = par.gas.EnvironmentData(
    temperature=np.array([298.15]),
    pressure=np.array([101325.0]),
    saturation_ratio=np.zeros((1, 3)),
)
print(gas_mixture)
print(environment)

# %% [markdown]
# ## Prescribed altitude history
#
# This diagnostic sweep holds gas mass concentrations fixed while prescribing
# temperature and pressure. It is not an expanding-parcel transport model.
# Background air sets the pressure without adding an implicit gas storage lane.
#
# ### Practical Example: Altitude Impact
#
# Consider a scenario where our gas mixture is transported from sea level to an altitude of 10 kilometers. Such a change in altitude significantly impacts both temperature and pressure, which in turn affects the behavior of each gas species in the mixture.
#
# #### Geopotential Height Equation
#
# The pressure and temperature changes with altitude can be approximated by using the geopotential height equation. Here's how you can calculate these changes:
#
# 1. **Pressure Change**: The pressure at a given altitude can be estimated by:
#
# $$
#    P = P_0 \left(1 - \frac{L \cdot h}{T_0}\right)^{\frac{g \cdot M}{R \cdot L}}
# $$
#
#    where:
#    - $ P $ is the pressure at altitude $ h $,
#    - $ P_0 $ is the reference pressure at sea level (101325 Pa),
#    - $ L $ is the standard temperature lapse rate (approximately 0.0065 K/m),
#    - $ h $ is the altitude in meters (10000 m for 10 km),
#    - $ T_0 $ is the reference temperature at sea level (288.15 K),
#    - $ g $ is the acceleration due to gravity (9.80665 m/s²),
#    - $ M $ is the molar mass of Earth's air (0.0289644 kg/mol),
#    - $ R $ is the universal gas constant (8.314 J/(mol·K)).
#
# 1. **Temperature Change**: The temperature decreases linearly with altitude at the lapse rate $ L $:
#
#    $$
#    T = T_0 - L h
#    $$
#
#    Using this formula, we can estimate the temperature at an altitude of 10 km:
#    - $T$ = 223.15 K
#    - $L$ = 0.0065 K/m
#    - $h$ = 10000 m
#

# %%
# Constants for calculations
sea_level_pressure = 101325  # Reference pressure at sea level (Pa)
sea_level_temperature = 288.15  # Reference temperature at sea level (K)
gravity = 9.80665  # Acceleration due to gravity (m/s^2)
molar_mass_air = 0.0289644  # Molar mass of Earth's air (kg/mol)
universal_gas_constant = 8.314  # Universal gas constant (J/(mol·K))
temperature_lapse_rate = 0.0065  # Standard temperature lapse rate (K/m)

# Generate an array of altitudes from sea level (0 meters) to 10 km (10000 meters), divided into 100 intervals
altitude_range = np.linspace(0, 10000, 100)

# Calculate the temperature at each altitude based on the linear temperature lapse rate
temperature_at_altitudes = (
    sea_level_temperature - temperature_lapse_rate * altitude_range
)

# Calculate the pressure at each altitude using the barometric formula
pressure_at_altitudes = sea_level_pressure * (
    (1 - temperature_lapse_rate * altitude_range / sea_level_temperature)
    ** (
        gravity
        * molar_mass_air
        / (universal_gas_constant * temperature_lapse_rate)
    )
)


# Only water is evaluated over this cold altitude range. The organic Antoine
# coefficients are not qualified here; gas storage still retains those lanes.
# Saturation-ratio lanes for organics remain their prescribed initial values.
saturation_ratio = np.zeros(len(altitude_range))
for index, (temperature, pressure) in enumerate(
    zip(temperature_at_altitudes, pressure_at_altitudes, strict=True)
):
    environment.temperature[0] = temperature
    environment.pressure[0] = pressure
    saturation = water_buck.saturation_concentration(
        gas_mixture.molar_mass[0], temperature
    )
    environment.saturation_ratio[0, 0] = (
        gas_mixture.concentration[0, 0] / saturation
    )
    saturation_ratio[index] = environment.saturation_ratio[0, 0]

# Plot the saturation ratio of water vapor at each altitude
fig, ax = plt.subplots()
ax.plot(saturation_ratio, altitude_range, label="Water")
ax.set_xscale("log")
ax.set_ylabel("Altitude (m)")
ax.set_xlabel("Water Saturation Ratio")
ax.set_title("Saturation Ratio of Water Vapor at Different Altitudes")
ax.legend()
plt.show()

# %% [markdown]
# We retain the complete gas order while updating held environment arrays.
# The water plot uses lane zero explicitly. A native Aerosol can hold these
# two containers together with particle data; processes own species mappings
# and validate their scientific requirements independently.
