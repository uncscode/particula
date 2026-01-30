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
# # Atmosphere Tutorial
#
# Gases, alongside particles, constitute the essential components of an aerosol system. In their natural state, gases are collections of molecules that move freely, not bound to one another. We introduce the `Atmosphere` class, a composite that encapsulates `GasSpecies`, with additional parameters for the atmospheric state.
#
# - **`Atmosphere`**: This class represents the atmospheric environment by detailing properties such as temperature and pressure, alongside a dynamic list of gas species
#     present.
# - **`AtmosphericBuilder`**: A builder class that simplifies the creation of `Atmosphere` objects.
#
# We'll continue with our organics and water example, combining the two into a single `Atmosphere` object.

# %%
# In Colab uncomment the following command to install particula:
# #!pip install particula[extra] --quiet
import matplotlib.pyplot as plt
import numpy as np

# import particula
import particula as par

# %% [markdown]
# ## Build Gas Species
#
# First we will build the, `GasSpecies` objects for the organics and water. Following the same procedure from previously in [`Gas Species`](./next_gas_species.ipynb).

# %%
# Define the coefficients for Butanol using the Antoine equation.
butanol_coefficients = {"a": 7.838, "b": 1558.19, "c": 196.881}
butanol_antione = par.gas.VaporPressureFactory().get_strategy(
    "antoine", butanol_coefficients
)
styrene_coefficients = {"a": 6.924, "b": 1420, "c": 226}
styrene_antione = par.gas.VaporPressureFactory().get_strategy(
    "antoine", styrene_coefficients
)

# Water uses a different model for vapor pressure calculation called the Buck equation.
water_buck = par.gas.VaporPressureFactory().get_strategy("water_buck")

# Create the GasSpecies using the GasSpeciesBuilder
# water species
water_species = (
    par.gas.GasSpeciesBuilder()
    .set_name("water")
    .set_molar_mass(18.01528e-3, "kg/mol")
    .set_vapor_pressure_strategy(water_buck)
    .set_partitioning(True)
    .set_concentration(1e-3, "kg/m^3")
    .build()
)

# organic species
organic_molar_mass = np.array([0.074121, 104.15e-3])
organic_vapor_pressure = [butanol_antione, styrene_antione]
organic_concentration = np.array([2e-6, 1e-9])
organic_names = np.array(["butanol", "styrene"])
organic_species = (
    par.gas.GasSpeciesBuilder()
    .set_name(organic_names)
    .set_molar_mass(organic_molar_mass, "kg/mol")
    .set_vapor_pressure_strategy(organic_vapor_pressure)
    .set_partitioning(True)
    .set_concentration(organic_concentration, "kg/m^3")
    .build()
)

# Print the species
print(water_species)
print(organic_species)

# %% [markdown]
# ## Atmosphere Builder
#
# The `AtmosphereBuilder` class is a builder class that simplifies the creation of `Atmosphere` objects. It provides a fluent interface for adding `GasSpecies` objects to the `Atmosphere` object. We will use it to build the `Atmosphere` object for the organics and water. The builder requries the following parameters:
#
# - `pressure`: The total pressure of the gas mixture, in Pascals, or provided pressure_units string for conversion.
# - `temperature`: The temperature of the gas mixture, in Kelvin, or provided temperature_units string for conversion.
# - `species`: A list of `GasSpecies` objects, representing the gases in the mixture. This can be added one by one using the `add_species` method.
#
# ### Air
#
# Air is assumed to be the non-specified component of the gas mixture, making up the remainder of the gas mixture. We do not explicitly add air to the gas mixture, but it is implicitly included in most calculations.

# %%
gas_mixture = (
    par.gas.AtmosphereBuilder()
    .set_more_partitioning_species(water_species)
    .set_more_partitioning_species(organic_species)
    .set_temperature(25, temperature_units="degC")
    .set_pressure(1, pressure_units="atm")
    .build()
)

print("Notice the units conversion to base SI:")
print(gas_mixture)

# %% [markdown]
# ## Iterating Over Gas Species
#
# Once the `Gas` object has been established, it enables us to iterate over each `GasSpecies` within the mixture. This functionality is particularly valuable for evaluating and adjusting properties dynamically, such as when changes in temperature and pressure occur due to environmental alterations.
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
# ### Application
# By iterating through each `GasSpecies`, we can apply these formulas to adjust their properties based on the calculated pressure and temperature at 10 km altitude, aiding in simulations or real-world applications where altitude plays a crucial role in gas behavior.
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


# Initialize a matrix to hold saturation ratios for each species at each
# altitude
saturation_ratio = np.zeros(len(altitude_range))

# Loop over each altitude's temperature and pressure
for index, (temperature, pressure) in enumerate(
    zip(temperature_at_altitudes, pressure_at_altitudes)
):
    # Set the current temperature and pressure of the gas mixture
    gas_mixture.temperature = temperature
    gas_mixture.total_pressure = pressure

    # Loop over water
    saturation_ratio[index] = (
        gas_mixture.partitioning_species.get_saturation_ratio(
            gas_mixture.temperature
        )[0]
    )


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
# # Summary
#
# In this notebook, we learned how to create an `Atmosphere` object using the `AtmosphereBuilder` class. We also explored how to iterate over each `GasSpecies` within the mixture, enabling us to adjust properties dynamically based on environmental changes. This functionality is particularly useful for simulating real-world scenarios where temperature and pressure variations significantly impact gas behavior.
#
# We now need to build the particle representation, so that combined with the `Atmosphere`, we can create an aerosol system.
