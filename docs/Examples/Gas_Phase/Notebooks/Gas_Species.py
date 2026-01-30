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
# # Gas Species Tutorial
#
# The `GasSpecies` is a class that represents a gas species in a simulation or calculation. It includes properties such as the species' name, molar mass, vapor pressure, and whether it is condensable. The class provides methods to set and retrieve these properties, ensuring that each species is fully defined and manageable within simulations.
#
# In this notebook, we will demonstrate how to use the `GasSpecies` class to create and manage gas species. This includes creating new species, setting their properties, and retrieving vapor pressure, concentration, and other properties.
#
# ## Key Classes
#
# - `GasSpecies`: Represents a gas species in a simulation or calculation.
# - `GasSpeciesBuilder`: A builder class that constructs instances of `GasSpecies` with well-defined properties.

# %%
# In Colab uncomment the following command to install particula:
# #!pip install particula[extra] --quiet
import matplotlib.pyplot as plt
import numpy as np

# Particula imports
import particula as par

# %% [markdown]
# ## Define Vapor Pressure Strategies
#
# In this section, we'll focus on defining vapor pressure strategies for gas species, specifically Butanol, Styrene, and Water, which were used in our previous examples. To streamline our analysis, we will group Butanol and Styrene into a single organic category, and consider Water separately.
#
# ### Strategy Assignment
#
# For calculating vapor pressures:
#
# - **Organics (Butanol and Styrene)**: We will utilize the Antoine equation, a widely recognized method for estimating the vapor pressure of organic compounds based on temperature.
# - **Water**: We will apply the Buck equation, which is specifically tailored to accurately calculate the vapor pressure of water across a range of temperatures.

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
# ## Using `GasSpeciesBuilder` to Construct Gas Species
#
# Now that we have defined the appropriate vapor pressure strategies for our gas species, we can proceed to construct the individual species using the `GasSpeciesBuilder`. This builder simplifies the process of defining and validating the properties of each gas species before their creation. We'll begin with Water, as it involves a straightforward application of the Buck equation.
#
# ### Building the Water Gas Species
#
# The `GasSpeciesBuilder` facilitates a structured approach to setting up a gas species. To build a Water gas species, the builder requires the following properties to be set:
#
# 1. **Name**: Identifies the species, which in this case is "Water".
# 2. **Molar Mass**: The molar mass of water, essential for calculations involving mass and moles.
# 3. **Vapor Pressure Strategy**: The specific strategy used to calculate vapor pressure; for Water, we use the Buck equation.
# 4. **Condensability**: Indicates whether the species can condense under certain atmospheric conditions. For Water, this is typically true.
# 5. **Concentration**: The initial concentration of Water in the mixture, which could vary based on the scenario.
#
# Here is how you can use the `GasSpeciesBuilder` to set up Water:

# %%
# Configure the builder with the necessary properties
water_species = (
    par.gas.GasSpeciesBuilder()
    .set_name("Water")
    .set_molar_mass(18.01528, molar_mass_units="g/mol")
    .set_vapor_pressure_strategy(water_buck)
    .set_partitioning(True)
    .set_concentration(1e2, concentration_units="ug/m^3")
    .build()
)


# molar mass in kg/mol, concentration in kg/m3

print(water_species)
print(
    f"Notice the units of the concentration are now in kg/m^3: {water_species.concentration}"
)
print(
    f"Also the units of the molar mass are now in kg/mol: {water_species.molar_mass}"
)

# %% [markdown]
# ### Building Gas Species for Organics
#
# Following Water, you can apply a similar process to build gas species for Organics like Butanol and Styrene. Each will have its set of properties based on the chemical's nature and the desired simulation context.
#
# When calling `.build()`, it checks that all required properties are set correctly, raising an error if any essential attribute is missing or improperly configured. This ensures that each `GasSpecies` instance is valid and ready usage.

# %%
# Define molar masses for organic species (Butanol and Styrene) in kilograms per mole (kg/mol).
organic_molar_mass = np.array(
    [0.074121, 104.15e-3]
)  # Molar mass for Butanol and Styrene respectively.

# List of vapor pressure strategies assigned to each organic species.
organic_vapor_pressure = [
    butanol_antione,
    styrene_antione,
]  # Using Antoine's equation for both.

# Define concentrations for each organic species in the mixture, in kilograms per cubic meter (kg/m^3).
organic_concentration = np.array(
    [2e-6, 1e-9]
)  # Concentration values for Butanol and Styrene respectively.

# Names of the organic species.
organic_names = np.array(["butanol", "styrene"])

# Using GasSpeciesBuilder to construct a GasSpecies object for organics.
# Notice how we can directly use arrays to set properties for multiple species.
organic_species = (
    par.gas.GasSpeciesBuilder()
    .set_name(organic_names)
    .set_molar_mass(organic_molar_mass, "kg/mol")
    .set_vapor_pressure_strategy(organic_vapor_pressure)
    .set_partitioning(True)
    .set_concentration(organic_concentration, "kg/m^3")
    .build()
)

# The `build()` method validates all the properties are set and returns the constructed GasSpecies object(s).
# Here, organic_species will contain the built GasSpecies instances for Butanol and Styrene.
print(organic_species)

# %% [markdown]
# ## Pure Vapor Pressures
#
# With the gas species defined, we can now calculate the pure vapor pressures of Butanol, Styrene, and Water using the respective strategies we assigned earlier. This will help us understand the vapor pressure behavior of each species individually, which is crucial for predicting their behavior in mixtures and under varying conditions.

# %%
temperature_range = np.linspace(
    273.15, 373.15, 100
)  # Temperature range from 0 to 100 degrees Celsius.

organic_pure_vapor_pressure = organic_species.get_pure_vapor_pressure(
    temperature_range
)
water_pure_vapor_pressure = water_species.get_pure_vapor_pressure(
    temperature_range
)

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
# ## Saturation Ratios
#
# Now that we have established the concentration of each gas species within the mixture, we can proceed to calculate the saturation ratio for each species. The saturation ratio is an essential parameter in determining the condensation behavior of gas species within a mixture.
#
# - **Above 1**: A saturation ratio greater than 1 indicates that the species is supersaturated and is likely to condense.
# - **Below 1**: Conversely, a saturation ratio below 1 suggests that the species will likely remain in the gas phase.
#
# ### Future Exploration
#
# In subsequent sections of this notebook series, we will delve deeper into how these saturation ratios reach equilibrium with a liquid phase, enhancing our understanding of the phase behavior under different conditions.

# %%
# Saturation ratio calculation
organic_saturation_ratio = organic_species.get_saturation_ratio(
    temperature_range
)
water_saturation_ratio = water_species.get_saturation_ratio(temperature_range)

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
# # Summary
#
# The `GasSpecies` module, along with the `GasSpeciesBuilder`, provides a robust framework for defining and managing gas species within a mixture. By assigning specific vapor pressure strategies and other essential properties, we can accurately model the behavior of individual species and their interactions in various scenarios. This module serves as a foundational component for more advanced simulations and analyses involving gas mixtures, condensation, and phase equilibrium.
#
# The next section is one more layer of abstraction, where we will define the `GasMixture` class to manage multiple gas species within a single mixture. This class will enable us to handle complex gas mixtures effectively and efficiently, paving the way particle to gas interactions.
