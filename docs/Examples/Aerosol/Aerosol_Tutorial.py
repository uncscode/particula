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
# # Aerosol Tutorial
#
# Aerosols are complex systems comprising both gaseous components and particulate matter. To accurately model such systems, we introduce the `Aerosol` class, which serves as a collection the `Atmosphere` and `ParticleRepresentation` objects.
#
# In this quick tutorial, we will demonstrate how to create an `Aerosol` object, as this is the key object that will track the state of the aerosol system during dynamics.

# %%
# In Colab uncomment the following command to install particula:
# #!pip install particula[extra] --quiet

import numpy as np
import particula as par

# %% [markdown]
# ## Gas->Atmosphere and Particles
#
# First we'll create a simple `Atmosphere` object, which will represent the gas phase of the aerosol system. We'll also create a `ParticleRepresentation` object, which will represent the particulate phase of the aerosol system.
#
# For the chemical species, we will use a pure component glycerol system.

# %%
# Glycerol gas
molar_mass_glycerol = 92.09382e-3  # kg/mol
parameters_clausius = {
    "latent_heat": 71.5 * molar_mass_glycerol,
    "latent_heat_units": "J/mol",
    "temperature_initial": 125.5,
    "temperature_initial_units": "degC",
    "pressure_initial": 1,
    "pressure_initial_units": "mmHg",
}
vapor_pressure_strategy = par.gas.VaporPressureFactory().get_strategy(
    "clausius_clapeyron", parameters_clausius
)

sat_concentration = vapor_pressure_strategy.saturation_concentration(
    molar_mass_glycerol, 298.15
)
print(f"Saturation concentration: {sat_concentration:.2e} kg/m^3")

sat_factor = 0.5  # 50% of saturation concentration
glycerol_gas = (
    par.gas.GasSpeciesBuilder()
    .set_molar_mass(molar_mass_glycerol, "kg/mol")
    .set_vapor_pressure_strategy(vapor_pressure_strategy)
    .set_concentration(sat_concentration * sat_factor, "kg/m^3")
    .set_name("Glycerol")
    .set_partitioning(True)
    .build()
)
print(glycerol_gas)

atmosphere = (
    par.gas.AtmosphereBuilder()
    .set_more_partitioning_species(glycerol_gas)
    .set_temperature(25, temperature_units="degC")
    .set_pressure(1, pressure_units="atm")
    .build()
)
print(atmosphere)

# Glycerol particle distribution
lognormal_rep = (
    par.particles.PresetParticleRadiusBuilder()
    .set_mode(np.array([100]), "nm")
    .set_geometric_standard_deviation(np.array([1.5]))
    .set_number_concentration(np.array([1e4]), "1/cm^3")
    .set_density(1.26, "g/cm^3")
    .build()
)

# %% [markdown]
# > Notice, that there are two different types of gas phase species possible. `partitioning` and `gas_only_spcies`. The `partitioning` species are the ones that will be partitioned between the gas and particle phase, while the `gas_only_species` are the ones that will only be in the gas phase.

# %% [markdown]
# ## Creating an Aerosol object
#
# With both the `Atmosphere` and `ParticleRepresentation` objects created, we can now create an `Aerosol` object. This object will contain both the gas and particle phase objects, and will be used to track the state of the aerosol system during dynamics.

# %%
aerosol = par.Aerosol(atmosphere=atmosphere, particles=lognormal_rep)

print(aerosol)

# %% [markdown]
# # Summary
#
# In this tutorial, we demonstrated how to create an `Aerosol` object, which is the key object that will track the state of the aerosol system during dynamics. It is pretty simple, as the `Aerosol` object is just a collection of the `Atmosphere` and `ParticleRepresentation` objects and only functions as a container for these objects. It can also iterate over the `Atmosphere` and `ParticleRepresentation` objects.
