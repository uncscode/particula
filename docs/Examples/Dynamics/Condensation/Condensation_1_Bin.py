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
# # Condensation Tutorial: Radius Bin
#
# *Work in progress, probably split into multiple notebooks, need to find a model system to test this on*
#
# Condensation, is the first process where this framework we have been building up is applied. Here we need to account for the gas phase, and the particle phase. Then ensure that the partial pressures of species at the surface of the particle are equal to the partial pressure in the gas.
#
# **Core Concepts:**
#
# - **Runnable**: An abstract base class defining the interface for aerosol transformation processes.
#   - Here and aerosol object is passed to the process, and the process is expected to modify the aerosol object in place, returning the modified object.
# - **MassCondensation**: A concrete class implementing the RunnableProcess interface for the condensation process. Is an implementation of a `Runnable` process that adds mass to the aerosol object based on the partial pressures of the gas phase and the particle phase. Then removes the mass from the gas phase.
#

# %%
# In Colab uncomment the following command to install particula:
# #!pip install particula[extra] --quiet
import numpy as np
import matplotlib.pyplot as plt

# particula imports
import particula as par

# %% [markdown]
# ## Setup Aerosol
#
# First we will repeat the aerosol object that we have been using in the previous notebooks. This object will be passed to the `Runnable` processes `MassCondensation`, and modified in place.

# %%
# Glycerol gas
molar_mass_glycerol = 92.09382e-3  # kg/mol
parameters_clausius = {
    "latent_heat": 71.5 * molar_mass_glycerol,
    "latent_heat_units": "kJ/mol",
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

sat_factor = 0.01  # 50% of saturation concentration
glycerol_gas = (
    par.gas.GasSpeciesBuilder()
    .set_molar_mass(molar_mass_glycerol, "kg/mol")
    .set_vapor_pressure_strategy(vapor_pressure_strategy)
    .set_concentration(sat_concentration * sat_factor, "kg/m^3")
    .set_name("Glycerol")
    .set_partitioning(True)
    .build()
)

atmosphere = (
    par.gas.AtmosphereBuilder()
    .set_more_partitioning_species(glycerol_gas)
    .set_temperature(25, temperature_units="degC")
    .set_pressure(1, pressure_units="atm")
    .build()
)

# Glycerol particle distribution
bins = np.logspace(-8, -5, 500)
lognormal_rep = (
    par.particles.PresetParticleRadiusBuilder()
    .set_mode(np.array([100]), "nm")
    .set_geometric_standard_deviation(np.array([1.3]))
    .set_number_concentration(np.array([1e4]), "1/cm^3")
    .set_density(1.26, "g/cm^3")
    .set_distribution_type("pmf")
    .set_radius_bins(bins, "m")
    .build()
)

aerosol = par.Aerosol(atmosphere=atmosphere, particles=lognormal_rep)

print(aerosol)

# %% [markdown]
# ## Condensation Process (Isothermal)
#
# In code this process is implemented as a `Runnable` process. This means that the process is expected to modify the aerosol object in place, returning the modified aerosol object. This is defined in `Particle_processes.py` as the `MassCondensation` class.
#
# The `MassCondensation` class takes a `CondensationStrategy` object as an input. This object defines and evaluates the $dm_{i}/dt$ equation for the condensation process. More strategies can be added into the `condensation.py` file.
#
# For now, let's just run it for a few time steps and see what happens.
#
# Note: We have a moving bin particle representation, so we would expect all the bins to move.

# %%
# define the condensation process using the builder API (M2 migration)
condensation_isothermal = (
    par.dynamics.CondensationIsothermalBuilder()
    .set_molar_mass(molar_mass_glycerol, "kg/mol")
    .set_diffusion_coefficient(2e-5, "m^2/s")
    .set_accommodation_coefficient(0.1)
    .build()
)
condensation_process = par.dynamics.MassCondensation(
    condensation_strategy=condensation_isothermal
)

# define the time array in seconds
time_step = 0.5
time_array = np.arange(0, 10, time_step)
total_mass = np.zeros_like(time_array)

# output arrays
aerosol_sim = []

rate = condensation_process.rate(aerosol)


# convenience helper to keep plotting arrays 1D and stable
def _flatten_radius_and_conc(aerosol_obj):
    radius = np.asarray(aerosol_obj.particles.get_radius()).reshape(-1)
    concentration = np.asarray(aerosol_obj.particles.concentration).reshape(-1)
    return radius, concentration


initial_radius, initial_conc = _flatten_radius_and_conc(aerosol)

fig, ax = plt.subplots(figsize=(8, 6))
ax.semilogx(initial_radius * 1e9, initial_conc, label="Initial")

# singe step
aerosol = condensation_process.execute(aerosol, time_step)
radius_1, conc_1 = _flatten_radius_and_conc(aerosol)
ax.semilogx(radius_1 * 1e9, conc_1, label="After 1 step")

# second step
aerosol = condensation_process.execute(aerosol, time_step)
radius_2, conc_2 = _flatten_radius_and_conc(aerosol)
ax.semilogx(radius_2 * 1e9, conc_2, label="After 2 steps")

# 5th step
aerosol = condensation_process.execute(aerosol, time_step)
aerosol = condensation_process.execute(aerosol, time_step)
aerosol = condensation_process.execute(aerosol, time_step)
radius_5, conc_5 = _flatten_radius_and_conc(aerosol)
ax.semilogx(radius_5 * 1e9, conc_5, label="After 5 steps")

plt.legend()
ax.set_xlabel("Radius (nm)")
ax.set_ylabel("Concentration (1/m^3)")
plt.show()

# %% [markdown]
# ## Summary
#
# We built out the condensation process, and the equations that define the process. We also defined the inputs and outputs of the process. Next we will build out coagulation and nucleation processes, to complete the aerosol dynamics framework.
