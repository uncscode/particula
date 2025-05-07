# %%

import numpy as np

import matplotlib.pyplot as plt
import particula as par

# move to particle properties and update to API style.
from particula.activity.species_density import organic_array


# %%

M_gmol = np.array(
    [200.0, 188.0, 216.0, 368.0, 186.0, 204.0, 195.0, 368.0, 158.0, 206.0]
)

OC_ratio = np.array(
    [0.40, 0.444, 0.50, 0.368, 0.444, 0.556, 0.857, 0.368, 0.375, 0.75]
)

HC_ratio = np.array(
    [1.60, 1.78, 1.60, 1.47, 1.56, 1.78, 1.75, 1.56, 1.75, 1.75]
)


density_organics_g_cm3 = organic_array(
    molar_mass=M_gmol,
    oxygen2carbon=OC_ratio,
    hydrogen2carbon=HC_ratio,
)  # g/cm^3

c_total_ug_per_m3 = np.array(
    [8.79, 3.98, 1.13, 4.07, 0.628, 0.919, 0.766, 1.02, 0.399, 0.313]
)

name = np.array(
    [
        "C107OOH",
        "C97OOH",
        "C108OOH",
        "ALDOL_dimer_C19H28O7",
        "PINIC",
        "C921OOH",
        "C812OOH",
        "ESTER_dimer",
        "C811OH",
        "C813OOH",
    ]
)

c_sat_ug_per_m3 = np.array(
    [
        8620.171693,
        522.7659518,
        231.757194,
        2.27e-06,
        24.13243017,
        3.131375563,
        1.107025816,
        2.97e-06,
        2197.484083,
        0.04398829,
    ]
)

# not needed for the moment
c_liquid_ug_per_m3 = np.array(
    [
        0.007057093,
        0.052085067,
        0.032911505,
        4.069606969,
        0.140058292,
        0.632546668,
        0.660729371,
        1.017406513,
        0.001254946,
        0.311210297,
    ]
)


# sulfate properties
sulfate_density = 1.77*1000  # kg/m^3
sulfate_molar_mass = 96.06  # g/mol

temperature_K = 298.15

# %% create gas phase species

# vapor pressures
vapor_pressure_strategies = []
for i in range(len(name)):
    vapor_pressure_organic = (
        par.gas.SaturationConcentrationVaporPressureBuilder()
        .set_molar_mass(M_gmol[i], "g/mol")
        .set_temperature(temperature_K, "K")
        .set_saturation_concentration(
            c_sat_ug_per_m3[i], "ug/m^3"
        )
        .build()
    )
    vapor_pressure_strategies.append(vapor_pressure_organic)

organics_gases = (
    par.gas.GasSpeciesBuilder()
    .set_name(name)
    .set_molar_mass(M_gmol, "g/mol")
    .set_vapor_pressure_strategy(vapor_pressure_strategies)
    .set_concentration(c_total_ug_per_m3, "ug/m^3")
    .set_partitioning(True)
    .build()
)

# sulfate vapor pressure
sulfate_vapor_pressure = (
    par.gas.ConstantVaporPressureBuilder()
    .set_vapor_pressure(1e-20, "Pa")
    .build()
)
sulfate_gas = (
    par.gas.GasSpeciesBuilder()
    .set_name("Sulfate")
    .set_molar_mass(96.06, "g/mol")
    .set_vapor_pressure_strategy(sulfate_vapor_pressure)
    .set_concentration(0.0, "ug/m^3")
    .set_partitioning(True)
    .build()
)

# create atmosphere
atmosphere = (
    par.gas.AtmosphereBuilder()
    .set_temperature(temperature_K, "K")
    .set_pressure(1.0, "atm")
    .set_more_partitioning_species(sulfate_gas)
    .set_more_partitioning_species(organics_gases)
    .build()
)

# %% particles

total_number_concentration = np.array([1e8]) # /m-3
particle_radius = np.logspace(-8, -6, 250)  # m

# create sulfate seeds
number_concentration = (
    par.particles.get_lognormal_pmf_distribution(
        x_values=particle_radius,
        mode=np.array([75e-9]),
        geometric_standard_deviation=np.array([1.4]),
        number_of_particles=total_number_concentration,
    )
)


# calculate mass in each bin
sulfate_volume_distribution = (
    number_concentration * 4.0 / 3.0 * np.pi * particle_radius**3
)
sulfate_mass_distribution = sulfate_volume_distribution * sulfate_density

organic_mass_distribution = np.zeros((len(sulfate_mass_distribution), len(M_gmol)), dtype=float)

mass_distribution = np.concatenate(
    (sulfate_mass_distribution[:, np.newaxis], organic_mass_distribution), axis=1
)

particle_molar_mass = np.append(sulfate_molar_mass, M_gmol)
particle_densities = np.append(sulfate_density, density_organics_g_cm3 * 1000)  # kg/m^3
activity_strategies = (
    par.particles.ActivityIdealMolarBuilder()
    .set_molar_mass(particle_molar_mass, "g/mol")
    .build()
)
surface_tension = np.append(0.072, np.ones(len(M_gmol)) * 0.03)  # N/m
surface_strategy = (
    par.particles.SurfaceStrategyVolumeBuilder()
    .set_surface_tension(surface_tension, "N/m")
    .set_density(particle_densities, "kg/m^3")
    .build()
)


particle_representation = (
    par.particles.ParticleMassRepresentationBuilder()
    .set_distribution_strategy(par.particles.MassBasedMovingBin())
    .set_activity_strategy(activity_strategies)
    .set_mass(mass_distribution, "kg")
    .set_concentration(number_concentration, "1/m^3")
    .set_charge(0)
    .set_density(particle_densities, "kg/m^3")
    .set_surface_strategy(surface_strategy)
    .build()
)

# Build aerosol

aerosol = (
    par.AerosolBuilder()
    .set_atmosphere(atmosphere)
    .set_particles(particle_representation)
    .build()
)
print(aerosol)

# %%
