# %%

import numpy as np
from tqdm import tqdm
import copy
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
sulfate_density = 1.77 * 1000  # kg/m^3
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
        .set_saturation_concentration(c_sat_ug_per_m3[i], "ug/m^3")
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
    .set_vapor_pressure(1e-24, "Pa")
    .build()
)
sulfate_gas = (
    par.gas.GasSpeciesBuilder()
    .set_name("Sulfate")
    .set_molar_mass(96.06, "g/mol")
    .set_vapor_pressure_strategy(sulfate_vapor_pressure)
    .set_concentration(1e-12, "ug/m^3")
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

total_number_concentration = np.array([1e8])  # /m-3
particle_radius = np.logspace(-8, -6, 400)  # m

# create sulfate seeds
number_concentration = par.particles.get_lognormal_pmf_distribution(
    x_values=particle_radius,
    mode=np.array([75e-9]),
    geometric_standard_deviation=np.array([1.4]),
    number_of_particles=total_number_concentration,
)


# calculate mass in each bin
sulfate_volume_distribution = (
    4.0 / 3.0 * np.pi * particle_radius**3
)
sulfate_mass_distribution = sulfate_volume_distribution * sulfate_density

organic_mass_distribution = np.zeros(
    (len(sulfate_mass_distribution), len(M_gmol)), dtype=float
)

mass_distribution = np.concatenate(
    (sulfate_mass_distribution[:, np.newaxis], organic_mass_distribution),
    axis=1,
)

particle_molar_mass = np.append(sulfate_molar_mass, M_gmol)
particle_densities = np.append(
    sulfate_density, density_organics_g_cm3 * 1000
)  # kg/m^3
activity_strategies = (
    par.particles.ActivityIdealMolarBuilder()
    .set_molar_mass(particle_molar_mass, "g/mol")
    .build()
)
surface_tension = np.append(1e-10, np.ones(len(M_gmol)) * 0.03)  # N/m
surface_strategy = (
    par.particles.SurfaceStrategyVolumeBuilder()
    .set_surface_tension(surface_tension, "N/m")
    .set_density(particle_densities, "kg/m^3")
    .build()
)


particle_representation = (
    par.particles.ParticleMassRepresentationBuilder()
    .set_distribution_strategy(par.particles.SpeciatedMassMovingBin())
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
fig, ax = plt.subplots(figsize=(7, 5))
ax.plot(
    particle_representation.get_radius(),
    particle_representation.get_concentration(),
    label="Initial Number Concentration",
)
ax.plot(
    particle_radius,
    number_concentration,
)
ax.set_xscale("log")


# %%

# Condensation Coagulation process setup
condensation_strategy = par.dynamics.CondensationIsothermal(
    molar_mass=particle_molar_mass,
    diffusion_coefficient=2e-5,
    accommodation_coefficient=0.1,
)
condensation_process = par.dynamics.MassCondensation(condensation_strategy)
# coagulation process setup
coagulation_strategy = (
    par.dynamics.BrownianCoagulationBuilder()
    .set_distribution_type("discrete")
    .build()
)
coagulation_process = par.dynamics.Coagulation(coagulation_strategy)
# setup runnable sequence
process_sequence = (
    par.RunnableSequence()
    # | coagulation_process
    | condensation_process
)


# Copy aerosol and define time bins
aerosol_initial = copy.deepcopy(aerosol)
time_sim = 10  # total simulation time in seconds
total_steps = 20_000  # total sub‑steps for finer resolution
time = np.linspace(0, time_sim, time_sim)  # 1‑second resolution
bins_lognormal = np.logspace(-8, -2, 200)  # radius bins from 10⁻⁸ to 10⁻² m
distribution_concentration = np.zeros((len(time), len(bins_lognormal) - 1))
aerosol_mass = np.zeros_like(time)

# Main simulation loop
sub_steps_per_sec = int(total_steps / time_sim)
for i, t in enumerate(tqdm(time, desc="Processing")):
    if i > 0:
        aerosol = process_sequence.execute(
            aerosol=aerosol, time_step=1, sub_steps=sub_steps_per_sec
        )
    # Record the size distribution at this time
    aerosol_mass[i] = aerosol.particles.get_mass_concentration(clone=True)
print(aerosol)

sat_ratio = aerosol.atmosphere.partitioning_species.get_saturation_ratio(
    temperature_K
)
print("Saturation Ratio: ", sat_ratio)

# %%
fig, ax = plt.subplots(figsize=(7, 5))
ax.plot(
    time,
    aerosol_mass,
    label="Total Mass Concentration",
)
ax.set_xlabel("Time (s)")
ax.set_ylabel("Mass Concentration (kg/m³)")

fig, ax = plt.subplots(figsize=(7, 5))
ax.plot(
    aerosol.particles.get_radius(),
    aerosol.particles.get_concentration(), label="Final Number Concentration",
    marker="o"
)
ax.plot(
    aerosol_initial.particles.get_radius(),
    aerosol_initial.particles.get_concentration(), label="Initial Number Concentration"
)
ax.set_xscale("log")
ax.set_yscale("log")


# %%Contour plot of log10(number concentration)
fig, ax = plt.subplots(figsize=(7, 5))
X, Y = np.meshgrid(time, edges[:-1])
log_conc = np.log10(
    concentrations,
    where=concentrations > 0,
    out=np.full_like(concentrations, np.nan),
)
cont = ax.contourf(X, Y, log_conc.T)

ax.set_yscale("log")
ax.set_ylim(1e-7, 1e-5)
ax.set_xlabel("Time (s)")
ax.set_ylabel("Particle Radius (m)")
fig.colorbar(cont, label="Log₁₀ Number Concentration")
plt.tight_layout()
plt.show()
