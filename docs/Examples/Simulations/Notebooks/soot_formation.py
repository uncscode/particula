# %% [markdown]
# Flame emissions and temperature dependence of vapor pressure and surface tension, in condensation and growth.
# Mixed with dust particles.

# %%
import copy
import matplotlib.pyplot as plt
import numpy as np
import particula as par
from tqdm import tqdm
from thermo import Chemical

# plot settings
TAILWIND = par.util.colors.TAILWIND
base_color = TAILWIND["gray"]["600"]
plt.rcParams.update(
    {
        "text.color": base_color,
        "axes.labelcolor": base_color,
        "figure.figsize": (5, 4),
        "font.size": 14,
        "axes.edgecolor": base_color,
        "axes.labelcolor": base_color,
        "xtick.color": base_color,
        "ytick.color": base_color,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    }
)

# %% [markdown]
# Search for chemicals


list_of_chemicals = [
    # 1–2 ring aromatics
    "Benzene",
    "Toluene",
    "Ethylbenzene",
    "p-Xylene",
    "Styrene",
    "Indene",
    # 3–4 ring PAHs
    "Naphthalene",
    "Acenaphthylene",
    "Acenaphthene",
    "Phenanthrene",
    "Anthracene",
    "Fluoranthene",
    "Pyrene",
    "Chrysene",
    "Benzo[a]anthracene",
    # 5–7 ring (large) PAHs
    "Benzo[b]fluoranthene",
    "Benzo[k]fluoranthene",
    "Benzo[a]pyrene",
    "Perylene",
    "Benzo[ghi]perylene",
    "Indeno[1,2,3-cd]pyrene",
    "Dibenzo[a,h]anthracene",
    "Coronene",
    # optional extra-large surrogate
    "Ovalene",
]

# Mass-fraction vector for particulate PAHs in combustion aerosol
# Order: 2–3-ring → 4-ring → 5–7-ring (large) species
initial_mass_fractions = np.array(
    [
        0.25,  # Naphthalene (2-ring)
        0.04,  # Acenaphthylene (3-ring)
        0.03,  # Acenaphthene (3-ring)
        0.08,  # Phenanthrene (3-ring)
        0.03,  # Anthracene (3-ring)
        0.06,  # Fluoranthene (4-ring)
        0.06,  # Pyrene (4-ring)
        0.03,  # Chrysene (4-ring)
        0.02,  # Benzo[a]anthracene (4-ring)
        0.03,  # Benzo[b]fluoranthene (5-ring)
        0.02,  # Benzo[k]fluoranthene (5-ring)
        0.02,  # Benzo[a]pyrene (5-ring)
        0.015,  # Perylene (5-ring)
        0.015,  # Benzo[ghi]perylene (6-ring)
        0.010,  # Indeno[1,2,3-cd]pyrene (6-ring)
        0.007,  # Dibenzo[a,h]anthracene (6-ring)
        0.005,  # Coronene (7-ring surrogate)
        0.003,  # Ovalene (optional ≥7-ring surrogate)
    ],
    dtype=np.float64,
)

density_array = np.array([])
molar_mass_array = np.array([])
surface_tension_array = np.array([])
cas_name = []
chemical_dict = {}
# Get the CAS numbers for each chemical
for chem in list_of_chemicals:
    cas = par.util.get_chemical_search(chem)
    print(f"{chem}: {cas}")
    chemical_dict[chem] = par.util.get_chemical_stp_properties(cas)
    # Store the molar mass and density
    molar_mass_array = np.append(
        molar_mass_array, chemical_dict[chem]["molar_mass"]
    )
    density_array = np.append(density_array, chemical_dict[chem]["density"])
    surface_tension_array = np.append(
        surface_tension_array, chemical_dict[chem]["surface_tension"]
    )
    cas_name.append(cas)

# Print the chemical properties
for chem, props in chemical_dict.items():
    print(f"{chem}:")
    print(f"  Molar Mass: {props['molar_mass']:.4f} kg/mol")
    print(f"  Density: {props['density']:.2f} kg/m³")
    print(f"  Surface Tension: {props['surface_tension']} N/m")
    print(f"  Vapor Pressure: {props['pure_vapor_pressure']} Pa\n")


# %%
# Reproducibility
np.random.seed(100)

# 1a. Species properties

number_of_samples = 10_000  # number of particles to sample
simulation_volume = 1e-8  # 1/m^3
temperature = 298.15  # K
temperature_range_table = np.linspace(
    200, 1200, 200
)  # temperature range for properties
vapor_pressure_strategy_list = []
surface_tension_table = np.zeros(
    (len(temperature_range_table), len(list_of_chemicals))
)
vapor_pressure_table = np.zeros(
    (len(temperature_range_table), len(list_of_chemicals))
)  # vapor pressure table for each chemical

for chem_i in list(chemical_dict.keys()):
    vapor_pressure_temp = par.util.get_chemical_vapor_pressure(
        chemical_identifier=chem_i,
        temperature=temperature_range_table,
    )
    vapor_pressure_table[:, list_of_chemicals.index(chem_i)] = (
        vapor_pressure_temp
    )
    # Vapor pressure strategies
    vapor_pressure_i = (
        par.gas.TableVaporPressureBuilder()
        .set_temperature_table(temperature_range_table, "K")
        .set_vapor_pressure_table(vapor_pressure_temp, "Pa")
        .build()
    )
    vapor_pressure_strategy_list.append(vapor_pressure_i)

    # Surface tension table
    surface_tension_table[:, list_of_chemicals.index(chem_i)] = (
        par.util.get_chemical_surface_tension(
            chemical_identifier=chem_i,
            temperature=temperature_range_table,
        )
    )
# replace nan values with 1e-6
surface_tension_table = np.nan_to_num(surface_tension_table, nan=1e-6)
surface_tension_table = np.clip(
    surface_tension_table, a_min=1e-6, a_max=None
)  # ensure no negative values

vapor_pressure_table = np.clip(vapor_pressure_table, a_min=1e-50, a_max=1e50)


# %% plot surface tension and vapor pressure
fig, ax = plt.subplots(2, 1, figsize=(8, 6), sharex=True, sharey=False)
# Plot surface tension
for i, chem in enumerate(list_of_chemicals):
    ax[0].plot(
        temperature_range_table,
        surface_tension_table[:, i],
        label=chem,
        linewidth=2,
    )
ax[0].set_ylabel("Surface Tension (N/m)")
ax[0].set_title("Surface Tension vs Temperature")
ax[0].grid()
ax[0].legend()
# Plot vapor pressure
for i, chem in enumerate(list_of_chemicals):
    ax[1].plot(
        temperature_range_table,
        vapor_pressure_table[:, i],
        label=chem,
        linewidth=2,
    )
ax[1].set_yscale("log")
ax[1].set_ylabel("Vapor Pressure (Pa)")
ax[1].set_xlabel("Temperature (K)")
ax[1].set_title("Vapor Pressure vs Temperature")
ax[1].grid()
ax[1].legend()
plt.tight_layout()
plt.show()

# %%
# Add mass to gas phase

concentration_gas = np.ones(len(list_of_chemicals)) * 1e-3  # kg/m^3

gas_species = (
    par.gas.GasSpeciesBuilder()
    .set_name(np.array(list_of_chemicals))
    .set_molar_mass(
        molar_mass_array,
        "kg/mol",
    )
    .set_partitioning(True)
    .set_vapor_pressure_strategy(vapor_pressure_strategy_list)
    .set_concentration(concentration_gas, "kg/m^3")
    .build()
)

atmosphere = (
    par.gas.AtmosphereBuilder()
    .set_more_partitioning_species(gas_species)
    .set_temperature(temperature, "K")
    .set_pressure(1, "atm")
    .build()
)

# %%

# Particle distributions
radii_seeds = par.particles.get_lognormal_sample_distribution(
    mode=np.array([5e-9]),
    geometric_standard_deviation=np.array([1.2]),
    number_of_particles=np.array([1]),
    number_of_samples=number_of_samples,
)

mass_seeds = (
    4 / 3 * np.pi * (radii_seeds**3) * density_array[-1]
)  # last chemical is the seed

mass_speciation = np.zeros((number_of_samples, len(list_of_chemicals)))
mass_speciation[:, -1] = mass_seeds  # last column is the seed mass
# Activity strategy
activity_strategy = (
    par.particles.ActivityIdealMolarBuilder()
    .set_molar_mass(molar_mass_array, "kg/mol")
    .build()
)

surface_strategy = (
    par.particles.SurfaceStrategyMassBuilder()
    .set_surface_tension(surface_tension_array, "N/m")
    .set_density(density_array, "kg/m^3")
    .set_phase_index(np.arange(len(list_of_chemicals)))
    .set_surface_tension_table(surface_tension_table, "N/m")
    .set_temperature_table(temperature_range_table, "K")
    .build()
)


# %% Create Particula particle object
resolved_masses = (
    par.particles.ResolvedParticleMassRepresentationBuilder()
    .set_distribution_strategy(par.particles.ParticleResolvedSpeciatedMass())
    .set_activity_strategy(activity_strategy)
    .set_surface_strategy(surface_strategy)
    .set_mass(mass_speciation, "kg")
    .set_density(density_array, "kg/m^3")  # density of each chemical
    .set_charge(0)  # no charge on particles
    .set_volume(simulation_volume, "m^3")
    .build()
)

# Create Aerosol object
aerosol = par.Aerosol(atmosphere=atmosphere, particles=resolved_masses)
print(aerosol)


# Plot initial size distribution
fig, ax = plt.subplots()
ax.hist(
    np.log10(resolved_masses.get_radius()), bins=50, density=False, alpha=0.5
)
ax.set_xlabel("log10(Diameter [m])")
ax.set_ylabel("Bin counts")
ax.set_title("Initial Particle Size Distribution")

# %%
# Condensation process setup
condensation_strategy = par.dynamics.CondensationIsothermal(
    molar_mass=np.array(molar_mass_array),
    diffusion_coefficient=2e-5,
    accommodation_coefficient=1,
    # skip_partitioning_indices=[],
)
condensation_process = par.dynamics.MassCondensation(condensation_strategy)

coagulation_strategy = (
    par.dynamics.BrownianCoagulationBuilder()
    .set_distribution_type("particle_resolved")
    .build()
)
coagulation_process = par.dynamics.Coagulation(coagulation_strategy)

aerosol_activated = copy.deepcopy(aerosol)
# Pre‑activation “spin‑up”
spin_up_time = 1e-5  # seconds
aerosol_activated = condensation_process.execute(
    aerosol=aerosol_activated, time_step=spin_up_time, sub_steps=1_000
)
# Print final state and water saturation

# %%
print(aerosol_activated)
aerosol_process = copy.deepcopy(aerosol_activated)
# simulation
total_time = 1  # seconds
total_steps = 200

coagulation_sub_step = 1
condensation_sub_step = 100

time_step = total_time / total_steps

time_array = np.linspace(spin_up_time, total_time, total_steps)
time_len = len(time_array)


# save bins for distribution
bins_lognormal = np.logspace(-7, -5, 50)  # radius bins from 10⁻⁸ to 10⁻² m
distribution_counts = np.zeros((time_len, len(bins_lognormal) - 1))

# temperature cooling rate
final_temperature = 200  # K
temperature_range = np.linspace(temperature, final_temperature, total_steps)
cooling_rate = (temperature - final_temperature) / total_time
print(f"Cooling rate: {cooling_rate:.2f} K/s")

# cooling process 2
aerosol_process2 = copy.deepcopy(aerosol_activated)
final_temperature = 500  # K
temperature_range2 = np.linspace(temperature, final_temperature, total_steps)
cooling_rate2 = (temperature - final_temperature) / total_time
print(f"Cooling rate 2: {cooling_rate2:.2f} K/s")


mass_concentration = np.zeros_like(time_array)
mode_diameter = np.zeros_like(time_array)
species_mass = np.zeros((time_len, len(list_of_chemicals)))
mass_concentration2 = np.zeros_like(time_array)
mode_diameter2 = np.zeros_like(time_array)
species_concentration2 = np.zeros((time_len, len(list_of_chemicals)))
species_mass2 = np.zeros((time_len, len(list_of_chemicals)))

for step, t in enumerate(
    tqdm(time_array, desc="Running Sim", mininterval=0.5)
):
    if step > 0:
        aerosol_process.atmosphere.temperature = temperature_range[step]
        aerosol_process2.atmosphere.temperature = temperature_range2[step]

        # execute condensation process, this is the slowest computationally
        # feel free to comment this out to see the effect of coagulation only
        aerosol_process = condensation_process.execute(
            aerosol=aerosol_process,
            time_step=time_step,
            sub_steps=condensation_sub_step,
        )
        aerosol_process2 = condensation_process.execute(
            aerosol=aerosol_process2,
            time_step=time_step,
            sub_steps=condensation_sub_step,
        )

        # # execute coagulation process
        # aerosol_process = coagulation_process.execute(
        #     aerosol=aerosol_process,
        #     time_step=time_step,
        #     sub_steps=coagulation_sub_step,
        # )

    # Record the size distribution at this time
    distribution_counts[step, :], edges = np.histogram(
        aerosol_process.particles.get_radius(clone=True),
        bins=bins_lognormal,
        density=False,
    )
    # Record the mass concentration
    mass_concentration[step] = (
        aerosol_process.particles.get_mass_concentration(clone=True)
    )
    # Record the mode diameter
    mode_diameter[step] = np.median(
        aerosol_process.particles.get_radius(clone=True)
    )
    # Record the species concentration
    species_temp = aerosol_process.particles.get_species_mass(clone=True)
    species_mass[step, :] = np.sum(species_temp, axis=0)
    # Record the mass concentration for the second process
    mass_concentration2[step] = (
        aerosol_process2.particles.get_mass_concentration(clone=True)
    )
    # Record the mode diameter for the second process
    mode_diameter2[step] = np.median(
        aerosol_process2.particles.get_radius(clone=True)
    )
    # Record the species concentration for the second process
    species_temp2 = aerosol_process2.particles.get_species_mass(clone=True)
    species_mass2[step, :] = np.sum(species_temp2, axis=0)

# Convert counts → number concentration (#/m³)
concentrations = distribution_counts / simulation_volume
print(aerosol_process)
# %%
fig, ax = plt.subplots(figsize=(7, 5))
# Plot the mass concentration over time
ax.plot(
    time_array,
    mass_concentration,
    color=TAILWIND["sky"]["800"],
    label=f"Fast Cooling {cooling_rate:1.0f} K/s",
    linewidth=6,
)
ax.plot(
    time_array,
    mass_concentration2,
    color=TAILWIND["sky"]["400"],
    label=f"Slow Cooling {cooling_rate2:1.0f} K/s",
    linewidth=2,
)
ax.set_xlabel("Time (s)")
ax.set_ylabel("Mass Concentration (kg/m³)")
ax.set_title("Mass Concentration Over Time")
ax.set_yscale("log")
ax.grid()
twinx = ax.twinx()
twinx.plot(
    time_array,
    temperature_range,
    color=TAILWIND["rose"]["800"],
    linestyle="--",
    linewidth=4,
    alpha=0.5,
)
twinx.plot(
    time_array,
    temperature_range2,
    color=TAILWIND["rose"]["400"],
    linestyle="--",
    linewidth=2,
    alpha=0.5,
)
twinx.set_ylabel("Temperature (K)")
twinx.tick_params(axis="y", labelcolor=TAILWIND["rose"]["500"])
ax.legend(loc="lower left")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()


fig, ax = plt.subplots(figsize=(7, 5))
ax.plot(
    time_array,
    mode_diameter,
    color=TAILWIND["sky"]["800"],
    label=f"Fast Cooling {cooling_rate:1.0f} K/s",
    linewidth=6,
)
ax.plot(
    time_array,
    mode_diameter2,
    color=TAILWIND["sky"]["400"],
    label=f"Slow Cooling {cooling_rate2:1.0f} K/s",
    linewidth=2,
)
ax.set_xlabel("Time (s)")
ax.set_ylabel("Mode Diameter (m)")
ax.set_title("Mode Diameter Over Time")
ax.set_yscale("log")
ax.grid()
twinx = ax.twinx()
twinx.plot(
    time_array,
    temperature_range,
    color=TAILWIND["rose"]["500"],
    linestyle="--",
    label="Temperature (K)",
    linewidth=4,
    alpha=0.5,
)
twinx.plot(
    time_array,
    temperature_range2,
    color=TAILWIND["rose"]["300"],
    linestyle="--",
    linewidth=2,
    alpha=0.5,
)
twinx.set_ylabel("Temperature (K)")
twinx.tick_params(axis="y", labelcolor=TAILWIND["rose"]["500"])
plt.xticks(rotation=45)
ax.legend(loc="lower left")
plt.tight_layout()
plt.show()

#  mass

mass_fraction = species_mass / np.sum(species_mass, axis=1, keepdims=True)
mass_fraction2 = species_mass2 / np.sum(species_mass2, axis=1, keepdims=True)


color_list = [
    TAILWIND["amber"]["600"],
    TAILWIND["amber"]["400"],
    TAILWIND["amber"]["800"],
    TAILWIND["sky"]["700"],
    TAILWIND["sky"]["500"],
    TAILWIND["slate"]["600"],
    TAILWIND["rose"]["900"],
]

fig, ax = plt.subplots(figsize=(8, 6))

# stackplot expects shape (n_series, n_times), so transpose
ax.stackplot(
    time_array,
    mass_fraction.T,  # each row is one species
    labels=list_of_chemicals,
    colors=color_list,
    alpha=1.0,
)

ax.set_xlabel("Time (s)")
ax.set_ylabel("Mean Mass Fraction in Particles")
# ax.set_title("Mass Fraction of Each Species Over Time")
ax.set_ylim(0, 1)
ax.grid(True)
twinx = ax.twinx()
twinx.plot(
    time_array,
    temperature_range,
    color=TAILWIND["rose"]["500"],
    linestyle="--",
    label="Temperature (K)",
    linewidth=4,
    alpha=0.5,
)
twinx.set_ylabel("Temperature (K)")
twinx.tick_params(axis="y", labelcolor=TAILWIND["rose"]["500"])
ax.legend(
    title="Species",
    loc="upper center",
    bbox_to_anchor=(0.5, -0.15),  # x=0.5 (center), y<0 pushes it below
    ncol=len(list_of_chemicals),  # lay out entries horizontally
    framealpha=0.95,
)
plt.tight_layout()
plt.show()


# %%
# fig, ax = plt.subplots(figsize=(7, 5))
# # Plot the final size distribution
# initial_distribution = aerosol.particles.get_radius(clone=True)
# activated_distribution = aerosol_activated.particles.get_radius(clone=True)
# final_distribution = aerosol_process.particles.get_radius(clone=True)
# bins = np.logspace(-9, -4, 50)  # radius bins from 10⁻⁸ to 10⁻² m
# ax.hist(
#     initial_distribution,
#     bins=bins,
#     density=False,
#     alpha=0.5,
#     color=TAILWIND["slate"]["500"],
#     label="Initial Size Distribution",
# )
# ax.hist(
#     activated_distribution,
#     bins=bins,
#     density=False,
#     alpha=0.5,
#     color=TAILWIND["sky"]["300"],
#     label="Activated Start Size Distribution",
# )
# ax.hist(
#     final_distribution,
#     bins=bins,
#     density=False,
#     alpha=0.5,
#     color=TAILWIND["sky"]["800"],
#     label="Final Size Distribution",
# )
# ax.set_xscale("log")
# ax.set_yscale("log")
# ax.set_xlabel("Particle Radius (m)")
# ax.set_ylabel("Concentration (m⁻³)")
# ax.set_title("Final Particle Size Distribution")
# ax.legend()
# # rotate x-ticks for better readability
# plt.tight_layout()
# plt.show()
