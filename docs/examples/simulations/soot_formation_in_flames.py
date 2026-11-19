# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.17.3
#   kernelspec:
#     display_name: particula
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Follow organic uptake onto seeds during imposed cooling
#
# How can cooling alter gas–particle partitioning while collisions change particle
# size? This example follows 24 organic species around a supplied seed population,
# applying condensation and Brownian coagulation under a prescribed temperature
# history. It connects volatility assumptions to mass and size diagnostics.
#
# Despite the historical filename, this is **not a soot-formation chemistry
# model**. There is no nucleation, reaction network, carbonization, oxidation,
# or aggregate morphology. Fullerene-labeled seeds already exist at the start.
# Material tables and cooling are illustrative, not a calibrated flame or plume.
# Read [organic partitioning](../thermodynamics/equilibria/equilibria_part1.py)
# and [resolved coagulation](../processes/coagulation/coagulation_3_particle_resolved_pattern.py)
# for the separate mechanisms.

# %%
# 🚧 Install Particula if you're running this in Google Colab
# Remove the comment below to enable installation
# Install from the repository root: `python -m pip install -e ".[extra]"`

# 📦 Import necessary libraries
import copy  # to safely duplicate Python objects
from typing import Union

import matplotlib.pyplot as plt  # for plotting
import numpy as np  # numerical tools
import particula as par  # the main Particula package
from matplotlib.colors import LogNorm
from numpy.typing import NDArray
from particula import plotting
from particula.dynamics.condensation.species_mapping import (
    CondensationSpeciesMap,
)
from tqdm import tqdm  # optional: for progress bars during simulation

plt.style.use("particula.default")

# 🎨 Set default plot styling using Tailwind-inspired colors from Particula
TAILWIND = par.util.colors.TAILWIND


# 🌌 Define the colors for the chemical species involved in the simulation
color_list = (
    # 1–2 ring aromatics (amber)
    [
        TAILWIND["amber"][shade]
        for shade in ["400", "500", "600", "700", "800", "900"]
    ]
    +
    # 3–4 ring PAHs (sky)
    [
        TAILWIND["sky"][shade]
        for shade in [
            "300",
            "400",
            "500",
            "600",
            "700",
            "800",
            "900",
            "200",
        ]
    ]
    +
    # 5–7 ring PAHs (rose)
    [
        TAILWIND["rose"][shade]
        for shade in [
            "300",
            "400",
            "500",
            "600",
            "700",
            "800",
            "900",
            "200",
        ]
    ]
    +
    # Extra-large PAH surrogate (slate)
    [TAILWIND["slate"]["800"]]
    +
    # Fullerene (indigo)
    [TAILWIND["indigo"]["700"]]
)

# %% [markdown]
# ## Define the species and audit the property lookup
#
# The list spans small aromatics, PAHs, and a fullerene seed label. Supplied gas
# mass fractions are normalized to one; they are not tied here to a measured
# emission profile. The utility resolves names before retrieving properties.
# A returned numerical value does not establish a correct identity or validity
# at flame temperatures; inspect the resolved identifiers and property methods
# before using this as a material model.
#
# Molar masses are kg/mol and densities kg/m³. The displayed standard-condition
# properties initialize part of the model; they do not mean all properties are
# updated consistently during cooling.

# %%
# Define species list grouped by volatility
list_of_chemicals = [
    # 1–2-ring aromatics (gas-phase precursors, ~65%)
    "Benzene",
    "Toluene",
    "Ethylbenzene",
    "p-Xylene",
    "Styrene",
    "Indene",
    # 2–4-ring PAHs (semi-volatile, ~35%)
    "Naphthalene",
    "Acenaphthylene",
    "Acenaphthene",
    "Anthracene",
    "Fluoranthene",
    "Pyrene",
    "Chrysene",
    "Benzo[a]anthracene",
    # 5–7-ring PAHs (low volatility)
    "Benzo[b]fluoranthene",
    "Benzo[k]fluoranthene",
    "Benzo[a]pyrene",
    "Perylene",
    "Benzo[ghi]perylene",
    "Indeno[1,2,3-cd]pyrene",
    "Dibenzo[a,h]anthracene",
    "Coronene",
    # Optional ≥7-ring surrogate and soot seed
    "Ovalene",
    "Fullerene",
]

# Initial mass fractions (normalized)
initial_mass_fractions = np.array(
    [
        # ------- 1–2-ring aromatics (~65%)
        0.20,
        0.15,
        0.10,
        0.10,
        0.05,
        0.05,
        # ------- 2–4-ring PAHs (~35%)
        0.12068966,
        0.01931034,
        0.01448276,
        0.01448276,
        0.02896552,
        0.02896552,
        0.01448276,
        0.00965517,
        # ------- 5–7-ring PAHs
        0.01448276,
        0.00965517,
        0.00965517,
        0.00724138,
        0.00724138,
        0.00482759,
        0.00337931,
        0.00241379,
        0.00144828,
        0.00072414,
    ],
    dtype=np.float64,
)

# Normalize to sum to 1
initial_mass_fractions /= np.sum(initial_mass_fractions)

# Initialize property arrays
density_array = np.array([])
molar_mass_array = np.array([])
surface_tension_array = np.array([])
cas_name = []
chemical_dict = {}

# Retrieve and store properties
for chem in list_of_chemicals:
    cas = par.util.get_chemical_search(chem)
    print(f"{chem}: {cas}")
    chemical_dict[chem] = par.util.get_chemical_stp_properties(cas)

    molar_mass_array = np.append(
        molar_mass_array, chemical_dict[chem]["molar_mass"]
    )
    density_array = np.append(density_array, chemical_dict[chem]["density"])
    surface_tension_array = np.append(
        surface_tension_array, chemical_dict[chem]["surface_tension"]
    )
    cas_name.append(cas)

# Display property summary
for chem, props in chemical_dict.items():
    print(f"{chem}:")
    print(f"  Molar Mass: {props['molar_mass']:.4f} kg/mol")
    print(f"  Density: {props['density']:.2f} kg/m³")
    print(f"  Surface Tension: {props['surface_tension']} N/m")
    print(f"  Vapor Pressure: {props['pure_vapor_pressure']} Pa\n")

# %% [markdown]
# ## Build temperature-dependent tables and expose their limits
#
# There are 5000 sampled particles at 10⁷ cm⁻³, represented in 5×10⁻¹⁰ m³.
# The total initial organic gas concentration is 5×10⁻⁵ kg/m³. These chosen
# conditions are not qualified as a particular combustion source.
#
# Vapor-pressure and surface-tension tables use 200 points from 200–1500 K.
# Surface tensions replace NaNs and impose a 0.02 N/m floor. Those numerical
# substitutions are model assumptions, not measurements at high temperature.
# The separately plotted vapor-pressure array is clipped after the strategies
# are constructed; that display operation should not be read as proof of the
# strategies' underlying property validity. The charts help locate strong trends
# and questionable extrapolations; they do not establish chemical stability.

# %%
# Reproducible initial sampling. Coagulation owns a separate random generator;
# trajectories are stochastic and are not asserted equal across runs.
np.random.seed(100)

# Simulation parameters
number_of_samples = 5_000  # Simulated particles
total_number_per_cm3 = 10_000_000  # #/cm³
simulation_volume = number_of_samples / total_number_per_cm3 * 1e-6  # m³

# Initial gas phase loading and temperature
total_mass_gas_phase = 5e-5  # kg/m³
temperature = 1200  # Initial gas temperature [K]

# Temperature range for property tables
temperature_range_table = np.linspace(200, 1500, 200)

# Preallocate property tables
vapor_pressure_strategy_list = []
surface_tension_table = np.zeros(
    (len(temperature_range_table), len(list_of_chemicals))
)
vapor_pressure_table = np.zeros(
    (len(temperature_range_table), len(list_of_chemicals))
)

# Loop through chemicals to build property tables
for chem_i in list(chemical_dict.keys()):
    idx = list_of_chemicals.index(chem_i)

    # Vapor pressure vs temperature
    vapor_pressure_temp = par.util.get_chemical_vapor_pressure(
        chemical_identifier=chem_i,
        temperature=temperature_range_table,
    )
    vapor_pressure_table[:, idx] = vapor_pressure_temp

    vapor_pressure_strategy = (
        par.gas.TableVaporPressureBuilder()
        .set_temperature_table(temperature_range_table, "K")
        .set_vapor_pressure_table(vapor_pressure_temp, "Pa")
        .build()
    )
    vapor_pressure_strategy_list.append(vapor_pressure_strategy)

    # Surface tension vs temperature
    surface_tension_temp = par.util.get_chemical_surface_tension(
        chemical_identifier=chem_i,
        temperature=temperature_range_table,
    )
    surface_tension_table[:, idx] = surface_tension_temp

# Clean up NaNs or extreme values
surface_tension_table = np.nan_to_num(surface_tension_table, nan=2e-2)
surface_tension_table = np.clip(surface_tension_table, a_min=2e-2, a_max=None)
vapor_pressure_table = np.clip(vapor_pressure_table, a_min=1e-50, a_max=1e100)

# 🔍 Plot property tables
fig, ax = plotting.subplots(2, 1, sharex=True, size="wide")

for i, chem in enumerate(list_of_chemicals):
    ax[0].plot(
        temperature_range_table,
        surface_tension_table[:, i],
        label=chem,
        linewidth=2,
        color=color_list[i],
    )
    ax[1].plot(
        temperature_range_table,
        vapor_pressure_table[:, i],
        label=chem,
        linewidth=2,
        color=color_list[i],
    )

ax[0].set_ylabel("Surface Tension (N/m)")
ax[0].set_title("Surface Tension vs Temperature")
ax[0].grid(True, axis="y")

ax[1].set_yscale("log")
ax[1].set_ylabel("Vapor Pressure (Pa)")
ax[1].set_xlabel("Temperature (K)")
ax[1].set_title("Vapor Pressure vs Temperature")
ax[1].grid(True, axis="y")

# Combined legend
fig.legend(list_of_chemicals, title="Chemicals", loc="outside right center")
plotting.fit(fig)
plt.show()

# %% [markdown]
# ## Put the available material into a closed gas inventory
#
# Each normalized fraction multiplies the total gas mass concentration. Native
# `GasData` stores that ordered inventory, while `EnvironmentData` stores pressure
# and temperature. The initial 1200 K value is replaced by the forcing below.
# The parcel volume and pressure remain fixed: prescribed cooling here does not
# also simulate expansion, entrainment, or dilution of material.

# %%
# Compute gas-phase concentration [kg/m³]
concentration_gas = initial_mass_fractions * total_mass_gas_phase

# Initialize gas-phase species with vapor pressure strategy
gas_species = par.gas.GasData(
    name=list_of_chemicals,
    molar_mass=molar_mass_array.copy(),
    partitioning=np.ones(len(list_of_chemicals), dtype=bool),
    concentration=concentration_gas[None, :].copy(),
)
environment = par.gas.EnvironmentData(
    temperature=np.array([float(temperature)]),
    pressure=np.array([101325.0]),
    saturation_ratio=np.zeros((1, len(list_of_chemicals))),
)

# %% [markdown]
# ## Supply the condensation sites rather than create them
#
# Two lognormal seed-radius modes at 1 and 5 nm, both with geometric standard
# deviation 1.8, populate the resolved sample. All initial particle mass occupies
# the final, fullerene-labeled lane. The initial histogram counts particles per
# log-radius bin; it does not show a nucleation rate.
#
# Ideal molar activity and a mass-weighted surface treatment are imposed. The
# seed has a separate phase label from the other organics. At the smallest sizes,
# continuum surface and transport assumptions themselves need scrutiny. The
# species-inventory helper includes gas mass times represented volume, so later
# closed material comparisons use kg per species.

# %%
# Generate bi-modal seed particle distribution [m]
radii_seeds = par.particles.get_lognormal_sample_distribution(
    mode=np.array([1e-9, 5e-9]),  # 1 nm and 5 nm modes
    geometric_standard_deviation=np.array([1.8, 1.8]),
    number_of_particles=np.array([1, 0.5]),  # relative mode strength
    number_of_samples=number_of_samples,
)

# Calculate seed mass assuming all are pure fullerene
mass_seeds = (4 / 3) * np.pi * (radii_seeds**3) * density_array[-1]

# Initialize mass speciation matrix: all mass in fullerene (last index)
mass_speciation = np.zeros((number_of_samples, len(list_of_chemicals)))
mass_speciation[:, -1] = mass_seeds

# Activity strategy (ideal molar)
activity_strategy = (
    par.particles.ActivityIdealMolarBuilder()
    .set_molar_mass(molar_mass_array, "kg/mol")
    .build()
)

# Phase indices: all condensed except the seed
phases = np.ones_like(density_array)
phases[-1] = 0  # last species (fullerene) is seed, not condensing

# Surface tension strategy (temperature-dependent)
surface_strategy = (
    par.particles.SurfaceStrategyMassBuilder()
    .set_surface_tension(surface_tension_array, "N/m")
    .set_density(density_array, "kg/m^3")
    .set_phase_index(phases)
    .set_surface_tension_table(surface_tension_table, "N/m")
    .set_temperature_table(temperature_range_table, "K")
    .build()
)

# Build the particle representation
resolved_masses = par.particles.create_resolved_particle_data(
    masses=mass_speciation,
    density=density_array,
    volume=simulation_volume,
)
aerosol = par.Aerosol(
    particles=resolved_masses, gas=gas_species, environment=environment
)


def species_inventory(aero):
    """Independent extensive species accounting including gas, kg."""
    return np.sum(
        aero.particles.masses * aero.particles.concentration[..., None], axis=1
    ) + (aero.gas.concentration * aero.particles.volume[:, None])


initial_inventory = species_inventory(aerosol)
print(aerosol)

# 📊 Plot the initial size distribution
fig, ax = plotting.subplots()
ax.hist(np.log10(resolved_masses.radii[0]), bins=50, alpha=0.5)
ax.set_xlabel("log10(Radius [m])")
ax.set_ylabel("Bin counts")
ax.set_title("Initial Particle Size Distribution")
plotting.fit(fig)
plt.show()

# %% [markdown]
# ## Apply mass transfer and collisions as separate operations
#
# Condensation uses a common diffusivity 2×10⁻⁵ m²/s and accommodation one.
# The fullerene lane (index 23) is excluded from partitioning, so its gas and
# particle inventories do not exchange through that process. Brownian coagulation
# can merge the existing resolved particles. Neither operation creates new
# chemical species or new seed particles.

# %%
# 🌀 Condensation strategy setup (isothermal)
condensation_strategy = par.dynamics.CondensationIsothermal(
    molar_mass=np.array(molar_mass_array),
    activity_strategy=activity_strategy,
    surface_strategy=surface_strategy,
    vapor_pressure_strategy=vapor_pressure_strategy_list,
    species_map=CondensationSpeciesMap(
        tuple(gas_species.name),
        tuple((i, i) for i in range(len(list_of_chemicals))),
    ),
    diffusion_coefficient=2e-5,  # m²/s
    accommodation_coefficient=1.0,  # perfect uptake
    skip_partitioning_indices=[23],  # skip "Fullerene"
)

# Wrap into condensation process object
condensation_process = par.dynamics.MassCondensation(condensation_strategy)

# ⚫ Coagulation strategy (Brownian motion)
coagulation_strategy = (
    par.dynamics.BrownianCoagulationBuilder()
    .set_distribution_type("particle_resolved")
    .build()
)

# Wrap into coagulation process object
coagulation_process = par.dynamics.Coagulation(coagulation_strategy)


# %% [markdown]
# ## Read the cooling equation before naming its temperatures
#
# The two exponential contributions add:
#
# $$T(t)=T_\infty+(T_0-T_\infty)e^{-t/\tau_1}
#  +(T_\mathrm{mix}-T_\infty)e^{-t/\tau_2}.$$
#
# Thus $T(0)=T_0+T_\mathrm{mix}-T_\infty$, not $T_0$. With the values used
# below, the initial stored state is 1600 K. The two terms impose timescales;
# they do not solve entrainment or an energy balance. The vapor-pressure table
# ends at 1500 K and its strategy clamps outside the table domain. This must be
# considered when changing the forcing or time grid.

# %%
# %% Temperature cooling setup


def temperature_profile(
    time: Union[float, NDArray[np.float64]],
    initial_temperature: float = 1200.0,
    intermediate_temperature: float = 700.0,
    ambient_temperature: float = 298.0,
    fast_e_folding_time: float = 0.05,
    slow_e_folding_time: float = 4.0,
) -> Union[float, NDArray[np.float64]]:
    """Compute a two-stage exponential cooling profile [K].

    Arguments:
        time: Time after emission [s].
        initial_temperature: Fast-component reference temperature [K].
        intermediate_temperature: Slow-component reference temperature [K].
        ambient_temperature: Ambient (final) temperature [K].
        fast_e_folding_time: Fast cooling e-folding time constant [s].
        slow_e_folding_time: Slow cooling e-folding time constant [s].

    Returns:
        Temperature as a function of time [K], with two exponential
        decay components toward the ambient temperature.
    """
    t_array = np.asanyarray(time, dtype=float)
    fast_decay = (initial_temperature - ambient_temperature) * np.exp(
        -t_array / fast_e_folding_time
    )
    slow_decay = (intermediate_temperature - ambient_temperature) * np.exp(
        -t_array / slow_e_folding_time
    )
    return ambient_temperature + fast_decay + slow_decay


# %% [markdown]
# ## Resolve the forcing separately from process substeps
#
# Store 500 states from 0 to 19.96 s, with 499 updates of 0.04 s. The forcing
# uses fast and slow time constants 0.05 and 6 s and approaches 300 K. The
# first state is recorded at 1600 K without a process advance; subsequent updates
# use their newly assigned temperatures.
#
# Each condensation call uses 100 substeps, but temperature is held at the
# outer-step value during that call. Increasing condensation substeps alone does
# not refine the imposed cooling history. The plotted temperature is an input,
# not evidence of simulated thermal feedback.

# %%
# Show initial aerosol state
print(aerosol)

# Make a copy to apply dynamic updates
aerosol_process = copy.deepcopy(aerosol)

# Simulation timing
total_time = 20  # seconds
total_steps = 500

condensation_sub_step = 100
coagulation_sub_step = 1

time_step = total_time / total_steps
time_array = np.arange(total_steps) * time_step  # recorded 0..19.96 s
# Alternative: use logspace for finer early-time resolution
# time_array = np.logspace(-6, np.log10(total_time), total_steps)

time_len = len(time_array)

# Temperature profile over time
temperature_array = temperature_profile(
    time_array,
    initial_temperature=1200.0,
    intermediate_temperature=700.0,
    ambient_temperature=300.0,
    fast_e_folding_time=0.05,
    slow_e_folding_time=6.0,
)

# Set initial temperature in atmosphere
aerosol_process.environment.temperature[0] = temperature_array[0]

# Plot the cooling profile
fig, ax = plotting.subplots(size="wide")
ax.plot(
    time_array,
    temperature_array,
    color=TAILWIND["rose"]["500"],
    label="Temperature Profile",
    linewidth=4,
    linestyle="--",
)
ax.set_xlabel("Time (s)")
ax.set_ylabel("Temperature (K)")
ax.set_title("Flame Emission Temperature Profile")
ax.grid(True, axis="y")
plotting.fit(fig)
plt.show()

# %% [markdown]
# ## Compare material accounting with evolving size
#
# For each update, set temperature, execute condensation, then execute one
# Brownian coagulation step. Record radius-bin counts, total particle mass
# density, particle species masses, and the mean positive radius. The variable
# `mode_diameter` actually stores that mean radius; plotting doubles it to produce
# a mean diameter, not a modal diameter.
#
# Every stored state checks the closed species inventories at rtol=10⁻¹² and
# atol=10⁻³⁰ kg. This is a stringent accounting check, but does not establish
# outer-step, stochastic, or property-table convergence.

# %%
# Set up histogram bins for radius distribution
bins_lognormal = np.logspace(-10, -6, 200)  # 0.1 nm to 1 μm
distribution_counts = np.zeros((time_len, len(bins_lognormal) - 1))

# Time series storage
mass_concentration = np.zeros_like(time_array)  # kg/m³
mode_diameter = np.zeros_like(time_array)  # m
species_mass = np.zeros((time_len, len(list_of_chemicals)))  # kg

# Simulation loop
for step, _t in enumerate(
    tqdm(time_array, desc="Running Sim", mininterval=0.5)
):
    if step > 0:
        # Update temperature
        aerosol_process.environment.temperature[0] = temperature_array[step]

        # --- Condensation process (optional toggle for speed) ---
        aerosol_process = condensation_process.execute(
            aerosol=aerosol_process,
            time_step=time_step,
            sub_steps=condensation_sub_step,
        )

        # --- Coagulation process ---
        aerosol_process = coagulation_process.execute(
            aerosol=aerosol_process,
            time_step=time_step,
            sub_steps=coagulation_sub_step,
        )

    # Record radius distribution
    distribution_counts[step, :], edges = np.histogram(
        aerosol_process.particles.radii[0].copy(),
        bins=bins_lognormal,
        density=False,
    )

    # Total particle-phase mass [kg/m³]
    mass_concentration[step] = np.sum(
        aerosol_process.particles.species_mass_density[0]
    )

    # Mean particle radius [m]
    radius_temp = aerosol_process.particles.radii[0].copy()
    mask_valid = radius_temp > 0
    mode_diameter[step] = np.mean(radius_temp[mask_valid])

    # Species-specific particle mass [kg]
    species_temp = aerosol_process.particles.masses[0].copy()
    species_mass[step, :] = np.sum(
        species_temp * aerosol_process.particles.concentration[0, :, None],
        axis=0,
    )
    np.testing.assert_allclose(
        species_inventory(aerosol_process),
        initial_inventory,
        rtol=1e-12,
        atol=1e-30,
    )

# Convert counts and masses to concentrations [#/m³] and [kg/m³]
concentrations = distribution_counts / simulation_volume
species_mass = species_mass / simulation_volume

# Final aerosol state
print(aerosol_process)

# %% [markdown]
# ## Distinguish mass uptake from a changing mean size
#
# The first plot reports particle mass concentration with imposed temperature on
# a separate axis. Condensation can change that mass by drawing from the finite
# gas reservoir; coagulation alone conserves it. The second plot's mean diameter
# can change under either process, so it cannot isolate their contributions.
#
# Any rapid variation must be assessed against the forcing resolution and process
# ordering before being interpreted physically. This run supplies no timestep
# sweep or matched process-off comparison that could attribute every feature.

# %%
dp_in_nm = 2 * 1e9  # Convert radius [m] → diameter [nm]

# Plot 1: Mass concentration over time
fig, ax = plotting.subplots(size="wide")
ax.plot(
    time_array,
    mass_concentration,
    color=TAILWIND["sky"]["800"],
    linewidth=6,
)
ax.set_xlabel("Time (s)")
ax.set_ylabel("Mass Concentration (kg/m³)")
ax.set_title("Mass Concentration Over Time")
ax.grid(True, axis="y")

# Add temperature on second axis
twinx = ax.twinx()
twinx.plot(
    time_array,
    temperature_array,
    color=TAILWIND["rose"]["500"],
    linestyle="--",
    linewidth=4,
)
twinx.set_ylabel("Temperature (K)", color=TAILWIND["rose"]["500"])
twinx.tick_params(axis="y", labelcolor=TAILWIND["rose"]["500"])

plt.xticks(rotation=45)
plotting.fit(fig)
plt.show()

# Plot 2: Mean diameter over time
fig, ax = plotting.subplots(size="wide")
ax.plot(
    time_array,
    mode_diameter * dp_in_nm,  # mean diameter [nm]
    color=TAILWIND["sky"]["800"],
    linewidth=6,
)
ax.set_xlabel("Time (s)")
ax.set_ylabel("Mean Diameter (nm)")
ax.set_title("Mean Particle Diameter Over Time")
ax.grid(True, axis="y")

# Add temperature on second axis
twinx = ax.twinx()
twinx.plot(
    time_array,
    temperature_array,
    color=TAILWIND["rose"]["500"],
    linestyle="--",
    linewidth=4,
    label="Temperature (K)",
)
twinx.set_ylabel("Temperature (K)", color=TAILWIND["rose"]["500"])
twinx.tick_params(axis="y", labelcolor=TAILWIND["rose"]["500"])

plt.xticks(rotation=45)
plotting.fit(fig)
plt.show()

# %% [markdown]
# ## Read absolute species mass before normalized composition
#
# The first stack plot shows particle species mass concentrations in kg/m³.
# The second divides each by total particle mass concentration. These are **bulk
# mass fractions**, not arithmetic means of particle-by-particle fractions,
# despite the retained “Mean” label. A rising fraction can result from another
# species leaving, so inspect absolute masses as well.
#
# The temperature overlay helps relate the imposed forcing to uptake, but no
# reaction converts one species into another. A change in the condensed mixture
# therefore describes partitioning of the supplied inventory, not soot chemistry.

# %%
# 📊 Plot 1: Absolute species mass concentration [kg/m³]
fig, ax = plotting.subplots(size="wide")

# stackplot expects shape (n_series, n_times), so transpose
ax.stackplot(
    time_array,
    species_mass.T,  # shape (n_species, n_time)
    labels=list_of_chemicals,
    colors=color_list,
    alpha=1.0,
)

ax.set_xlabel("Time (s)")
ax.set_ylabel("Species Mass Concentration (kg/m³)")
ax.set_title("Particle-Phase Species Mass Over Time")
ax.grid(True)

# Overlay temperature curve
twinx = ax.twinx()
twinx.plot(
    time_array,
    temperature_array,
    color=TAILWIND["rose"]["500"],
    linestyle="--",
    label="Temperature (K)",
    linewidth=4,
)
twinx.set_ylabel("Temperature (K)", color=TAILWIND["rose"]["500"])
twinx.tick_params(axis="y", labelcolor=TAILWIND["rose"]["500"])

# External legend
fig.legend(list_of_chemicals, title="Chemicals", loc="outside right center")
plotting.fit(fig)
plt.show()

# 📊 Plot 2: Mass fraction (normalized by total particle mass)
mass_fraction = species_mass / np.sum(species_mass, axis=1, keepdims=True)

fig, ax = plotting.subplots(size="wide")
ax.stackplot(
    time_array,
    mass_fraction.T,
    labels=list_of_chemicals,
    colors=color_list,
    alpha=1.0,
)

ax.set_xlabel("Time (s)")
ax.set_ylabel("Mean Mass Fraction in Particles")
ax.set_title("Species Mass Fractions in Particle Phase")
ax.grid(True)

# Overlay temperature
twinx = ax.twinx()
twinx.plot(
    time_array,
    temperature_array,
    color=TAILWIND["rose"]["500"],
    linestyle="--",
    label="Temperature (K)",
    linewidth=4,
)
twinx.set_ylabel("Temperature (K)", color=TAILWIND["rose"]["500"])
twinx.tick_params(axis="y", labelcolor=TAILWIND["rose"]["500"])

fig.legend(list_of_chemicals, title="Chemicals", loc="outside right center")
plotting.fit(fig)
plt.show()

# %% [markdown]
# ## Retain size information alongside bulk mass
#
# The final map shows number concentration per radius bin, in m⁻³, on a
# logarithmic color scale. It is not dN/dr. Empty merged slots fall outside the
# positive-radius histogram. Use this map with the mean-diameter plot: a single
# mean can hide a changing width or a sparse large-particle tail.
#
# The initial nanometre seeds and modeled growth are not evidence of resolved
# molecular-cluster formation. The representation and property assumptions still
# bound interpretation of the smallest sizes.

# %%
fig, ax = plotting.subplots(size="wide")

# Create mesh grid: time vs radius bins (left edges only)
X, Y = np.meshgrid(time_array, edges[:-1])

# Plot number concentration on a log scale
ax.grid(False)
mesh = ax.pcolormesh(
    X,
    Y,
    concentrations.T,  # shape (n_bins, n_times)
    shading="auto",
    norm=LogNorm(
        vmin=concentrations[concentrations > 0].min(),
        vmax=concentrations.max(),
    ),
    cmap=plotting.colormap("magnitude"),
)

ax.set_yscale("log")
ax.set_ylim(5e-10, 1e-7)
ax.set_xlabel("Time (s)")
ax.set_ylabel("Particle Radius (m)")
ax.set_title("Time-Evolving Particle Size Distribution")

# Colorbar for number concentration
fig.colorbar(mesh, label="Number Concentration (#/m³)")

plotting.fit(fig)
plt.show()

# %% [markdown]
# ## Use cooling-driven uptake as a testable model component
#
# The calculation couples an imposed temperature history to organic mass transfer
# and collisions among existing seeds, with closed per-species accounting.
# It does not model how a flame creates soot or predict a real emission spectrum.
# That distinction makes the example useful as one component of a larger study.
#
# A next comparison could hold the initial population fixed, vary the two cooling
# timescales, and separate condensation-only from condensation-plus-coagulation
# runs. Compare condensed species mass, number density, and size quantiles while
# refining the outer time grid. Validate property identities, temperature domains,
# and chemical stability before comparing with measured post-flame uptake; add
# reaction or nucleation physics only when the research question requires it.
