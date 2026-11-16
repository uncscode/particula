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
# # Audit a respiratory-droplet illustration before using its predictions
#
# An evaporation calculation can conserve mass and still represent the wrong
# materials. This example follows a multicomponent droplet population in initially
# 30% RH air, while exposing a necessary earlier step: checking that requested
# chemical names resolve to the intended species.
#
# **The retained property lookup fails that identity check.** In the saved output,
# “Lysozyme” resolves to hexachlorophene and “IgA” to isoguanine; the requested
# calcium and magnesium bicarbonates resolve to carbonates. Those are different
# substances, not acceptable protein or salt surrogates established by this example.
# Consequently the results illustrate the workflow and its input-audit failure;
# they must not be interpreted as quantitative respiratory-droplet or exposure
# predictions.
#
# Read [condensation and evaporation](../processes/condensation/condensation_3_massresolved.py)
# for the mass-transfer mechanism. The useful question here is what additional
# checks are required before adapting that mechanism to a material mixture.

# %%
# 🚧 Install Particula if you're running this in Google Colab
# Remove the comment below to enable installation
# Install from the repository root: `python -m pip install -e ".[extra]"`

# 📦 Import necessary libraries
import copy  # to safely duplicate Python objects

import matplotlib.pyplot as plt  # for plotting
import numpy as np  # numerical tools
import particula as par  # the main Particula package
from particula.dynamics.condensation.species_mapping import (
    CondensationSpeciesMap,
)
from particula.execution.condensation_ode import CondensationODE
from tqdm import tqdm  # optional: for progress bars during simulation

# 🎨 Set default plot styling using Tailwind-inspired colors from Particula
TAILWIND = par.util.colors.TAILWIND
base_color = TAILWIND["gray"]["600"]

plt.rcParams.update(
    {
        "text.color": base_color,
        "axes.labelcolor": base_color,
        "figure.figsize": (5, 4),
        "font.size": 14,
        "axes.edgecolor": base_color,
        "xtick.color": base_color,
        "ytick.color": base_color,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    }
)


# %% [markdown]
# ## Compare requested labels with resolved identities
#
# The twelve requested labels span water, salts, small organics, and biological
# materials. The utility first attempts an exact lookup and can fall back to a
# similar name. Printing the returned name, CAS identifier, and structure is
# therefore a scientific input check, not merely a status message. The mismatches
# identified above invalidate the claim that these arrays describe the requested
# respiratory-fluid composition.
#
# The independently sampled mass ranges are illustrative and lack a cited
# composition dataset here. Molar masses are kg/mol, densities kg/m³, surface
# tensions N/m, and vapor pressures Pa. The NaCl surface-tension override is an
# assumed constant; it is not a concentration-dependent electrolyte model.
# A verified identity and a numerical property value are separate requirements.

# %%
list_of_chemicals = [
    "Water",
    "NaCl",
    "potassium bicarbonate (KHCO3)",
    "potassium dihydrogen phosphate (KH2PO4)",  # K+ + H2PO4–
    "calcium bicarbonate [Ca(HCO3)2]",
    "magnesium bicarbonate [Mg(HCO3)2]",
    "Urea",
    "Lactate",
    "Glucose",
    "Lysozyme",
    "IgA",
    "cholesterol",
]
initial_mass_fraction_bounds = [
    (0.985, 0.995),  # Water
    (0.0006, 0.0023),  # NaCl
    (0.0007, 0.0030),  # KHCO3
    (0.0002, 0.0027),  # KH2PO4
    (0.00002, 0.00045),  # Ca(HCO3)2
    (0.000006, 0.00003),  # Mg(HCO3)2
    (0.0002, 0.0015),  # Urea
    (0.00005, 0.0004),  # Lactate
    (0.00001, 0.0002),  # Glucose
    (0.000005, 0.00002),  # Lysozyme
    (0.00002, 0.0002),  # IgA
    (0.00005, 0.0005),  # Cholesterol
]

density_array = np.array([])
molar_mass_array = np.array([])
surface_tension_array = np.array([])
short_name = []
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
    short_name.append(chemical_dict[chem]["name"])

# correct NaCl surface tension
surface_tension_array[1] = 0.091  # N/m, typical value

# Print the chemical properties
for chem, props in chemical_dict.items():
    print(f"{chem}:")
    print(f"  Name: {props['name']}")
    print(f"  CAS: {props['cas_number']}")
    print(f"  Smiles: {props['smiles']}")
    print(f"  Molar Mass: {props['molar_mass']:.4f} kg/mol")
    print(f"  Density: {props['density']:.2f} kg/m³")
    print(f"  Surface Tension: {props['surface_tension']} N/m")
    print(f"  Vapor Pressure: {props['pure_vapor_pressure']} Pa\n")

# %% [markdown]
# ## Distinguish reproducible inputs from validated inputs
#
# The sample contains 1000 particles at 10⁻⁴ cm⁻³ (100 m⁻³), corresponding to
# 10 m³ represented volume, at 298.15 K. The fixed sampling seed controls this
# realization, not the accuracy of the material model.
#
# Missing vapor pressures are replaced by 10⁻³⁰ Pa, making those lanes nearly
# nonvolatile. Missing surface tensions receive 0.072 N/m. These substitutions
# allow execution; they do not resolve missing measurements or incorrect
# identities. Record them as assumptions before interpreting any partitioning.

# %%
# %% Step 1: Define system size, temperature, and vapor pressure strategies

# 🎲 Ensure reproducibility
np.random.seed(100)

# 🔢 Simulation settings
number_of_samples = 1_000  # Number of particles to simulate
total_number_per_cm3 = 1e-4  # Particle number concentration [#/cm³]
simulation_volume = (
    number_of_samples / total_number_per_cm3 * 1e-6
)  # Simulation volume in m³ (converted from cm³)

temperature = 298.15  # System temperature in Kelvin (25°C)

# 🌡 Build vapor pressure strategies for each chemical
vapor_pressure_strategy_list = []

for chem_i, props in chemical_dict.items():
    vapor_pressure_temp = props["pure_vapor_pressure"]

    if vapor_pressure_temp is None:
        print(f"⚠️ Warning: No vapor pressure data for {chem_i}, using default.")
        vapor_pressure_temp = 1e-30  # fallback for non-volatile species

    vapor_pressure_i = (
        par.gas.ConstantVaporPressureBuilder()
        .set_vapor_pressure(vapor_pressure_temp, "Pa")
        .build()
    )

    vapor_pressure_strategy_list.append(vapor_pressure_i)

# 💧 Ensure surface tension values are set
surface_tension_array = np.array(
    [0.072 if x is None else x for x in surface_tension_array]
)

# %% [markdown]
# ## Initialize a finite gas reservoir at 30% saturation
#
# Water gas concentration is 0.3 times its modeled saturation concentration;
# other vapor lanes start near zero. Gas names retain the requested labels even
# when property lookup returned another substance, so names in a container do
# not certify the underlying material identity.
#
# Temperature is fixed at 298.15 K and pressure at 101325 Pa. Humidity is only
# initialized here. The active loop does not reset it to 30% at each step.

# %% Step 2: Define gas-phase concentrations and thermodynamic environment

# 💨 Initialize gas-phase concentrations near zero
concentration_gas = np.ones(len(list_of_chemicals)) * 1e-30  # kg/m³

# 💧 Set water vapor to 30% RH
isat_water = vapor_pressure_strategy_list[0].saturation_concentration(
    molar_mass_array[0], temperature
)
concentration_gas[0] = isat_water * 0.3  # 30% RH

# 🧪 Build gas-phase species with vapor strategies and concentrations
gas_species = par.gas.GasData(
    name=list_of_chemicals,
    molar_mass=molar_mass_array.copy(),
    partitioning=np.ones(len(list_of_chemicals), dtype=bool),
    concentration=concentration_gas[None, :].copy(),
)
environment = par.gas.EnvironmentData(
    temperature=np.array([temperature]),
    pressure=np.array([101325.0]),
    saturation_ratio=np.zeros((1, len(list_of_chemicals))),
)

# %% [markdown]
# ## Track what the composition sampling actually constructs
#
# The three supplied radius modes are 0.3, 4, and 40 µm with relative weights
# 15, 2, and 0.2. They define an illustrative sample, not a validated cough-size
# distribution. Water-equivalent spherical mass is multiplied independently by
# each sampled composition factor. Those factors are not jointly normalized,
# so the final total mass and reconstructed radius need not exactly equal the
# water-only sample from which they were built.
#
# The κ array is assumed, with water in lane zero. Each species receives its
# own surface-phase label, so this configuration retains species-specific surface
# tensions rather than averaging them across one shared liquid phase. It does
# not calculate physical phase separation. The inventory helper counts weighted
# particle mass plus gas mass in the represented volume.

# %% Step 3: Define particle composition and create aerosol object

# 💠 3a. Lognormal size distributions for cough droplet modes
# Modes: bronchiolar, laryngeal, oropharyngeal
modes = np.array([0.6, 8.0, 80.0]) / 2 * 1e-6  # radii in meters
geometric_standard_deviations = np.array([1.65, 2.4, 1.6])
relative_number = np.array([15, 2, 0.2])  # relative weights

# Generate sampled radii
radii_cough = par.particles.get_lognormal_sample_distribution(
    mode=modes,
    geometric_standard_deviation=geometric_standard_deviations,
    number_of_particles=relative_number,
    number_of_samples=number_of_samples,
)

# 💧 3b. Assign initial water mass and scale by chemical mass fractions
mass_water = 4 / 3 * np.pi * (radii_cough**3) * density_array[0]

mass_speciation = np.repeat(
    mass_water[:, np.newaxis], len(list_of_chemicals), axis=1
)

# Apply random mass fractions from defined bounds
for i, (_chem, bounds) in enumerate(
    zip(list_of_chemicals, initial_mass_fraction_bounds)
):
    mass_fraction = np.random.uniform(bounds[0], bounds[1], number_of_samples)
    mass_speciation[:, i] *= mass_fraction

# ⚗️ 3c. Define activity strategy using kappa parameters
kappa_parameter = np.array(
    [0.0, 1.1, 0.6, 0.6, 0.6, 0.6, 0.3, 0.2, 0.1, 0.1, 0.1, 0.01]
)

activity_strategy = (
    par.particles.ActivityKappaParameterBuilder()
    .set_molar_mass(molar_mass_array, "kg/mol")
    .set_density(density_array, "kg/m^3")
    .set_kappa(kappa_parameter)
    .set_water_index(0)
    .build()
)

# Define surface tension strategy
surface_strategy = (
    par.particles.SurfaceStrategyMassBuilder()
    .set_surface_tension(surface_tension_array, "N/m")
    .set_density(density_array, "kg/m^3")
    .set_phase_index(np.arange(len(list_of_chemicals)))
    .build()
)

# 🧱 3d. Build resolved particle object
resolved_masses = par.particles.create_resolved_particle_data(
    masses=mass_speciation,
    density=density_array,
    volume=simulation_volume,
)
aerosol = par.Aerosol(
    particles=resolved_masses, gas=gas_species, environment=environment
)


def species_inventory(aero):
    """Independent species inventory in kg, including represented volume."""
    return np.sum(
        aero.particles.masses * aero.particles.concentration[..., None], axis=1
    ) + (aero.gas.concentration * aero.particles.volume[:, None])


initial_inventory = species_inventory(aerosol)

# Optional: print summary of aerosol system
print("Particle slots:", aerosol.particles.masses.shape[1])
print("Gas species:", ", ".join(aerosol.gas.name))

# %% [markdown]
# ## Inspect the initialized distribution on its own terms
#
# The histogram counts sampled **radii** per logarithmic bin. It is not yet a
# number concentration or a diameter spectrum. The native `ParticleData` stores
# the species masses and represented counts; its derived radii include the
# sampled composition and densities. This is the appropriate initialized state
# to compare with later sizes.

# %% Step 4: Plot initial particle size distribution

fig, ax = plt.subplots()

# Log-scale histogram of particle radii
ax.hist(
    np.log10(aerosol.particles.radii[0]),  # convert to log10
    bins=50,
    density=False,
    alpha=0.5,
)

ax.set_xlabel("log₁₀(Radius [m])")
ax.set_ylabel("Bin Counts")
ax.set_title("Initial Particle Size Distribution")
plt.tight_layout()
plt.show()

# %% [markdown]
# ## Separate a configured process from an executed process
#
# Condensation/evaporation maps all twelve lanes, with a common diffusivity
# 2×10⁻⁵ m²/s and accommodation one. These are simplifying assumptions across
# a chemically diverse set of requested labels. A Brownian process is also
# constructed, but its execution is commented out in the loop below. The saved
# calculation therefore contains mass transfer alone, without coagulation,
# ventilation, sedimentation removal, or temperature feedback.

# %%
# %% Step 5: Set up physical aerosol processes

# 💧 Isothermal Condensation
condensation_strategy = par.dynamics.CondensationIsothermal(
    molar_mass=np.array(molar_mass_array),
    activity_strategy=activity_strategy,
    surface_strategy=surface_strategy,
    vapor_pressure_strategy=vapor_pressure_strategy_list,
    species_map=CondensationSpeciesMap(
        tuple(gas_species.name),
        tuple((i, i) for i in range(len(list_of_chemicals))),
    ),
    diffusion_coefficient=2e-5,  # typical air value [m²/s]
    accommodation_coefficient=1,  # full sticking efficiency
    # skip_partitioning_indices=[]      # optional: exclude species
)
condensation_process = par.dynamics.MassCondensation(condensation_strategy)

# ⚪ Brownian Coagulation
coagulation_strategy = (
    par.dynamics.BrownianCoagulationBuilder()
    .set_distribution_type("particle_resolved")  # use individual particles
    .build()
)
coagulation_process = par.dynamics.Coagulation(coagulation_strategy)

# %% [markdown]
# ## Record 20 seconds of adaptive closed mass transfer
#
# There are 201 stored states, including both endpoints of the full 20 s run.
# ROS3 adapts condensation steps inside each 0.1 s output interval, with rtol
# 10⁻⁵, particle atol 10⁻²⁵ kg and gas atol 10⁻¹² kg/m³. Water exchanges with a
# finite gas reservoir. Better integration does not repair uncertain chemical
# properties or validate respiratory-fluid predictions.
#
# At every stored state, the assertion compares each gas-plus-particle inventory
# with its initial value at rtol=10⁻¹² and atol=10⁻³⁰ kg. This checks material
# accounting for the supplied arrays; it cannot detect that a property array
# belongs to the wrong chemical. Radius-bin counts, bulk particle species masses,
# and vapor saturation ratios support distinct diagnostics below.

# %% Step 6: Run the simulation loop

print("Initial gas concentration (kg/m^3):", aerosol.gas.concentration)

# 🔁 Deep copy for simulation
aerosol_process = copy.deepcopy(aerosol)

# Simulation parameters
total_time = 20  # seconds
total_steps = 200

time_step = total_time / total_steps
time_array = np.linspace(0.0, total_time, total_steps + 1)
time_len = len(time_array)

# 📊 Initialize arrays for tracking outputs
bins_lognormal = np.logspace(-8, -4, 100)
distribution_counts = np.zeros((time_len, len(bins_lognormal) - 1))
species_mass = np.zeros((time_len, len(list_of_chemicals)))
vapor_saturation = np.zeros((time_len, len(list_of_chemicals)))

simulation = CondensationODE(
    aerosol_process,
    condensation_strategy,
    method="ros3",
    rtol=1e-5,
    particle_atol=1e-25,
    gas_atol=1e-12,
    first_step=1e-6,
    max_step=time_step,
)

# 🌀 Time integration loop
for step, _t in enumerate(
    tqdm(time_array, desc="Running Sim", mininterval=0.5)
):
    if step > 0:
        while simulation.time < float(_t):
            simulation.step(float(_t))
        aerosol_process = simulation.snapshot()

    # Record size distribution
    distribution_counts[step, :], edges = np.histogram(
        aerosol_process.particles.radii[0].copy(),
        bins=bins_lognormal,
        density=False,
    )

    # Record species mass in particles
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

    # Record vapor saturation ratios
    vapor_saturation[step, :] = np.array(
        [
            aerosol_process.gas.concentration[0, i]
            / strategy.saturation_concentration(
                molar_mass_array[i], aerosol_process.environment.temperature[0]
            )
            for i, strategy in enumerate(vapor_pressure_strategy_list)
        ]
    )

# Normalize counts to number concentration (#/m³)
concentrations = distribution_counts / simulation_volume

# Final system state and actual adaptive work
print("Final gas concentration (kg/m^3):", aerosol_process.gas.concentration)
print("Adaptive condensation work:", simulation.solver.statistics)

# %% [markdown]
# <a id="initial-and-final-sizes-what-to-look-for"></a>
#
# ## Compare initial and final sizes
#
# The dashed and solid curves use the same radius bins. Their vertical values
# are number concentration **per bin** in m⁻³, not dN/dr. A shift toward smaller
# radii is consistent with net evaporation under the modeled inputs; it does not
# by itself prove equilibrium or a validated droplet lifetime. Particles moving
# outside the histogram's limits would also change the plotted bin sum.

# %% Step 7: Plot initial vs final size distributions

fig, ax = plt.subplots(figsize=(7, 5))

# Get radii before and after simulation
initial_distribution = aerosol.particles.radii[0].copy()
final_distribution = aerosol_process.particles.radii[0].copy()

# Compute binned counts
counts_initial, _ = np.histogram(
    initial_distribution, bins=bins_lognormal, density=False
)
counts_final, _ = np.histogram(
    final_distribution, bins=bins_lognormal, density=False
)

# Plot initial distribution
ax.plot(
    bins_lognormal[:-1],
    counts_initial / simulation_volume,
    label="Initial Size Distribution",
    color=TAILWIND["sky"]["800"],
    linestyle="--",
)

# Plot final distribution
ax.plot(
    bins_lognormal[:-1],
    counts_final / simulation_volume,
    label="Final Size Distribution",
    color=TAILWIND["rose"]["400"],
    linestyle="-",
)

# Set axis scales and labels
ax.set_xscale("log")
ax.set_xlabel("Particle Radius (m)")
ax.set_ylabel("Concentration (#/m⁻³)")
ax.legend()
plt.tight_layout()
plt.show()

# %% [markdown]
# ## Read gas saturation separately from particle equilibrium
#
# The water saturation ratio divides gas concentration by pure-water saturation
# concentration at the fixed temperature. Unity means saturation relative to that
# reference, not equilibrium for every curved, multicomponent particle.
#
# The small reservoir change in the saved run reflects this population's low
# loading relative to gas water. It is an outcome of the chosen volume and
# concentrations, not a general claim that humidity stays constant near a cough.
# The lower axis limit is 0.3, so inspect the actual vertical scale when judging
# the change.

# %% Step 8: Plot water vapor saturation ratio over time

fig, ax = plt.subplots(figsize=(7, 5))

# Plot water vapor saturation ratio (species index 0)
ax.plot(
    time_array,
    vapor_saturation[:, 0],
    color=TAILWIND["sky"]["800"],
    label="Vapor Saturation Ratio (Water)",
)

ax.set_ylim(bottom=0.3)
ax.set_xlabel("Time (s)")
ax.set_ylabel("Vapor Saturation Ratio")
ax.set_title("Vapor Saturation Ratio for Water Over Time")
ax.legend()

plt.tight_layout()
plt.show()

# %% [markdown]
# ## Identify the weighting behind a composition plot
#
# The stack plot divides **summed particle species mass** by **summed total
# particle mass**. It is a bulk mass fraction, not an arithmetic average of each
# particle's composition, despite the retained “Mean” axis label. Larger
# particles can dominate this diagnostic. It cannot on its own identify which
# size class retains water longer.
#
# Legend names come from the resolved property entries. Their disagreement with
# the requested labels is further evidence of the identity problem, not a change
# of species caused by evaporation.

# %% Step 9: Plot particle mass fraction evolution

# Compute total mass fraction over time
mass_fraction = species_mass / np.sum(species_mass, axis=1, keepdims=True)

# Set color palette (one color per species)
color_list = [
    TAILWIND["sky"]["800"],  # Water
    TAILWIND["rose"]["400"],  # NaCl
    TAILWIND["red"]["800"],  # KHCO3
    TAILWIND["red"]["600"],  # KH2PO4
    TAILWIND["amber"]["500"],  # Ca(HCO3)2
    TAILWIND["orange"]["400"],  # Mg(HCO3)2
    TAILWIND["teal"]["900"],  # Urea
    TAILWIND["teal"]["700"],  # Lactate
    TAILWIND["green"]["500"],  # Glucose
    TAILWIND["indigo"]["600"],  # Lysozyme
    TAILWIND["indigo"]["400"],  # IgA
    TAILWIND["cyan"]["500"],  # Cholesterol
]

# Create stackplot of species mass fraction
fig, ax = plt.subplots(figsize=(8, 5))
ax.stackplot(
    time_array,
    mass_fraction.T,  # Each row = one species
    labels=short_name,  # Species names
    colors=color_list,
    alpha=1.0,
)

ax.set_xlabel("Time (s)")
ax.set_ylabel("Mean Mass Fraction in Particles")
ax.set_ylim(0, 1)
ax.grid(True)
ax.legend(
    title="Species",
    loc="lower left",
    framealpha=0.9,
    fontsize=10,
)
plt.tight_layout()
plt.show()

# %% [markdown]
# <a id="time-evolution-what-to-look-for"></a>
#
# ## Follow size redistribution without overinterpreting a plateau
#
# The contour displays log₁₀ of number concentration per radius bin through time.
# It complements the bulk mass-fraction plot by retaining size information.
# A slowly changing band is not sufficient evidence of thermodynamic equilibrium;
# that would require examining mass-transfer rates and convergence as well.
# The plotted range omits radii below 0.1 µm even though the stored histogram
# extends lower.

# %% Step 10: Contour plot of particle size distribution over time

fig, ax = plt.subplots(figsize=(7, 5))

# Create 2D mesh grid for time and particle radius
X, Y = np.meshgrid(time_array, edges[:-1])

# Convert number concentration to log10 scale (avoid log(0) issues)
log_conc = np.log10(
    concentrations,
    where=concentrations > 0,
    out=np.full_like(concentrations, np.nan),  # mask zeros
)

# Contour plot of log concentration
cont = ax.contourf(X, Y, log_conc.T)

# Axis settings
ax.set_yscale("log")
ax.set_ylim(1e-7, 1e-4)
ax.set_xlabel("Time (s)")
ax.set_ylabel("Particle Radius (m)")

# Add colorbar
fig.colorbar(cont, label="Log₁₀ Number Concentration")
plt.tight_layout()
plt.show()

# %% [markdown]
# ## Carry forward the audit, not an unsupported material prediction
#
# The workflow links sampled composition, finite-reservoir evaporation, and size
# and mass diagnostics. Its inventory checks pass for the supplied numerical
# species, but the identity mismatches prevent interpreting those results as
# respiratory-fluid behavior. Successful execution is not material validation.
#
# Before a research adaptation, replace fuzzy matches with verified identifiers
# and documented properties or explicitly justified macromolecular surrogates.
# Normalize the intended starting composition, choose a defensible activity and
# surface model, then rerun the full calculation. Compare size-resolved water
# loss across humidity and timestep, using measured evaporation data to test the
# material assumptions. None of that validation is supplied by the retained run.
