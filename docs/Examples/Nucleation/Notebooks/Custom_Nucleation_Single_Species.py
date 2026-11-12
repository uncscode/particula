# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.17.3
#   kernelspec:
#     display_name: particula_dev312
#     language: python
#     name: python3
# ---

# %%
"""Demonstrate an illustrative custom single-species nucleation workflow.

The notebook builds an ammonium-sulfate aerosol, applies a hand-built
saturation-based mass source, and couples condensation and coagulation
runnables. Its native fixed-slot custom source is illustrative only; use
``../cpu_nucleation.py`` for the supported bounded CPU-only nucleation API.
"""

# %% [markdown]
# # Custom Nucleation: Single Species
#
# In this *How-to Guide*, we demonstrate an illustrative custom single-species
# workflow with a saturation-dependent rate. Its custom slot activation and
# saturation-based mass source are not the supported E6-F7 public API. Use the
# supported `cpu_nucleation.py` script and CPU Nucleation Strategy System guide
# for public `Nucleation` runnable usage.
#
# The 100 seeded initial particles share 100,000 fixed-capacity slots.
# Unit count weights preserve the resolved coagulation contract. Gas debit
# includes the physical volume; no arrays are squeezed or methods patched.
# The plotted range includes both the 400 nm initial mode and 2 nm nuclei.
#
# **Imports**

# %%
# In Colab uncomment the following command to install particula:
# Use the selected E9 checkout: `python -m pip install -e ".[extra]"`
import matplotlib.pyplot as plt
import numpy as np

# particula
import particula as par
from particula.dynamics.condensation.species_mapping import (
    CondensationSpeciesMap,
)
from particula.particles.slot_management import activate_slots

# %% [markdown]
# ## Aerosol Setup
#
# We will begin by setting up ammonium sulfate vapor alongside a few pre-existing particles. The vapor phase will include a constant vapor pressure for ammonium sulfate, and a lognormal distribution will be used to represent the initial particle population.
#
# The pre-existing particles are also necessary as, the zero particle case is not supported in the current version of the model.

# %%
# Shared properties and initial builders
molar_mass_ammonium_sulfate = 132.14e-3  # kg/mol
density_ammonium_sulfate = 1.77e3  # kg/m^3
volume_sim = 0.1 * par.util.get_unit_conversion("cm^3", "m^3")  # m^3

parameters_vapor = {
    "vapor_pressure": 4e-12,  # atmosphere
    "vapor_pressure_units": "atm",
}

vapor_pressure_sulfate = par.gas.VaporPressureFactory().get_strategy(
    "constant", parameters_vapor
)

sulfate_saturation = vapor_pressure_sulfate.saturation_concentration(
    molar_mass=molar_mass_ammonium_sulfate,
    temperature=298.15,  # Kelvin
)
initial_sulfate_concentration = 0.5 * sulfate_saturation  # kg/m^3

# Generate a lognormal particle size distribution
np.random.seed(42)
particle_sample = par.particles.get_lognormal_sample_distribution(
    mode=np.array([400e-9]),  # Modal particle radius of 400 nm
    geometric_standard_deviation=np.array([1.4]),  # GSD of 1.4
    number_of_particles=np.array([1e4]),
    number_of_samples=100,
)
particle_mass_sample = (
    4 / 3 * np.pi * particle_sample**3 * density_ammonium_sulfate
)


def build_aerosol() -> par.Aerosol:
    """Build one native resolved box with explicit free-slot capacity."""
    masses = np.zeros((100_000, 1))
    counts = np.zeros(100_000)
    masses[: particle_mass_sample.size, 0] = particle_mass_sample
    counts[: particle_mass_sample.size] = 1.0
    particles = par.particles.create_resolved_particle_data(
        masses=masses,
        concentration=counts,
        density=density_ammonium_sulfate,
        volume=volume_sim,
    )
    gas = par.gas.GasData(
        name=["sulfate"],
        molar_mass=np.array([molar_mass_ammonium_sulfate]),
        partitioning=np.array([True]),
        concentration=np.array([[float(initial_sulfate_concentration)]]),
    )
    environment = par.gas.EnvironmentData(
        temperature=np.array([298.15]),
        pressure=np.array([101325.0]),
        saturation_ratio=np.zeros((1, 1)),
    )
    return par.Aerosol(particles=particles, gas=gas, environment=environment)


def current_saturation(aerosol):
    """Evaluate the current gas state with the explicit vapor strategy."""
    return float(aerosol.gas.concentration[0, 0] / sulfate_saturation)


def total_inventory(aerosol):
    """Independent extensive particle-plus-gas inventory in kg."""
    return float(
        np.sum(
            aerosol.particles.masses
            * aerosol.particles.concentration[..., None]
        )
        + aerosol.gas.concentration[0, 0] * volume_sim
    )


def inject_custom_nuclei(aerosol, rate, duration, nucleus_mass):
    """Apply the illustrative mass-rate law using unit-count free slots.

    The rate is kg/m³/s. Multiply by physical volume exactly once to obtain
    represented mass, then floor to whole nuclei. Available gas limits the
    request. Capacity failure occurs before the gas debit; there is no resize
    or claim of rollback for a whole multi-process time step.
    """
    count = int(
        np.floor(
            min(float(rate) * duration, aerosol.gas.concentration[0, 0])
            * volume_sim
            / nucleus_mass
        )
    )
    if count == 0:
        return 0
    print(f"Activating {count} unit-count nuclei")
    before = total_inventory(aerosol)
    activate_slots(
        aerosol.particles,
        np.full((1, count, 1), nucleus_mass),
        np.ones((1, count)),
        np.zeros((1, count)),
        np.array([count], dtype=np.int32),
    )
    aerosol.gas.concentration[0, 0] -= count * nucleus_mass / volume_sim
    np.testing.assert_allclose(
        total_inventory(aerosol), before, rtol=1e-12, atol=1e-30
    )
    return count


aerosol = build_aerosol()
print(aerosol)

# %% [markdown]
# ## Simulation
#
# This section performs a step in the simulation using a manual stepping method. The steps include:
#
# 1. Adding more vapors to the gas phase.
# 2. Calculating the new saturation ratio.
# 3. Calculating the nucleation rate based on the saturation difference.
# 4. Determining the number of new particles nucleated.
# 5. Determining the number of resolved particles to be added to the aerosol.
# 6. Creating and adding the new particles to the aerosol.
# 7. Performing a condensation step to account for gas-phase condensation onto existing particles.
# 8. Performing a coagulation step to account for particle-particle interactions.
#
# And before we start, we also need to initialize the condensation and coagulation runnables.

# %%
# setup dynamics for condensation
condensation_strategy = par.dynamics.CondensationIsothermal(
    molar_mass=molar_mass_ammonium_sulfate,
    accommodation_coefficient=1,
    diffusion_coefficient=2e-5,
    activity_strategy=par.particles.ActivityIdealMass(),
    surface_strategy=par.particles.SurfaceStrategyVolume(),
    vapor_pressure_strategy=vapor_pressure_sulfate,
    species_map=CondensationSpeciesMap(("sulfate",), ((0, 0),)),
)
condensation_runnable = par.dynamics.MassCondensation(condensation_strategy)
# setup dynamics for coagulation
coagulation_strategy = (
    par.dynamics.BrownianCoagulationBuilder()
    .set_distribution_type("particle_resolved")
    .build()
)
coagulation_runnable = par.dynamics.Coagulation(
    coagulation_strategy=coagulation_strategy
)

step_count = 0

# %% [markdown]
# You can repeatedly run the next cell to see the evolution of the aerosol system.

# %%
# Initialize or increment step counter
step_count += 1
print(f"Step {step_count}")

# Define key parameters
vapor_production = (
    sulfate_saturation * 0.2
)  # Adding 20% of saturation concentration per second
base_nucleation_rate = (
    1e-8 * density_ammonium_sulfate
)  # Base nucleation rate in kg/m^3/s
mass_nucleated_particle = (
    4 / 3 * np.pi * (2e-9) ** 3 * density_ammonium_sulfate
)  # Mass of a 4 nm diameter particle in kg
exponent_nucleation = 2  # Nucleation rate exponent (empirical)
time_step = 1  # Time step in seconds

# 1. Add more vapor to the gas phase (e.g., by external sources)
print(
    "Current sulfate concentration: ",
    aerosol.gas.concentration[0, 0],
)
aerosol.gas.concentration[0, 0] += vapor_production * time_step
print(
    "New sulfate concentration: ",
    aerosol.gas.concentration[0, 0],
)

# 2. Calculate the new saturation ratio for sulfate in the atmosphere
saturation_ratio = current_saturation(aerosol)
print(f"Saturation ratio: {saturation_ratio}")

# 3. Calculate the nucleation rate based on the saturation ratio
# Ensure the saturation ratio is above 1, nucleation only occurs above saturation
saturation_difference = np.maximum(
    saturation_ratio - 1, 0
)  # No nucleation if S ≤ 1
# Calculate the nucleation rate using the exponential form (custom)
# note this is mass based, if you have a volume based nucleation rate, you need to convert it
# to mass, as the resolved particles are mass based
nucleation_rate = (
    base_nucleation_rate * (saturation_difference / 500) ** exponent_nucleation
)
print(
    f"Nucleation rate [mass concentration per sec, kg/m^3/s]: {nucleation_rate}"
)

# 4. Calculate the number of new particles nucleated
# Floor division ensures we only get whole particles
number_of_new_particles = time_step * nucleation_rate // mass_nucleated_particle
print(f"Number of new particles nucleated: {number_of_new_particles}")

# 5. Determine the number of resolved particles to create (based on simulation volume)
single_resolved_particle = 1.0 / volume_sim
number_of_new_resolved_particles = int(
    np.asarray(number_of_new_particles // single_resolved_particle).flat[0]
)

# %% [markdown]
# ## Time Loop
#
# Now that we see the simulation is working, we can put that into a loop and save out the distribution of particles at each time step.
#
# We'll first reset the aerosol system to its initial state, create a output matrix, then run the previous simulation in a for loop.

# %%
# Reset to the same seeded initial sample and fixed capacity.
aerosol = build_aerosol()

# Print the properties of the created aerosol system
print(aerosol)


# Set up time and sub-steps for the coagulation process
total_time = 5
time_step = 1
sub_steps = 2

# bins
bins_lognormal = np.logspace(-9, -5, 200)

# output arrays
time = np.arange(0, total_time, time_step)
total_mass_resolved = np.ones_like(time, dtype=np.float64)
number_distribution_binned = np.zeros((len(time), len(bins_lognormal) - 1))
total_number_resolved = np.ones_like(time, dtype=np.float64)
saturation_ratio_output = np.ones_like(time, dtype=np.float64)

print(f"Total iterations to do: {len(time) * sub_steps}")

# %%
# Define key parameters
vapor_production = (
    sulfate_saturation * 0.2
)  # Adding 20% of saturation concentration per second
base_nucleation_rate = (
    1e-8 * density_ammonium_sulfate
)  # Base nucleation rate in kg/m^3/s
mass_nucleated_particle = (
    4 / 3 * np.pi * (2e-9) ** 3 * density_ammonium_sulfate
)  # Mass of a 4 nm diameter particle in kg
exponent_nucleation = 2  # Nucleation rate exponent (empirical)


bin_edges = bins_lognormal

initial_inventory = total_inventory(aerosol)
for i, _ in enumerate(time):
    if i > 0:
        # 1. Add more vapor to the gas phase (e.g., by external sources)
        aerosol.gas.concentration[0, 0] += vapor_production * time_step

        # 2. Calculate the new saturation ratio for sulfate in the atmosphere
        saturation_ratio = current_saturation(aerosol)

        # 3. Calculate the nucleation rate based on the saturation ratio
        # Ensure the saturation ratio is above 1, nucleation only occurs above saturation
        saturation_difference = np.maximum(
            saturation_ratio - 1, 0
        )  # No nucleation if S ≤ 1
        # Calculate the nucleation rate using the exponential form (custom)
        # note this is mass based, if you have a volume based nucleation rate, you need to convert it
        # to mass, as the resolved particles are mass based
        nucleation_rate = (
            base_nucleation_rate
            * (saturation_difference / 500) ** exponent_nucleation
        )

        # The custom law uses explicit native fixed-slot activation.
        number_of_new_resolved_particles = inject_custom_nuclei(
            aerosol, nucleation_rate, time_step, mass_nucleated_particle
        )

        # 7. Perform the condensation step
        condensation_runnable.execute(aerosol, time_step, sub_steps)
        # 8. Perform the coagulation step
        coagulation_runnable.execute(aerosol, time_step, sub_steps)

    np.testing.assert_allclose(
        total_inventory(aerosol),
        initial_inventory + i * vapor_production * time_step * volume_sim,
        rtol=1e-12,
        atol=1e-30,
    )
    total_mass_resolved[i] = aerosol.particles.species_mass_density[0, 0]
    number_distribution = aerosol.particles.radii[0].copy()
    number_distribution_binned[i, :], edges = np.histogram(
        number_distribution, bins=bins_lognormal
    )
    if i == 0:
        bin_edges = edges
    total_number_resolved[i] = np.sum(number_distribution > 0)
    saturation_ratio_output[i] = np.asarray(current_saturation(aerosol)).flat[0]

    if i % 20 == 0:
        # Retrieve and print the total number of resolved particles simulated
        total_resolved_particles_in_simulation = (
            aerosol.particles.concentration.sum()
        )
        print(
            f"Index {i}: Total resolved particles in simulation: {total_resolved_particles_in_simulation}"
        )

# ensure bin_edges is defined for plotting
bin_edges = bin_edges if "bin_edges" in locals() else bins_lognormal
plot_edges = bin_edges

number_distribution_binned = number_distribution_binned / volume_sim

# %% [markdown]
# ## Graphing
#
# In this section, we will visualize the nucleation events over time. The initial particles will be displayed, followed by their coagulated pairs. As the simulation progresses, particle growth results from both coagulation and condensation processes.

# %%
fig, ax = plt.subplots(figsize=(8, 5))

# Swap X and Y to reverse axes
X, Y = np.meshgrid(
    time, plot_edges[:-1]
)  # Now time is on the x-axis and edges on the y-axis

# Plot the contour with updated X and Y
log_of_number_distribution_binned = np.log10(
    number_distribution_binned,
    out=np.nan * np.ones_like(number_distribution_binned),
    where=number_distribution_binned > 0,
)
contour = ax.contourf(
    X,
    Y,
    log_of_number_distribution_binned.T,
    cmap="viridis",
    vmin=5,
)

# Add the color bar
cbar = fig.colorbar(contour)
cbar.set_label("Log10 of Number concentration (m^-3)")

ax.set_ylim(1e-9, 1e-5)  # Set limits for y-axis

# Set axis labels
ax.set_yscale("log")  # Log scale for particle radius on y-axis
ax.set_xlabel("Time (s)")
ax.set_ylabel("Particle radius (m)")
fig.tight_layout()
plt.show(block=False)

# %%
# plot the total mass and water saturation on twin y-axis
fig, ax1 = plt.subplots(figsize=(8, 5))

ax1.plot(time, total_mass_resolved, label="Total mass", color="blue")
ax1.set_xlabel("Time (s)")
ax1.set_ylabel("Total Particle mass (kg/m^3)", color="blue")
ax1.tick_params(axis="y", labelcolor="blue")

ax2 = ax1.twinx()
ax2.plot(time, saturation_ratio_output, label="Saturation Ratio", color="red")
ax2.set_ylabel("Saturation Ratio", color="red")
ax2.tick_params(axis="y", labelcolor="red")

fig.tight_layout()
plt.show(block=False)

# %% [markdown]
# ## Conclusion
#
# In this guide, we demonstrated how to integrate custom nucleation processes into the aerosol simulation. This shows the flexibility of the aerosol model, allowing for the addition of new processes before they are fully integrated into the core framework.
#
# *Note*: Custom nucleation, particularly at high rates, can significantly
# increase the number of particles simulated and slow computation. The supported
# CPU-only `Nucleation` runnable uses its configured fixed-capacity exhaustion
# policy to handle source-slot pressure.
