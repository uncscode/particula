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

# %% [markdown]
# # Staggered Condensation Stepping
#
# This notebook demonstrates the **staggered ODE stepping framework** for particle-resolved condensation simulations. The examples compare gas-feedback ordering and check mass conservation for the specified workloads.
#
# ## What You'll Learn
#
# 1. **Theta Modes**: Compare three stepping modes (`"half"`, `"random"`, `"batch"`)
# 2. **Mass Conservation**: Verify that staggered stepping conserves mass
# 3. **Stability**: Compare stability with simultaneous stepping at large time steps
# 4. **Parameter Selection**: Guidelines for choosing `theta_mode` and `num_batches`
#
# ## Why Staggered Stepping?
#
# In particle-resolved condensation, all particles compete for a shared vapor reservoir. Simultaneous updates can cause:
# - **Numerical instability** at large time steps
# - **Mass conservation errors** from all particles "seeing" the same gas concentration
#
# Staggered stepping addresses these issues by splitting each timestep into two passes and updating particles in batches, allowing the gas field to respond between updates.

# %%
# In Colab uncomment the following command to install particula:
# Install from the repository root: `python -m pip install -e ".[extra]"`

import matplotlib.pyplot as plt
import numpy as np
import particula as par
from particula.dynamics.condensation.species_mapping import (
    CondensationSpeciesMap,
)

# Set random seed for reproducibility
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

# %% [markdown]
# ## Setup Native Gas and Environment
#
# First, we set up the gas phase with water vapor at supersaturated conditions to drive condensation.

# %%
# Physical constants
molar_mass_water = 18.015e-3  # kg/mol

# Vapor pressure strategy for water
vapor_pressure_water = par.gas.VaporPressureFactory().get_strategy("water_buck")

# Calculate water saturation concentration
temperature = 298.15  # K
saturation_concentration = vapor_pressure_water.saturation_concentration(
    molar_mass=molar_mass_water, temperature=temperature
)

# Supersaturated conditions (102.5% RH) to drive condensation
initial_rh = 1.025
water_concentration = saturation_concentration * initial_rh

# Build gas species
gas_species = (
    par.gas.GasDataBuilder()
    .set_molar_mass([molar_mass_water], "kg/mol")
    .set_concentration([water_concentration], "kg/m^3")
    .set_names(["H2O"])
    .set_partitioning([True])
    .build()
)

# Build atmosphere
environment = par.gas.EnvironmentData(
    temperature=[temperature],
    pressure=[101325.0],
    saturation_ratio=[[initial_rh]],
)

print(f"Initial water concentration: {water_concentration:.6e} kg/m^3")
print(f"Saturation concentration: {saturation_concentration:.6e} kg/m^3")

# %% [markdown]
# ## Setup Particles
#
# Next, we create a lognormal particle distribution using the particle-resolved representation.

# %%
# Particle parameters
density = 1000.0  # kg/m^3 (water droplets)
volume_sim = 1e-6  # m^3 (1 cm^3)
n_particles = 1000  # Moderate number for demonstration

# Generate lognormal particle distribution
rng = np.random.default_rng(RANDOM_SEED)
particle_radii = rng.lognormal(mean=np.log(100e-9), sigma=0.5, size=n_particles)

# Calculate particle masses
particle_masses = (4 / 3) * np.pi * particle_radii**3 * density
mass_speciation = particle_masses.reshape(-1, 1)  # Single species
densities = np.array([density])  # Single species density array

# Activity and surface strategies
activity_strategy = par.particles.ActivityIdealMass()
surface_strategy = par.particles.SurfaceStrategyVolume(
    surface_tension=0.072, density=density
)

# Build particle representation
particles = par.particles.create_resolved_particle_data(
    masses=mass_speciation,
    density=densities,
    volume=volume_sim,
)
# Every strategy receives its own explicit process configuration.
configuration = dict(
    activity_strategy=activity_strategy,
    surface_strategy=surface_strategy,
    vapor_pressure_strategy=vapor_pressure_water,
    species_map=CondensationSpeciesMap(("H2O",), ((0, 0),)),
)

print(f"Created {n_particles} particles")
print(f"Initial total particle mass: {np.sum(particle_masses):.6e} kg")
print(f"Mean particle radius: {np.mean(particle_radii) * 1e9:.1f} nm")


# %% [markdown]
# ## Comparing Theta Modes
#
# The `theta_mode` parameter controls how the timestep is split between two passes:
#
# | Mode | Theta Value | Characteristics |
# |------|-------------|-----------------|
# | `"half"` | theta = 0.5 | Deterministic, symmetric splitting |
# | `"random"` | theta ~ U[0,1] | Stochastic, best mass conservation |
# | `"batch"` | theta = 1.0 | Staggering via batch updates only |
#
# Let's run the same simulation with each mode and compare the results.


# %%
def run_simulation(strategy, aerosol, time_step, n_steps, sub_steps=1):
    """Run condensation simulation and track mass over time."""
    condensation = par.dynamics.MassCondensation(condensation_strategy=strategy)

    # Copy all native containers for an independent simulation
    current_aerosol = par.Aerosol(
        gas=aerosol.gas.copy(),
        environment=aerosol.environment.copy(),
        particles=aerosol.particles.copy(),
    )

    particle_mass = float(
        np.sum(current_aerosol.particles.species_mass_inventory)
    )
    gas_mass = np.asarray(
        current_aerosol.gas.concentration
        * current_aerosol.particles.volume[:, None]
    ).item()
    mass_history = [particle_mass]
    gas_mass_history = [gas_mass]

    for _step in range(n_steps):
        # Execute condensation
        current_aerosol = condensation.execute(
            current_aerosol, time_step, sub_steps
        )

        # Record masses after step
        particle_mass = float(
            np.sum(current_aerosol.particles.species_mass_inventory)
        )
        gas_mass = np.asarray(
            current_aerosol.gas.concentration
            * current_aerosol.particles.volume[:, None]
        ).item()
        mass_history.append(particle_mass)
        gas_mass_history.append(gas_mass)

    combined = np.array(mass_history) + np.array(gas_mass_history)
    np.testing.assert_allclose(combined, combined[0], rtol=1e-12, atol=1e-30)
    return np.array(mass_history), np.array(gas_mass_history)


# Create base aerosol
aerosol_base = par.Aerosol(
    gas=gas_species, environment=environment, particles=particles
)

# Simulation parameters
time_step = 0.01  # seconds
n_steps = 100
sub_steps = 1

# Run with each theta mode
modes = ["half", "random", "batch"]
results = {}

for mode in modes:
    strategy = par.dynamics.CondensationIsothermalStaggered(
        molar_mass=molar_mass_water,
        **configuration,
        theta_mode=mode,
        num_batches=10,
        shuffle_each_step=True,
        random_state=RANDOM_SEED,
    )
    particle_mass, gas_mass = run_simulation(
        strategy, aerosol_base, time_step, n_steps, sub_steps
    )
    results[mode] = {
        "particle_mass": particle_mass,
        "gas_mass": gas_mass,
        "total_mass": particle_mass + gas_mass,
    }
    print(f"Mode '{mode}': Final particle mass = {particle_mass[-1]:.6e} kg")

# %% [markdown]
# ## Mass Conservation Verification
#
# One key advantage of staggered stepping is improved mass conservation. Let's verify by tracking total system mass (particles + gas) over time and computing the relative error:
#
# $$\text{Relative Error} = \frac{|M(t) - M(0)|}{M(0)}$$
#
# Good mass conservation should show relative error < 10^-10.

# %%
time = np.arange(n_steps + 1) * time_step

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

# Plot total mass over time
for mode, data in results.items():
    ax1.plot(time, data["total_mass"], label=f"{mode}", linewidth=2)

ax1.set_xlabel("Time (s)")
ax1.set_ylabel("Total System Mass (kg)")
ax1.set_title("Total Mass Over Time")
ax1.legend()
ax1.ticklabel_format(style="scientific", axis="y", scilimits=(0, 0))

# Plot relative mass error
for mode, data in results.items():
    initial_mass = data["total_mass"][0]
    relative_error = np.abs(data["total_mass"] - initial_mass) / initial_mass
    ax2.plot(time, relative_error, label=f"{mode}", linewidth=2)

ax2.set_xlabel("Time (s)")
ax2.set_ylabel("Relative Mass Error")
ax2.set_title("Mass Conservation Error")
ax2.set_yscale("log")
ax2.legend()
ax2.axhline(
    y=1e-10, color="gray", linestyle="--", alpha=0.5, label="Target (1e-10)"
)

plt.tight_layout()
plt.show()

# Print final relative errors
print("\nFinal Relative Mass Errors:")
for mode, data in results.items():
    initial_mass = data["total_mass"][0]
    final_error = np.abs(data["total_mass"][-1] - initial_mass) / initial_mass
    print(f"  {mode}: {final_error:.2e}")

# %% [markdown]
# ## Stability Comparison: Staggered vs Simultaneous
#
# At large time steps, simultaneous condensation (where all particles update at once) can become unstable. Let's compare:
#
# - **Simultaneous**: `CondensationIsothermal` - all particles see same gas concentration
# - **Staggered**: `CondensationIsothermalStaggered` - particles update in sequence with gas feedback
#
# We use a larger time step to compare these two trajectories; this is not
# a general stability or convergence study.

# %%
# Larger time step for the bounded comparison
large_time_step = 0.1  # 10x larger
n_steps_stability = 50

# Staggered strategy
staggered_strategy = par.dynamics.CondensationIsothermalStaggered(
    molar_mass=molar_mass_water,
    **configuration,
    theta_mode="random",
    num_batches=10,
    random_state=RANDOM_SEED,
)

# Simultaneous strategy
simultaneous_strategy = par.dynamics.CondensationIsothermal(
    molar_mass=molar_mass_water,
    **configuration,
    diffusion_coefficient=2e-5,
    accommodation_coefficient=1.0,
    update_gases=True,
)

# Run both strategies
staggered_mass, staggered_gas = run_simulation(
    staggered_strategy, aerosol_base, large_time_step, n_steps_stability
)
simultaneous_mass, simultaneous_gas = run_simulation(
    simultaneous_strategy, aerosol_base, large_time_step, n_steps_stability
)

# Plot comparison
time_stability = np.arange(n_steps_stability + 1) * large_time_step

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

# Particle mass evolution
ax1.plot(
    time_stability, staggered_mass, label="Staggered (random)", linewidth=2
)
ax1.plot(
    time_stability,
    simultaneous_mass,
    label="Simultaneous",
    linewidth=2,
    linestyle="--",
)
ax1.set_xlabel("Time (s)")
ax1.set_ylabel("Particle Mass (kg)")
ax1.set_title(f"Particle Mass (dt = {large_time_step}s)")
ax1.legend()

# Mass conservation comparison
staggered_total = staggered_mass + staggered_gas
simultaneous_total = simultaneous_mass + simultaneous_gas

staggered_error = (
    np.abs(staggered_total - staggered_total[0]) / staggered_total[0]
)
simultaneous_error = (
    np.abs(simultaneous_total - simultaneous_total[0]) / simultaneous_total[0]
)

ax2.plot(time_stability, staggered_error, label="Staggered", linewidth=2)
ax2.plot(
    time_stability,
    simultaneous_error,
    label="Simultaneous",
    linewidth=2,
    linestyle="--",
)
ax2.set_xlabel("Time (s)")
ax2.set_ylabel("Relative Mass Error")
ax2.set_title("Mass Conservation at Large Time Steps")
ax2.set_yscale("log")
ax2.legend()

plt.tight_layout()
plt.show()

# %% [markdown]
# ## Parameter Selection Guide
#
# ### Choosing the theta mode
#
# | Use Case | Recommended Mode |
# |----------|------------------|
# | General purpose, good default | `"half"` |
# | Best mass conservation | `"random"` |
# | Maximum staggering effect | `"batch"` |
# | Reproducible results needed | `"half"` or `"random"` with fixed seed |
#
# ### Choosing the number of batches
#
# More batches change how frequently gas feedback is applied. Accuracy and
# cost depend on the workload and require separate convergence/timing studies.
#
# - **Small systems (<100 particles)**: `num_batches=1-5`
# - **Medium systems (100-1000 particles)**: `num_batches=10-20`
# - **Large systems (>1000 particles)**: `num_batches=50-100`
#
# The batch count is automatically clipped to the number of particles to avoid empty batches.

# %%
# Demonstrate effect of num_batches
batch_counts = [1, 5, 10, 20]
batch_results = {}

for num_batches in batch_counts:
    strategy = par.dynamics.CondensationIsothermalStaggered(
        molar_mass=molar_mass_water,
        **configuration,
        theta_mode="half",
        num_batches=num_batches,
        random_state=RANDOM_SEED,
    )
    particle_mass, gas_mass = run_simulation(
        strategy, aerosol_base, time_step, n_steps
    )
    batch_results[num_batches] = particle_mass + gas_mass

# Plot mass conservation vs batch count
fig, ax = plt.subplots(figsize=(8, 5))

for num_batches, total_mass in batch_results.items():
    initial = total_mass[0]
    error = np.abs(total_mass - initial) / initial
    ax.plot(time, error, label=f"num_batches={num_batches}", linewidth=2)

ax.set_xlabel("Time (s)")
ax.set_ylabel("Relative Mass Error")
ax.set_title("Effect of Batch Count on Mass Conservation")
ax.set_yscale("log")
ax.legend()
plt.tight_layout()
plt.show()

# %% [markdown]
# ## Summary
#
# This tutorial demonstrated the **staggered condensation stepping** feature in particula:
#
# ### Key Takeaways
#
# 1. **Staggered stepping** uses sequential particle updates with gas feedback.
# 2. The `half`, `random`, and `batch` modes give distinct update orderings.
# 3. These workloads check extensive particle-plus-gas mass with `rtol=1e-12`
#    and `atol=1e-30`; observed roundoff does not rank the modes' accuracy.
# 4. Both methods remain finite in this comparison. Choosing a time step still
#    requires a convergence study for the intended physical case.
#
# ### Quick Reference
#
# ```python
# # Reuse the explicit species mapping and physical configuration above
# strategy = par.dynamics.CondensationIsothermalStaggered(
#     molar_mass=0.018,  # kg/mol
#     **configuration,
#     theta_mode="random",
#     num_batches=10,
#     shuffle_each_step=True,
#     random_state=42,  # for reproducibility
# )
#
# # Use with MassCondensation runnable
# condensation = par.dynamics.MassCondensation(condensation_strategy=strategy)
# aerosol = condensation.execute(aerosol, time_step=0.1, sub_steps=10)
# ```
#
# ### Related Resources
#
# - [Condensation 3: Masses Resolved](condensation_3_massresolved.ipynb) - Basic particle-resolved condensation
# - [CondensationIsothermalStaggered API](https://uncscode.github.io/particula/API/particula/dynamics/condensation/condensation_strategies/) - Full API documentation
