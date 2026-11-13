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
# An aerosol holds three native containers: particle material and population,
# ordered gas species, and environment. Physics configuration stays separate.
# Here we initialize glycerol at half its saturation concentration and a
# lognormal particle population. Construction retains the supplied objects.

# %%
import numpy as np
import particula as par

# %% [markdown]
# ## Glycerol thermodynamics
#
# A vapor-pressure strategy computes the initial saturation concentration.
# It is not stored in gas data; a subsequent process owns its configuration.

# %%
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

sat_factor = 0.5
glycerol_gas = (
    par.gas.GasDataBuilder()
    .set_names(["Glycerol"])
    .set_molar_mass([molar_mass_glycerol], "kg/mol")
    .set_concentration([sat_concentration * sat_factor], "kg/m^3")
    .set_partitioning([True])
    .build()
)
environment = par.gas.EnvironmentData(
    temperature=np.array([298.15]),
    pressure=np.array([101325.0]),
    saturation_ratio=np.array([[sat_factor]]),
)

# %% [markdown]
# ## Native particle initialization
#
# Discrete bins store number density in m^-3 at reference volume 1 m^3.
# Each mass lane is per-particle mass in kg, not population mass. The preset
# retains the original 100 nm mode, geometric spread, and number density.

# %%
particles = par.particles.create_lognormal_binned_particle_data(
    distribution_type="discrete",
    mode=[100e-9],
    geometric_standard_deviation=[1.5],
    number_concentration=[1e10],  # 1e4 cm^-3
    density=1260.0,  # 1.26 g/cm^3
)

# %% [markdown]
# ## Assemble and inspect the held state
#
# Partitioning flags distinguish condensable species from inert gas lanes in
# one ordered gas container. No species list is filtered during construction.

# %%
aerosol = par.Aerosol(
    particles=particles, gas=glycerol_gas, environment=environment
)
assert aerosol.particles is particles
assert aerosol.gas is glycerol_gas
assert aerosol.environment is environment
print(aerosol)
print("Number density [m^-3]:", particles.concentration.sum(axis=1))
print(
    "Particle mass density [kg/m^3]:",
    np.sum(particles.masses * particles.concentration[:, :, None], axis=1),
)

# %% [markdown]
# The aggregate exposes its held containers through properties. Mutating their
# arrays changes the held state; use `copy()` for detached inspection. Processes
# validate their own scientific requirements before using this state.

# %% [markdown]
# ## Two boxes through the same CPU runnable
#
# Boxes share one population type and material layout. Each resolved box has
# its own physical volume, environment, gas inventory, and particle counts.
# Runnables iterate boxes; direct strategy methods still operate on one box.

# %%
box_particles = par.particles.create_resolved_particle_data(
    masses=np.array([[[1e-18], [2e-18]], [[1e-18], [2e-18]]]),
    density=1000.0,
    concentration=np.array([[10.0, 20.0], [30.0, 40.0]]),
    volume=np.array([0.25, 4.0]),
)
box_gas = (
    par.gas.GasDataBuilder()
    .set_names(["inert"])
    .set_molar_mass([0.028])
    .set_concentration([[1e-6], [2e-6]])
    .set_partitioning([False])
    .build()
)
box_aerosol = par.Aerosol(
    particles=box_particles,
    gas=box_gas,
    environment=par.gas.EnvironmentData(
        [280.0, 310.0], [101325.0, 90000.0], [[0.0], [0.0]]
    ),
)
dilution = par.dynamics.Dilution(par.dynamics.DilutionStrategy(0.1))
initial_counts = box_particles.concentration.copy()
initial_gas = box_gas.concentration.copy()
box_rates = dilution.rate(box_aerosol)
assert box_rates.shape == (2, 2)
np.testing.assert_allclose(
    box_rates, -0.1 * initial_counts / box_particles.volume[:, None]
)
assert dilution.execute(box_aerosol, time_step=2.0, sub_steps=2) is box_aerosol
np.testing.assert_allclose(
    box_particles.concentration, initial_counts * np.exp(-0.2)
)
np.testing.assert_allclose(box_gas.concentration, initial_gas * np.exp(-0.2))
print(
    "Two-box number densities after dilution [m^-3]:",
    box_particles.number_density,
)
print("Runnable rate shape (boxes, slots):", box_rates.shape)

# %% [markdown]
# Stochastic multi-box execution requires stable logical box IDs. Reuse this
# context to continue streams; if arrays are reordered, use a matching
# `context.with_box_ids(...)` view. Reusing a seed does not reset a context.
# A failed process substep commits neither box state nor random streams.

# %%
context = par.CPUExecutionContext(
    root_seed=42, box_ids=("chamber-a", "chamber-b")
)
wall_loss = par.dynamics.WallLoss(
    par.dynamics.SphericalWallLossStrategy(
        wall_eddy_diffusivity=0.001,
        chamber_radius=0.5,
        distribution_type="particle_resolved",
    )
)
wall_loss.execute(box_aerosol, time_step=0.1, context=context)
print("Per-box volumes [m^3]:", box_particles.volume)
print("Number densities after wall loss [m^-3]:", box_particles.number_density)
