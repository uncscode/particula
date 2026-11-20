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
# # From a population to an evolving aerosol
#
# What information must be shared before particles and gas can evolve together?
# An `Aerosol` holds particle, gas, and environment containers; a process supplies
# the physical rules. This example first assembles a single-box glycerol state,
# then uses a separate two-box population to demonstrate dilution and wall loss.
# You will learn to distinguish stored counts, concentrations, and material mass
# before interpreting a process result. Read [particle construction](particles/particle_data.py)
# and [gas storage](gas/gas_data.py) if those quantities are unfamiliar.

# %%
import numpy as np
import particula as par

# %% [markdown]
# ## Specify an initial vapor reservoir
#
# The vapor-pressure strategy converts a reference pressure and temperature into
# an initial saturation concentration at 298.15 K. Half of that value is put in
# the gas container. The retained glycerol parameters are arithmetic demonstration
# inputs, not a calibrated glycerol property model; the printed concentration
# should not be used as a reference measurement. The strategy is configuration,
# not a field stored in `GasData`.

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
# The identity assertions show that assembly holds the supplied containers.
# Consequently, editing a held array edits the aerosol. A detached `copy()` is
# needed for a control case. The printed number and mass densities describe the
# initialized population; no glycerol condensation step has been executed.

# %% [markdown]
# ## Compare two boxes under the same dilution law
#
# The next population is particle-resolved, so its weights are represented counts.
# The boxes have different counts and volumes (0.25 and 4 m³); equal raw weights
# would therefore not imply equal number densities. Dilution at 0.1 s⁻¹ over 2 s
# multiplies each inventory by exp(−0.2). The assertions compare with that analytic
# factor, while `rate` returns slot number-density rates in m⁻³ s⁻¹.
#
# This gives a useful controlled comparison: the fractional change is the same
# in both boxes even though their absolute concentrations differ.

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
# ## Add a stochastic loss process
#
# Spherical wall loss now acts on the diluted population. Stable logical box IDs
# keep random streams associated with the boxes. Reuse the context to continue
# those streams; a seed is not a request to restart them on every call. The printed
# post-loss densities describe this realization, not an ensemble-average loss law.

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

# %% [markdown]
# ## What the assembled state lets you investigate
#
# The example separates three questions: what is stored, how a deterministic
# process changes it, and how a stochastic step samples change. The dilution
# assertions support the specified exponential decay; they do not validate the
# wall-loss model or the illustrative glycerol parameters.
#
# For a chamber comparison, vary physical volume while preserving initial number
# density, then compare normalized losses across repeated wall-loss realizations.
# That requires adjusting represented counts and checking material inventories,
# not merely changing `volume`. Continue with [state access and replacement](native_state_access.py)
# or the [wall-loss examples](../../examples/particle-processes/wall-loss/index.md).
