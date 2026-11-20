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
# # Will this population initially grow or evaporate?
#
# Before integrating a condensation model, inspect the direction and meaning of
# its initial mass-transfer rate. A class named `MassCondensation` can describe
# evaporation too: the sign follows the gas-to-surface vapor-pressure difference.
#
# This CPU example constructs a glycerol population and evaluates its initial
# rate at 298.15 K. It does not advance time. The aim is to distinguish the
# population's initial size distribution from its tendency to gain or lose mass.
# Read [size distributions](../building-an-aerosol/particles/aerosol_distributions.py)
# and [vapor pressure](../building-an-aerosol/gas/vapor_pressure.py) for prerequisites.

# %%
# In Colab uncomment the following command to install particula:
# Install from the repository root: `python -m pip install -e ".[extra]"`
import matplotlib.pyplot as plt
import numpy as np

# particula imports
import particula as par
from particula import plotting
from particula.dynamics.condensation.species_mapping import (
    CondensationSpeciesMap,
)

plt.style.use("particula.default")

# %% [markdown]
# ## Specify the particles and their vapor reservoir
#
# The example uses a single material, spherical particles with density
# 1,260 kg m⁻³, a lognormal radius distribution, and total number concentration
# 10¹⁰ m⁻³. Radii are in metres; `discrete` concentration stores number per volume
# in each bin. The 500 radius points span 10 nm–10 µm.
#
# Gas concentration starts at 1% of the saturation concentration calculated by
# the configured Clausius–Clapeyron relation. The vapor-pressure parameters are
# demonstration inputs, not a calibration established here. This strongly
# undersaturated setting gives a clear reason to inspect evaporation. Surface
# curvature and activity enter the equilibrium surface pressure separately.

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

sat_factor = 0.01  # 1% of saturation concentration
glycerol_gas = (
    par.gas.GasDataBuilder()
    .set_names(["Glycerol"])
    .set_molar_mass([molar_mass_glycerol], "kg/mol")
    .set_concentration([sat_concentration * sat_factor], "kg/m^3")
    .set_partitioning([True])
    .build()
)
environment = par.gas.EnvironmentData(
    temperature=[298.15],
    pressure=[101325.0],
    saturation_ratio=[[sat_factor]],
)
bins = np.logspace(-8, -5, 500)
particles = par.particles.create_lognormal_binned_particle_data(
    distribution_type="discrete",
    radius_bins=bins,
    mode=[100e-9],
    geometric_standard_deviation=[1.3],
    number_concentration=[1e10],
    density=1260.0,
)
aerosol = par.Aerosol(
    particles=particles,
    gas=glycerol_gas,
    environment=environment,
)

print(aerosol)

# %% [markdown]
# ## Evaluate a rate without advancing the state
#
# `CondensationIsothermal` combines vapor pressure, ideal activity, surface
# curvature, transport parameters, and an explicit gas-to-particle species map.
# The map connects the glycerol gas lane to the glycerol particle material.
# `MassCondensation` wraps that strategy for native aerosol state.
#
# The call below uses `rate(aerosol)`. Its output retains box, particle-slot,
# and material axes, even with one box and one material. For this binned
# population, entries are contributions to mass-density change in kg m⁻³ s⁻¹,
# including each bin's number concentration. They are not individual-particle
# rates in kg s⁻¹. Summing over slots would give the population rate; the printed
# mean is only an average of bin contributions.
#
# Positive entries indicate uptake and negative entries evaporation. The
# pressure driving force is described in the [condensation equations](../../../theory/dynamics/condensation-equations.md).
# The figure shows the initial PMF, not a growth trajectory.

# %%
# define the condensation process
condensation_isothermal = par.dynamics.CondensationIsothermal(
    molar_mass=glycerol_gas.molar_mass,
    activity_strategy=par.particles.ActivityIdealMass(),
    surface_strategy=par.particles.SurfaceStrategyVolume(density=1260.0),
    vapor_pressure_strategy=vapor_pressure_strategy,
    species_map=CondensationSpeciesMap(("Glycerol",), ((0, 0),)),
    diffusion_coefficient=2e-5,
    accommodation_coefficient=0.1,  # makes things go slower/faster
)
condensation_process = par.dynamics.MassCondensation(
    condensation_strategy=condensation_isothermal
)
print(f"Strategy: {condensation_isothermal}")
print(f"Runnable: {condensation_process}")

# %%
# Get the initial state for plotting
initial_radii = aerosol.particles.radii[0].copy()
initial_concentration = aerosol.particles.slot_concentration_density[0].copy()

# Calculate the condensation rate (mass change per particle per second)
rate = condensation_process.rate(aerosol)
# Runnable results retain (boxes, slots, particle species), even for B=1.
print(f"Condensation rate shape: {rate.shape}")
print(f"Mean rate: {np.mean(rate):.2e} kg/m^3/s per slot")

# %%
# Plot the initial particle distribution
fig, ax = plotting.subplots(size="wide")
ax.semilogx(
    initial_radii * 1e9,
    initial_concentration,
    label="Initial PMF",
    linewidth=2,
)
ax.set_xlabel("Radius (nm)")
ax.set_ylabel("Concentration (1/m^3)")
plotting.legend(ax)
ax.set_title("Initial Particle Distribution (PMF)")
ax.grid(alpha=0.3)
plotting.fit(fig)
plt.show()

# %% [markdown]
# ## Interpret the displayed state and rate
#
# The initial spectrum identifies where particles are placed in radius bins.
# Its ordinate is bin number concentration, so it should be summed rather than
# integrated against radius. It cannot show growth because no timestep has run.
#
# Use the printed signed rate to interpret the initial tendency in the chosen
# undersaturated environment. Neither the mean rate nor the initial PMF tells
# us how long evaporation takes: radii, surface pressure, and gas concentration
# would change during integration.

# %% [markdown]
# ## From an initial tendency to a research question
#
# This calculation connects a specified aerosol and vapor reservoir to an
# initial mass-transfer tendency. It establishes neither an equilibrium endpoint
# nor a numerically converged time history.
#
# A useful next investigation is to vary the initial saturation factor and
# compare the rate by radius, checking where its sign changes. Keep the species
# map and units fixed; account for curvature before equating flat-surface
# saturation with zero rate for every particle. This is a proposed extension,
# not a scan executed here.
#
# For an actual time history, continue to the
# [staggered condensation example](../../examples/particle-processes/condensation/staggered_condensation_example.py).
# Check gas-plus-particle inventories and timestep sensitivity separately:
# conservation alone does not establish trajectory accuracy.
