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
# # Water uptake from a maintained vapor reservoir
#
# Can a hygroscopic particle grow when the gas is below pure-water saturation?
# This example integrates composition-dependent uptake at prescribed 80% water
# saturation. Activity lowers the surface equilibrium pressure; curvature opposes
# uptake. The gas reservoir is maintained (`update_gases=False`), so particle mass
# can increase without a matching decrease in the stored gas concentration.
#
# Begin with the [initial-rate example](condensation_1_bin.py) and
# [activity models](../../examples/thermodynamics/activity/index.md). Here each moving bin
# stores a component-mass vector and a number concentration. Different bins may
# have different compositions; the model does not represent a composition
# distribution within one bin.

# %%
# In Colab uncomment the following command to install particula:
# Install from the repository root: `python -m pip install -e ".[extra]"`
import matplotlib.pyplot as plt
import numpy as np
import particula as par
from particula import plotting
from particula.dynamics.condensation.species_mapping import (
    CondensationSpeciesMap,
)

plt.style.use("particula.default")

# %% [markdown]
# ## Define the reservoir and component assumptions
#
# Water starts at 0.8 times its Buck saturation concentration at 298.15 K and
# 101325 Pa. The second component is assigned negligible vapor pressure. The
# retained label `NH4HSO4` is a storage identifier; the molar mass, density, and
# κ parameters are illustrative and should not be read as a consistent chemical
# specification of ammonium bisulfate. A material-specific study must reconcile
# those inputs with its chosen substance.

# %%
# Ammonium sulfate and water vapor pressure
molar_mass_ammonium_sulfate = 132.14e-3  # kg/mol
molar_mass_water = 18.015e-3  # kg/mol
parameters_vapor = {
    "vapor_pressure": 1e-20,
    "vapor_pressure_units": "Pa",
}
vapor_pressure_ammonium = par.gas.VaporPressureFactory().get_strategy(
    "constant", parameters_vapor
)
vapor_pressure_water = par.gas.VaporPressureFactory().get_strategy("water_buck")

water_sat = vapor_pressure_water.saturation_concentration(
    molar_mass=molar_mass_water, temperature=298.15
)
water_concentration = 0.8 * water_sat

combo_gas = (
    par.gas.GasDataBuilder()
    .set_molar_mass(
        np.array([molar_mass_water, molar_mass_ammonium_sulfate]), "kg/mol"
    )
    .set_concentration(np.array([water_concentration, 0.0]), "kg/m^3")
    .set_names(["H2O", "NH4HSO4"])
    .set_partitioning([True, True])
    .build()
)

environment = par.gas.EnvironmentData(
    temperature=[298.15],
    pressure=[101325.0],
    saturation_ratio=[[water_concentration / water_sat, 0.0]],
)

# %% [markdown]
# ## Sample sizes, then assign physical abundance
#
# The first figure is a probability density per metre of radius for 1,000 sampled
# sizes. The two lognormal radius scales and relative sampling weights determine
# its shape. Each resulting computational bin is subsequently assigned 10 cm⁻³,
# so the physical number concentration is determined by these assigned weights,
# not by the normalized histogram height.
#
# Material lanes are water followed by the hygroscopic component. The κ model
# and volume-weighted surface model connect evolving composition to equilibrium
# surface pressure. Follow this ordering into the explicit gas-to-material map.

# %%
# sample
np.random.seed(42)
particles_sample = par.particles.get_lognormal_sample_distribution(
    mode=np.array([100, 1000]) * 1e-9,
    geometric_standard_deviation=np.array([1.3, 1.5]),
    number_of_particles=np.array([1e3, 1e2]),
    number_of_samples=1000,
)

# histogram lognormal
bins_lognormal = np.logspace(-8, -4, 100)
bins, edges = np.histogram(particles_sample, bins=bins_lognormal, density=True)
# plot
fig, ax = plotting.subplots(size="wide")
ax.bar(edges[:-1], bins, width=np.diff(edges), align="edge")
ax.set_xscale("log")
ax.set_xlabel("Radius (m)")
ax.set_ylabel("Probability density per metre of radius")
plotting.fit(fig)
plt.show()

# %%
# particle radis to mass
density = 1.26e3  # kg/m^3
particle_mass = density * 4 / 3 * np.pi * particles_sample**3
mass_speciation = np.array(
    [particle_mass * 0, particle_mass]
).T  # water, ammonium sulfate
concentration = np.ones_like(particles_sample) * 1e1
densities = np.array([1000, 1.26e3])  # kg/m^3

print(mass_speciation.shape)

activity_strat = (
    par.particles.ActivityKappaParameterBuilder()
    .set_density(densities, "kg/m^3")
    .set_kappa(np.array([0.0, 0.61]))
    .set_molar_mass(
        np.array([molar_mass_water, molar_mass_ammonium_sulfate]), "kg/mol"
    )
    .set_water_index(0)
    .build()
)

surface_strat = (
    par.particles.SurfaceStrategyVolumeBuilder()
    .set_density(densities, "kg/m^3")
    .set_surface_tension(np.array([0.072, 0.092]), "N/m")
    .build()
)

particles = (
    par.particles.ParticleDataBuilder()
    .set_distribution_type("discrete")
    .set_masses(mass_speciation)
    .set_density(densities)
    .set_charge(np.zeros(len(concentration)))
    .set_concentration(concentration, "1/cm^3")
    .set_volume(np.array([1.0]))
    .build()
)
aerosol = par.Aerosol(
    particles=particles,
    gas=combo_gas,
    environment=environment,
)

print(aerosol)

# %% [markdown]
# ## Advance mass while keeping the reservoir fixed
#
# The accommodation coefficient is 0.1. Each `execute` call advances 11 s with
# 10,000 substeps; the two calls therefore display 11 s and 22 s after initialization.
# The unused time-array setup does not define the displayed timeline. Temperature
# and gas concentration stay prescribed.
#
# The overlay counts computational bins by radius, unlike the earlier normalized
# probability-density plot. All bins have equal number concentration here, so its
# shape also tracks the population shape up to a common factor. It does not show
# component mass or prove an equilibrium endpoint.

# %%
# define the condensation process
condensation_isothermal = par.dynamics.CondensationIsothermal(
    activity_strategy=activity_strat,
    surface_strategy=surface_strat,
    vapor_pressure_strategy=[vapor_pressure_water, vapor_pressure_ammonium],
    species_map=CondensationSpeciesMap(("H2O", "NH4HSO4"), ((0, 0), (1, 1))),
    molar_mass=np.array(
        [molar_mass_water, molar_mass_ammonium_sulfate]
    ),  # kg/mol
    accommodation_coefficient=0.1,  # makes things go slower/faster
    update_gases=False,
)
condensation_process = par.dynamics.MassCondensation(
    condensation_strategy=condensation_isothermal
)

# define the time array in seconds
time_step = 11
sub_steps = 10000
time_array = np.arange(0, 5, time_step)
total_mass = np.zeros_like(time_array)

# output arrays
aerosol_sim = []


fig, ax = plotting.subplots(size="wide")
bins, edges = np.histogram(aerosol.particles.radii[0], bins=bins_lognormal)
ax.bar(edges[:-1], bins, width=np.diff(edges), align="edge", label="Initial")

print(aerosol)
# one step
initial_gas = combo_gas.concentration.copy()
initial_water_mass = particles.species_mass_density[0, 0]
aerosol = condensation_process.execute(aerosol, time_step, sub_steps)
bins, edges = np.histogram(aerosol.particles.radii[0], bins=bins_lognormal)
ax.bar(
    edges[:-1],
    bins,
    width=np.diff(edges),
    align="edge",
    label="After 1 step",
    alpha=0.8,
)

print(aerosol)
# 10 seconds
aerosol = condensation_process.execute(aerosol, time_step, sub_steps)
bins, edges = np.histogram(aerosol.particles.radii[0], bins=bins_lognormal)
ax.bar(
    edges[:-1],
    bins,
    width=np.diff(edges),
    align="edge",
    label="After 2 steps",
    alpha=0.7,
)
print(aerosol)

ax.set_xscale("log")
# ax.set_yscale("log")
plotting.legend(plt.gca())
ax.set_xlabel("Radius (m)")
ax.set_ylabel("Computational bins")
plotting.fit(fig)
plt.show()

# %% [markdown]
# ## Check the open-system interpretation
#
# The following checks require unchanged gas concentration, finite nonnegative
# particle masses, and positive net water uptake. They match the intended maintained
# reservoir. A gas-plus-particle closed-inventory assertion would be inappropriate:
# water supplied from outside this modeled inventory supports the growth.

# %%
# Open-system checks: the fixed reservoir supplies net condensed water.
np.testing.assert_array_equal(combo_gas.concentration, initial_gas)
if not np.all(np.isfinite(particles.masses)) or np.any(particles.masses < 0):
    raise AssertionError("Nonphysical particle masses")
if particles.species_mass_density[0, 0] <= initial_water_mass:
    raise AssertionError("Expected water uptake from the prescribed reservoir")

# %% [markdown]
# ## Turn uptake into a comparison
#
# The calculation demonstrates why pure-water subsaturation need not imply drying
# for a hygroscopic solution. Its evidence is net water uptake under the specified
# activity, curvature, and maintained-vapor assumptions, not validation of a
# particular salt or convergence of the 22 s state.
#
# To ask whether a chamber's finite vapor supply limits growth, compare with the
# [particle-resolved finite-reservoir example](condensation_3_massresolved.py).
# Hold initial composition and physical number density fixed when making that
# comparison, track gas-plus-particle water, and refine the timestep separately
# from checking conservation. The published examples have different initial
# conditions and cannot be interpreted as a controlled method comparison as-is.
