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
# # When a size-distribution plot hides a composition error
#
# Can a representation that conserves each species during condensation remain
# suitable after coagulation begins? This example follows organic uptake onto
# sulfate seeds, then exposes a limitation of fixed-composition binned coagulation.
# The question matters when interpreting secondary organic aerosol: a plausible
# size curve does not guarantee a valid species inventory.
#
# **This is a method-boundary demonstration, not a reliable composition forecast.**
# Condensation creates size-dependent composition. The subsequent binned
# coagulation changes bin populations without transporting donor species into
# destination bins. Its saved per-species inventory changes range from about
# −59% to +35%. Those are observed defects, not acceptable tolerances, and a
# smaller timestep cannot supply the missing species transport.
#
# Read the [coagulation contract](../../guide/processes/coagulation.md). Use a
# [supported resolved representation](../processes/coagulation/coagulation_3_particle_resolved_pattern.py)
# when collisions must mix size-dependent species composition.

# %% [markdown]
# ## Prepare the calculation and the comparison
#
# The imports support native aerosol construction, process execution, and plotting.
# Keep the full 250-bin grid and 100,000 condensation substeps when reproducing
# the saved run. The important comparison is between inventories before and after
# each stage, not merely whether the script finishes.

# %%

import copy

import matplotlib.pyplot as plt
import numpy as np
import particula as par
from particula import plotting
from particula.dynamics.condensation.species_mapping import (
    CondensationSpeciesMap,
)

# Organic density estimation for species
from particula.particles import get_organic_density_array
from tqdm import tqdm

plt.style.use("particula.default")

# plot settings
TAILWIND = par.util.colors.TAILWIND

# %% [markdown]
# ## Make the illustrative inputs explicit
#
# Ten organic surrogates have specified molar masses in g/mol, O:C and H:C
# ratios, and dry saturation concentrations in µg/m³. Density estimates use those
# elemental descriptors. The initial gas concentrations are multiplied by 10,000;
# this intentionally concentrated setup is not a typical ambient loading or a
# validated experimental mixture. The source does not supply traceable material
# references for every numerical input.
#
# Saturation concentration characterizes pure-component volatility at the chosen
# temperature. It does not mean that a mixture is at equilibrium whenever its gas
# concentration reaches that value: activity and curvature also enter the
# particle-surface pressure.

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


density_organics_kg_m3 = get_organic_density_array(
    molar_mass=M_gmol,
    oxygen2carbon=OC_ratio,
    hydrogen2carbon=HC_ratio,
)  # kg/m^3

c_total_ug_per_m3 = (
    np.array([8.79, 3.98, 1.13, 4.07, 0.628, 0.919, 0.766, 1.02, 0.399, 0.313])
    * 10000
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

# sulfate properties
sulfate_density = 1.77 * 1000  # kg/m^3
sulfate_molar_mass = 96.06  # g/mol

temperature_K = 298.15

# %% [markdown]
# ## Convert saturation concentration to a pressure model
#
# Each organic receives a strategy constructed from its saturation concentration
# and molar mass at 298.15 K. The builder converts µg/m³ and g/mol to consistent
# pressure inputs. A conversion at this reference temperature does not itself
# supply a measured temperature dependence or a vapor-pressure validation.

# %%
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

# %% [markdown]
# ## Align gas and particle species
#
# The unified order is sulfate followed by the ten organics. Gas concentrations
# are converted to kg/m³; temperature and pressure occupy the environment
# container. Sulfate starts with zero gas concentration and an effectively
# nonvolatile vapor-pressure input. Its later exclusion from condensation is
# explicit, rather than inferred from its name.

# %%
sulfate_vapor_pressure = (
    par.gas.ConstantVaporPressureBuilder()
    .set_vapor_pressure(1e-30, "Pa")
    .build()
)
# Unified gas order matches particle species: sulfate then ten organics.
gas = par.gas.GasData(
    name=["Sulfate", *name.tolist()],
    molar_mass=np.append(sulfate_molar_mass, M_gmol) * 1e-3,
    partitioning=np.ones(11, dtype=bool),
    concentration=np.append(0.0, c_total_ug_per_m3)[None, :] * 1e-9,
)
environment = par.gas.EnvironmentData(
    temperature=np.array([temperature_K]),
    pressure=np.array([101325.0]),
    saturation_ratio=np.zeros((1, 11)),
)

# %% [markdown]
# ## Construct seeds with a known number measure
#
# The 250 radii span about 32–501 nm. A lognormal PMF assigns number concentration
# **per bin**, with nominal total 10¹³ m⁻³, mode radius 100 nm, and geometric
# standard deviation 1.4. Per-particle sulfate mass comes from spherical volume;
# organic masses initially vanish. Summing bin concentrations, rather than
# integrating them as dN/dr, gives the represented number density.
#
# Ideal molar activity is assumed. Sulfate and organics occupy separate surface
# phase labels; the low organic surface tension of 0.0003 N/m is an illustrative
# assumption, not a measured property. These inputs limit the physical
# interpretation even before the coagulation representation becomes unsuitable.

# %%
total_number_concentration = np.array([1e13])  # /m-3
particle_radius = np.logspace(-7.5, -6.3, 250)  # m

# create sulfate seeds
number_concentration = par.particles.get_lognormal_pmf_distribution(
    x_values=particle_radius,
    mode=np.array([100e-9]),
    geometric_standard_deviation=np.array([1.4]),
    number_of_particles=total_number_concentration,
)

# calculate mass in each bin
sulfate_volume_distribution = 4.0 / 3.0 * np.pi * particle_radius**3
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
    sulfate_density, density_organics_kg_m3
)  # kg/m^3
activity_strategies = (
    par.particles.ActivityIdealMolarBuilder()
    .set_molar_mass(particle_molar_mass, "g/mol")
    .build()
)

# Retained illustrative surface-tension parameters. The low organic values
# are modeling assumptions, not measured or independently validated inputs.
surface_tension = np.append(0.09, np.ones(len(M_gmol)) * 0.0003)  # N/m
surface_strategy = (
    par.particles.SurfaceStrategyVolumeBuilder()
    .set_surface_tension(surface_tension, "N/m")
    .set_density(particle_densities, "kg/m^3")
    .set_phase_index(np.append(0, np.ones(len(M_gmol))))
    .build()
)

# %% [markdown]
# ## Identify what each process can change
#
# Native discrete `ParticleData` stores per-bin number density at unit reference
# volume. Condensation changes species masses and hence radius. The later binned
# coagulation changes number density at fixed per-bin masses and compositions.
# This distinction is the central test: after unequal uptake across sizes,
# reweighting fixed compositions need not conserve individual species.

# %%
particle_data = (
    par.particles.ParticleDataBuilder()
    .set_distribution_type("discrete")
    .set_masses(mass_distribution, "kg")
    .set_concentration(number_concentration, "1/m^3")
    .set_charge(np.zeros(len(number_concentration)))
    .set_density(particle_densities, "kg/m^3")
    .set_volume(np.array([1.0]))
    .build()
)

# %% [markdown]
# ## Record the reference species inventory
#
# Assemble particles, gas, and environment, then independently sum particle
# species mass times bin number density and add gas concentration. The result is
# a species mass density in kg/m³. This reference includes vapor, so transfer
# between gas and particles should not change it in the closed condensation stage.

# %%
# Build aerosol
aerosol = par.Aerosol(particles=particle_data, gas=gas, environment=environment)


def species_inventory(aero):
    """Independent species mass density for this reference-volume PMF."""
    return (
        np.sum(
            aero.particles.masses * aero.particles.concentration[..., None],
            axis=1,
        )
        + aero.gas.concentration
    )


initial_inventory = species_inventory(aerosol)
print(aerosol)

# %% [markdown]
# ## Configure closed isothermal condensation
#
# The species map aligns all eleven lanes, while `skip_partitioning_indices=[0]`
# excludes sulfate. Diffusivity is 2×10⁻⁵ m²/s and accommodation is one.
# Updating gas concentrations accounts for vapor removed by condensation or
# returned by evaporation. Separate copies preserve the initial and post-uptake
# states for comparison.

# %%
# Condensation Coagulation process setup
condensation_strategy = par.dynamics.CondensationIsothermal(
    # Process inputs are SI kg/mol; particle_molar_mass above is g/mol.
    molar_mass=gas.molar_mass.copy(),
    activity_strategy=activity_strategies,
    surface_strategy=surface_strategy,
    vapor_pressure_strategy=[
        sulfate_vapor_pressure,
        *vapor_pressure_strategies,
    ],
    species_map=CondensationSpeciesMap(
        tuple(gas.name), tuple((i, i) for i in range(11))
    ),
    diffusion_coefficient=2e-5,
    accommodation_coefficient=1,
    update_gases=True,
    skip_partitioning_indices=[0],  # no partitioning for sulfate
)
condensation_process = par.dynamics.MassCondensation(condensation_strategy)

sequence_condensation = par.RunnableSequence() | condensation_process

# Copy aerosol and define time bins
aerosol_initial = copy.deepcopy(aerosol)
aerosol_soa = copy.deepcopy(aerosol)

# %% [markdown]
# ## Inspect uptake before collisions begin
#
# Advance condensation for 20 s with 100,000 substeps, giving 0.0002 s per
# substep. The assertion checks each mapped gas-plus-particle species inventory
# at rtol=10⁻¹² and atol=10⁻³⁰ kg/m³. It establishes this conservation check,
# not convergence or equilibrium.
#
# The first plot places unchanged bin populations at their grown radii. A shift
# in radius therefore describes mass uptake; it should not be interpreted as a
# newly formed population or a number-density-per-radius curve.

# %%
aerosol_soa = sequence_condensation.execute(
    aerosol=aerosol_soa,
    time_step=20,
    sub_steps=100_000,
)

np.testing.assert_allclose(
    species_inventory(aerosol_soa), initial_inventory, rtol=1e-12, atol=1e-30
)
print(aerosol_soa)

fig, ax = plotting.subplots(size="wide")
ax.plot(
    aerosol_initial.particles.radii[0],
    aerosol_initial.particles.concentration[0],
    label="Initial Seeds",
)
ax.plot(
    aerosol_soa.particles.radii[0],
    aerosol_soa.particles.concentration[0],
    label="Seeds+SOA",
)
plotting.legend(ax, loc="upper left")
ax.set_xscale("log")
ax.set_xlabel("Particle radius (m)")
ax.set_ylabel("Number concentration per bin (m$^{-3}$)")
ax.set_xlim(left=1e-8, right=1e-6)
ax.grid(alpha=0.25)
plotting.fit(fig)
plt.show()

# %% [markdown]
# ## Apply the binned Brownian method at its boundary
#
# The discrete Brownian strategy models thermally driven collisions. Here it is
# applied after condensation has produced size-dependent composition, so this
# stage tests a known representation mismatch. Changing the collision kernel
# would not resolve the missing species transport.

# %%
# coagulation process setup
coagulation_strategy = (
    par.dynamics.BrownianCoagulationBuilder()
    .set_distribution_type("discrete")
    .build()
)
coagulation_process = par.dynamics.Coagulation(coagulation_strategy)

# %% [markdown]
# ## Run coagulation alone for 600 s
#
# The loop contains 60 steps of 10 s with one coagulation substep each.
# Condensation is commented out throughout this stage. The unused condensation
# substep setting does not make the run a coupled or alternating calculation.
# The final gas equality check confirms that the gas reservoir remains unchanged.

# %%
# %% staggered time stepping

aerosol_process = copy.deepcopy(aerosol_soa)

total_time = 600  # seconds
total_steps = 60

coagulation_sub_step = 1
condensation_sub_step = 50_000

time_step = total_time / total_steps

for _ in tqdm(range(total_steps), desc="Running Sim", mininterval=0.5):
    # Condensation stays disabled during this retained coagulation-only stage.
    # aerosol_process = condensation_process.execute(
    #     aerosol=aerosol_process,
    #     time_step=time_step,
    #     sub_steps=condensation_sub_step,
    # )

    # execute coagulation process
    aerosol_process = coagulation_process.execute(
        aerosol=aerosol_process,
        time_step=time_step,
        sub_steps=coagulation_sub_step,
    )

# %% [markdown]
# ## Compare distributions without treating them as validation
#
# The three curves show initial seeds, the result after 20 s of condensation,
# and the result after 600 s of coagulation alone. The latter redistributes bin
# number, but the final inventory diagnostic is needed to judge its species
# meaning. All vertical values remain number concentration per bin, not dN/dr;
# a visual peak shift is not an independently calculated median radius.

# %%
fig, ax = plotting.subplots(size="wide")
ax.plot(
    aerosol_initial.particles.radii[0],
    aerosol_initial.particles.concentration[0],
    label="Initial Seeds",
)
ax.plot(
    aerosol_soa.particles.radii[0],
    aerosol_soa.particles.concentration[0],
    label="20 sec SOA partitioning",
)
ax.plot(
    aerosol_process.particles.radii[0],
    aerosol_process.particles.concentration[0],
    label="600 sec coagulation",
    # marker=".",
)
plotting.legend(ax, loc="upper left")
ax.set_xscale("log")
ax.set_xlabel("Particle radius (m)")
ax.set_ylabel("Number concentration per bin (m$^{-3}$)")
ax.grid(alpha=0.25)
ax.set_xlim(left=1e-8, right=1e-6)
plotting.fit(fig)
plt.show()

# %% [markdown]
# ## Check the quantity a size plot cannot show
#
# The final block requires finite, nonnegative populations, a reduction in total
# number, and unchanged gas. It then reports each species' relative inventory
# drift without accepting that drift as conservation. This distinguishes a run
# that produces a size distribution from one that can support composition-based
# research conclusions.

# %%
assert (
    0
    < aerosol_process.particles.concentration.sum()
    < aerosol_soa.particles.concentration.sum()
)
assert np.all(np.isfinite(aerosol_process.particles.concentration))
assert np.all(aerosol_process.particles.concentration >= 0)
np.testing.assert_array_equal(
    aerosol_process.gas.concentration, aerosol_soa.gas.concentration
)
coagulation_inventory_drift = (
    species_inventory(aerosol_process) - species_inventory(aerosol_soa)
) / species_inventory(aerosol_soa)
print("Per-species relative inventory drift during binned coagulation:")
print(coagulation_inventory_drift)

# %% [markdown]
# ## Use the failure to choose a representation
#
# Condensation passes the stated closed-species check, but fixed-composition
# binned coagulation fails to preserve the size-dependent species inventories
# created by that uptake. Consequently the final composition is not a trustworthy
# SOA prediction, even though particle number decreases and the plot looks smooth.
#
# A useful next study would initialize an equivalent particle-resolved population,
# run the same staged forcing, and compare both size spectra and each species'
# gas-plus-particle inventory. Test timestep and sampling convergence separately.
# Only after the representation passes those checks should measured volatility,
# surface tension, and loading be used to investigate a particular SOA system.
