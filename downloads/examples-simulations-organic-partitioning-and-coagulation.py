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
#  # Organic Partitioning & Coagulation Simulation
#
# **Important limitation:** condensation conserves each mapped species, but
# the subsequent fixed-composition binned coagulation does not transport
# species between bins. Condensation creates size-dependent composition, so
# this 600 s coagulation stage is a demonstration of the retained method's
# limitation, **not a quantitatively reliable species-composition forecast**.
# In the saved run, relative species-inventory changes range from about
# -59% to +35%. These are observed defects, not acceptable error tolerances.
# Reducing the timestep does not supply the missing species transport.
# Use a supported particle-resolved workflow when composition varies by size;
# see the [coagulation contract](../../guide/processes/coagulation.md)
# and [resolved example](../processes/coagulation/coagulation_3_particle_resolved_pattern.ipynb).

#
#  This notebook demonstrates how to form secondary organic aerosol (SOA)
#  from a mixture of gas-phase organics and to follow its evolution by
#  Brownian coagulation.
#
#  We will use a moving bin representation of particles, with vapor-liquid partitioning of organic vapours.
#
#  **Road-map**
#  1. Define physicochemical inputs
#  2. Build gas species and the atmosphere
#  3. Create seed particles and a particle representation
#  4. Assemble an `Aerosol` object
#  5. Run isothermal condensation (20 s)
#  6. Add Brownian coagulation and continue for 10 min
#  7. Visualise & discuss results
#
#
#  > **Concept – Secondary Organic Aerosol (SOA):** particles that originate when low-volatility organic vapours condense onto pre-existing seeds.
#
#  > **Concept – Partitioning:** the reversible transfer of a compound between the gas phase and the particle phase until thermodynamic equilibrium is reached.
#
#  > **Concept – Brownian Coagulation:** random collisions driven by thermal motion that merge individual particles into larger ones, reducing number concentration while conserving mass.

# %% [markdown]
# <details> <summary>▶️ Install Particula</summary>
# try:
#     import google.colab
# except ImportError:
#     # not Colab → do nothing
#     pass
# else:
#     # in Colab → install quietly
# Install from the repository root: `python -m pip install -e ".[extra]"`
# </details>

# %%

import copy

import matplotlib.pyplot as plt
import numpy as np
import particula as par
from particula.dynamics.condensation.species_mapping import (
    CondensationSpeciesMap,
)

# Organic density estimation for species
from particula.particles import get_organic_density_array
from tqdm import tqdm

# plot settings
TAILWIND = par.util.colors.TAILWIND
base_color = TAILWIND["gray"]["600"]
plt.rcParams.update(
    {
        "text.color": base_color,
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
#  ## Physicochemical inputs
#
#  Molecular weight, O:C & H:C ratios, saturation concentrations and
#  total gas-phase concentrations gathered from the literature.
#
#  >
#  > **O:C ratio (oxygen-to-carbon):** proxy for oxidation state – a
#  > higher ratio usually implies higher polarity and lower volatility.
#  >
#  > **c_sat:** saturation concentration (often noted *C\**); at this
#  > concentration the compound is in equilibrium between gas and
#  > particle phases.
#  >
#  > All concentrations are specified in µg m⁻³ to mimic experimental
#  > data and are converted to SI units by the builders.

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
#  ## Build gas-phase species
#  Each species receives a vapour-pressure *strategy* so that the code
#  can evaluate temperature-dependent properties on the fly.
#  Note: the saturation-concentration builder is a convenient wrapper
#  converting **μg m⁻³ → Pa**.
#
#
#  > **Strategy pattern:** every species owns a *strategy* object that
#  > knows how to compute its pure vapour pressure – we can switch
#  > correlations without touching core logic.
#  >
#  > `SaturationConcentrationVaporPressureBuilder` back-calculates
#  > vapour pressure from the published saturation concentration *C\**.

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
#  ## Compose gas and environment state
#
#  Keep the ordered gas mixture in `GasData` and temperature and pressure
#  in `EnvironmentData`; the native aerosol holds both containers.
#
#
#  > **EnvironmentData:** central thermodynamic context (temperature &
#  > pressure).  All processes query these values rather than carrying
#  > their own copies.
#  sulfate vapor pressure

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
# ## Seed particles
#
#  Generate a log-normal size distribution of sulfate seeds and assign
#  species-resolved mass bins.
#  Tip: for speed you may reduce the radius grid – 250 points is rather
#  fine for demonstration purposes.
#
#  > **Log-normal distribution:** widely observed for atmospheric
#  > aerosols; characterised by a modal radius and geometric standard
#  > deviation (GSD).
#  >
#  > We convert the number distribution to a probability-mass function
#  > (PMF) so the integral equals the desired particle count.

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
# ## Particle representation
#
# Native discrete `ParticleData` stores per-bin number density at unit
# reference volume. Condensation changes species masses and therefore radii.
# Coagulation subsequently updates number density at fixed per-bin masses;
# it does not mix donor compositions into destination bins.
#
# **Activity & surface strategies:** thermodynamics (ideal,
# etc.) and surface tension models are swappable components, keeping
# the main algorithm clean.

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
# ## Assemble aerosol
#
#  Wrap particles, gas and environment into a single object that the
#  dynamical processes will modify in place.
#
# **Big picture:** `Aerosol` holds native particle, gas, and environment data.
# Every subsequent *Runnable* process expects a single aerosol object
# and modifies it in place.

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
#  ## Condensation process
#
#  We use an isothermal Fuchs–Sutugin formulation.
#  `skip_partitioning_indices=[0]` excludes sulfate from condensation.
#
#  > **Performance hint** – the number of sub-steps controls accuracy.
#  >
#  > `update_gases=True` removes vapours from the gas pool as they
#  > condense, enforcing mass conservation.
#  >
#  > The `RunnableSequence` uses the `|` pipe operator for a fluent
#  > interface

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
#  ### Single 20 s condensation pulse
#
#  Useful for inspecting how much SOA forms before coagulation starts.
#
#  > We pick a very small internal sub-step
#  > (`Δt_sub = 20 s / 100 000 ≈ 2×10⁻⁴ s`) to capture the rapid initial
#  > uptake of vapours onto nano-particles.
#  20 s condensation pulse

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

fig, ax = plt.subplots(figsize=(7, 5))
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
ax.legend(loc="upper left")
ax.set_xscale("log")
ax.set_xlabel("Particle radius (m)")
ax.set_ylabel("Number concentration per bin (m$^{-3}$)")
ax.set_xlim(left=1e-8, right=1e-6)
ax.grid(alpha=0.25)
fig.tight_layout()

# %% [markdown]
#  ## Coagulation process
#
#  Builder selects a discrete Brownian kernel.  Other kernels are
#  available and can be swapped by changing the builder called.
#
#  > **Brownian kernel:** collision frequency determined solely by random
#  > thermal motion; other kernels
#  > can be swapped by changing a single builder call.

# %%
# coagulation process setup
coagulation_strategy = (
    par.dynamics.BrownianCoagulationBuilder()
    .set_distribution_type("discrete")
    .build()
)
coagulation_process = par.dynamics.Coagulation(coagulation_strategy)

# %% [markdown]
#  ## Coagulation after the condensation pulse (10 min)
#
#  We alternate long, finely-resolved condensation steps with coarse
#  coagulation steps.
#  Rule of thumb: `time_step / sub_steps ≈ 0.01 s` keeps condensation
#  stable while minimising overhead.
#
# Experiment: toggle condensation on to see if coagulation+condensation changes the particle size distribution.
#
#
#  > We loop over 60 outer steps of 10 s each to reach 10 min.  Inside
#  > every step we *could* run condensation with many sub-steps while
#  > keeping coagulation coarse – a trade-off between accuracy and
#  > speed.
# .

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
#  ## Results
#
#  The figure contrasts (i) the initial seed distribution, (ii) mass
#  growth after 20 s of SOA partitioning, and (iii) its further
#  evolution through coagulation.
#
#
#  > Observe how number concentration drops (coagulation) while the
#  > median radius grows (condensation + coagulation).
#  > The 600 s stage already runs coagulation alone. Its binned numerical
#  > mass drift is reported below; this is not exact inventory conservation.

# %%
fig, ax = plt.subplots(figsize=(7, 5))
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
ax.legend(loc="upper left")
ax.set_xscale("log")
ax.set_xlabel("Particle radius (m)")
ax.set_ylabel("Number concentration per bin (m$^{-3}$)")
ax.grid(alpha=0.25)
ax.set_xlim(left=1e-8, right=1e-6)
fig.tight_layout()

# %% [markdown]
# ## Summary
#
# This notebook demonstrates how to simulate the formation of secondary
# organic aerosol (SOA) from gas-phase organics and its evolution by
# Brownian coagulation.  We used a moving bin representation of particles
# with vapour-liquid partitioning, allowing us to model the complex
# interactions between gas-phase species and particle growth.
#

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
