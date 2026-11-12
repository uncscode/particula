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
# ## Biomass Burning Aerosol with Cloud Interactions
#
# Welcome to the **Biomass Burning Aerosol with Cloud Interactions** notebook, your step‑by‑step guide to setting up and running multi‑component aerosol simulations using **Particula**. Even if you’re new to aerosol modeling or cloud microphysics, this notebook will walk you through:
#
# 1. **What This Notebook Is About**
#    This notebook demonstrates how to set up and run a multi‑component particle resolved aerosol simulation using Particula. The example focuses on biomass burning aerosols, which are a mixture of soot, organics, and water. The simulation will explore how these aerosols behave under cloud conditions, including their activation into cloud droplets and subsequent evolution through condensation and coagulation processes.
#
# 2. **Why It Matters**
#    Biomass burning aerosols (soot + organics + water) play a critical role in cloud formation, climate forcing, and air quality. Understanding how these mixed particles activate into cloud droplets and evolve through condensation and coagulation helps bridge experiments and models.
#
# 3. **What You’ll Learn**
#    - Configuring species properties (molar masses, densities, hygroscopicity)
#    - Building gas‑phase and particle‑phase objects
#    - Running isothermal condensation and four coagulation scenarios:
#      - **Condensation Only**
#      - **Condensation + Brownian Coagulation**
#      - **Condensation + Brownian + Sedimentation Coagulation**
#      - **Condensation + Turbulent DNS Coagulation**
#    - Visualizing size distributions, number concentrations, and mass fractions
#    - Comparing saturation ratios across processes
#
# > No prior experience with Particula is required—simply follow the markdown cells and code examples to explore how mixed‐phase aerosols behave under cloud‐like conditions.

# %%
# In Colab uncomment the following command to install particula:
# Use the v0.3.0 development checkout: `python -m pip install -e ".[extra]"`

import copy

import matplotlib.pyplot as plt
import numpy as np
import particula as par
from particula.dynamics.condensation.species_mapping import (
    CondensationSpeciesMap,
)
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
# ## Setup for Biomass‐Burning Organic Aerosol (BBOA) in a Cloud Environment
#
# In this section we configure a particle‑resolved BBOA simulation under cloud‑like conditions:
#
# 1. **Reproducibility**
#    – Fix the random seed (`np.random.seed(100)`) for consistent sampling across runs.
#
# 2. **Species Properties**
#    – Molar masses (organics, soot, water)
#    – Densities (kg m⁻³)
#    – Hygroscopicity parameters (κ values)
#
# 3. **Simulation Parameters**
#    – `number_of_samples` (particles)
#    – Simulation volume (`1e‑6 m³`)
#    – Initial water activity (`1.02`) and temperature (`298.15 K`)
#
# 4. **Vapor‑Pressure Strategies**
#    – Constant builder for organics and soot
#    – Buck equation for water vapor
#
# 5. **Gas‑Phase Composition**
#    – Build a `GasData` object with condensable vapors and set concentrations
#    – Store temperature and pressure separately in `EnvironmentData`
#
# 6. **Particle Size Distributions**
#    – Draw lognormal samples for organics and soot modes
#    – Initialize water mass as the sum of organic and soot masses
#
# 7. **Mass Speciation & Parameterization**
#    – Stack species masses into an (N×3) array
#    – Define a κ‑Köhler activity strategy and a surface‑volume strategy
#
# 8. **Particle‑Resolved Representation**
#    – Use `create_resolved_particle_data` to combine mass, density, charge (0), and volume
#
# 9. **Aerosol Object & Initial Visualization**
#    – Instantiate `Aerosol(particles, gas, environment)`
#    – Plot a histogram of log₁₀ particle radii to verify the initial distribution
#
# > **Pro tip:** For fast prototyping, begin with `5 000 – 10 000` particles. Once your setup is validated, increase `number_of_samples`, simulation length, or time resolution.
#

# %%
# Reproducibility
np.random.seed(100)

# 1a. Species properties
molar_mass_organics = 250e-3  # kg/mol
molar_mass_soot = 1000e-3  # kg/mol
molar_mass_water = 18.01528e-3  # kg/mol

density_organics = 1400.0  # kg/m^3
density_soot = 1800.0
density_water = 1000.0

kappa_organics = 0.15
kappa_soot = 0.01
kappa_water = 0.01

number_of_samples = 10_000  # number of particles to sample
simulation_volume = 1e-6  # m^3
water_activity = 1.02  # initial water activity
temperature = 298.15

# Vapor pressure strategies
vapor_organics = (
    par.gas.ConstantVaporPressureBuilder()
    .set_vapor_pressure(2e-12, "Pa")
    .build()
)
vapor_soot = (
    par.gas.ConstantVaporPressureBuilder()
    .set_vapor_pressure(1e-30, "Pa")
    .build()
)
vapor_water = par.gas.WaterBuckVaporPressureBuilder().build()

# Gas species
water_sat = vapor_water.saturation_concentration(molar_mass_water, temperature)
water_conc = water_sat * water_activity

gas_species = par.gas.GasData(
    name=["organics", "soot", "water"],
    molar_mass=np.array(
        [molar_mass_organics, molar_mass_soot, molar_mass_water]
    ),
    partitioning=np.array([True, True, True]),
    concentration=np.array([[1e-12, 1e-12, water_conc]]),
)
environment = par.gas.EnvironmentData(
    temperature=np.array([temperature]),
    pressure=np.array([101325.0]),
    saturation_ratio=np.zeros((1, 3)),
)

# Particle distributions
radii_organics = par.particles.get_lognormal_sample_distribution(
    mode=np.array([30e-9, 110e-9]),
    geometric_standard_deviation=np.array([1.3, 1.2]),
    number_of_particles=np.array([0.2, 0.6]),
    number_of_samples=number_of_samples,
)
radii_soot = par.particles.get_lognormal_sample_distribution(
    mode=np.array([75e-9, 120e-9]),
    geometric_standard_deviation=np.array([1.3, 1.4]),
    number_of_particles=np.array([2, 1]),
    number_of_samples=number_of_samples,
)

mass_organic = 4 / 3 * np.pi * (radii_organics**3) * density_organics
mass_soot = 4 / 3 * np.pi * (radii_soot**3) * density_soot
mass_water = mass_organic + mass_soot

mass_speciation = np.column_stack((mass_organic, mass_soot, mass_water))

# Activity strategy
activity_strategy = (
    par.particles.ActivityKappaParameterBuilder()
    .set_density([density_organics, density_soot, density_water], "kg/m^3")
    .set_molar_mass(
        [molar_mass_organics, molar_mass_soot, molar_mass_water], "kg/mol"
    )
    .set_kappa(np.array([kappa_organics, kappa_soot, kappa_water]))
    .set_water_index(2)
    .build()
)

surface_strategy = par.particles.SurfaceStrategyVolume()

# Create Particula particle object
resolved_masses = par.particles.create_resolved_particle_data(
    masses=mass_speciation,
    density=np.array([density_organics, density_soot, density_water]),
    volume=simulation_volume,
)
aerosol = par.Aerosol(
    particles=resolved_masses, gas=gas_species, environment=environment
)


def species_inventory(aero):
    """Independent extensive material inventory, kg per species."""
    return (
        np.sum(
            aero.particles.masses * aero.particles.concentration[..., None],
            axis=1,
        )
        + aero.gas.concentration * aero.particles.volume[:, None]
    )


def water_saturation(aero):
    """Current water saturation from explicit process thermodynamics."""
    return aero.gas.concentration[0, 2] / vapor_water.saturation_concentration(
        molar_mass_water, aero.environment.temperature[0]
    )


initial_inventory = species_inventory(aerosol)
print(aerosol)


# Plot initial size distribution
fig, ax = plt.subplots()
ax.hist(np.log10(resolved_masses.radii[0]), bins=50, density=False, alpha=0.5)
ax.set_xlabel("log10(Radius [m])")
ax.set_ylabel("Bin counts")
ax.set_title("Initial Particle Size Distribution")

# %% [markdown]
# ## Simulation: Isothermal Condensation
#
# In this section we run an isothermal condensation on our activated aerosol, track how the size distribution evolves over time, and visualize the result as a contour of number concentration.
#
# 1. **Define the Condensation Process**
#    We use `CondensationIsothermal` with specified molar masses, diffusion coefficient, and accommodation coefficient, wrapped by `MassCondensation`.
#
# 2. **Pre‑activation Step**
#    A small “spin‑up” condensation to initialize vapors on the dry aerosol.
#
# 3. **Time Loop**
#    - Divide the total simulation time into equal “sub‑step” chunks.
#    - At each time step, execute the condensation process and record the particle radii histogram.
#
# 4. **Post‑processing & Plotting**
#    - Convert raw counts to number concentration (# m⁻³).
#    - Build a 2D mesh of time vs. radius and plot `log10` of concentration on a log‑radius axis.
#
# > **Pro tip:** If the water saturation ratio is much lower than 1, in a cloud environment, the time step may be too large. Consider reducing the time step to ensure accurate results.

# %%
# Condensation process setup
condensation_strategy = par.dynamics.CondensationIsothermal(
    molar_mass=np.array(
        [molar_mass_organics, molar_mass_soot, molar_mass_water]
    ),
    diffusion_coefficient=2e-5,
    accommodation_coefficient=1,
    activity_strategy=activity_strategy,
    surface_strategy=surface_strategy,
    vapor_pressure_strategy=[vapor_organics, vapor_soot, vapor_water],
    species_map=CondensationSpeciesMap(
        tuple(gas_species.name), ((0, 0), (1, 1), (2, 2))
    ),
)
condensation_process = par.dynamics.MassCondensation(condensation_strategy)

# Copy aerosol and define time bins
aerosol_activated = copy.deepcopy(aerosol)
time_step = 600  # total simulation time in seconds
total_steps = 5_000  # total sub‑steps for finer resolution
time = np.arange(time_step)  # recorded 0..599 s at one-second intervals
bins_lognormal = np.logspace(-8, -2, 200)  # radius bins from 10⁻⁸ to 10⁻² m
distribution_counts = np.zeros((len(time), len(bins_lognormal) - 1))

# Pre‑activation “spin‑up”
aerosol_activated = condensation_process.execute(
    aerosol=aerosol_activated, time_step=0.01, sub_steps=3_000
)

# Main simulation loop
sub_steps_per_sec = int(total_steps / time_step)
for i, _t in enumerate(tqdm(time, desc="Condensing")):
    if i > 0:
        aerosol_activated = condensation_process.execute(
            aerosol=aerosol_activated, time_step=1, sub_steps=sub_steps_per_sec
        )
    # Record the size distribution at this time
    distribution_counts[i, :], edges = np.histogram(
        aerosol_activated.particles.radii[0].copy(), bins=bins_lognormal
    )

# Print final state and water saturation
print(aerosol_activated)
print(
    "Final water saturation ratio:",
    water_saturation(aerosol_activated),
)

np.testing.assert_allclose(
    species_inventory(aerosol_activated),
    initial_inventory,
    rtol=1e-12,
    atol=1e-30,
)

# Convert counts → number concentration (#/m³)
concentrations = distribution_counts / simulation_volume

# Contour plot of log10(number concentration)
fig, ax = plt.subplots(figsize=(7, 5))
X, Y = np.meshgrid(time, edges[:-1])
log_conc = np.log10(
    concentrations,
    where=concentrations > 0,
    out=np.full_like(concentrations, np.nan),
)
cont = ax.contourf(X, Y, log_conc.T)

ax.set_yscale("log")
ax.set_ylim(1e-7, 1e-5)
ax.set_xlabel("Time (s)")
ax.set_ylabel("Particle Radius (m)")
fig.colorbar(cont, label="Log₁₀ Number Concentration")
plt.tight_layout()
plt.show()

# %% [markdown]
# ## Four Coagulation + Condensation Scenarios
#
# We apply each coagulation strategy in sequence with isothermal condensation to see how different particle interactions impact the final water saturation ratio:
#
# 1. **Condensation Only**
#    No coagulation—particles grow only by condensation.
#
# 2. **Brownian Coagulation**
#    Particle‐resolved Brownian collisions followed by condensation.
#
# 3. **Brownian + Sedimentation Coagulation**
#    Combines Brownian motion and gravitational settling before condensation.
#
# 4. **Turbulent DNS Coagulation**
#    Uses the DNS turbulent collision kernel before condensation. This branch does not add the separate Brownian strategy.
#

# %%
# Define coagulation strategies
brownian = (
    par.dynamics.BrownianCoagulationBuilder()
    .set_distribution_type("particle_resolved")
    .build()
)
sedimentation = (
    par.dynamics.SedimentationCoagulationBuilder()
    .set_distribution_type("particle_resolved")
    .build()
)
combined_brown_sed = (
    par.dynamics.CombineCoagulationStrategyBuilder()
    .set_strategies([brownian, sedimentation])
    .build()
)

turbulent_dns = (
    par.dynamics.TurbulentDNSCoagulationBuilder()
    .set_distribution_type("particle_resolved")
    .set_relative_velocity(0, "m/s")
    .set_reynolds_lambda(72, "dimensionless")
    .set_turbulent_dissipation(1000, "cm^2/s^3")
    .set_fluid_density(1.225, "kg/m^3")
    .build()
)

# Define Sequences for each Scenarios
seq_condensation = par.RunnableSequence() | condensation_process
seq_brown = (
    par.RunnableSequence()
    | par.dynamics.Coagulation(brownian)
    | condensation_process
)
seq_comb = (
    par.RunnableSequence()
    | par.dynamics.Coagulation(combined_brown_sed)
    | condensation_process
)
seq_turb = (
    par.RunnableSequence()
    | par.dynamics.Coagulation(turbulent_dns)
    | condensation_process
)

# Run each
results = {}
for name, seq in [
    ("CondensationOnly", seq_condensation),
    ("Brownian", seq_brown),
    ("Brownian+Sed", seq_comb),
    ("TurbulentDNS", seq_turb),
]:
    print("Started simulation:", name)
    obj = copy.deepcopy(aerosol_activated)
    obj = seq.execute(aerosol=obj, time_step=3600, sub_steps=10_000)
    np.testing.assert_allclose(
        species_inventory(obj), initial_inventory, rtol=1e-12, atol=1e-30
    )
    assert np.all(np.isfinite(obj.particles.masses))
    assert np.all(obj.particles.masses >= 0)
    results[name] = obj
    print(obj)
    print(
        f"{name} water saturation ratio:",
        water_saturation(obj),
    )
    print("\n")

# %% [markdown]
# ## Processing Distributions and Final Graphs
#
# Before plotting, we convert each aerosol object into a `dN/dlogDp` distribution:
#
# 1. **Define histogram bins**
#    Create `num_bins+1` logarithmically spaced diameter edges from 0.01 to 100 µm.
#
# 2. **Utility functions**
#    - `get_wet_diameters(aero_obj)`:
#      Extract nonzero wet diameters (µm) from an aerosol’s radius array.
#    - `compute_dndlogdp(diameters, volume_m3)`:
#      - Bin diameters → raw counts
#      - Convert counts to number concentration (# / cm³)
#      - Convert to dN/dlogDp using Particula’s distribution strategy
#
# 3. **Assemble distributions**
#    Loop over the “Initial,” “Start,” and each coagulation scenario to populate a `distributions` dict mapping label → `(diameter_centers, dN/dlogDp)`.
#

# %%
num_bins = 140
BIN_EDGES = np.logspace(np.log10(0.01), np.log10(100), num_bins + 1)


def get_wet_diameters(aero_obj):
    """Return an array of nonzero wet diameters (µm)."""
    radii_m = aero_obj.particles.radii[0].copy()
    radii_um = radii_m * par.util.get_unit_conversion("m", "um")
    diameters_um = radii_um[radii_um > 0] * 2
    return diameters_um


def compute_dndlogdp(diameters, volume_m3):
    """Convert diameter samples into dN/dlogDp (#/cm³)."""
    counts, _ = np.histogram(diameters, bins=BIN_EDGES)
    m3_to_cm3 = par.util.get_unit_conversion("m^3", "cm^3")
    dn_number = counts / (volume_m3 * m3_to_cm3)
    centers = (BIN_EDGES[:-1] + BIN_EDGES[1:]) / 2
    strategy = par.particles.get_distribution_conversion_strategy(
        "dN/dlogDp", "pmf"
    )
    dndlogdp = strategy.convert(centers, dn_number, inverse=True)
    return centers, dndlogdp


# Compute distributions for all cases
distributions = {}
cases = [("Initial", aerosol), ("Start", aerosol_activated)] + list(
    results.items()
)

for label, aero_obj in cases:
    diameters = get_wet_diameters(aero_obj)
    volume = aero_obj.particles.volume[0]
    centers, dnd = compute_dndlogdp(diameters, volume)
    distributions[label] = (centers, dnd)

# %% [markdown]
# ### Graph Size Distribution Comparison
#
# In this section we plot the wet‑diameter distributions for each scenario on a single log–log axis:
#
# 1. **Extract and clean each distribution**
#    - Retrieve `(centers, dN/dlogDp)` for “Initial”, “Start”, “CondensationOnly”, “Brownian”, “Brownian+Sed”, and “TurbulentDNS”.
#    - Replace zeros with `NaN` so they don’t appear as flat lines on the log scale.
#
# 2. **Plot each case**
#    - **Initial**: thick, semi‑transparent slate line
#    - **Activated start**: very thick sky‑blue line
#    - **Condensation Only**: medium-thick sky‑blue line
#    - **Brownian Only**: dashed light-slate line
#    - **with Sedimentation**: rose‑colored circles
#    - **with turbulentDNS**: amber solid line

# %%
# 1. Create figure & axes
fig, ax = plt.subplots(figsize=(8, 4))

# 2. Extract & clean data for each case
centers_init, dnd_init = distributions["Initial"]
dnd_init = np.where(dnd_init == 0, np.nan, dnd_init)

centers_start, dnd_start = distributions["Start"]
dnd_start = np.where(dnd_start == 0, np.nan, dnd_start)

centers_cond, dnd_cond = distributions["CondensationOnly"]
dnd_cond = np.where(dnd_cond == 0, np.nan, dnd_cond)

centers_brown, dnd_brown = distributions["Brownian"]
dnd_brown = np.where(dnd_brown == 0, np.nan, dnd_brown)

centers_sed, dnd_sed = distributions["Brownian+Sed"]
dnd_sed = np.where(dnd_sed == 0, np.nan, dnd_sed)

centers_turb, dnd_turb = distributions["TurbulentDNS"]
dnd_turb = np.where(dnd_turb == 0, np.nan, dnd_turb)

# 3. Plot each line explicitly
ax.plot(
    centers_init,
    dnd_init,
    linewidth=3,
    alpha=0.5,
    color=TAILWIND["slate"]["800"],
    label="Initial",
)

ax.plot(
    centers_start,
    dnd_start,
    linewidth=15,
    alpha=0.8,
    color=TAILWIND["sky"]["300"],
    label="Activated start",
)

ax.plot(
    centers_cond,
    dnd_cond,
    linewidth=6,
    alpha=0.5,
    color=TAILWIND["sky"]["800"],
    label="Condensation Only",
)

ax.plot(
    centers_brown,
    dnd_brown,
    linewidth=2,
    linestyle="--",
    color=TAILWIND["slate"]["100"],
    label="Brownian Only",
)

ax.plot(
    centers_sed,
    dnd_sed,
    marker="o",
    markersize=4,
    linewidth=5,
    alpha=1,
    color=TAILWIND["rose"]["300"],
    linestyle="",
    label="with Sedimentation",
)

ax.plot(
    centers_turb,
    dnd_turb,
    linewidth=3,
    alpha=0.75,
    color=TAILWIND["amber"]["900"],
    label="with turbulentDNS",
)

# 5. Configure axes, grid, legend, and limits
ax.set_xscale("log")
ax.set_yscale("log")
ax.set_xlim(0.05, 100)
ax.set_ylim(10, 3e5)
ax.set_xlabel("Wet Diameter [µm]")
ax.set_ylabel("dN/dlogDp [#/cm³]")
ax.grid(alpha=0.5, which="both")
ax.legend(
    loc="center left",  # anchor the legend’s “center left” corner
    bbox_to_anchor=(
        1.02,
        0.5,
    ),  # (x, y) in axes coordinates: x=1.02 is just past the right edge
    borderaxespad=0.5,  # padding between axes and legend
)
plt.tight_layout()
plt.show()

# %% [markdown]
# #### Interpretation of Final Size Distributions
#
# The initial particles already contain water. The activated state follows
# the shared condensation spin-up; all four scenarios begin from its copy.
# Condensation redistributes water between vapor and particles. Coagulation
# merges particles and reduces their number while conserving material.
# Sedimentation here is a collision mechanism, not removal to chamber walls.
# The DNS branch uses its turbulent kernel alone; it does not compose a
# separate Brownian or sedimentation kernel. Compare the saved curves for
# this sampled case rather than treating their peak positions as universal.
#

# %% [markdown]
# ### Graph Organic Mass Fraction vs. Wet Diameter
#
# Here we compare each particle’s organic mass fraction before and after the turbulent DNS coagulation:
#
# 1. **Extract species masses**
#    - Get the final species‐mass array from the “TurbulentDNS” result and the initial aerosol.
#    - Zero out the water column so fractions are computed from organics + soot only.
#
# 2. **Compute per‑particle organic fraction**
#    - f_org = m_organics ∕ (m_organics + m_soot)
#    - Apply a mask to exclude any zero‑radius particles.
#
# 1. **Convert to wet diameters**
#    Use `get_wet_diameters(...)` to obtain each particle’s diameter in µm.
#
# 2. **Scatter plot**
#    - **After DNS** (amber, semi‑transparent): shows how turbulence‐enhanced coagulation shifts organic fraction across sizes.
#    - **Initial** (slate, very faint): baseline before any coagulation.
#

# %%
#  1. Retrieve and prepare mass arrays
combined_dns_mass = results["TurbulentDNS"].particles.masses[0].copy()
initial_mass = aerosol.particles.masses[0].copy()

mask = results["TurbulentDNS"].particles.radii[0] > 0
combined_dns_mass[:, -1] = 0.0  # drop water
initial_mass[:, -1] = 0.0

dns_mass_fraction = combined_dns_mass[mask, 0] / combined_dns_mass[mask].sum(
    axis=1
)
initial_mass_fraction = initial_mass[:, 0] / initial_mass.sum(axis=1)

# 2. Compute wet diameters
dns_dp = get_wet_diameters(results["TurbulentDNS"])
initial_dp = get_wet_diameters(aerosol)

# 3. Plot scatter
fig, ax = plt.subplots(figsize=(5, 4))

ax.scatter(
    dns_dp,
    dns_mass_fraction,
    s=1,
    alpha=0.3,
    color=TAILWIND["amber"]["800"],
    label="After DNS",
)
ax.scatter(
    initial_dp,
    initial_mass_fraction,
    s=1,
    alpha=0.1,
    color=TAILWIND["slate"]["500"],
    label="Initial",
)

# 4. Aesthetics
ax.set_xscale("log")
ax.set_xlim(0.05, 100)
ax.set_ylim(0, 1.0)
ax.set_xlabel("Wet Diameter [µm]")
ax.set_ylabel("Organic Mass Fraction")
ax.grid(alpha=0.5, which="both")
ax.legend(loc="upper right")

plt.tight_layout()
plt.show()

# %% [markdown]
# #### Impact of Turbulent DNS on Size–Composition Space
#
# This scatter plot shows each particle’s organic mass fraction versus wet diameter **before** (brown) and **after** (blue) applying the Turbulent DNS coagulation sequence.
#
# **What this tells us:**
# Turbulence‑enhanced coagulation preferentially drives mixed droplets to collide and merge into a narrower, organic‑rich population, while leaving a distinct soot‑rich mode of small particles. In other words, turbulent DNS “sorts” particles by both size and composition. The largest mixed droplets then concentrate organics into a tighter size band.
#

# %% [markdown]
# ## Take‑Home Messages
#
# - **End‑to‑End Workflow:**
#   We walked through a complete Particula pipeline—from defining species properties and gas/particle builders, to running time‑resolved condensation and four coagulation scenarios—culminating in clear visualizations of size distributions and mass fractions.
#
# - **Key Insights:**
#   - **Condensation Only** produces the highest droplet peaks, since no particles are lost to collisions.
#   - **Brownian Coagulation** alone has a minimal smoothing effect on the bimodal droplet distribution.
#   - **Sedimentation** preferentially coagulates larger droplets, with smaller. On this time scale and droplet size the effect is similar to Turbulent DNS.
#   - **Turbulent DNS** Broadens the size spectrum and grows large‐diameter droplets, illustrating the combined importance of Brownian, gravitational, and turbulent collisions.
#
# - **Particle‐Resolved Mass Analysis:**
#   By plotting per‑particle organic mass fraction vs. wet diameter, we saw how turbulent coagulation alters composition across size bins—an approach you can adapt to other species or processes.
#
# - **Extensibility:**
#   This modular, object‑oriented approach can be extended to:
#   - Other dynamic processes (e.g., activation kinetics, heterogeneous chemistry)
#   - Different aerosol types and multicomponent mixtures
#   - Automated code lookup and suggestion via an LLM‑backed vector store
#
# > **Bottom Line:** Particula makes it straightforward to prototype complex aerosol–cloud interactions, compare physical mechanisms quantitatively, and extract particle‐resolved insights—all within a few hundred lines of Python.
#
