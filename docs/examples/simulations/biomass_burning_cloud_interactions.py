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
# # How do collision models change a water-grown mixed aerosol?
#
# A mixed aerosol can change size through water exchange and through collisions.
# How can we compare collision assumptions without confusing them with different
# starting populations? This example gives four scenarios copies of the same
# water-grown, particle-resolved aerosol, then compares their size spectra and
# one scenario's particle composition.
#
# The organics–soot–water mixture is inspired by biomass-burning aerosol, but its
# properties and concentrations are illustrative. It is not a validated smoke
# plume or cloud parcel: temperature stays fixed, and there is no spatial flow,
# chemical aging, or wall removal. Use [resolved coagulation](../processes/coagulation/coagulation_3_particle_resolved_pattern.py)
# and [condensation](../processes/condensation/condensation_3_massresolved.py)
# first if the separate process contracts are unfamiliar.

# %%
# In Colab uncomment the following command to install particula:
# Install from the repository root: `python -m pip install -e ".[extra]"`

import copy

import matplotlib.pyplot as plt
import numpy as np
import particula as par
from particula.dynamics.condensation.species_mapping import (
    CondensationSpeciesMap,
)
from particula.execution.condensation_ode import CondensationODE
from particula.execution.splitting import condensation_flow
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
# ## Build one mixed population and one finite vapor reservoir
#
# Sample 10,000 particles in 10⁻⁶ m³, giving 10¹⁰ m⁻³ initially. Independent
# organic and soot radius samples become masses in the same particle slots;
# these are not two externally mixed populations. Initial water mass equals
# organic plus soot mass, so particles already contain 50% water by mass.
#
# Densities are kg/m³ and molar masses kg/mol. Organic and soot κ values are
# 0.15 and 0.01; water occupies lane 2 and its own κ entry is unused. The default
# surface strategy supplies a water-like surface treatment, not a fitted
# composition-dependent smoke surface model.
#
# The variable named `water_activity` sets the initial **gas saturation ratio**
# to 1.02 at 298.15 K. It is not a prescribed liquid activity or humidity
# controller. Gas water can be depleted by condensation. The first histogram
# counts sampled radii per logarithmic bin; it has not yet been normalized by
# the represented volume.

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
print("Particle slots:", aerosol.particles.masses.shape[1])
print("Gas species:", ", ".join(aerosol.gas.name))


# Plot initial size distribution
fig, ax = plt.subplots()
ax.hist(np.log10(resolved_masses.radii[0]), bins=50, density=False, alpha=0.5)
ax.set_xlabel("log10(Radius [m])")
ax.set_ylabel("Bin counts")
ax.set_title("Initial Particle Size Distribution")

# %% [markdown]
# ## Establish the shared water-grown starting state
#
# Adaptive ROS3 integrates the same continuous condensation physics through
# a 0.01 s spin-up and a full 600 s preparation. The 601 stored states include
# both endpoints at one-second intervals after spin-up. The state labeled
# “Start” below includes this preparation; it is not the original dry population.
#
# The contour divides histogram counts by represented volume, giving number
# concentration per radius bin in m⁻³. It shows size redistribution, not dN/dr.
# The final species-inventory comparison includes both gas and weighted particle
# mass at rtol=10⁻¹² and atol=10⁻³⁰ kg. Conservation does not by itself establish
# timestep convergence, equilibration, or a physically calibrated activation
# threshold. A low final saturation ratio alone cannot diagnose numerical error.

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
time = np.arange(time_step + 1)  # include the full 600 s endpoint
bins_lognormal = np.logspace(-8, -2, 200)  # radius bins from 10⁻⁸ to 10⁻² m
distribution_counts = np.zeros((len(time), len(bins_lognormal) - 1))

# Adaptive preparation retains its accepted controller between observations.
preparation = CondensationODE(
    aerosol_activated,
    condensation_strategy,
    method="ros3",
    rtol=1e-5,
    particle_atol=1e-25,
    gas_atol=1e-12,
    first_step=1e-6,
    max_step=1.0,
)
while preparation.time < 0.01:
    preparation.step(0.01)
for i, _t in enumerate(tqdm(time, desc="Condensing")):
    bound = 0.01 + float(_t)
    while preparation.time < bound:
        preparation.step(bound)
    aerosol_activated = preparation.snapshot()
    # Record the size distribution at this time
    distribution_counts[i, :], edges = np.histogram(
        aerosol_activated.particles.radii[0].copy(), bins=bins_lognormal
    )

# Print final state and water saturation
print(
    "Prepared gas concentration (kg/m^3):", aerosol_activated.gas.concentration
)
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
# ## Change the collision model while retaining condensation
#
# Each branch starts from a copy of the same shared state and advances a further
# 3600 s in 10,000 substeps (0.36 s each). The branches are condensation alone,
# Brownian plus condensation, combined Brownian/sedimentation plus condensation,
# and a turbulent-DNS-kernel model plus condensation.
#
# The DNS branch does not add the separate Brownian or sedimentation strategies.
# It uses Reynolds lambda 72 and dissipation 1000 cm²/s³ (0.1 m²/s³), with zero
# imposed relative velocity. This selects a collision parameterization; it does
# not run a direct numerical flow simulation. Sedimentation here causes
# collisions through differential settling, not removal from the box.
#
# The assertions test finite nonnegative masses and closed material inventories
# for each branch. Differences between one stochastic realization of each branch
# are not an ensemble estimate of a kernel's effect.

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

# Collision draws occur once per existing 0.36 s outer interval. Adaptive
# condensation trials never execute or repeat a stochastic collision operation.
outer_dt = 3600.0 / 10_000
growth = condensation_flow(
    "water_exchange",
    condensation_strategy,
    method="ros3",
    rtol=1e-5,
    particle_atol=1e-25,
    gas_atol=1e-12,
    first_step=outer_dt,
    max_step=outer_dt,
)
results = {}
for name, collision in [
    ("CondensationOnly", None),
    ("Brownian", par.dynamics.Coagulation(brownian)),
    ("Brownian+Sed", par.dynamics.Coagulation(combined_brown_sed)),
    ("TurbulentDNS", par.dynamics.Coagulation(turbulent_dns)),
]:
    print("Started simulation:", name)
    obj = copy.deepcopy(aerosol_activated)
    for step in tqdm(range(10_000), desc=name):
        if collision is not None:
            obj = collision.execute(obj, time_step=outer_dt)
        obj = growth.advance(obj, 600.01 + step * outer_dt, outer_dt)
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
# ## Convert counts into a comparable diameter spectrum
#
# Exclude empty slots, double radius to obtain wet diameter, and bin particles
# between 0.01 and 100 µm. Divide counts by sample volume in cm³ and convert the
# bin totals into dN/dlogDp. This coordinate differs from the initial count
# histogram and the radius-bin contour, so compare integrated number with the
# appropriate measure rather than comparing raw vertical magnitudes.
#
# “Initial” means the original wet seeds; “Start” means the shared result of the
# condensation preparation. The four final curves add one hour to that state.

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
# ## Read the scenario overlay as a controlled model comparison
#
# The overlay uses the same binning and units for every scenario. Empty bins are
# hidden on logarithmic axes. Condensation moves sizes by exchanging vapor;
# coagulation reduces occupied particle count by merging material. A peak height
# alone does not measure total number or establish which process is dominant.
#
# The plot label “Brownian Only” identifies its **collision kernel only**;
# condensation also runs in that branch. Likewise, “with Sedimentation” combines
# Brownian and sedimentation kernels, while “with turbulentDNS” uses the DNS
# kernel without the separate Brownian strategy.

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
# ## Keep the comparison's evidence boundary in view
#
# All branches share their initial material inventory and retain the same
# condensation model. Their differing spectra therefore motivate a collision-model
# sensitivity study under these inputs. The plots and conservation checks do not
# establish a universal ordering of peak sizes or validate any kernel against a
# measured cloud. Repeated random seeds and temporal convergence are separate
# requirements for a quantitative comparison.

# %% [markdown]
# ## Relate wet size to dry composition
#
# The scatter plot compares the original wet seeds with the final DNS-branch
# particles. Its horizontal coordinate is wet diameter, while its vertical
# coordinate is the **dry** organic fraction: organic mass divided by organic
# plus soot mass. Water is excluded from that denominator only in copied arrays;
# the simulation state remains unchanged.
#
# The comparison includes both the shared condensation preparation and the later
# DNS-plus-condensation stage. It is not a before/after test of turbulence alone.
# A particle can shift horizontally by taking up water without changing its dry
# organic fraction.

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
# ## Avoid inferring a composition mechanism from size alone
#
# Merging particles can mix their dry fractions, while condensation changes wet
# size. A narrower band in the final scatter could reflect those effects and the
# sampled population. This plot alone does not demonstrate that turbulence
# selects organic-rich particles or leaves a distinct soot-rich mode. That claim
# would require a matched control and a quantified composition diagnostic.

# %% [markdown]
# ## Turn the overlays into a research comparison
#
# The example connects a common water-grown population to several collision
# assumptions and shows why both size and composition diagnostics are useful.
# A conserved material inventory supports interpretation of transfers, but does
# not choose the most accurate cloud model.
#
# For a next study, compare every branch with the same post-preparation state,
# repeat independent seeds, and report total number, wet-size quantiles, and dry
# organic-fraction distributions. Vary turbulent dissipation while holding other
# inputs fixed, then test timestep and sampling convergence. Measured size and
# composition data would be needed to attribute a real smoke–cloud response to
# one collision mechanism.
