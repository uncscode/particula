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
# # Coagulation Basic 4: Particle Resolved (Builder + Coagulation)
#
# This notebook demonstrates Brownian coagulation for a **particle-resolved**
# representation using the builder pattern and the public `par.dynamics`
# `Coagulation` runnable. It mirrors the pattern notebooks but keeps sample sizes
# small for quick execution (<120 s).

# %%
# In Colab uncomment the following command to install particula:
# #!pip install particula[extra] --quiet
import numpy as np
import matplotlib.pyplot as plt

import particula as par

# %% [markdown]
# ## Build a particle-resolved aerosol
#
# We sample radii from a modest bimodal lognormal distribution, assign mass and
# density, and combine with a simple atmosphere.

# %%
# Sample radii (small sample for speed)
radii_sample = par.particles.get_lognormal_sample_distribution(
    mode=np.array([80e-9, 200e-9]),
    geometric_standard_deviation=np.array([1.4, 1.5]),
    number_of_particles=np.array([4e3, 2e3]),
    number_of_samples=12_000,
)

# Mass from radius assuming 1 g/cm^3
density = np.array([1_000.0])
mass_sample = 4 / 3 * np.pi * radii_sample**3 * density

# Atmosphere (ambient)
atmosphere = (
    par.gas.AtmosphereBuilder()
    .set_temperature(25, "degC")
    .set_pressure(1, "atm")
    .build()
)

# Particle-resolved representation
particles_resolved = (
    par.particles.ResolvedParticleMassRepresentationBuilder()
    .set_distribution_strategy(par.particles.ParticleResolvedSpeciatedMass())
    .set_activity_strategy(par.particles.ActivityIdealMass())
    .set_surface_strategy(par.particles.SurfaceStrategyVolume())
    .set_mass(mass_sample, mass_units="kg")
    .set_density(density, density_units="kg/m^3")
    .set_charge(0)
    .set_volume(1e-3, volume_units="m^3")
    .build()
)

aerosol_resolved = par.Aerosol(
    atmosphere=atmosphere, particles=particles_resolved
)

# %% [markdown]
# ## Configure coagulation (particle_resolved)

# %%
coagulation_strategy = (
    par.dynamics.BrownianCoagulationBuilder()
    .set_distribution_type("particle_resolved")
    .build()
)
coagulation_process = par.dynamics.Coagulation(
    coagulation_strategy=coagulation_strategy
)

print(coagulation_process)

# %% [markdown]
# ## Execute a short coagulation step

# %%
time_step = 200  # seconds
sub_steps = 5

radii_before = aerosol_resolved.particles.get_radius()

_aerosol_after = coagulation_process.execute(
    aerosol_resolved, time_step=time_step, sub_steps=sub_steps
)
radii_after = _aerosol_after.particles.get_radius()

# %% [markdown]
# ## Plot: particle-resolved histogram before/after

# %%
bins = np.logspace(
    np.log10(radii_before.min()), np.log10(radii_before.max()), 80
)

fig, ax = plt.subplots(figsize=(8, 5))
ax.hist(
    radii_before,
    bins=bins,
    histtype="step",
    color="tab:blue",
    label="Before",
    density=True,
)
ax.hist(
    radii_after,
    bins=bins,
    histtype="step",
    color="tab:orange",
    label="After",
    density=True,
)
ax.set_xscale("log")
ax.set_xlabel("Particle radius (m)")
ax.set_ylabel("PDF of sampled particles")
ax.set_title("Particle-resolved coagulation (Brownian)")
ax.legend()
ax.grid(alpha=0.3)
plt.show()

# %% [markdown]
# ## Summary
#
# - Strategy: `BrownianCoagulationBuilder().set_distribution_type("particle_resolved").build()`
# - Runnable: `par.dynamics.Coagulation`
# - Representation: particle-resolved sampled radii/mass
# - Runtime guardrails: ~12k samples, short horizon, few sub-steps
