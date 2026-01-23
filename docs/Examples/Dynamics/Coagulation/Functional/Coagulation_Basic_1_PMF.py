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
# # Coagulation Basic 1: PMF (Builder + Coagulation)
#
# This notebook demonstrates Brownian coagulation for a **discrete/PMF**
# representation using the public `par.dynamics` API:
#
# 1. Build a Brownian coagulation strategy with `distribution_type="discrete"`.
# 2. Wrap it in `par.dynamics.Coagulation`.
# 3. Execute a short, lightweight step on a PMF aerosol and visualize the change.
#
# The setup mirrors the pattern notebooks (see `Coagulation_1_PMF_Pattern`) but
# keeps grids small so it runs quickly (<120 s).

# %%
# In Colab uncomment the following command to install particula:
# #!pip install particula[extra] --quiet
import numpy as np
import matplotlib.pyplot as plt

import particula as par

# %% [markdown]
# ## Build a PMF aerosol
#
# We create a lognormal PMF on a modest grid and attach it to a simple
# atmosphere. The PMF uses number concentration (m⁻³) per bin.

# %%
# Radius bins (log-spaced, modest resolution for speed)
radius_bins = np.logspace(-8, -6, 120)  # m

# Lognormal PMF (concentration per bin)
concentration_pmf = par.particles.get_lognormal_pmf_distribution(
    x_values=radius_bins,
    mode=np.array([80e-9]),
    geometric_standard_deviation=np.array([1.5]),
    number_of_particles=np.array([5e8]),  # m^-3 total
)

# Atmosphere (ambient, non-condensing)
atmosphere = (
    par.gas.AtmosphereBuilder()
    .set_temperature(25, "degC")
    .set_pressure(1, "atm")
    .build()
)

# PMF particle representation
particles_pmf = (
    par.particles.PresetParticleRadiusBuilder()
    .set_distribution_type("pmf")
    .set_radius_bins(radius_bins, radius_bins_units="m")
    .set_concentration(concentration_pmf, concentration_units="m^-3")
    .set_density(np.array([1_000.0]), density_units="kg/m^3")
    .set_charge(np.zeros_like(radius_bins))
    .build()
)

# Aerosol object
pmf_aerosol = par.Aerosol(atmosphere=atmosphere, particles=particles_pmf)

# %% [markdown]
# ## Configure coagulation (discrete)
#
# Use the Brownian builder with `distribution_type="discrete"`, then wrap in the
# public `Coagulation` runnable.

# %%
coagulation_strategy = (
    par.dynamics.BrownianCoagulationBuilder()
    .set_distribution_type("discrete")
    .build()
)
coagulation_process = par.dynamics.Coagulation(
    coagulation_strategy=coagulation_strategy
)

print(coagulation_process)

# %% [markdown]
# ## Execute a short coagulation step
#
# We keep the time step and sub-steps small for quick execution. The
# `Coagulation` runnable updates the aerosol in-place and returns it.

# %%
time_step = 200  # seconds
time_substeps = 10

pmf_before = pmf_aerosol.particles.get_concentration()
radii = pmf_aerosol.particles.get_radius()

pmf_after_aerosol = coagulation_process.execute(
    pmf_aerosol, time_step=time_step, sub_steps=time_substeps
)
pmf_after = pmf_after_aerosol.particles.get_concentration()

# %% [markdown]
# ## Plot: concentration before/after

# %%
fig, ax = plt.subplots(figsize=(8, 5))
ax.plot(radii, pmf_before, label="Before", color="tab:blue")
ax.plot(radii, pmf_after, label="After", color="tab:orange")
ax.set_xscale("log")
ax.set_xlabel("Particle radius (m)")
ax.set_ylabel(r"Number concentration ($m^{-3}$)")
ax.set_title("PMF coagulation (Brownian, discrete)")
ax.legend()
ax.grid(alpha=0.3)
plt.show()

# %% [markdown]
# ## Summary
#
# - Strategy: `BrownianCoagulationBuilder().set_distribution_type("discrete").build()`
# - Runnable: `par.dynamics.Coagulation`
# - Representation: PMF (bin-based number concentration)
# - Runtime guardrails: small grid, short horizon for fast execution
