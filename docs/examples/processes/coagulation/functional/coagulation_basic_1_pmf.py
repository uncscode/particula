# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
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
# 1. Build a Brownian coagulation strategy with
#    `distribution_type="discrete"`.
# 2. Wrap it in `par.dynamics.Coagulation`.
# 3. Run a short, lightweight step on a PMF aerosol and visualize
#    the change.
#
# The setup mirrors the pattern notebooks (see `Coagulation_1_PMF_Pattern`) but
# keeps grids small so it runs quickly (<120 s).

# %%
# In Colab uncomment the following command to install particula:
# Use the v0.3.0 development checkout: `python -m pip install -e ".[extra]"`
import matplotlib.pyplot as plt
import numpy as np
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

# Native ambient environment. GasData requires a species axis, so a
# zero-concentration, non-partitioning carrier fills that axis. Coagulation
# does not consume it; ambient pressure is specified independently.
environment = par.gas.EnvironmentData(
    temperature=np.array([298.15]),
    pressure=np.array([101325.0]),
    saturation_ratio=np.zeros((1, 1)),
)
gas = par.gas.GasData(
    name=["carrier"],
    molar_mass=np.array([0.02897]),
    partitioning=np.array([False]),
    concentration=np.zeros((1, 1)),
)

# PMF particle representation
particles_pmf = par.particles.create_particle_data_from_radii(
    radii=radius_bins,
    concentration=concentration_pmf,
    distribution_type="discrete",
    density=1000.0,
)

# Aerosol object
pmf_aerosol = par.Aerosol(
    particles=particles_pmf, gas=gas.copy(), environment=environment.copy()
)

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

pmf_before = pmf_aerosol.particles.concentration[0].copy()
radii = pmf_aerosol.particles.radii[0].copy()

pmf_after_aerosol = coagulation_process.execute(
    pmf_aerosol, time_step=time_step, sub_steps=time_substeps
)
pmf_after = pmf_after_aerosol.particles.concentration[0].copy()

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
# - Strategy: `BrownianCoagulationBuilder().set_distribution_type(`
#   "discrete").build()
# - Runnable: `par.dynamics.Coagulation`
# - Runnable: `par.dynamics.Coagulation`
# - Representation: PMF (bin-based number concentration)
# - Runtime guardrails: small grid, short horizon for fast execution

# %% [markdown]
# Native construction uses the explicitly generated distribution above. The old
# preset construction regenerated default modal inputs instead. Finite-grid
# coagulation has numerical mass drift; these curves illustrate number loss,
# not exact conservation or a time-step/grid-convergence claim.
