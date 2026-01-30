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

"""Showcase a PMF-based Brownian coagulation example.

This notebook builds a lognormal PMF aerosol, configures a Brownian
coagulation strategy with ``distribution_type="discrete"``, runs a short,
safe step, and visualizes the PMF concentration before and after the
coagulation runnable executes.

Example:
    Execute this script to generate before/after concentration curves for
a discrete PMF aerosol.
"""


# %% [markdown]
# # Coagulation Pattern 1: PMF
#
# Brownian coagulation using the public `par.dynamics` API with a
# discrete/PMF representation:
#
# 1. Build a Brownian coagulation strategy with
#    `distribution_type="discrete"`.
# 2. Wrap it in `par.dynamics.Coagulation`.
# 3. Run a short, fast step on a PMF aerosol and visualize the change.

# %%
# In Colab uncomment the following command to install particula:
# #!pip install particula[extra] --quiet
import matplotlib.pyplot as plt
import numpy as np
import particula as par

# %% [markdown]
# ## Build a PMF aerosol
#
# We create a modest lognormal PMF and attach it to an ambient atmosphere.

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
# Use the Brownian builder with `distribution_type="discrete"`, then wrap in
# the public `Coagulation` runnable.

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
# Keep the time step and sub-steps small for quick execution. The runnable
# updates the aerosol in-place and returns it.

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
# - Strategy: `BrownianCoagulationBuilder().set_distribution_type("discrete")`
# - Runnable: `par.dynamics.Coagulation`
# - Representation: PMF (bin-based number concentration)
# - Runtime guardrails: small grid, short horizon for fast execution
