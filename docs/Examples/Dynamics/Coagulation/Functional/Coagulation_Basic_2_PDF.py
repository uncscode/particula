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
# # Coagulation Basic 2: PDF (Builder + Coagulation)
#
# This notebook shows Brownian coagulation for a **continuous/PDF** distribution
# using the public builder pattern:
#
# 1. Build a Brownian coagulation strategy with `distribution_type="continuous_pdf"`.
# 2. Wrap it in `par.dynamics.Coagulation`.
# 3. Run a short, lightweight step on a PDF aerosol and visualize the change.
#
# Grid sizes and time horizon are kept small for quick execution (<120 s).

# %%
# In Colab uncomment the following command to install particula:
# #!pip install particula[extra] --quiet
import numpy as np
import matplotlib.pyplot as plt

import particula as par

# %% [markdown]
# ## Build a PDF aerosol
#
# We build a lognormal PDF on a modest grid. PDF units are m⁻³·m⁻¹ (number per
# volume per size).

# %%
# Radius bins (log-spaced)
radius_bins = np.logspace(-8, -6, 140)  # m

# Lognormal PDF (concentration per unit size)
concentration_pdf = par.particles.get_lognormal_pdf_distribution(
    x_values=radius_bins,
    mode=np.array([120e-9]),
    geometric_standard_deviation=np.array([1.5]),
    number_of_particles=np.array([8e8]),  # m^-3 total when integrated
)

# Atmosphere (ambient)
atmosphere = (
    par.gas.AtmosphereBuilder()
    .set_temperature(25, "degC")
    .set_pressure(1, "atm")
    .build()
)

# PDF particle representation
particles_pdf = (
    par.particles.PresetParticleRadiusBuilder()
    .set_distribution_type("pdf")
    .set_radius_bins(radius_bins, radius_bins_units="m")
    .set_concentration(concentration_pdf, concentration_units="m^-3")
    .set_density(np.array([1_000.0]), density_units="kg/m^3")
    .set_charge(np.zeros_like(radius_bins))
    .build()
)

aerosol_pdf = par.Aerosol(atmosphere=atmosphere, particles=particles_pdf)

# %% [markdown]
# ## Configure coagulation (continuous PDF)

# %%
coagulation_strategy = (
    par.dynamics.BrownianCoagulationBuilder()
    .set_distribution_type("continuous_pdf")
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
sub_steps = 10

pdf_before = aerosol_pdf.particles.get_concentration()
radii = aerosol_pdf.particles.get_radius()

aerosol_pdf_after = coagulation_process.execute(
    aerosol_pdf, time_step=time_step, sub_steps=sub_steps
)
pdf_after = aerosol_pdf_after.particles.get_concentration()

# %% [markdown]
# ## Plot: PDF concentration before/after

# %%
fig, ax = plt.subplots(figsize=(8, 5))
ax.plot(radii, pdf_before, label="Before", color="tab:blue")
ax.plot(radii, pdf_after, label="After", color="tab:orange")
ax.set_xscale("log")
ax.set_xlabel("Particle radius (m)")
ax.set_ylabel(r"PDF concentration ($m^{-4}$)")
ax.set_title("PDF coagulation (Brownian, continuous)")
ax.legend()
ax.grid(alpha=0.3)
plt.show()

# %% [markdown]
# ## Summary
#
# - Strategy: `BrownianCoagulationBuilder().set_distribution_type("continuous_pdf").build()`
# - Runnable: `par.dynamics.Coagulation`
# - Representation: PDF (continuous number concentration per size)
# - Runtime guardrails: small grid, short horizon
