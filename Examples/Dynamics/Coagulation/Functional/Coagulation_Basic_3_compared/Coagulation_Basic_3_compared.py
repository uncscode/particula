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
# # Coagulation Basic 3: PMF vs PDF (Builder + Coagulation)
#
# This notebook compares Brownian coagulation for PMF (discrete) and PDF
# (continuous) representations using the public builder pattern:
#
# - `distribution_type="discrete"` → PMF
# - `distribution_type="continuous_pdf"` → PDF
#
# Both are wrapped in `par.dynamics.Coagulation` and run over a short horizon for
# quick execution (<120 s). We also show how to convert between PMF and PDF using
# `par.particles.get_pdf_distribution_in_pmf` for like-for-like comparisons.

# %%
# In Colab uncomment the following command to install particula:
# #!pip install particula[extra] --quiet
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import trapezoid

import particula as par

# %% [markdown]
# ## Build aerosols (PMF and PDF)

# %%
# Shared grid (modest resolution)
radius_bins = np.logspace(-8, -6, 140)  # m

# Lognormal parameters (kept small for speed)
mode = np.array([120e-9])
gsd = np.array([1.5])
number_conc = np.array([8e8])  # m^-3 total

# PMF (number per bin)
concentration_pmf = par.particles.get_lognormal_pmf_distribution(
    x_values=radius_bins,
    mode=mode,
    geometric_standard_deviation=gsd,
    number_of_particles=number_conc,
)

# PDF (number per size)
concentration_pdf = par.particles.get_lognormal_pdf_distribution(
    x_values=radius_bins,
    mode=mode,
    geometric_standard_deviation=gsd,
    number_of_particles=number_conc,
)

# Atmosphere
atmosphere = (
    par.gas.AtmosphereBuilder()
    .set_temperature(25, "degC")
    .set_pressure(1, "atm")
    .build()
)

# PMF particles
particles_pmf = (
    par.particles.PresetParticleRadiusBuilder()
    .set_distribution_type("pmf")
    .set_radius_bins(radius_bins, radius_bins_units="m")
    .set_concentration(concentration_pmf, concentration_units="m^-3")
    .set_density(np.array([1_000.0]), density_units="kg/m^3")
    .set_charge(np.zeros_like(radius_bins))
    .build()
)

aerosol_pmf = par.Aerosol(atmosphere=atmosphere, particles=particles_pmf)

# PDF particles
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
# ## Configure coagulation runnables

# %%
coagulation_pmf = par.dynamics.Coagulation(
    coagulation_strategy=(
        par.dynamics.BrownianCoagulationBuilder()
        .set_distribution_type("discrete")
        .build()
    )
)

coagulation_pdf = par.dynamics.Coagulation(
    coagulation_strategy=(
        par.dynamics.BrownianCoagulationBuilder()
        .set_distribution_type("continuous_pdf")
        .build()
    )
)

print(coagulation_pmf)
print(coagulation_pdf)

# %% [markdown]
# ## Execute short coagulation steps

# %%
time_step = 200  # seconds
sub_steps = 10

# PMF run
pmf_before = aerosol_pmf.particles.get_concentration()
aerosol_pmf_after = coagulation_pmf.execute(
    aerosol_pmf, time_step=time_step, sub_steps=sub_steps
)
pmf_after = aerosol_pmf_after.particles.get_concentration()

# PDF run
pdf_before = aerosol_pdf.particles.get_concentration()
aerosol_pdf_after = coagulation_pdf.execute(
    aerosol_pdf, time_step=time_step, sub_steps=sub_steps
)
pdf_after = aerosol_pdf_after.particles.get_concentration()

# %% [markdown]
# ## Convert for apples-to-apples comparison
#
# We convert PDF⇄PMF using `par.particles.get_pdf_distribution_in_pmf` so both
# representations can be compared on the same units.

# %%
pdf_as_pmf_before = par.particles.get_pdf_distribution_in_pmf(
    x_array=radius_bins, distribution=pdf_before, to_pdf=False
)
pdf_as_pmf_after = par.particles.get_pdf_distribution_in_pmf(
    x_array=radius_bins, distribution=pdf_after, to_pdf=False
)

pmf_as_pdf_before = par.particles.get_pdf_distribution_in_pmf(
    x_array=radius_bins, distribution=pmf_before, to_pdf=True
)
pmf_as_pdf_after = par.particles.get_pdf_distribution_in_pmf(
    x_array=radius_bins, distribution=pmf_after, to_pdf=True
)

# Quick consistency check
number_pmf = pmf_before.sum()
number_pdf = trapezoid(pdf_before, x=radius_bins)
print(f"Total number PMF: {number_pmf:.2e} m^-3")
print(f"Total number PDF: {number_pdf:.2e} m^-3 (integrated)")

# %% [markdown]
# ## Plots

# %%
fig, ax = plt.subplots(figsize=(8, 5))
ax.plot(radius_bins, pmf_before, label="PMF before", color="tab:blue")
ax.plot(radius_bins, pmf_after, label="PMF after", color="tab:orange")
ax.plot(
    radius_bins,
    pdf_as_pmf_before,
    label="PDF→PMF before",
    color="tab:green",
    linestyle="--",
)
ax.plot(
    radius_bins,
    pdf_as_pmf_after,
    label="PDF→PMF after",
    color="tab:red",
    linestyle="--",
)
ax.set_xscale("log")
ax.set_xlabel("Particle radius (m)")
ax.set_ylabel(r"Number concentration ($m^{-3}$)")
ax.set_title("PMF vs PDF coagulation (converted to PMF units)")
ax.legend()
ax.grid(alpha=0.3)
plt.show()

# %%
fig, ax = plt.subplots(figsize=(8, 5))
ax.plot(
    radius_bins, pmf_as_pdf_before, label="PMF→PDF before", color="tab:blue"
)
ax.plot(
    radius_bins, pmf_as_pdf_after, label="PMF→PDF after", color="tab:orange"
)
ax.plot(
    radius_bins,
    pdf_before,
    label="PDF before",
    color="tab:green",
    linestyle="--",
)
ax.plot(
    radius_bins, pdf_after, label="PDF after", color="tab:red", linestyle="--"
)
ax.set_xscale("log")
ax.set_xlabel("Particle radius (m)")
ax.set_ylabel(r"PDF concentration ($m^{-4}$)")
ax.set_title("PMF vs PDF coagulation (PDF units)")
ax.legend()
ax.grid(alpha=0.3)
plt.show()

# %% [markdown]
# ## Summary
#
# - PMF: `distribution_type="discrete"` via `BrownianCoagulationBuilder`
# - PDF: `distribution_type="continuous_pdf"`
# - Both run through `par.dynamics.Coagulation`
# - Conversions use `par.particles.get_pdf_distribution_in_pmf` for consistent
#   comparisons across representations.
# - Runtime kept short with modest grids and a single 200 s step.
