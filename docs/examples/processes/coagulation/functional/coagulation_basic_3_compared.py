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
# # Coagulation Basic 3: PMF vs PDF (Builder + Coagulation)
#
# This notebook compares Brownian coagulation for PMF (discrete) and PDF
# (continuous) representations using the public builder pattern:
#
# - `distribution_type="discrete"` → PMF
# - `distribution_type="continuous_pdf"` → PDF
#
# Both are wrapped in `par.dynamics.Coagulation`.
# We run them over a short horizon for quick execution (<120 s).
# We also show how to convert between PMF and PDF using
# `par.particles.get_pdf_distribution_in_pmf` for like-for-like comparisons.

# %%
# In Colab uncomment the following command to install particula:
# Use the v0.3.0 development checkout: `python -m pip install -e ".[extra]"`
import matplotlib.pyplot as plt
import numpy as np
import particula as par
from scipy.integrate import trapezoid

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

# PMF particles
particles_pmf = par.particles.create_particle_data_from_radii(
    radii=radius_bins,
    concentration=concentration_pmf,
    distribution_type="discrete",
    density=1000.0,
)

aerosol_pmf = par.Aerosol(
    particles=particles_pmf, gas=gas.copy(), environment=environment.copy()
)

# PDF particles
particles_pdf = par.particles.create_particle_data_from_radii(
    radii=radius_bins,
    concentration=concentration_pdf,
    distribution_type="continuous_pdf",
    density=1000.0,
)

aerosol_pdf = par.Aerosol(
    particles=particles_pdf, gas=gas.copy(), environment=environment.copy()
)

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
pmf_before = aerosol_pmf.particles.concentration[0].copy()
aerosol_pmf_after = coagulation_pmf.execute(
    aerosol_pmf, time_step=time_step, sub_steps=sub_steps
)
pmf_after = aerosol_pmf_after.particles.concentration[0].copy()

# PDF run
pdf_before = aerosol_pdf.particles.concentration[0].copy()
aerosol_pdf_after = coagulation_pdf.execute(
    aerosol_pdf, time_step=time_step, sub_steps=sub_steps
)
pdf_after = aerosol_pdf_after.particles.concentration[0].copy()

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

# %% [markdown]
# Native construction uses the explicitly generated distribution above. The old
# preset construction regenerated default modal inputs instead. Finite-grid
# coagulation has numerical mass drift; these curves illustrate number loss,
# not exact conservation or a time-step/grid-convergence claim.
