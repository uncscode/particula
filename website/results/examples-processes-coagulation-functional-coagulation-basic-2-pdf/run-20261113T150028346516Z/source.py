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
# # Coagulation Basic 2: PDF (Builder + Coagulation)
#
# This notebook shows Brownian coagulation for a **continuous/PDF** distribution
# using the public builder pattern:
#
# 1. Build a Brownian coagulation strategy with
#    `distribution_type="continuous_pdf"`.
# 2. Wrap it in `par.dynamics.Coagulation`.
# 3. Run a short, lightweight step on a PDF aerosol and visualize
#    change.
#
# Grid sizes and time horizon are kept small for quick execution (<120 s).

# %%
# In Colab uncomment the following command to install particula:
# Install from the repository root: `python -m pip install -e ".[extra]"`
import matplotlib.pyplot as plt
import numpy as np
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

# PDF particle representation
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

pdf_before = aerosol_pdf.particles.concentration[0].copy()
radii = aerosol_pdf.particles.radii[0].copy()

aerosol_pdf_after = coagulation_process.execute(
    aerosol_pdf, time_step=time_step, sub_steps=sub_steps
)
pdf_after = aerosol_pdf_after.particles.concentration[0].copy()

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
# - Strategy: `BrownianCoagulationBuilder().set_distribution_type(`
#   "continuous_pdf").build()
# - Runnable: `par.dynamics.Coagulation`
# - Representation: PDF (continuous number concentration per size)
# - Runtime guardrails: small grid, short horizon

# %% [markdown]
# Native construction uses the explicitly generated distribution above. The old
# preset construction regenerated default modal inputs instead. Finite-grid
# coagulation has numerical mass drift; these curves illustrate number loss,
# not exact conservation or a time-step/grid-convergence claim.
