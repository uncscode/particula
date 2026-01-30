# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.17.3
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Rectangular Wall Loss Strategy (interactive quick start)
#
# This notebook shows how to use `particula.dynamics.RectangularWallLossStrategy`
# to compute wall loss rates and apply time stepping in a box-shaped chamber.
# Runtime is kept small (<1 s) with tiny arrays and short integration horizons.

# %% [markdown]
# ## Prerequisites
# - Python 3.12+
# - `particula` installed (pip or conda)
# - NumPy for light array handling
#
# To run locally:
# ```bash
# pip install particula
# ```

# %% [markdown]
# ## Imports and helpers

# %%
import numpy as np
import particula as par

np.set_printoptions(precision=3, suppress=True)

# %% [markdown]
# ## 1) Build a discrete particle distribution
# We use the preset radius-binned distribution to keep setup minimal.

# %%
particle = par.particles.PresetParticleRadiusBuilder().build()
print("Bins:", particle.get_radius().size)
print("Initial total concentration:", f"{particle.get_total_concentration():.3e} 1/m^3")

# %% [markdown]
# ## 2) Configure `RectangularWallLossStrategy`
# Dimensions are given as `(x, y, z)` in meters.

# %%
wall_loss = par.dynamics.RectangularWallLossStrategy(
    wall_eddy_diffusivity=1e-3,
    chamber_dimensions=(1.0, 1.2, 0.8),
    distribution_type="discrete",
)
T = 298.15  # K
P = 101325.0  # Pa

# %% [markdown]
# ## 3) Compute instantaneous wall loss rate
# This returns a per-bin rate array (1/s).

# %%
rate = wall_loss.rate(particle=particle, temperature=T, pressure=P)
print("Rate shape:", rate.shape)
print("First three rates (1/s):", rate[:3])

# %% [markdown]
# ## 4) Apply a short transient (30 min, 5 s steps)
# Keeps runtime small while showing concentration decay.

# %%
dt = 5.0  # seconds
steps = int((30 * 60) / dt)
history = np.zeros(steps + 1)
history[0] = particle.get_total_concentration()

for i in range(1, steps + 1):
    particle = wall_loss.step(
        particle=particle, temperature=T, pressure=P, time_step=dt
    )
    history[i] = particle.get_total_concentration()

print("Final concentration:", f"{history[-1]:.3e} 1/m^3")
print("Relative decay:", f"{history[-1] / history[0]:.3f}")

# %% [markdown]
# ### Optional: quick visualization
# Uncomment to view a normalized decay curve (kept lightweight).

# %%
# from matplotlib import pyplot as plt
# minutes = np.arange(steps + 1) * dt / 60.0
# plt.plot(minutes, history / history[0])
# plt.xlabel("Time (min)")
# plt.ylabel("Normalized concentration")
# plt.title("Rectangular wall loss (1.0 x 1.2 x 0.8 m)")
# plt.grid(True)
# plt.tight_layout()
# plt.show()

# %% [markdown]
# ## 5) Notes on other distribution types
# The same API works for `"continuous_pdf"` and `"particle_resolved"`.
# Only the underlying particle representation changes.

# %%
# Continuous PDF example (same call pattern)
wall_loss_pdf = par.dynamics.RectangularWallLossStrategy(
    wall_eddy_diffusivity=1e-3,
    chamber_dimensions=(1.0, 1.2, 0.8),
    distribution_type="continuous_pdf",
)
# rate_pdf = wall_loss_pdf.rate(particle, temperature=T, pressure=P)

# Particle-resolved example
resolved = par.particles.PresetResolvedParticleMassBuilder().build()
wall_loss_resolved = par.dynamics.RectangularWallLossStrategy(
    wall_eddy_diffusivity=1e-3,
    chamber_dimensions=(1.0, 1.2, 0.8),
    distribution_type="particle_resolved",
)
# resolved = wall_loss_resolved.step(
#     particle=resolved, temperature=T, pressure=P, time_step=dt
# )

# %% [markdown]
# ## Summary
# - Configure rectangular chambers with `chamber_dimensions` (x, y, z).
# - Use `rate` for instantaneous loss coefficients.
# - Use `step` to update particle concentrations over time.
# - Switch `distribution_type` to target discrete, continuous PDF, or
#   particle-resolved representations.
