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
# # Wall Loss Builders and Factory (interactive)
#
# Try the new wall loss builders and the `WallLossFactory` with unit-aware
# setters and distribution type options.

# %% [markdown]
# ## Prerequisites
#
# - Python 3.12+
# - `particula` installed (`pip install particula`)
# - NumPy included with `particula` for sample particle presets

# %%
import particula as par

# Light-weight particle distribution for rate calculations
particle = par.particles.PresetParticleRadiusBuilder().build()
T = 298.0  # K
P = 101325.0  # Pa

# %% [markdown]
# ## Builder chaining with units
#
# Convert `1/hr` to `1/s` and `cm` to `m` while keeping distribution type at
# the default (`"discrete"`).

# %%
spherical = (
    par.dynamics.SphericalWallLossBuilder()
    .set_wall_eddy_diffusivity(3.6, "1/hr")
    .set_chamber_radius(60.0, "cm")
    .build()
)

rate = spherical.rate(particle=particle, temperature=T, pressure=P)

print("eddy diffusivity [1/s]", f"{spherical.wall_eddy_diffusivity:.3e}")
print("chamber radius [m]", f"{spherical.chamber_radius:.2f}")
print("distribution type", spherical.distribution_type)
print("rate[0]", f"{rate[0]:.3e} 1/s")

# %% [markdown]
# ## Factory creation for rectangular chambers
#
# `WallLossFactory.get_strategy` applies `set_parameters` (including unit
# suffixes) and `distribution_type`. Strategy names are case-insensitive.

# %%
factory = par.dynamics.WallLossFactory()

rectangular = factory.get_strategy(
    "rectangular",
    {
        "wall_eddy_diffusivity": 1.2,
        "wall_eddy_diffusivity_units": "1/hr",
        "chamber_dimensions": (180.0, 150.0, 120.0),
        "chamber_dimensions_units": "cm",
        "distribution_type": "particle_resolved",
    },
)

print("rectangular dims [m]", rectangular.chamber_dimensions)
print("distribution type", rectangular.distribution_type)
print("diffusivity [1/s]", f"{rectangular.wall_eddy_diffusivity:.3e}")

# %% [markdown]
# ## Notes
#
# - Supported `distribution_type` values: `discrete` (default),
#   `continuous_pdf`, `particle_resolved`.
# - Negative or zero inputs for diffusivity or geometry raise `ValueError`.
# - Unknown strategy names raise `ValueError` in the factory.
