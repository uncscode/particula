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
# # Configure chamber loss without a unit mismatch
#
# A chamber radius entered in centimetres instead of metres changes the physical
# problem. This compact how-to shows unit-aware construction of spherical and
# rectangular wall-loss strategies. The goal is to verify configuration before
# interpreting a loss rate or running a trajectory.

# %% [markdown]
# ## Use a known population for the configuration check
#
# Run with Particula and its example dependencies installed, including the unit
# conversion dependency. The native preset supplies a discrete, two-mode population
# for the spherical rate calculation. Temperature is 298 K and pressure 101325 Pa;
# this page does not evolve that population.

# %%
import particula as par

# Light-weight particle distribution for rate calculations
particle = par.particles.create_lognormal_binned_particle_data(
    distribution_type="discrete"
)
T = 298.0  # K
P = 101325.0  # Pa

# %% [markdown]
# ## Convert geometry and the wall-mixing coefficient explicitly
#
# The spherical builder converts 3.6 h⁻¹ to 10⁻³ s⁻¹ and 60 cm to 0.6 m.
# The wall eddy parameter in this model has units s⁻¹; it is not a gas diffusivity
# in m² s⁻¹. The discrete rate is concentration change per bin in m⁻³ s⁻¹.
# A tiny first-bin rate can reflect negligible occupancy, so it is not a summary
# of loss across the whole distribution.

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
print("rate[0]", f"{rate[0]:.3e} m^-3 s^-1")

# %% [markdown]
# ## Express the same choices as a configuration dictionary
#
# The factory constructs a rectangular strategy with dimensions 1.8 × 1.5 × 1.2 m
# and coefficient 1.2 h⁻¹ = 3.333…×10⁻⁴ s⁻¹. It selects particle-resolved semantics,
# but no resolved population is advanced here. Strategy configuration and the
# representation of the data must match when the strategy is used.

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
# ## Verify configuration before interpreting chamber decay
#
# The printed values confirm unit conversion and distribution selection; they
# are not a spherical-versus-rectangular physics comparison. Constructors reject
# invalid required geometry and coefficient inputs, and the factory rejects
# unknown strategy names.
#
# For a chamber-specific study, enter measured geometry and a justified wall-mixing
# parameter, then use the [geometry comparison](wall_loss_tutorial.py) with matched
# initial populations. Vary the coefficient over its uncertainty range and compare
# size-dependent survival, rather than treating a successful builder call as a
# calibrated deposition model.
