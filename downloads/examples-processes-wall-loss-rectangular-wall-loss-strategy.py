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
# # Track removal in a rectangular chamber
#
# How does a size-dependent wall-loss law change total particle abundance while
# leaving individual particle masses fixed? This example follows a prescribed
# binned population for 30 minutes in a rectangular chamber. It connects an
# instantaneous loss rate to a numerical decay trajectory and an explicit check
# of the update formula.

# %% [markdown]
# ## Prepare a local calculation
#
# Use Python 3.12 or newer with Particula and its example dependencies installed.
# The example prints diagnostics; the plotting cell is commented out and may be
# enabled locally. For the distinction between removal and particle shrinkage,
# see [CPU dilution](../dilution/cpu_dilution.py).

# %% [markdown]
# ## Keep units visible
#
# All geometry below is in metres, time in seconds, temperature in kelvin, and
# pressure in pascals. NumPy printing suppresses very small values, so displayed
# negative zeros in the first few rates do not establish exactly zero loss.

# %%
import numpy as np
import particula as par

np.set_printoptions(precision=3, suppress=True)

# %% [markdown]
# ## Initialize a distribution whose total has a clear meaning
#
# The preset uses 250 radius nodes from 1 nm–100 µm, lognormal radius
# scales 100 nm and 1 µm, geometric standard deviations 1.2 and 1.4, and modal
# number densities 10¹⁰ and 10⁹ m⁻³. Density is 1000 kg m⁻³. These are demonstration
# inputs, not a chamber measurement. Bin concentrations are in m⁻³ and sum to
# total number density.

# %%
particle = par.particles.create_lognormal_binned_particle_data(
    distribution_type="discrete"
)
print("Bins:", particle.radii[0].size)
print(
    "Initial total concentration:",
    f"{particle.number_density[0]:.3e} 1/m^3",
)

# %% [markdown]
# <a id="2-configure-rectangularwalllossstrategy"></a>
#
# ## Specify geometry and environmental conditions
#
# The rectangular dimensions are 1.0 × 1.2 × 0.8 m. The wall eddy parameter is
# 10⁻³ s⁻¹ at 298.15 K and 101325 Pa. These fixed conditions determine a loss
# coefficient kᵢ for each radius. The model contains no growth, sources, dilution,
# or collisions.

# %%
wall_loss = par.dynamics.RectangularWallLossStrategy(
    wall_eddy_diffusivity=1e-3,
    chamber_dimensions=(1.0, 1.2, 0.8),
    distribution_type="discrete",
)
T = 298.15  # K
P = 101325.0  # Pa

# %% [markdown]
# ## Distinguish coefficient from rate
#
# `loss_coefficient` returns kᵢ in s⁻¹. `rate` combines that coefficient with
# current bin concentration, giving −kᵢ Cᵢ in m⁻³ s⁻¹. Consequently a sparsely
# occupied bin can have a tiny rate even when its fractional loss is large.

# %%
rate = wall_loss.rate(particle=particle, temperature=T, pressure=P)
print("Rate shape:", rate.shape)
print("First three rates (m^-3 s^-1):", rate.ravel()[:3])

# %% [markdown]
# ## Advance 30 minutes with a positivity bound
#
# Output intervals are 5 s. Internal substeps are selected so the largest kᵢ Δt
# is at most 0.9, avoiding negative concentrations in the explicit Euler update.
# The saved run uses nine substeps per interval. This is a positivity condition,
# not an accuracy criterion.
#
# Because radii and conditions remain fixed, the final array can be checked
# against repeated Euler decay Cᵢ(0)(1−kᵢ Δt)ⁿ. The assertion checks that discrete
# update and unchanged individual masses; convergence to continuous exponential
# decay would require a separate timestep study.

# %%
dt = 5.0  # seconds
steps = int((30 * 60) / dt)
# Native Euler updates reject negative concentrations. Retain the output
# interval and full duration, but bound each internal step by max(k) * dt < 1.
sub_steps = max(
    1,
    int(
        np.ceil(
            np.max(
                wall_loss.loss_coefficient(particle, temperature=T, pressure=P)
            )
            * dt
            / 0.9
        )
    ),
)
print("Internal substeps per output interval:", sub_steps)
initial_counts = particle.concentration.copy()
initial_masses = particle.masses.copy()
loss_coefficients = wall_loss.loss_coefficient(
    particle, temperature=T, pressure=P
)
history = np.zeros(steps + 1)
history[0] = particle.number_density[0]

for i in range(1, steps + 1):
    for _substep in range(sub_steps):
        particle = wall_loss.step(
            particle=particle,
            temperature=T,
            pressure=P,
            time_step=dt / sub_steps,
        )
    history[i] = particle.number_density[0]

# Fixed radii give an independent repeated-Euler decay prediction.
np.testing.assert_allclose(
    particle.concentration,
    initial_counts
    * (1 - loss_coefficients * dt / sub_steps) ** (steps * sub_steps),
    rtol=1e-12,
    atol=1e-30,
)
np.testing.assert_array_equal(particle.masses, initial_masses)
assert np.all(particle.concentration >= 0)

print("Final concentration:", f"{history[-1]:.3e} 1/m^3")
print("Relative decay:", f"{history[-1] / history[0]:.3f}")

# %% [markdown]
# ### Optional view of the aggregate decay
#
# The commented plot would show total number divided by its initial value.
# That total sums bins with different loss coefficients, so it need not follow
# a single exponential. No figure is emitted by this cell in the published run.

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
# ## Keep representation semantics separate
#
# The PDF example below calculates an instantaneous rate and checks its sign and
# finiteness. Its concentration measure differs from bin totals. The resolved
# example constructs state and strategy, but its advance is commented out;
# this page therefore does not demonstrate a resolved trajectory.

# %%
# Continuous PDF example (same call pattern)
wall_loss_pdf = par.dynamics.RectangularWallLossStrategy(
    wall_eddy_diffusivity=1e-3,
    chamber_dimensions=(1.0, 1.2, 0.8),
    distribution_type="continuous_pdf",
)
particle_pdf = par.particles.create_lognormal_binned_particle_data(
    distribution_type="continuous_pdf"
)
rate_pdf = wall_loss_pdf.rate(particle_pdf, temperature=T, pressure=P)
assert np.all(np.isfinite(rate_pdf)) and np.all(rate_pdf <= 0)

# Particle-resolved example
np.random.seed(42)
resolved = par.particles.create_lognormal_resolved_particle_data()
wall_loss_resolved = par.dynamics.RectangularWallLossStrategy(
    wall_eddy_diffusivity=1e-3,
    chamber_dimensions=(1.0, 1.2, 0.8),
    distribution_type="particle_resolved",
)
# resolved = wall_loss_resolved.step(
#     particle=resolved, temperature=T, pressure=P, time_step=dt
# )

# %% [markdown]
# ## What the checked trajectory supports
#
# The saved binned run decreases from 1.100×10¹⁰ to 1.065×10¹⁰ m⁻³, retaining
# about 96.8% after 30 minutes. The update preserves individual masses while
# reducing abundance. It is evidence for the specified numerical calculation,
# not validation against a chamber experiment.
#
# To estimate a size-dependent chamber lifetime, compare per-bin survival and
# repeat with finer substeps. Then vary geometry or wall-mixing strength within
# measurement uncertainty. The [spherical example](spherical_wall_loss_strategy.py)
# uses different geometry and duration; match those comparison choices explicitly
# rather than attributing every difference to shape.
