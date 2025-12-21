# Wall Loss Strategies with `SphericalWallLossStrategy`

Learn how to use the new strategy-based wall loss API for chamber simulations in **particula**.

This guide focuses on the `WallLossStrategy` abstract base class and the
`SphericalWallLossStrategy` implementation, and shows how to:

- Build a simple particle distribution using `PresetParticleRadiusBuilder`.
- Configure `SphericalWallLossStrategy` for a spherical chamber.
- Integrate wall loss over time and visualize concentration decay.
- Connect wall loss strategies with other dynamics components.
- Extend to charged scenarios with `ChargedWallLossStrategy` when wall
  potentials or applied electric fields matter.

For a full forward simulation including dilution and coagulation, see the
[Chamber Forward Simulation](Notebooks/Chamber_Forward_Simulation.ipynb).

## Prerequisites

Before starting, ensure you have:

- Python 3.9 or later
- `particula` installed (from PyPI or conda):

```bash
pip install particula
# or
conda install -c conda-forge particula
```

- Basic familiarity with:
  - NumPy arrays
  - Matplotlib plotting

## Overview: Function vs Strategy APIs for Wall Loss

Historically, wall loss in particula was exposed via standalone rate
functions such as:

- `particula.dynamics.get_spherical_wall_loss_rate(...)`
- `particula.dynamics.get_rectangle_wall_loss_rate(...)`

The new wall loss strategy system introduces an object-oriented API that is
consistent with condensation and coagulation strategies:

- `particula.dynamics.WallLossStrategy` – abstract base class.
- `particula.dynamics.SphericalWallLossStrategy` – spherical chamber
  implementation.

Key advantages of the strategy-based API:

- Operates directly on `ParticleRepresentation` objects.
- Supports all distribution types: `"discrete"`, `"continuous_pdf"`, and
  `"particle_resolved"`.
- Exposes a consistent interface: `loss_coefficient`, `rate`, and `step`.

## Runnable entry point: `WallLoss` wrapper

Use `particula.dynamics.WallLoss` when you want a runnable that splits
`time_step` across `sub_steps`, clamps concentrations to non-negative values,
and plugs into runnable chaining with `|`.

```python
import particula as par

particle = par.particles.PresetParticleRadiusBuilder().build()
atmosphere = par.gas.Atmosphere(
    temperature=298.15,
    total_pressure=101325.0,
    partitioning_species=[],
    gas_only_species=[],
)
aerosol = par.Aerosol(atmosphere=atmosphere, particles=particle)

strategy = par.dynamics.SphericalWallLossStrategy(
    wall_eddy_diffusivity=1e-3,
    chamber_radius=0.5,
    distribution_type="discrete",
)
wall_loss = par.dynamics.WallLoss(wall_loss_strategy=strategy)

# Split 60 s into 3 sub-steps; concentrations clamp after each sub-step
aerosol = wall_loss.execute(aerosol, time_step=60.0, sub_steps=3)
print("Total concentration:", aerosol.particles.get_total_concentration())
```

### Chaining with other runnables

Combine wall loss with coagulation (or any other runnable) using the `|`
operator.

```python
coag_strategy = (
    par.dynamics.BrownianCoagulationBuilder()
    .set_distribution_type("discrete")
    .build()
)
coagulation = par.dynamics.Coagulation(
    coagulation_strategy=coag_strategy,
)

# Wall loss runs first, then Brownian coagulation in one call
combined = wall_loss | coagulation
aerosol = combined.execute(aerosol, time_step=60.0, sub_steps=3)
```

Switch `SphericalWallLossStrategy` for `RectangularWallLossStrategy`, or
change the distribution type as needed; the runnable handles the sub-step
splitting and clamping for any supported strategy.

## Quick Start: Spherical Wall Loss on a Lognormal Distribution

The fastest way to try the strategy-based API is to:

1. Build a lognormal particle distribution with
   `PresetParticleRadiusBuilder`.
2. Create a `SphericalWallLossStrategy`.
3. Apply wall loss over a short time window.

```python
import numpy as np
from matplotlib import pyplot as plt

import particula as par

# 1. Build a simple radius-based particle distribution (discrete bins)
particle = par.particles.PresetParticleRadiusBuilder().build()

print("Total concentration (initial):",
      f"{particle.get_total_concentration():.3e} 1/m^3")

# 2. Configure spherical wall loss strategy
wall_loss = par.dynamics.SphericalWallLossStrategy(
    wall_eddy_diffusivity=1e-3,  # m^2/s
    chamber_radius=0.5,          # m
    distribution_type="discrete",
)

# 3. Compute instantaneous wall loss rate
T = 298.15  # K
P = 101325.0  # Pa

rate = wall_loss.rate(
    particle=particle,
    temperature=T,
    pressure=P,
)

print("Rate array shape:", rate.shape)
print("Example rate[0]:", f"{rate[0]:.3e} 1/(m^3 s)")
```

At this point you have:

- A `ParticleRepresentation` with radius bins and number concentration.
- A `SphericalWallLossStrategy` that computes a first-order wall loss rate
  for each bin.

## Step-by-Step: Time Integration and Concentration Decay

Next, integrate wall loss over time and visualize the decay in total
concentration.

```python
# Simulation setup

# Integrate for 1 hour with 10-second time steps
final_time = 3600.0  # s

dt = 10.0  # s
n_steps = int(final_time / dt)

times = np.arange(n_steps + 1) * dt

total_concentration = np.zeros_like(times, dtype=float)

# Record initial concentration
total_concentration[0] = particle.get_total_concentration()

# Time loop: apply only wall loss
for i in range(1, n_steps + 1):
    particle = wall_loss.step(
        particle=particle,
        temperature=T,
        pressure=P,
        time_step=dt,
    )
    total_concentration[i] = particle.get_total_concentration()

print("Total concentration (final):",
      f"{total_concentration[-1]:.3e} 1/m^3")
```

### Visualizing Normalized Concentration Decay

```python
fig, ax = plt.subplots(figsize=(6, 4))

ax.plot(times / 60.0, total_concentration / total_concentration[0])

ax.set_xlabel("Time (min)")
ax.set_ylabel("Normalized total concentration")
ax.set_title("Spherical wall loss: concentration decay")
ax.grid(True)

plt.tight_layout()
plt.show()
```

You should see a monotonic decay in total particle concentration as particles
are deposited onto the spherical chamber walls.

### Comparing Initial vs Final Size Distributions

```python
# Rebuild the initial distribution for comparison
initial_particle = par.particles.PresetParticleRadiusBuilder().build()

fig, ax = plt.subplots(figsize=(6, 4))

ax.plot(
    initial_particle.get_radius(),
    initial_particle.get_concentration(),
    label="Initial",
)
ax.plot(
    particle.get_radius(),
    particle.get_concentration(),
    label="After wall loss",
)

ax.set_xscale("log")
ax.set_xlabel("Particle radius (m)")
ax.set_ylabel("Number concentration (1/m^3)")
ax.set_title("Size distribution before/after wall loss")
ax.grid(True)
ax.legend()

plt.tight_layout()
plt.show()
```

Because wall loss rates depend on particle size, you should see size-dependent
changes in the distribution, not just a uniform scaling.

## Using Different Distribution Types

`SphericalWallLossStrategy` supports all three distribution types used in
particula dynamics.

### Discrete (`"discrete"`)

This is the case used above: a radius-binned distribution where
concentration is stored per bin.

```python
wall_loss_discrete = par.dynamics.SphericalWallLossStrategy(
    wall_eddy_diffusivity=1e-3,
    chamber_radius=0.5,
    distribution_type="discrete",
)
```

### Continuous PDF (`"continuous_pdf"`)

Use this when your `ParticleRepresentation` stores a continuous probability
density (e.g., when using PDF-based strategies). The API is identical:

```python
wall_loss_pdf = par.dynamics.SphericalWallLossStrategy(
    wall_eddy_diffusivity=1e-3,
    chamber_radius=0.5,
    distribution_type="continuous_pdf",
)

rate_pdf = wall_loss_pdf.rate(particle, temperature=T, pressure=P)
particle = wall_loss_pdf.step(
    particle=particle,
    temperature=T,
    pressure=P,
    time_step=dt,
)
```

### Particle-Resolved (`"particle_resolved"`)

For particle-resolved simulations, wall loss is interpreted as a
probabilistic removal process, implemented via a deterministic survival
probability update.

```python
# Example: particle-resolved builder (see Particle Phase examples for details)
resolved_particle = (
    par.particles.PresetResolvedParticleMassBuilder()
    .set_volume(1.0, "m^3")
    .build()
)

wall_loss_resolved = par.dynamics.SphericalWallLossStrategy(
    wall_eddy_diffusivity=1e-3,
    chamber_radius=0.5,
    distribution_type="particle_resolved",
)

resolved_particle = wall_loss_resolved.step(
    particle=resolved_particle,
    temperature=T,
    pressure=P,
    time_step=dt,
)
```

For particle-resolved distributions, the strategy updates the internal
concentration using a survival probability derived from the wall loss rate.

## Composing Wall Loss with Other Dynamics

Wall loss strategies are designed to plug into the broader dynamics system.
You can combine wall loss with coagulation, condensation, or dilution
strategies in a single time integration loop.

The general pattern is:

```python
# Pseudo-code sketch: combine wall loss with another dynamics process

# 1. Build particle representation (as shown above)
particle = par.particles.PresetParticleRadiusBuilder().build()

# 2. Configure wall loss and another dynamics strategy
wall_loss = par.dynamics.SphericalWallLossStrategy(
    wall_eddy_diffusivity=1e-3,
    chamber_radius=0.5,
    distribution_type="discrete",
)

condensation = par.dynamics.CondensationIsothermal(
    molar_mass=180e-3,              # kg/mol
    diffusion_coefficient=2e-5,     # m^2/s
    accommodation_coefficient=1.0,
)

# 3. Time loop: apply both processes each step
for _ in range(n_steps):
    # First update composition/size via condensation
    particle = condensation.step(
        particle=particle,
        temperature=T,
        pressure=P,
        time_step=dt,
    )

    # Then apply wall loss to updated distribution
    particle = wall_loss.step(
        particle=particle,
        temperature=T,
        pressure=P,
        time_step=dt,
    )
```

In a full chamber model, you would typically:

- Wrap one or more strategies in higher-level process classes.
- Combine wall loss with dilution and coagulation.
- Use `Aerosol` objects to keep gas and particle phases synchronized.

The existing [Chamber Forward Simulation](Notebooks/Chamber_Forward_Simulation.ipynb)
notebook shows a function-based version of this workflow; the strategy-based
API lets you perform the same type of simulation using
`SphericalWallLossStrategy`.

## Configuration Options

| Option                   | Description                                              | Default        |
|--------------------------|----------------------------------------------------------|----------------|
| `wall_eddy_diffusivity` | Wall eddy diffusivity controlling wall mixing [m^2/s].  | Required (no default) |
| `chamber_radius`        | Radius of the spherical chamber [m].                     | Required (no default) |
| `distribution_type`     | Distribution type: `"discrete"`, `"continuous_pdf"`, or `"particle_resolved"`. | `"discrete"` |

## Troubleshooting

### ImportError: `SphericalWallLossStrategy` not found

Make sure you are using particula 0.2.6 or later and importing from the
`dynamics` namespace:

```python
import particula as par

wall_loss = par.dynamics.SphericalWallLossStrategy(
    wall_eddy_diffusivity=1e-3,
    chamber_radius=0.5,
)
```

### ValueError: Invalid distribution type

Ensure `distribution_type` is exactly one of:

- `"discrete"`
- `"continuous_pdf"`
- `"particle_resolved"`

Any other string will raise a `ValueError` during initialization.

### Concentration does not change over time

Check the following:

- Time step `dt` is not zero and simulation runs for more than one step.
- `wall_eddy_diffusivity` and `chamber_radius` are positive and
  physically reasonable.
- You are updating `particle` with the return value of `step` inside the
  loop.

## Next Steps

- Run the interactive notebook version:
  - [Spherical Wall Loss Strategy Notebook](Notebooks/Spherical_Wall_Loss_Strategy.ipynb)
- Explore the chamber forward simulation:
  - [Chamber Forward Simulation](Notebooks/Chamber_Forward_Simulation.ipynb)
- Learn more about particle representations:
  - [Particle Phase Examples](../Particle_Phase/index.md)
- Read more about wall loss strategies in the main docs:
  - [Dynamics and wall loss overview][wall-loss-overview]

[wall-loss-overview]: ../../index.md#wall-loss-strategies-and-runnable
