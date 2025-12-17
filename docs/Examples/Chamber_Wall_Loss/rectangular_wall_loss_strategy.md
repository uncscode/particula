# Wall Loss Strategies with `RectangularWallLossStrategy`

Learn how to use the rectangular chamber wall loss strategy in **particula**
for fast, size-resolved wall loss calculations in box-shaped chambers.

This guide shows how to:

- Instantiate `RectangularWallLossStrategy` with `chamber_dimensions`.
- Compute wall loss rates for a discrete distribution.
- Apply time stepping with `step` to update concentrations.
- Adapt the same API for `continuous_pdf` or `particle_resolved` cases.

For spherical chambers, see the companion
[`SphericalWallLossStrategy` guide](wall_loss_strategy.md).

## Prerequisites

Before starting, ensure you have:

- Python 3.9 or later
- `particula` installed:

```bash
pip install particula
# or
conda install -c conda-forge particula
```

- Basic familiarity with NumPy arrays.

## Quick Start: Rectangular Wall Loss on a Discrete Distribution

The fastest way to try `RectangularWallLossStrategy` is to:

1. Build a radius-binned particle distribution with
   `PresetParticleRadiusBuilder`.
2. Create `RectangularWallLossStrategy` with (x, y, z) chamber dimensions.
3. Compute the instantaneous wall loss rate and apply one time step.

```python
import numpy as np
import particula as par

# 1) Build a simple discrete particle distribution (radius bins)
particle = par.particles.PresetParticleRadiusBuilder().build()

# 2) Configure rectangular wall loss strategy (meters)
wall_loss = par.dynamics.RectangularWallLossStrategy(
    wall_eddy_diffusivity=1e-3,  # m^2/s
    chamber_dimensions=(1.0, 1.2, 0.8),  # (x, y, z) in meters
    distribution_type="discrete",
)

T = 298.15  # K
P = 101325.0  # Pa

# 3a) Compute instantaneous wall loss rate per bin
rate = wall_loss.rate(particle=particle, temperature=T, pressure=P)
print("Rate shape:", rate.shape)
print("Example rate[0]:", f"{rate[0]:.3e} 1/(m^3 s)")

# 3b) Apply one small time step
updated = wall_loss.step(
    particle=particle,
    temperature=T,
    pressure=P,
    time_step=10.0,  # seconds
)

print("Total concentration before:", f"{particle.get_total_concentration():.3e} 1/m^3")
print("Total concentration after:", f"{updated.get_total_concentration():.3e} 1/m^3")
```

**Expected output (approximate):**

```
Rate shape: (50,)
Example rate[0]: 1.7e-03 1/(m^3 s)
Total concentration before: 1.000e+06 1/m^3
Total concentration after: 9.831e+05 1/m^3
```

Numbers will vary slightly depending on the preset distribution, but the
concentration should decrease after `step`.

## Step-by-Step: Short Transient Simulation

Integrate wall loss over a 30-minute window with 5-second steps. This keeps
runtime well under a second on a laptop.

```python
import numpy as np

# Reuse `wall_loss` and `particle` from above
final_time = 30 * 60.0  # 30 minutes, seconds

dt = 5.0  # s
n_steps = int(final_time / dt)

history = np.zeros(n_steps + 1)
history[0] = particle.get_total_concentration()

for i in range(1, n_steps + 1):
    particle = wall_loss.step(
        particle=particle,
        temperature=T,
        pressure=P,
        time_step=dt,
    )
    history[i] = particle.get_total_concentration()

print("Final concentration:", f"{history[-1]:.3e} 1/m^3")
print("Relative decay:", f"{history[-1] / history[0]:.3f}")
```

Plotting (optional, matplotlib):

```python
from matplotlib import pyplot as plt

minutes = np.arange(n_steps + 1) * dt / 60.0

plt.plot(minutes, history / history[0])
plt.xlabel("Time (min)")
plt.ylabel("Normalized concentration")
plt.title("Rectangular wall loss (1.0 x 1.2 x 0.8 m)")
plt.grid(True)
plt.tight_layout()
plt.show()
```

## Using Other Distribution Types

`RectangularWallLossStrategy` supports the same distribution options as the
spherical strategy:

- `"discrete"` (shown above)
- `"continuous_pdf"`
- `"particle_resolved"`

### Continuous PDF

Use the same API; the rate and step operate on the continuous distribution
fields of your `ParticleRepresentation`:

```python
wall_loss_pdf = par.dynamics.RectangularWallLossStrategy(
    wall_eddy_diffusivity=1e-3,
    chamber_dimensions=(1.0, 1.2, 0.8),
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

### Particle-Resolved

For particle-resolved simulations, wall loss is applied as a survival
probability per particle. The API is identical; only the underlying
representation differs.

```python
resolved = par.particles.PresetResolvedParticleMassBuilder().build()

wall_loss_resolved = par.dynamics.RectangularWallLossStrategy(
    wall_eddy_diffusivity=1e-3,
    chamber_dimensions=(1.0, 1.2, 0.8),
    distribution_type="particle_resolved",
)

resolved = wall_loss_resolved.step(
    particle=resolved,
    temperature=T,
    pressure=P,
    time_step=dt,
)
```

## Configuration Reference

| Option | Description | Default |
| --- | --- | --- |
| `wall_eddy_diffusivity` | Wall eddy diffusivity [m^2/s]; must be positive. | Required |
| `chamber_dimensions` | Tuple `(x, y, z)` in meters; all three must be positive. | Required |
| `distribution_type` | One of `"discrete"`, `"continuous_pdf"`, `"particle_resolved"`. | `"discrete"` |

## Troubleshooting

- **ValueError: invalid chamber dimensions** — ensure the tuple has length 3
  and all entries are positive.
- **ValueError: invalid distribution type** — set `distribution_type` to one
  of the supported values exactly.
- **No concentration change over time** — verify `time_step > 0`, update the
  `particle` variable with the return from `step`, and use reasonable
  `wall_eddy_diffusivity` and dimensions.

## Next Steps

- Run the interactive notebook version for this guide:
  - [Rectangular Wall Loss Strategy Notebook](Notebooks/Rectangular_Wall_Loss_Strategy.ipynb)
- Compare with the spherical chamber example:
  - [Spherical Wall Loss Strategy Notebook](Notebooks/Spherical_Wall_Loss_Strategy.ipynb)
- Learn more about particle representations:
  - [Particle Phase Examples](../Particle_Phase/index.md)
- Read more about wall loss strategies in the main docs:
  - [Wall loss strategies](../../index.md#wall-loss-strategies)
