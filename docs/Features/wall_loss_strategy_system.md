# Wall Loss Strategy System

> Strategy-based wall loss for chamber simulations that plugs directly into particula's dynamics workflow.

## Overview

The wall loss strategy system lets you model particle deposition onto chamber walls using the same object-oriented patterns as condensation and coagulation. Instead of calling standalone rate functions, you work with strategy objects that operate on `ParticleRepresentation`, support multiple distribution types, and expose a consistent `rate` / `step` interface.

This feature is built around user-facing APIs exposed via `particula.dynamics`:

- `WallLossStrategy` – abstract base class for wall loss models.
- `SphericalWallLossStrategy` and `RectangularWallLossStrategy` – chamber implementations using existing wall loss coefficient utilities.
- `SphericalWallLossBuilder` and `RectangularWallLossBuilder` – validated, unit-aware builders for configuring strategies.
- `WallLossFactory` – factory for selecting a wall loss geometry by name with builder defaults.

## Key Benefits

- **Consistent dynamics workflow**: Use the same strategy-based API (`rate`, `step`, `distribution_type`) as for condensation and coagulation.
- **Builder/factory parity with validation**: Configure wall loss using the same unit-aware builder and factory patterns as other dynamics modules, with built-in checks for geometry and distribution types.
- **Direct integration with ParticleRepresentation**: Apply wall loss directly to particle distributions without writing glue code.
- **Extensible wall loss models**: Add new chamber geometries or wall loss models as additional `WallLossStrategy` implementations.

## Who It's For

This feature is designed for:

- **Chamber simulation users**: Running time-dependent simulations where wall deposition competes with processes like condensation, coagulation, and dilution.
- **Model developers**: Implementing new wall loss parameterizations while reusing particula's particle and dynamics infrastructure.
- **Experiment interpreters**: Matching chamber experiments with simulations that treat wall loss as a first-class process.

## Capabilities

### Unified wall loss API in `particula.dynamics`

Wall loss is exposed alongside other dynamics components:

```python
import particula as par

# Abstract interface (not instantiated directly)
par.dynamics.WallLossStrategy

# Concrete chamber implementations
par.dynamics.SphericalWallLossStrategy
par.dynamics.RectangularWallLossStrategy

# Builder and factory API (also re-exported via particula.dynamics.wall_loss)
par.dynamics.SphericalWallLossBuilder
par.dynamics.RectangularWallLossBuilder
par.dynamics.WallLossFactory
```

All wall loss strategies share a common shape:

- Initialize with physical parameters (e.g., wall eddy diffusivity, chamber radius).
- Call `rate(particle, temperature, pressure)` to compute instantaneous loss rate.
- Call `step(particle, temperature, pressure, time_step)` to advance the system.

### Spherical wall loss strategy

`SphericalWallLossStrategy` models deposition in a well-mixed spherical chamber. It:

- Reuses existing wall loss coefficient utilities for the underlying physics.
- Works with any `ParticleRepresentation` compatible with particula dynamics.
- Computes size-dependent loss rates and updates particle concentration over time.

Typical initialization:

```python
wall_loss = par.dynamics.SphericalWallLossStrategy(
    wall_eddy_diffusivity=1e-3,  # m^2/s
    chamber_radius=0.5,          # m
    distribution_type="discrete",
)
```

### Rectangular wall loss strategy

`RectangularWallLossStrategy` models deposition in box-shaped chambers. Pair it with `RectangularWallLossBuilder` to validate a (length, width, height) tuple and convert units before constructing the strategy.

### Builder and factory workflow

Builders and the factory give you a validated, unit-aware way to construct wall loss strategies and keep parity with other dynamics modules. Geometry lengths convert to meters and wall eddy diffusivity converts to 1/s; setters validate positivity and enforce a length-3 tuple for rectangular chambers. Distribution types are restricted to the supported set and default to `"discrete"`.

```python
import particula as par

# Chained builder with unit conversion and validation
wall_loss = (
    par.dynamics.SphericalWallLossBuilder()
    .set_wall_eddy_diffusivity(1e-3, "1/s")
    .set_chamber_radius(50.0, "cm")  # converts to 0.5 m
    .set_distribution_type("discrete")
    .build()
)
```

For rectangular chambers, use `set_chamber_dimensions((L, W, H), units)`; each side must be positive and provided as a 3-tuple.

```python
factory = par.dynamics.WallLossFactory()
rectangular = factory.get_strategy(
    strategy_type="rectangular",
    parameters={
        "wall_eddy_diffusivity": 1e-4,
        "chamber_dimensions": (1.0, 0.5, 0.5),
        "distribution_type": "continuous_pdf",
    },
)
```

`WallLossFactory` is exported via both `particula.dynamics.wall_loss` and `particula.dynamics`, letting you select a geometry by name without manually instantiating builders.

### Support for multiple distribution types

The strategy system operates on the same distribution types used elsewhere in particula:

- "discrete" – radius-binned distributions.
- "continuous_pdf" – continuous probability-density representations.
- "particle_resolved" – ensembles of individual particles.

You select the appropriate mode at initialization (or via builder/factory parameters) with `distribution_type`, and the strategy adjusts its `step` behavior to match the representation.


## Getting Started

### Quick start: wall loss on a discrete distribution

```python
import particula as par

# 1. Build a radius-binned particle distribution
particle = par.particles.PresetParticleRadiusBuilder().build()

# 2. Configure spherical wall loss
wall_loss = par.dynamics.SphericalWallLossStrategy(
    wall_eddy_diffusivity=1e-3,
    chamber_radius=0.5,
    distribution_type="discrete",
)

# 3. Compute instantaneous wall loss rate
T = 298.15  # K
P = 101325.0  # Pa
rate = wall_loss.rate(particle, temperature=T, pressure=P)

# 4. Advance the system by one time step
particle = wall_loss.step(
    particle=particle,
    temperature=T,
    pressure=P,
    time_step=10.0,  # s
)
```

### Prerequisites

- `particula` version 0.2.6 or later installed.
- A `ParticleRepresentation` instance (e.g., from one of the preset builders).
- Basic familiarity with particula's dynamics and particle-phase examples.

## Typical Workflows

### 1. Build a ParticleRepresentation

Start by constructing a particle distribution using existing builders:

```python
particle = (
    par.particles.PresetParticleRadiusBuilder()
    .set_volume(1.0, "m^3")
    .build()
)
```

You can also use particle-resolved or continuous-PDF builders from the particle phase examples when you need finer control.

### 2. Configure a SphericalWallLossStrategy

Choose physical parameters for your chamber and distribution type:

```python
wall_loss = par.dynamics.SphericalWallLossStrategy(
    wall_eddy_diffusivity=1e-3,
    chamber_radius=0.5,
    distribution_type="discrete",  # or "continuous_pdf" / "particle_resolved"
)
```

At this point you can:

- Inspect `wall_loss.rate(...)` to understand size-dependent loss.
- Call `wall_loss.step(...)` in a loop to model concentration decay.

### 3. Combine wall loss with other dynamics

Wall loss strategies are designed to compose with other dynamics, such as condensation and coagulation, in a single time-stepping loop:

```python
condensation = par.dynamics.CondensationIsothermal(
    molar_mass=180e-3,
    diffusion_coefficient=2e-5,
    accommodation_coefficient=1.0,
)

# Time loop (pseudo-code)
for _ in range(n_steps):
    # Update particle size/composition
    particle = condensation.step(
        particle=particle,
        temperature=T,
        pressure=P,
        time_step=dt,
    )

    # Apply wall loss to the updated distribution
    particle = wall_loss.step(
        particle=particle,
        temperature=T,
        pressure=P,
        time_step=dt,
    )
```

This pattern matches how other dynamics strategies are combined in chamber simulations.

## Use Cases

### Use case 1: Standalone chamber wall loss

**Scenario:** You want to understand how quickly particles are lost to the walls of a well-mixed spherical chamber.

**Solution:** Build a radius-binned `ParticleRepresentation`, configure `SphericalWallLossStrategy`, and integrate `step` over the experiment duration to track normalized concentration decay.

### Use case 2: Full chamber dynamics with wall loss

**Scenario:** You are modeling a chamber experiment where condensation growth, coagulation, and wall loss all act simultaneously.

**Solution:** Combine `SphericalWallLossStrategy` with existing condensation or coagulation strategies in a shared time loop. This lets you track how wall loss interacts with growth and collisions without custom coupling code.

## Configuration

| Option                   | Description                                              | Default        |
|--------------------------|----------------------------------------------------------|----------------|
| `wall_eddy_diffusivity` | Wall eddy diffusivity controlling wall mixing [m^2/s].  | Required       |
| `chamber_radius`        | Radius of the spherical chamber [m] (spherical builder). | Required       |
| `chamber_dimensions`    | Length, width, height of rectangular chamber [m].        | Required (rectangular) |
| `distribution_type`     | `"discrete"`, `"continuous_pdf"`, or `"particle_resolved"`. | `"discrete"` |

## Best Practices

1. **Match distribution type to your builder**: Ensure `distribution_type` matches how your `ParticleRepresentation` was constructed to avoid unintended behavior.
2. **Use builder/factory validation**: Set parameters through the builders or factory to enforce positivity, 3D chamber dimensions, and allowed distribution types with automatic unit conversion.
3. **Use physically reasonable parameters**: Choose `wall_eddy_diffusivity`, `chamber_radius`, or `chamber_dimensions` consistent with your experimental setup.
4. **Compose processes explicitly**: When combining wall loss with other dynamics, keep a clear, ordered time loop so you can reason about which processes act first in each step.

## Limitations

- Supports spherical and rectangular chambers; other geometries require new strategies/builders to be added to the factory.
- Factory selection is limited to registered strategy names.
- Does not include a high-level chamber orchestrator; you are responsible for building the main time-stepping loop that combines multiple strategies.

## Related Documentation

- **Design details**: [Agent feature: wall loss strategy system](../Agent/feature/P2-wall-loss-strategy-system.md)
- **Hands-on guide**: [Chamber wall loss example](../Examples/Chamber_Wall_Loss/wall_loss_strategy.md)
- **Notebooks**: [Spherical wall loss strategy](../Examples/Chamber_Wall_Loss/Notebooks/Spherical_Wall_Loss_Strategy.ipynb)
- **Dynamics overview**: [Wall loss strategies](../index.md#wall-loss-strategies)

## FAQ

### Should I use the function-based or strategy-based wall loss API?

Use `SphericalWallLossStrategy` whenever you are already working with `ParticleRepresentation` and other dynamics strategies. The function-based API remains available for lower-level or legacy workflows.

### How do I add a new wall loss model or geometry?

Subclass `WallLossStrategy` in your own code or contribute a new strategy to particula that implements `loss_coefficient` using the appropriate wall loss physics, then expose it through `particula.dynamics`.

## See Also

- [Condensation strategies](../Examples/Simulations/index.md)
- [Particle phase examples](../Examples/Particle_Phase/index.md)
