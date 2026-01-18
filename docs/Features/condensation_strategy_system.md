# Condensation Strategy System

> Strategy-based condensation and evaporation with unified dynamics APIs, staggered Gauss-Seidel updates for stability, and runnable integration.

## Overview

The condensation strategy system models mass transfer between gas and particle
phases using the same object-oriented patterns as wall loss and coagulation.
Strategies expose `rate` and `step`, work with `ParticleRepresentation` across
"discrete", "continuous_pdf", and "particle_resolved" modes, and plug directly
into runnable pipelines. You can choose simultaneous isothermal updates or
staggered two-pass Gauss-Seidel sweeps with theta-controlled first-pass
fractions, batch partitioning, and gas-field updates for stability.

This feature is built around user-facing APIs exposed via `particula.dynamics`:

- `CondensationStrategy` – abstract base defining `rate` / `step`.
- `CondensationIsothermal` – simultaneous isothermal mass transfer.
- `CondensationIsothermalStaggered` – two-pass staggered (Gauss-Seidel) update
  with theta modes (`"half"`, `"random"`, `"batch"`) and batching.
- `CondensationIsothermalBuilder`, `CondensationIsothermalStaggeredBuilder` –
  fluent builders with validation and unit handling.
- `CondensationFactory` – factory selecting a condensation strategy by name.
- `MassCondensation` runnable – delegates to a condensation strategy, splits
  `time_step` across `sub_steps`, and composes in pipelines.
- Mass-transfer helpers – `get_mass_transfer_rate`, `get_first_order_mass_
  transport_k`, `get_mass_transfer`, `get_radius_transfer_rate` for reference
  and lower-level workflows.

## Key Benefits

- **Consistent dynamics workflow**: Same strategy-based API (`rate`, `step`,
  builders, factory) as wall loss and coagulation.
- **Stability for particle-resolved runs**: Staggered two-pass Gauss-Seidel with
  theta modes and batch clipping reduces lag and preserves mass.
- **Gas-particle coupling control**: Toggle `update_gases` and
  `skip_partitioning_indices` to steer which species condense and whether gas
  fields are depleted.
- **Pipeline-ready**: Use `MassCondensation` with `sub_steps` for tight coupling
  to other runnables in a single pipeline.

## Who It's For

This feature is designed for:

- **Chamber and box-model users**: Time-dependent condensation combined with wall
  loss and coagulation.
- **Particle-resolved modelers**: Needing Gauss-Seidel batches, theta control,
  and deterministic or shuffled ordering for stability.
- **Multi-species workflows**: Tracking several vapors with selective
  partitioning and optional gas-field depletion.

## Capabilities

### Unified condensation API in `particula.dynamics`

Condensation is exported alongside other dynamics components:

```python
import particula as par

# Abstract interface
par.dynamics.CondensationStrategy

# Concrete implementations
par.dynamics.CondensationIsothermal
par.dynamics.CondensationIsothermalStaggered

# Builders and factory
par.dynamics.CondensationIsothermalBuilder
par.dynamics.CondensationIsothermalStaggeredBuilder
par.dynamics.CondensationFactory
```

All strategies share a common shape: initialize with physical parameters (molar
mass, diffusion coefficient, accommodation coefficient) and coupling controls
(`update_gases`, `skip_partitioning_indices`); call `rate(...)` for
instantaneous transfer; call `step(...)` to advance.

### Runnable entry point: `MassCondensation`

`MassCondensation` is a `RunnableABC` exported as `par.dynamics.MassCondensation`.
It operates on an `Aerosol`, delegates to the configured strategy, splits
`time_step` across `sub_steps`, and composes with other runnables.

```python
import particula as par

condensation = par.dynamics.MassCondensation(
    condensation_strategy=par.dynamics.CondensationIsothermal(
        molar_mass=0.018,
        diffusion_coefficient=2e-5,
        accommodation_coefficient=1.0,
    ),
)
wall_loss = par.dynamics.WallLoss(
    wall_loss_strategy=par.dynamics.SphericalWallLossStrategy(
        wall_eddy_diffusivity=1e-3,
        chamber_radius=0.5,
        distribution_type="discrete",
    ),
)
workflow = condensation | wall_loss
updated = workflow.execute(aerosol, time_step=60.0, sub_steps=4)
```

### CondensationIsothermal (simultaneous update)

`CondensationIsothermal` evaluates the mass-transfer equation in one shot for all
particles/bins:

- dm/dt = 4π × r × D × M × f(Kn, α) × Δp / (R × T)

Radii are filled when zero, clipped to the minimum physical size (`1e-10 m`),
pressure deltas (Δp) are sanitized (NaN/inf → 0) before computing rates, and gas
mass is optionally depleted (`update_gases=True`).

```python
iso = par.dynamics.CondensationIsothermal(
    molar_mass=0.18,
    diffusion_coefficient=2e-5,
    accommodation_coefficient=0.9,
    update_gases=True,
)
particle, gas = iso.step(
    particle=particle,
    gas_species=gas,
    temperature=298.15,
    pressure=101325.0,
    time_step=1.0,
)
```

### CondensationIsothermalStaggered (two-pass Gauss-Seidel)

`CondensationIsothermalStaggered` splits each timestep into two passes. Theta
controls the first-pass fraction; batches are clipped to the particle count and
optionally shuffled each step. Gas concentration is updated after every batch to
reduce lag. Radii are clamped and Δp is sanitized before computing dm/dt.

```python
staggered = par.dynamics.CondensationIsothermalStaggered(
    molar_mass=0.018,
    diffusion_coefficient=2e-5,
    accommodation_coefficient=1.0,
    theta_mode="random",
    num_batches=8,
    shuffle_each_step=True,
    random_state=1234,
)
particle, gas = staggered.step(
    particle=particle,
    gas_species=gas,
    temperature=298.0,
    pressure=101325.0,
    time_step=0.5,
)
```

#### Theta modes and batching

- `"half"`: deterministic θ = 0.5 (symmetric two-pass).
- `"random"`: θ ~ U[0,1] using the configured RNG (`random_state`).
- `"batch"`: θ = 1.0; staggering comes from batch ordering instead of θ.

```python
half = par.dynamics.CondensationIsothermalStaggered(
    molar_mass=0.018,
    theta_mode="half",
    num_batches=1,
)

batch = par.dynamics.CondensationIsothermalStaggered(
    molar_mass=0.018,
    theta_mode="batch",
    num_batches=16,
    shuffle_each_step=False,  # deterministic ordering
)

rand = par.dynamics.CondensationIsothermalStaggered(
    molar_mass=0.018,
    theta_mode="random",
    num_batches=4,
    random_state=2024,
)
```

### Builder and factory workflow

Builders provide validation, units, and consistent naming; the factory selects a
strategy by string while reusing builder validation.

```python
import particula as par

iso = (
    par.dynamics.CondensationIsothermalBuilder()
    .set_molar_mass(0.018, "kg/mol")
    .set_diffusion_coefficient(2.1e-5, "m^2/s")
    .set_accommodation_coefficient(1.0)
    .set_update_gases(True)
    .build()
)

staggered = (
    par.dynamics.CondensationIsothermalStaggeredBuilder()
    .set_molar_mass(0.018, "kg/mol")
    .set_diffusion_coefficient(2e-5, "m^2/s")
    .set_accommodation_coefficient(0.95)
    .set_theta_mode("batch")
    .set_num_batches(12)
    .set_shuffle_each_step(True)
    .set_random_state(7)
    .set_update_gases(False)
    .build()
)

factory = par.dynamics.CondensationFactory()
iso_factory = factory.get_strategy(
    strategy_type="isothermal",
    parameters={
        "molar_mass": 0.018,
        "diffusion_coefficient": 2e-5,
        "accommodation_coefficient": 1.0,
        "update_gases": True,
    },
)
```

### Skip-partitioning indices and gas updates

Use `skip_partitioning_indices` to zero-out selected species during rate/step
without touching others; combine with `update_gases` to control depletion.

```python
iso_skip = par.dynamics.CondensationIsothermal(
    molar_mass=[0.1, 0.2],
    diffusion_coefficient=[2e-5, 1.5e-5],
    accommodation_coefficient=[1.0, 0.8],
    skip_partitioning_indices=[1],  # second species stays in gas
    update_gases=True,
)
rates = iso_skip.rate(particle, gas, temperature=298.0, pressure=101325.0)
```

### Multi-species and particle-resolved batching

```python
multi = par.dynamics.CondensationIsothermalStaggered(
    molar_mass=[0.018, 0.046],
    diffusion_coefficient=[2e-5, 1.2e-5],
    accommodation_coefficient=[1.0, 0.7],
    theta_mode="batch",
    num_batches=24,
    shuffle_each_step=True,
)
particle, gas = multi.step(
    particle=particle,
    gas_species=gas,
    temperature=296.0,
    pressure=101325.0,
    time_step=0.25,
)
```

### Sub-steps in runnable pipelines

`MassCondensation.execute` splits `time_step` by `sub_steps`, calling the
strategy `step` each sub-iteration. Use this to interleave condensation tightly
with other processes.

```python
cond = par.dynamics.MassCondensation(condensation_strategy=iso)
coag = par.dynamics.Coagulation(
    coagulation_strategy=par.dynamics.BrownianCoagulationStrategy(
        distribution_type="discrete",
    ),
)
workflow = cond | coag | wall_loss
updated = workflow.execute(aerosol, time_step=120.0, sub_steps=6)
```

### Mass conservation, stability, and sanitization

- **Two-pass Gauss-Seidel** (staggered): theta split plus batch sweeps, with gas
  updated after each batch.
- **Batch clipping**: `num_batches` is clipped to the particle count to avoid
  empty batches.
- **Minimum radius clamp**: Radii are clipped to `1e-10 m`; zeros filled to avoid
  divide-by-zero.
- **Δp sanitization**: Non-finite partial-pressure deltas are zeroed before
  computing dm/dt.
- **Inventory limits**: `get_mass_transfer` clips condensation/evaporation by gas
  and particle inventory and per-bin limits.
- **Gas updates optional**: `update_gases=False` leaves gas concentrations
  unchanged.

## Getting Started

### Quick start: isothermal condensation on a discrete distribution

```python
import particula as par

particle = par.particles.PresetParticleRadiusBuilder().build()
gas = par.gas.GasSpecies(molar_mass=0.018, concentration=1e-6)
condensation = par.dynamics.CondensationIsothermal(
    molar_mass=0.018,
    diffusion_coefficient=2e-5,
    accommodation_coefficient=1.0,
)
rate = condensation.rate(
    particle, gas, temperature=298.15, pressure=101325.0
)
particle, gas = condensation.step(
    particle=particle,
    gas_species=gas,
    temperature=298.15,
    pressure=101325.0,
    time_step=10.0,
)
```

### Staggered quick start with theta and batches

```python
staggered = par.dynamics.CondensationIsothermalStaggered(
    molar_mass=0.018,
    diffusion_coefficient=2e-5,
    accommodation_coefficient=1.0,
    theta_mode="half",
    num_batches=4,
)
particle, gas = staggered.step(
    particle=particle,
    gas_species=gas,
    temperature=298.15,
    pressure=101325.0,
    time_step=5.0,
)
```

## Prerequisites

- `particula` version 0.2.6 or later.
- A `ParticleRepresentation` (discrete, continuous PDF, or particle-resolved).
- Gas species configured via `par.gas.GasSpecies` with vapor properties.
- Familiarity with particula dynamics and examples.

## Typical Workflows

### 1. Configure via builder

```python
condensation = (
    par.dynamics.CondensationIsothermalBuilder()
    .set_molar_mass(0.05, "kg/mol")
    .set_diffusion_coefficient(1.5e-5, "m^2/s")
    .set_accommodation_coefficient(0.85)
    .set_update_gases(True)
    .build()
)
```

### 2. Run inside a runnable pipeline

```python
cond = par.dynamics.MassCondensation(condensation_strategy=condensation)
wall = par.dynamics.WallLoss(
    wall_loss_strategy=par.dynamics.SphericalWallLossStrategy(
        wall_eddy_diffusivity=1e-3,
        chamber_radius=0.5,
        distribution_type="discrete",
    ),
)
workflow = cond | wall
aerosol = workflow.execute(aerosol, time_step=30.0, sub_steps=3)
```

### 3. Particle-resolved batching with deterministic ordering

```python
staggered = (
    par.dynamics.CondensationIsothermalStaggeredBuilder()
    .set_molar_mass(0.018)
    .set_diffusion_coefficient(2e-5)
    .set_accommodation_coefficient(1.0)
    .set_theta_mode("batch")
    .set_num_batches(32)
    .set_shuffle_each_step(False)
    .build()
)
aerosol = par.dynamics.MassCondensation(staggered).execute(
    aerosol,
    time_step=60.0,
    sub_steps=2,
)
```

## Use Cases

### Use case 1: Chamber growth with gas depletion

**Scenario:** Track condensational growth of secondary organic aerosol while
reducing gas phase mass.

**Solution:** Use `CondensationIsothermal` with `update_gases=True` and couple to
wall loss in one pipeline.

```python
cond = par.dynamics.MassCondensation(
    condensation_strategy=par.dynamics.CondensationIsothermal(
        molar_mass=0.15,
        diffusion_coefficient=1.8e-5,
        accommodation_coefficient=0.95,
        update_gases=True,
    )
)
workflow = cond | wall_loss
updated = workflow.execute(aerosol, time_step=120.0, sub_steps=4)
```

### Use case 2: Particle-resolved stability with batch staggering

**Scenario:** Particle-resolved simulation oscillates when large particles
dominate.

**Solution:** Switch to `CondensationIsothermalStaggered` with `theta_mode`
`"batch"` and multiple batches to spread updates and reduce lag.

```python
stable = par.dynamics.CondensationIsothermalStaggered(
    molar_mass=0.018,
    theta_mode="batch",
    num_batches=24,
    shuffle_each_step=True,
)
aerosol = par.dynamics.MassCondensation(stable).execute(
    aerosol,
    time_step=10.0,
    sub_steps=2,
)
```

### Use case 3: Mixed-species with selective partitioning

**Scenario:** Only one of two vapors should condense; the other stays in gas.

**Solution:** Provide vector properties and set `skip_partitioning_indices` for
the species that should remain in gas; keep `update_gases` enabled for the
condensing species.

```python
selective = par.dynamics.CondensationIsothermal(
    molar_mass=[0.018, 0.05],
    diffusion_coefficient=[2e-5, 1.6e-5],
    accommodation_coefficient=[1.0, 0.5],
    skip_partitioning_indices=[1],
    update_gases=True,
)
aerosol, gas = selective.step(
    particle=particle,
    gas_species=gas,
    temperature=300.0,
    pressure=101325.0,
    time_step=5.0,
)
```

## Configuration

| Option | Description | Default |
|--------|-------------|---------|
| `molar_mass` | Molar mass of condensing species [kg/mol]. | Required |
| `diffusion_coefficient` | Vapor diffusion coefficient [m^2/s]. | `2e-5` |
| `accommodation_coefficient` | Mass accommodation coefficient (unitless). | `1.0` |
| `update_gases` | Whether to deplete gas concentrations during step. | `True` |
| `skip_partitioning_indices` | Species indices to exclude from partitioning. | `None` |
| `theta_mode` | Staggered theta selection: `"half"`, `"random"`, `"batch"`. | `"half"` (staggered) |
| `num_batches` | Gauss-Seidel batch count (clipped to particle count). | `1` |
| `shuffle_each_step` | Shuffle particle order each step (staggered). | `True` |
| `random_state` | Seed / RNG for theta draws and shuffling. | `None` |
| `sub_steps` | Runnable-only: split `time_step` into `sub_steps`. | `1` |

## Best Practices

1. **Match strategy to stability needs**: Use simultaneous isothermal for bulk
   runs; switch to staggered with batches when particle-resolved lag or
   oscillations appear.
2. **Choose sensible batches**: Start with `num_batches` near √N for
   particle-resolved cases; never exceed particle count.
3. **Control gas coupling explicitly**: Set `update_gases=False` when gas is
   externally prescribed; leave it on for conservation.
4. **Use skip indices sparingly**: Limit `skip_partitioning_indices` to species
   that should remain entirely in gas.
5. **Sub-step when composing processes**: Increase `sub_steps` to reduce
   operator-splitting error when coupling with wall loss or coagulation.

## Limitations

- Staggered solver is Gauss-Seidel only; other solvers are not exposed.
- Factory supports `"isothermal"` and `"isothermal_staggered"` only.
- No latent-heat or temperature feedback; condensation is isothermal.
- Minimum-radius clamp (1e-10 m) enforces continuum validity; sub-continuum
  physics is out of scope.

## Related Documentation

- **Mass transfer helpers**: [adw-docs/architecture_reference.md](../../adw-docs/architecture_reference.md)
- **Dynamics overview**: [Wall loss strategy system](./wall_loss_strategy_system.md)
- **Examples**: [docs/Examples/Simulations/index.md](../Examples/Simulations/index.md)
- **Theory**: [docs/Theory/index.md](../Theory/index.md)
- **API reference**: [adw-docs/code_style.md](../../adw-docs/code_style.md)

## FAQ

### When should I choose staggered over isothermal?

Choose `CondensationIsothermalStaggered` when particle-resolved runs show
numerical oscillations or lag, or when you need reproducible Gauss-Seidel
ordering with batches and theta control.

### How do theta modes differ?

- `"half"` uses θ = 0.5 for symmetric two-pass updates.
- `"random"` draws θ per particle with the configured RNG to reduce ordering
  bias.
- `"batch"` uses θ = 1.0 and relies on batch ordering for staggering.

### How is mass conserved?

Mass changes are limited by `get_mass_transfer` to available gas and particle
inventory, batches update working gas each pass, radii are clamped, and Δp is
sanitized before computing dm/dt.

## See Also

- [Wall loss strategy system](./wall_loss_strategy_system.md)
- [Particle phase examples](../Examples/Particle_Phase/index.md)
- [Simulations overview](../Examples/Simulations/index.md)
