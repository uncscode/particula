# Condensation Strategy System

For the supported CUDA configuration of this runnable, use
[prepared GPU simulations](../gpu/prepared.md). The CPU and direct-kernel
contracts below retain their own, sometimes broader, capabilities.

For aggregate construction, population units and explicit cross-process calls,
see [native scientific process calls](index.md).

## Complete direct-process context

The direct condensation call begins the illustrative [explicit-transfer sequence](https://github.com/Gorkowski/particula/blob/main/docs/examples/gpu/direct/gpu_complete_process_sequence.py). The [direct-process validation evidence](https://github.com/Gorkowski/particula/blob/main/particula/gpu/tests/process_sequence_test.py) and [example regression](https://github.com/Gorkowski/particula/blob/main/examples_tests/gpu_complete_process_sequence_example_test.py) preserve caller-owned sidecars and do not claim a high-level coordinator, resident loop, hidden transfer, or CPU fallback.

> Strategy-based condensation and evaporation with unified dynamics APIs, staggered Gauss-Seidel updates for stability, and runnable integration.

## Overview

The condensation strategy system models mass transfer between gas and particle
phases using the same object-oriented patterns as wall loss and coagulation.
Strategies expose `rate` and `step`, work with native `ParticleData` across
"discrete", "continuous_pdf", and "particle_resolved" modes, and plug directly
into runnable pipelines. You can choose simultaneous isothermal updates or
staggered two-pass Gauss-Seidel sweeps with theta-controlled first-pass
fractions, batch partitioning, and gas-field updates for stability.

CondensationLatentHeat mirrors the isothermal workflow for particle-resolved
runs while adding latent-heat-aware mass transfer and per-step energy
diagnostics. The step now supports a `dynamic_viscosity` override and records
`last_latent_heat_energy` for consistency with the isothermal API plus energy
tracking.

This feature is built around user-facing APIs exposed via
`particula.dynamics` and `particula.dynamics.condensation`:

- `CondensationStrategy` – abstract base defining `rate` / `step`, exposed via
  `particula.dynamics.condensation`.
- `CondensationIsothermal` – simultaneous isothermal mass transfer.
- `CondensationIsothermalStaggered` – two-pass staggered (Gauss-Seidel) update
  with theta modes (`"half"`, `"random"`, `"batch"`) and batching.
- `CondensationLatentHeat` – latent-heat-corrected rate with per-step energy
  diagnostics.
- `CondensationIsothermalBuilder`, `CondensationIsothermalStaggeredBuilder`,
  `CondensationLatentHeatBuilder` – fluent builders with validation and unit
  handling, exported via both `particula.dynamics` and
  `particula.dynamics.condensation`.
- `CondensationFactory` – factory selecting a condensation strategy by name,
  exported via both `particula.dynamics` and
  `particula.dynamics.condensation`.
- `MassCondensation` runnable – delegates to a condensation strategy, splits
  `time_step` across `sub_steps`, and composes in pipelines.
- Mass-transfer helpers – `get_mass_transfer_rate`, `get_first_order_mass_
  transport_k`, `get_mass_transfer`, `get_radius_transfer_rate` for reference
  and lower-level workflows.

## Native data contract (v0.3.0)

Direct native calls to `CondensationIsothermal`,
`CondensationIsothermalStaggered`, and `CondensationLatentHeat` accept tagged
`ParticleData` and `GasData`. Supply process-owned activity, surface and vapor
pressure models plus an explicit `CondensationSpeciesMap` from
`particula.dynamics.condensation.species_mapping`. Its expected full gas-name
order and unique `(gas_index, particle_material_index)` pairs are checked at
admission. Equal species widths do not establish an identity map. Transport,
vapor-pressure and latent parameters follow full gas order; activity and
surface composition use all particle materials, including nonvolatile ones.

Native `mass_transfer_rate` is kg/s per particle and material. Native `rate`
returns each node's contribution in kg/m³/s: resolved counts use count/volume,
PMF uses stored bin density, and radius PDF uses full-grid trapezoidal weights.
PMF/PDF require volume exactly 1 m³; resolved volume is positive and finite.
PDF inventory integrates mass times dn/dr on its explicit radius grid.

Native steps stage particle and mapped gas updates, bound evaporation by
remaining particle mass, and limit uptake by available mapped gas plus actual
released inventory. Unmapped/skipped/nonpartitioning lanes remain unchanged.
`update_gases=False` keeps the external reservoir unchanged while retaining the
finite-stage availability limiter. Native latent energy is extensive J, using
counts (density integrated over volume for PMF/PDF), rather than energy density.

When a resolved particle fully evaporates (every material mass is zero), its
count and charge are cleared after gas and energy accounting. This releases a
fixed slot that subsequent coagulation and wall-loss calls can consume. A
remaining unmapped or nonvolatile material keeps the particle active. Clearing
charge records removal of the particle; no gas-phase charge transport is
modeled. Resolved count and charge storage must therefore also be writable
and nonoverlapping. PMF/PDF retain their prescribed bin populations.

The native staggered step preserves two theta sweeps, seeded or caller-owned
random generators, shuffle and batching. Each node uses its own local theta
duration. Material/radius evaluation is frozen from the original particle
state during the two sweeps; current private gas and remaining mass determine
coupling and limiting. Full-grid PDF weights are computed before selecting
batches. Physical arrays publish only after successful staging; late model
failure does not promise rollback of already consumed random draws.
Fully evaporated resolved particles leave subsequent staggered batches/sweeps.

Direct CPU strategy calls require one box and revalidate mutable state at each
entry. `MassCondensation` iterates a batch through the shared transactional
driver and reports rates with shape `(B, N, Sp)`. Latent-energy diagnostics on
the runnable have shape `(B,)` for the last successful substep; multi-box runs
set the strategy scalar diagnostic to `None`.
They preserve carrier and destination-array identities and reject protected
storage overlap before writes. CPU condensation accepts tagged `ParticleData`
and `GasData` only; activity, surface and vapor-pressure laws and the species
map belong to the process. Facade dispatch, untagged CPU execution, implicit
physics lookup and legacy normalization helpers have been removed. Empty
populations retain their native shapes and gas state while still validating
configuration. The examples below use native state and process-owned physics.

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
- **Latent heat diagnostics**: Track per-step energy release for
  particle-resolved runs with `CondensationLatentHeat`.

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

Concrete condensation implementations are exported alongside other dynamics
components, while the abstract base interface remains available from the
condensation subpackage:

Run the following setup once before the other Python snippets on this page.
Each recipe then creates fresh aligned state with the indicated molar masses;
no recipe silently reuses a different species layout. The shared illustrative PMF uses
64 nodes over 50–500 nm, mode 100 nm, GSD 1.5 and 1e8 m⁻³; resolved examples
use two unit-count slots in 1e-6 m³. The discrete quick start below retains the
default two-mode, 250-bin initializer. Vapor pressures are prescribed at
1e-3 Pa for this illustration, not calibrated species properties. No grid or
timestep convergence is claimed. Composed wall-loss runs preserve their stated
elapsed durations with 0.25-second substeps to avoid negative native proposals.

```python
import particula as par

# Abstract interface
par.dynamics.condensation.CondensationStrategy

# Concrete implementations
par.dynamics.CondensationIsothermal
par.dynamics.CondensationIsothermalStaggered
par.dynamics.CondensationLatentHeat

# Builders and factory
par.dynamics.CondensationIsothermalBuilder
par.dynamics.CondensationIsothermalStaggeredBuilder
par.dynamics.CondensationLatentHeatBuilder
par.dynamics.CondensationFactory

import numpy as np
from particula.dynamics.condensation.species_mapping import CondensationSpeciesMap


def make_example(molar_mass, *, resolved=False):
    """Create illustrative state and explicit process-owned configuration."""
    molar_mass = np.atleast_1d(molar_mass).astype(float)
    names = [f"vapor_{i}" for i in range(molar_mass.size)]
    if resolved or molar_mass.size > 1:
        particles = par.particles.create_resolved_particle_data(
            masses=np.full((2, molar_mass.size), 1e-18),
            density=np.full(molar_mass.size, 1000.0),
            concentration=[1.0, 1.0], volume=1e-6,
        )
    else:
        particles = par.particles.create_lognormal_binned_particle_data(
            distribution_type="discrete",
            radius_bins=np.geomspace(50e-9, 500e-9, 64),
            mode=[100e-9], geometric_standard_deviation=[1.5],
            number_concentration=[1e8],
        )
    gas = par.gas.GasData(
        name=names, molar_mass=molar_mass,
        concentration=np.full((1, molar_mass.size), 1e-6),
        partitioning=np.ones(molar_mass.size, dtype=bool),
    )
    environment = par.gas.EnvironmentData(
        temperature=[298.15], pressure=[101325.0],
        saturation_ratio=np.ones((1, molar_mass.size)),
    )
    aerosol = par.Aerosol(particles=particles, gas=gas, environment=environment)
    physics = dict(
        activity_strategy=par.particles.ActivityIdealMass(),
        surface_strategy=par.particles.SurfaceStrategyVolume(
            density=np.full(molar_mass.size, 1000.0)
        ),
        vapor_pressure_strategy=par.gas.ConstantVaporPressureStrategy(1e-3),
        species_map=CondensationSpeciesMap(
            tuple(names), tuple((i, i) for i in range(molar_mass.size))
        ),
    )
    return aerosol, physics

```

All strategies share a common shape: initialize with physical parameters (molar
mass, diffusion coefficient, accommodation coefficient) and coupling controls
(`update_gases`, `skip_partitioning_indices`); call `rate(...)` for
instantaneous transfer; call `step(...)` to advance.

### Runnable entry point: `MassCondensation`

`MassCondensation` is a `RunnableABC` exported as `par.dynamics.MassCondensation`.
It requires a native `Aerosol` holding `ParticleData`, `GasData` and
`EnvironmentData`, delegates to the configured strategy, splits `time_step`
across `sub_steps`, and composes with other runnables. Every substep reads the
currently held particle, gas and environment state. The strategy must return
the same particle and gas containers; replacement results reject. Physics and
species mappings belong to the strategy. Non-native aggregates reject before
strategy dispatch.

```python
aerosol, physics = make_example(0.018)
particle, gas = aerosol.particles, aerosol.gas
import particula as par

condensation = par.dynamics.MassCondensation(
    condensation_strategy=par.dynamics.CondensationIsothermal(
        **physics,
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
updated = workflow.execute(aerosol, time_step=60.0, sub_steps=240)
```

### CondensationIsothermal (simultaneous update)

`CondensationIsothermal` evaluates the mass-transfer equation in one shot for all
particles/bins:

- dm/dt = 4π × r × D × M × f(Kn, α) × Δp / (R × T)

Radii are filled when zero, clipped to the minimum physical size (`1e-10 m`),
pressure deltas (Δp) are sanitized (NaN/inf → 0) before computing rates, and gas
mass is optionally depleted (`update_gases=True`).

```python
aerosol, physics = make_example(0.18)
particle, gas = aerosol.particles, aerosol.gas
iso = par.dynamics.CondensationIsothermal(
    **physics,
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

### CondensationLatentHeat (energy diagnostics)

`CondensationLatentHeat` mirrors the isothermal step but applies a latent-heat
correction when a latent heat strategy is provided. Direct manual
`CondensationLatentHeat(...)` construction is supported, and the constructor
also accepts a positive scalar `latent_heat` compatibility fallback. For most
public user workflows, prefer the builder/factory path below when you want
validated parameter loading or a strategy object. When both
`latent_heat_strategy` and `latent_heat` are supplied, the explicit strategy
takes precedence. It records `last_latent_heat_energy` each step (positive for
condensation, negative for evaporation) and accepts a `dynamic_viscosity`
override for particle-resolved workflows.

```python
aerosol, physics = make_example(0.018)
particle, gas = aerosol.particles, aerosol.gas
latent = par.dynamics.CondensationLatentHeat(
    **physics,
    molar_mass=0.018,
    diffusion_coefficient=2e-5,
    accommodation_coefficient=1.0,
    latent_heat=2.4e6,  # J/kg fallback
)
particle, gas = latent.step(
    particle=particle,
    gas_species=gas,
    temperature=298.15,
    pressure=101325.0,
    time_step=1.0,
    dynamic_viscosity=1.8e-5,
)
energy_released = latent.last_latent_heat_energy  # total energy [J]
```

When neither a `latent_heat_strategy` nor a positive scalar `latent_heat`
fallback is configured, the step follows the isothermal path and reports
`last_latent_heat_energy = 0.0`.

For native state, `last_latent_heat_energy` is signed extensive energy in J:
sum the applied per-particle material transfer times represented counts and
latent heat. For PMF/PDF, integrate the population density over the canonical
reference volume. Condensation is positive and evaporation negative. The
example regression checks this against the actual particle inventory change;
it is a per-call diagnostic, not a temperature-evolution model.

For the current shipped support boundary, CPU latent-heat conservation is the executable CPU
integration baseline for future Epic D GPU parity work:
`particula/integration_tests/condensation_latent_heat_conservation_test.py`.
It verifies only a finite nonzero condensation transfer, particle water
gain, gas water loss, total water conservation, and final-step
`last_latent_heat_energy` agreement with the constant-latent-heat
bookkeeping path. This baseline is CPU-only and diagnostic/reference only.

The bounded, low-level direct GPU condensation path separately supports an
optional per-species latent-rate correction in each of its four equal
substeps, with CPU-oracle/Warp parity coverage. Omitting `latent_heat`, or
using a zero entry for a species, retains that species' isothermal rate path.
This is diagnostic bookkeeping, not temperature feedback or strategy/runnable
thermal coupling. The direct path mutates gas concentration and its inventory finalization
finalization conserves particle-plus-gas inventory per box and species. It does
not provide strategy/runnable-level latent-heat support; the separate
`energy_transfer` diagnostic is described below.

### CondensationIsothermalStaggered (two-pass Gauss-Seidel)

`CondensationIsothermalStaggered` splits each timestep into two passes. Theta
controls the first-pass fraction; batches are clipped to the particle count and
optionally shuffled each step. Gas concentration is updated after every batch to
reduce lag. Radii are clamped and Δp is sanitized before computing dm/dt.

```python
aerosol, physics = make_example(0.018)
particle, gas = aerosol.particles, aerosol.gas
staggered = par.dynamics.CondensationIsothermalStaggered(
    **physics,
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
aerosol, physics = make_example(0.018)
particle, gas = aerosol.particles, aerosol.gas
half = par.dynamics.CondensationIsothermalStaggered(
    **physics,
    molar_mass=0.018,
    theta_mode="half",
    num_batches=1,
)

batch = par.dynamics.CondensationIsothermalStaggered(
    **physics,
    molar_mass=0.018,
    theta_mode="batch",
    num_batches=16,
    shuffle_each_step=False,  # deterministic ordering
)

rand = par.dynamics.CondensationIsothermalStaggered(
    **physics,
    molar_mass=0.018,
    theta_mode="random",
    num_batches=4,
    random_state=2024,
)
```

### Builder and factory workflow

Builders provide validation, units, and consistent naming; the factory selects
a strategy by string while reusing builder validation. The public builder and
factory entry points are available from both import surfaces:

- `particula.dynamics.CondensationLatentHeatBuilder`
- `particula.dynamics.condensation.CondensationLatentHeatBuilder`
- `particula.dynamics.CondensationFactory`
- `particula.dynamics.condensation.CondensationFactory`

The shipped latent-heat factory key is `"latent_heat"`.

```python
aerosol, physics = make_example(0.018)
particle, gas = aerosol.particles, aerosol.gas
import particula as par
from particula.gas.latent_heat_strategies import ConstantLatentHeat

iso = (
    par.dynamics.CondensationIsothermalBuilder()
    .set_molar_mass(0.018, "kg/mol")
    .set_diffusion_coefficient(2.1e-5, "m^2/s")
    .set_accommodation_coefficient(1.0)
    .set_update_gases(True)
    .set_activity_strategy(physics["activity_strategy"])
    .set_surface_strategy(physics["surface_strategy"])
    .set_vapor_pressure_strategy(physics["vapor_pressure_strategy"])
    .set_species_map(physics["species_map"])
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
    .set_activity_strategy(physics["activity_strategy"])
    .set_surface_strategy(physics["surface_strategy"])
    .set_vapor_pressure_strategy(physics["vapor_pressure_strategy"])
    .set_species_map(physics["species_map"])
    .build()
)

latent_strategy = ConstantLatentHeat(latent_heat_ref=2.4e6)

latent = (
    par.dynamics.CondensationLatentHeatBuilder()
    .set_molar_mass(0.018, "kg/mol")
    .set_diffusion_coefficient(2e-5, "m^2/s")
    .set_accommodation_coefficient(1.0)
    .set_latent_heat_strategy(latent_strategy)
    .set_update_gases(True)
    .set_activity_strategy(physics["activity_strategy"])
    .set_surface_strategy(physics["surface_strategy"])
    .set_vapor_pressure_strategy(physics["vapor_pressure_strategy"])
    .set_species_map(physics["species_map"])
    .build()
)

factory = par.dynamics.CondensationFactory()
latent_factory_with_strategy = factory.get_strategy(
    strategy_type="latent_heat",
    parameters={
        **physics,
        "molar_mass": 0.018,
        "molar_mass_units": "kg/mol",
        "diffusion_coefficient": 2e-5,
        "diffusion_coefficient_units": "m^2/s",
        "accommodation_coefficient": 1.0,
        "latent_heat_strategy": latent_strategy,
    },
)

iso_factory = factory.get_strategy(
    strategy_type="isothermal",
    parameters={
        **physics,
        "molar_mass": 0.018,
        "molar_mass_units": "kg/mol",
        "diffusion_coefficient": 2e-5,
        "diffusion_coefficient_units": "m^2/s",
        "accommodation_coefficient": 1.0,
    },
)
latent_factory = factory.get_strategy(
    strategy_type="latent_heat",
    parameters={
        **physics,
        "molar_mass": 0.018,
        "molar_mass_units": "kg/mol",
        "diffusion_coefficient": 2e-5,
        "diffusion_coefficient_units": "m^2/s",
        "accommodation_coefficient": 1.0,
        "latent_heat": 2.4e6,
    },
)
```

Use direct `CondensationLatentHeat(...)` construction when you want to wire the
strategy manually. For the supported public workflow, prefer
`CondensationLatentHeatBuilder` or
`CondensationFactory().get_strategy("latent_heat", parameters=...)`. The
builder/factory path supports both documented latent-heat inputs:

- `latent_heat_strategy`: pass an explicit latent heat strategy object.
- `latent_heat`: pass a positive scalar fallback that the builder forwards to
  `CondensationLatentHeat`.

If both are supplied, the explicit `latent_heat_strategy` remains active and
the scalar value is preserved only as constructor input metadata.

### Skip-partitioning indices and gas updates

Use `skip_partitioning_indices` to zero-out selected species during rate/step
without touching others; combine with `update_gases` to control depletion.

```python
aerosol, physics = make_example([0.1, 0.2])
particle, gas = aerosol.particles, aerosol.gas
iso_skip = par.dynamics.CondensationIsothermal(
    **physics,
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
aerosol, physics = make_example([0.018, 0.046])
particle, gas = aerosol.particles, aerosol.gas
multi = par.dynamics.CondensationIsothermalStaggered(
    **physics,
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
aerosol, physics = make_example(0.018)
particle, gas = aerosol.particles, aerosol.gas
iso = par.dynamics.CondensationIsothermal(
    molar_mass=gas.molar_mass.copy(), **physics,
)
wall_loss = par.dynamics.WallLoss(
    par.dynamics.SphericalWallLossStrategy(1e-3, 0.5, "discrete")
)
cond = par.dynamics.MassCondensation(condensation_strategy=iso)
coag = par.dynamics.Coagulation(
    coagulation_strategy=par.dynamics.BrownianCoagulationStrategy(
        distribution_type="discrete",
    ),
)
workflow = cond | coag | wall_loss
updated = workflow.execute(aerosol, time_step=120.0, sub_steps=480)
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
aerosol, physics = make_example(0.018)
particle, gas = aerosol.particles, aerosol.gas
particle = par.particles.create_lognormal_binned_particle_data(
    distribution_type="discrete",
)
aerosol.particles = particle

condensation = par.dynamics.CondensationIsothermal(
    **physics,
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
aerosol, physics = make_example(0.018)
particle, gas = aerosol.particles, aerosol.gas
staggered = par.dynamics.CondensationIsothermalStaggered(
    **physics,
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

- Native `ParticleData` with an explicit population kind.
- `GasData`, aligned `EnvironmentData`, and process-owned vapor properties and mapping.
- Familiarity with particula dynamics and examples.

## Typical Workflows

### 1. Configure via builder

```python
aerosol, physics = make_example(0.05)
particle, gas = aerosol.particles, aerosol.gas
condensation = (
    par.dynamics.CondensationIsothermalBuilder()
    .set_molar_mass(0.05, "kg/mol")
    .set_diffusion_coefficient(1.5e-5, "m^2/s")
    .set_accommodation_coefficient(0.85)
    .set_update_gases(True)
    .set_activity_strategy(physics["activity_strategy"])
    .set_surface_strategy(physics["surface_strategy"])
    .set_vapor_pressure_strategy(physics["vapor_pressure_strategy"])
    .set_species_map(physics["species_map"])
    .build()
)
```

### 2. Run inside a runnable pipeline

```python
aerosol, physics = make_example(0.05)
particle, gas = aerosol.particles, aerosol.gas
condensation = par.dynamics.CondensationIsothermal(
    molar_mass=gas.molar_mass.copy(), **physics,
)
cond = par.dynamics.MassCondensation(condensation_strategy=condensation)
wall = par.dynamics.WallLoss(
    wall_loss_strategy=par.dynamics.SphericalWallLossStrategy(
        wall_eddy_diffusivity=1e-3,
        chamber_radius=0.5,
        distribution_type="discrete",
    ),
)
workflow = cond | wall
aerosol = workflow.execute(aerosol, time_step=30.0, sub_steps=120)
```

### 3. Particle-resolved batching with deterministic ordering

```python
aerosol, physics = make_example(0.018, resolved=True)
particle, gas = aerosol.particles, aerosol.gas
staggered = (
    par.dynamics.CondensationIsothermalStaggeredBuilder()
    .set_molar_mass(0.018, "kg/mol")
    .set_diffusion_coefficient(2e-5, "m^2/s")
    .set_accommodation_coefficient(1.0)
    .set_theta_mode("batch")
    .set_num_batches(32)
    .set_shuffle_each_step(False)
    .set_activity_strategy(physics["activity_strategy"])
    .set_surface_strategy(physics["surface_strategy"])
    .set_vapor_pressure_strategy(physics["vapor_pressure_strategy"])
    .set_species_map(physics["species_map"])
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
aerosol, physics = make_example(0.15)
particle, gas = aerosol.particles, aerosol.gas
wall_loss = par.dynamics.WallLoss(
    par.dynamics.SphericalWallLossStrategy(1e-3, 0.5, "discrete")
)
cond = par.dynamics.MassCondensation(
    condensation_strategy=par.dynamics.CondensationIsothermal(
        **physics,
        molar_mass=0.15,
        diffusion_coefficient=1.8e-5,
        accommodation_coefficient=0.95,
        update_gases=True,
    )
)
workflow = cond | wall_loss
updated = workflow.execute(aerosol, time_step=120.0, sub_steps=480)
```

### Use case 2: Particle-resolved stability with batch staggering

**Scenario:** Particle-resolved simulation oscillates when large particles
dominate.

**Solution:** Switch to `CondensationIsothermalStaggered` with `theta_mode`
`"batch"` and multiple batches to spread updates and reduce lag.

```python
aerosol, physics = make_example(0.018, resolved=True)
particle, gas = aerosol.particles, aerosol.gas
stable = par.dynamics.CondensationIsothermalStaggered(
    **physics,
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
aerosol, physics = make_example([0.018, 0.05])
particle, gas = aerosol.particles, aerosol.gas
selective = par.dynamics.CondensationIsothermal(
    **physics,
    molar_mass=[0.018, 0.05],
    diffusion_coefficient=[2e-5, 1.6e-5],
    accommodation_coefficient=[1.0, 0.5],
    skip_partitioning_indices=[1],
    update_gases=True,
)
particle, gas = selective.step(
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
| `latent_heat_strategy` | Optional latent heat strategy. | `None` |
| `latent_heat` | Scalar fallback latent heat [J/kg]. | `0.0` |
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
- Factory supports `"isothermal"`, `"isothermal_staggered"`, and
  `"latent_heat"` only.
- Direct low-level GPU latent heat is a per-species rate correction only; it
  has no temperature feedback or strategy/`Runnable` integration. Its signed
  `energy_transfer` sidecar is finalized-transfer diagnostic bookkeeping only.
- The minimum-radius clamp (1e-10 m) is a numerical safeguard, not evidence
  of physical validity at that scale; sub-continuum physics is out of scope.

## Related Documentation

- **Dynamics overview**: [Wall loss strategy system](wall-loss.md)
- **Examples**: [docs/examples/simulations/index.md](../../examples/simulations/index.md)
- **Latent-heat example (latent-heat example)**: [CPU latent-heat condensation bookkeeping](../../examples/processes/condensation/condensation_latent_heat.ipynb)
- **CPU integration baseline**: `particula/integration_tests/condensation_latent_heat_conservation_test.py`
- **Theory**: [docs/theory/index.md](../../theory/index.md)

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

- [Wall loss strategy system](wall-loss.md)
- [Particle phase examples](../../examples/fundamentals/particles/index.md)
- [Simulations overview](../../examples/simulations/index.md)
