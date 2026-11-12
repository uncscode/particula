# Condensation Strategy System

For aggregate construction, population units and explicit cross-process calls,
see [native scientific process calls](native_scientific_processes.md).

## Complete direct-process context

The direct condensation call begins the illustrative [explicit-transfer sequence](https://github.com/Gorkowski/particula/blob/main/docs/Examples/gpu_complete_process_sequence.py). The [private P2 evidence](https://github.com/Gorkowski/particula/blob/main/particula/gpu/tests/process_sequence_test.py) and [P3 regression](https://github.com/Gorkowski/particula/blob/main/examples_tests/gpu_complete_process_sequence_example_test.py) preserve caller-owned sidecars and do not claim a high-level coordinator, resident loop, hidden transfer, or CPU fallback.

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

## Native data contract (E9-M3)

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

CPU native calls require one box and revalidate mutable state at each entry.
They preserve carrier and destination-array identities and reject protected
storage overlap before writes. The examples below use native state and process-owned physics. Legacy facade
execution remains available for compatibility until the separate M6 removal.

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
no recipe silently reuses a different species layout. These examples require
the selected E9 checkout (`pip install -e .`). The shared illustrative PMF uses
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
It operates on an `Aerosol`, delegates to the configured strategy, splits
`time_step` across `sub_steps`, and composes with other runnables.

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

For the current shipped support boundary, E3-F7 is the executable CPU
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
thermal coupling. The direct path mutates gas concentration and its P2
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

- The selected E9 checkout installed with `pip install -e .`.
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

### Low-level Warp GPU condensation

#### Tagged native populations and species binding

Tagged discrete/resolved GPU calls require `species_binding=`. Create it
explicitly with `prepare_condensation_species_map` from
`particula.gpu.species_mapping`, using the exact Warp carriers, actual complete
gas-name order retained at transfer, and the process's
`CondensationSpeciesMap`. The binding owns fresh device lane maps and pins
array identities, layouts, devices and population metadata. Keep its map
contents unchanged; new storage, names, mapping or metadata requires fresh
preparation. Full-gas partitioning remains live device state.

| Native quantity | Species order / units |
|---|---|
| Particle masses and work/total transfer | Full material order; kg per particle, `(boxes, slots, materials)` |
| Activity kappas, reference molar masses and surface tension | Full material order; mapped reference masses must match the declared gas species |
| Thermodynamics, accommodation, diffusion and latent/thermal arrays | Full gas order, including unused/nonpartitioning gas species |
| Demand/release/scale scratch and energy output | Full gas width, `(boxes, gas_species)` |
| Gas inventory coupling | kg/m³: transfer weighted by N/V for resolved counts, stored density for PMF |
| Native energy output | Extensive J: applied transfer weighted by counts and mapped latent heat |

The direct and private prepared routes use the same four equal substeps. Each
refreshes full-gas vapor pressure, evaluates full particle composition, gates
mapped proposals, bounds evaporation and gas-limits uptake, then couples the
representable particle change to gas. Whole-call transfer and energy outputs
are cleared at entry and accumulated from the applied substeps. PMF requires
V=1; resolved volumes need not be one. PDF execution rejects explicitly.

Resolved N/V must be finite at admission. Evaporation contributes only its
representable release to the uptake budget. Before either inventory update,
rounded positive additions are reconciled against available gas; a bounded
downward correction suppresses uptake if it cannot fit, and a nonfinite
reduction suppresses that species' transfer. Fully evaporated resolved slots
have their counts and charge cleared after whole-call gas and latent-energy
accounting. Empty gas storage with an empty map is a no-op that clears transfer
outputs and launches no gas-indexed work.

Setup validates configuration, physical state and nonoverlapping writers;
validation may launch read-only device scans and read private status flags.
Enqueue uses retained references without allocation, upload, readback,
synchronization or rebinding. No rollback is promised after a writer launches.
The standalone private inventory-finalization test helper remains legacy-only;
it is not a mapped native entry point. Untagged legacy direct/prepared callers
retain the same-width contract described below. Native transfer does not
establish resident/checkpoint support by itself. See the separately exercised
[resident walkthrough](../Examples/gpu_resident_multi_timestep.py) for explicit
exact-device restart and fresh species bindings.

Native scientific evidence is in
`particula/gpu/kernels/tests/condensation_native_test.py`: independent scalar
four-substep equations, separate particle/gas changes, V=.25/1/4 conservation,
unequal counts, unequal species widths, latent energy, activity/surface modes,
rejection snapshots and prepared execution. Warp CPU is required when Warp is
installed; CUDA is optional and skips explicitly when unavailable.

#### Temporary untagged direct path

This is a direct low-level Warp path, separate from the CPU strategies and the
`MassCondensation` runnable described above. Its canonical step import is:

```python
from particula.gpu.kernels import condensation_step_gpu
```

When reusable scratch is needed, import `CondensationScratchBuffers` only from
`particula.gpu.kernels.condensation`. That concrete-module-only sidecar is not
a second step entry point. Its transfer fields have shape
`(n_boxes, n_particles, n_species)` and its property fields have shape
`(n_boxes,)`. Each supplied field must be active-device, stable-shape
`wp.float64`; fields may be omitted independently and use fallback allocations.
Supplied buffers preserve identity and callers must keep them alive and
unmodified through launch completion.

Every successful call executes exactly four `time_step / 4.0` substeps. Each
substep optionally refreshes composition-weighted surface tension, overwrites
`gas.vapor_pressure`, refreshes environment properties, produces a raw transfer
proposal, finalizes that proposal against available inventory, applies the
finalized transfer to particle mass, accumulates it, and mutates gas
concentration by the matching particle-concentration-weighted amount. Later
proposals therefore see gas concentration coupled by earlier substeps.
Vapor-pressure refresh is temperature-driven and does not read gas
concentration. The resolved total transfer buffer is cleared once after
preflight, accumulates P2-finalized transfer, and is returned by identity when
supplied. Work storage keeps only the final raw proposal. Production
calculations make no hidden data CPU↔Warp transfers. CUDA validation reads small
device status flags with `.numpy()`, which synchronizes execution but does not
transfer or mutate caller-owned simulation buffers.

Invalid primary P2 state or supplied P2 sidecars fail during aggregate
preflight, before launches or caller-buffer mutation. A non-finite fresh raw
proposal fails before P2 finalization for that substep; completed earlier
substeps remain applied. The failed cycle may already have written raw work,
`gas.vapor_pressure`, and supplied dynamic-viscosity, mean-free-path, or
composition-weighted surface-tension workspaces, but not P2 inventories or
energy output. Retain or restore caller-owned snapshots when a failed call must
be retried from its original state.

This direct low-level GPU step uses an optional keyword-only caller-owned
active-device `wp.float64` `latent_heat` sidecar with shape `(n_species,)` in
each fixed substep. Its shipped rate is
`dm_i/dt = isothermal_rate_i / correction_i`, where
`isothermal_rate_i = k_i * Delta_p_i * M_i / (R * T)`,
`R_specific_i = R / M_i`, and
`correction_i = thermal_factor_i / (R_specific_i * T)`. The source thermal
resistance expression is
`thermal_factor_i = (D_i * L_i * p_surface_i / (T * k_thermal)) * (L_i / (T * R_specific_i) - 1) + R_specific_i * T`,
using the activity- and Kelvin-adjusted `p_surface_i`. Omitting latent heat, or
setting a species entry to exactly zero, takes that species through the exact
isothermal branch.

Issue #1272 also ships optional keyword-only caller-owned active-device
`wp.float64` `energy_transfer`, shape `(n_boxes, n_species)`, as write-only
diagnostic output. It requires valid latent heat, is overwritten only after
successful preflight, and is not a third return item. Its signed whole-call
identity is `Q[box, species] = sum_particles(Delta m_finalized) * L[species]`:
finalized `Delta m` is kg, `L` is J/kg, and `Q` is J, so condensation is
positive and evaporation negative (`kg * J/kg = J`). The #1272 diagnostic uses strict
`rtol=1e-12, atol=1e-18` evidence. `thermal_work` is validated but remains
deferred and unused.

This is diagnostic bookkeeping, not temperature feedback, gas mutation or
gas/full-system conservation. The direct step couples finalized particle and
gas transfer through its stated P2-finalized direct coupling, but does not add strategy/
`Runnable` integration, adaptive stepping, graph capture/replay, autodiff
guarantees, or complete E4-F6 cross-device certification.

The [GPU condensation parity walkthrough](../Examples/gpu_condensation_parity_walkthrough.py)
separately reports physics, conservation, and energy evidence for this
fixed-four-substep low-level direct-kernel comparison. Its
[condensation parity walkthrough ownership record](Roadmap/condensation-parity-walkthrough.md)
keeps deferred capabilities explicit. Warp CPU is the installed-Warp baseline;
CUDA is optional additive evidence. The caller-owned, write-only
`energy_transfer` output is a diagnostic, not a return value and not temperature
feedback.

Focused fixed-four-substep direct-kernel evidence commands are:
`python docs/Examples/gpu_condensation_parity_walkthrough.py` and
`pytest examples_tests/gpu_condensation_parity_walkthrough_test.py -q -Werror`.

#### P1--P4 direct-condensation evidence

| Evidence | Discoverable wrapper | Bounded result and backend |
| --- | --- | --- |
| P1 parity | `particula/gpu/kernels/tests/condensation_test.py` | An independent NumPy fixed-four-substep oracle separately compares final particle masses and gas concentrations for one/multi-box and multi-species cases. Warp CPU is the normal backend when Warp is installed; CUDA is optional additive evidence. |
| P2 conservation and contracts | `particula/gpu/kernels/tests/condensation_test.py` | Separate concentration-weighted particle-plus-gas inventory checks use `rtol=1e-12`, distinct from P1 parity. The wrapper also covers in-place mutation and caller-owned output contracts. |
| P3 capture limit | `particula/gpu/kernels/tests/condensation_graph_capture_test.py` | Warp CPU capture is capability-skipped because Warp capture requires CUDA. On CUDA, only the public-step host-validation readback inside capture is strict-xfailed; setup and normal calls must still pass. This records an unsupported capability, not replay support. |
| P4 raw-rate autodiff | `particula/gpu/kernels/tests/condensation_autodiff_test.py` | A one-box fp64 out-of-place `condensation_mass_transfer_kernel` Tape derivative is compared with centered finite differences only in the interior, where P2 positivity and inventory boundaries are inactive. Warp CPU is the normal backend; CUDA is optional. |

This matrix is evidence for `condensation_step_gpu` (P1--P3) or the raw
`condensation_mass_transfer_kernel` (P4) only. It establishes no CPU strategy
or `Runnable` equivalence, performance target, adaptive stepping, new API,
general graph replay, or general differentiability claim. For focused,
warning-clean coverage, run:

```bash
pytest particula/gpu/kernels/tests/condensation_test.py -q -Werror
pytest particula/gpu/kernels/tests/condensation_graph_capture_test.py -q -Werror
pytest particula/gpu/kernels/tests/condensation_autodiff_test.py -q -Werror

# Optional CUDA selections; unavailable CUDA must skip cleanly.
pytest particula/gpu/kernels/tests/condensation_test.py -q -m "warp and cuda" -Werror
pytest particula/gpu/kernels/tests/condensation_graph_capture_test.py -q -m "warp and cuda" -Werror
pytest particula/gpu/kernels/tests/condensation_autodiff_test.py -q -m "warp and cuda" -Werror
```

Warp-backed tests may skip when Warp is missing. CUDA evidence is optional when
CUDA is unavailable; a skip is not GPU execution. P3's CPU skip and narrowly
scoped CUDA strict expected failure document the public-step host-readback
limitation rather than a successful capture replay.

## Backend-selected condensation

Selected condensation is a bounded concrete adapter layer over the existing
native CPU `MassCondensation` runnable and the direct
`condensation_step_gpu` boundary. It is not new physics, a new public runnable,
a scheduler, a resident session, or automatic backend selection. Ownership and
selection boundaries are described in the
[execution-selection guide](data-containers-and-gpu-foundations.md#execution-selection)
and its [GPU thermodynamics and condensation refresh record](data-containers-and-gpu-foundations.md#gpu-thermodynamics-and-condensation-refresh).

| Selection/backend case | Contract |
| --- | --- |
| CPU semantic catalogue | All 36 equal-step/staggered, latent, activity, and surface semantic combinations are declared, including `NONREPRESENTABLE` categories. This is catalogue support only. |
| Selected CPU execution | The concrete adapter accepts declared catalogue metadata but does not bind it to the supplied runnable; it is isothermal only (`latent_heat=False`) and calls the caller-owned `MassCondensation` once. Equal-step or staggered semantics remain CPU-side. |
| Selected Warp execution | Exactly 8 declared profiles: equal-step, latent heat enabled or disabled, ideal or kappa activity, and static or composition-weighted surface tension. Profiles are capability/catalogue selection and do not validate supplied sidecars against their semantics. |
| Unavailable selected Warp mapping | Staggered/Gauss-Seidel and every `NONREPRESENTABLE`/BAT activity or surface mapping reject during capability-profile preflight before lazy kernel resolution, native dispatch, or adapter-driven writes; no conversion, approximation, or fallback occurs. |

Warp delegation executes exactly four equal `time_step / 4.0` substeps. Each
substep overwrites and refreshes temperature-driven vapor pressure, then applies
P2 inventory-limited particle/gas coupling. It mutates particle masses, gas
concentration, and derived GPU vapor pressure in place. The caller owns supplied
finalized total-transfer output, scratch/work storage, `latent_heat`, write-only
`energy_transfer`, and validated but deferred `thermal_work` sidecars. Omitted
optional scratch or output buffers may use direct-kernel fallback allocation;
validation may perform permitted device scans or status readbacks. Adapter
dispatch itself performs no hidden transfer, restore, retry, or CPU fallback,
and callers synchronize before host observation.

Bounded integration fixtures use an independent fixed-four-substep P2 oracle
that separately compares particle mass and gas concentration at
`rtol=1e-10, atol=1e-30`. Inventory is checked separately. Warp CPU is the
installed-Warp baseline, and CUDA rows are optional guarded evidence. This does
not claim general
CPU-strategy parity, performance, adaptive stepping, graph capture/replay,
autodiff, multi-GPU, resizing, or full scheduler support. The
[Epic G handoff](Roadmap/data-oriented-gpu.md#epic-g-backend-selection-and-gpu-resident-simulation)
defers E7-F5 deterministic scheduling.

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

- **Dynamics overview**: [Wall loss strategy system](./wall_loss_strategy_system.md)
- **Examples**: [docs/Examples/Simulations/index.md](../Examples/Simulations/index.md)
- **Latent-heat example (E3-F6)**: [CPU latent-heat condensation bookkeeping](../Examples/Dynamics/Condensation/Condensation_Latent_Heat.ipynb)
- **CPU integration baseline**: `particula/integration_tests/condensation_latent_heat_conservation_test.py`
- **Theory**: [docs/Theory/index.md](../Theory/index.md)

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
