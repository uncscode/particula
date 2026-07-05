---
title: ParticleData and GasData Migration
---

# ParticleData and GasData Migration Guide

This guide explains how to migrate from the legacy facades
(`ParticleRepresentation`, `GasSpecies`) to the new data containers
(`ParticleData`, `GasData`). The facades remain available for backward
compatibility, but they are deprecated and emit log warnings to guide you
toward the data-first workflow.

If you arrived here from the legacy path `docs/migration/particle-data.md`,
that page now redirects to this canonical guide.

## Overview

Before adding container fields or changing CPU↔GPU conversion behavior, review
the roadmap's
[canonical shape conventions for container workflows](Roadmap/data-oriented-gpu.md#canonical-shape-conventions-for-container-workflows)
and its surrounding [authoritative field ownership
decisions](Roadmap/data-oriented-gpu.md#authoritative-field-ownership-decisions).
Downstream implementers should then read the
[final downstream handoff map for sibling
features](Roadmap/data-oriented-gpu.md#final-downstream-handoff-map-for-sibling-features)
for the exact inherited field, shape, ownership, and CPU↔GPU transfer rules.

The migration moves **data** into dedicated containers and leaves **behavior**
in strategies and runnables:

- `ParticleData` stores per-particle arrays with an explicit batch dimension.
- `GasData` stores gas species arrays with an explicit box dimension; it does
  not own per-box thermodynamic state.
- `EnvironmentData` now provides the shipped CPU-side owner of per-box
  thermodynamic state with `temperature -> (n_boxes,)`,
  `pressure -> (n_boxes,)`, and `saturation_ratio -> (n_boxes, n_species)`.
- `ParticleData.volume` remains the authoritative per-box simulation-volume
  owner; this migration does not move simulation volume into
  `EnvironmentData`.
- `ParticleRepresentation` and `GasSpecies` remain as facades so existing
  workflows keep working while you migrate.

!!! note
    `EnvironmentData` is the shipped CPU container for per-box thermodynamic
    state, not a separate gas facade. It is available from
    `particula.gas.environment_data` and is exported from `particula.gas` for
    package-level imports. It requires at least one box at construction time.
    The only shipped CPU↔GPU transfer boundary is the explicit helper trio
    `particula.gpu.WarpEnvironmentData`,
    `particula.gpu.to_warp_environment_data()`, and
    `particula.gpu.from_warp_environment_data()`. The shape contract stays
    fixed across that boundary:
    `temperature`/`pressure -> (n_boxes,)` and
    `saturation_ratio -> (n_boxes, n_species)`. Current regression coverage
    checks one-box and multi-box CPU→Warp→CPU round trips, always on Warp `cpu`
    and additionally on Warp `cuda` when available. Those transfers remain
    explicit helper calls only: kernels and runnables do not perform hidden
    CPU↔GPU environment synchronization or movement, and CUDA is optional
    rather than required. Current tests also cover default synchronized
    restore, manual `sync=False` restore after explicit synchronization, and
    schema validation failures. This documents the helper surface only, not a
    broader automatic runtime integration.

!!! warning
    GPU→CPU gas restore is intentionally lossy unless you preserve ordered
    species metadata outside the GPU container. `WarpGasData` excludes string
    fields and `from_warp_gas_data()` can validate only the supplied name-list
    length today; it does not verify that restored names still match the
    original species ordering, and GPU helper state such as
    `WarpGasData.vapor_pressure` is dropped on CPU restore. Use the roadmap's
    [authoritative field ownership decisions](Roadmap/data-oriented-gpu.md#authoritative-field-ownership-decisions),
    [canonical shape conventions for container workflows](Roadmap/data-oriented-gpu.md#canonical-shape-conventions-for-container-workflows),
    and [final downstream handoff map for sibling features](Roadmap/data-oriented-gpu.md#final-downstream-handoff-map-for-sibling-features)
    as the authoritative contract for this restore boundary.

## Why migrate

- **Clear data/behavior split**: data containers keep state, strategies keep
  physics.
- **Multi-box ready**: batch dimensions make CFD and multi-box simulations
  first-class.
- **Fewer implicit conversions**: attributes are explicit arrays rather than
  getter methods.

## Quick migration

### Particle data (before → after)

```python
import numpy as np
import particula as par

# Legacy facade
rep = par.particles.ParticleRepresentation(
    strategy=par.particles.MassBasedMovingBin(),
    activity=par.particles.ActivityIdealMass(),
    surface=par.particles.SurfaceStrategyMass(),
    distribution=np.array([1e-18, 2e-18, 3e-18]),
    density=np.array([1200.0]),
    concentration=np.array([1e5, 1e5, 1e5]),
    charge=np.zeros(3),
    volume=1e-6,
)

# New data container
from particula.particles.particle_data import ParticleData

data = ParticleData(
    # (n_boxes, n_particles, n_species)
    masses=rep.get_species_mass()[None, ...],
    concentration=rep.get_concentration()[None, ...],
    charge=rep.get_charge()[None, ...],
    density=rep.get_density(),
    volume=np.array([rep.get_volume()]),
)
```

### Gas data (before → after)

```python
import numpy as np
import particula as par

# Legacy facade
species = par.gas.GasSpecies(
    name="Water",
    molar_mass=0.018,
    vapor_pressure_strategy=par.gas.ConstantVaporPressureStrategy(2330.0),
    concentration=1e-6,
)

# New data container
from particula.gas.gas_data import GasData

gas_data = GasData(
    name=["Water"],
    molar_mass=np.array([0.018]),
    concentration=np.array([[1e-6]]),  # (n_boxes, n_species)
    partitioning=np.array([True]),
)
```

## ParticleRepresentation → ParticleData

### Constructor mapping

| Legacy input | ParticleData field | Notes |
| --- | --- | --- |
| `distribution` | `masses` | Convert to per-species masses. |
| `density` | `density` | 1D array of species densities. |
| `concentration` | `concentration` | Shape `(n_boxes, n_particles)`. |
| `charge` | `charge` | Shape `(n_boxes, n_particles)`. |
| `volume` | `volume` | Shape `(n_boxes,)`. |
| `strategy` | _behavior_ | Keep strategy separate from data. |
| `activity` | _behavior_ | Remains in activity strategies. |
| `surface` | _behavior_ | Remains in surface strategies. |

### Property/method mapping

| Legacy API | ParticleData equivalent | Notes |
| --- | --- | --- |
| `get_radius()` | `data.radii` | Computed from mass and density. |
| `get_mass()` | `data.total_mass` | Total per particle. |
| `get_species_mass()` | `data.masses` | Per-species masses. |
| `get_density()` | `data.density` | Density per species. |
| `get_concentration()` | `data.concentration` | 2D array by box. |
| `get_charge()` | `data.charge` | 2D array by box. |
| `get_volume()` | `data.volume` | Per-box volume. |
| `get_effective_density()` | `data.effective_density` | Mass-weighted density. |
| `get_total_concentration()` | `data.concentration.sum(axis=1)` | Per box. |

!!! note
    `ParticleData` keeps the batch dimension. If you used a single-box facade,
    index `data.concentration[0]` or `data.radii[0]` to get the legacy shape.

## GasSpecies → GasData

### Constructor mapping

| Legacy input | GasData field | Notes |
| --- | --- | --- |
| `name` | `name` | List of species names. |
| `molar_mass` | `molar_mass` | 1D array of molar masses. |
| `concentration` | `concentration` | Shape `(n_boxes, n_species)`. |
| `partitioning` | `partitioning` | 1D boolean array. |
| `vapor_pressure_strategy` | _behavior_ | Remains on the facade. |

### Property/method mapping

| Legacy API | GasData equivalent | Notes |
| --- | --- | --- |
| `get_name()` | `data.name` | List of names. |
| `get_molar_mass()` | `data.molar_mass` | 1D array. |
| `get_concentration()` | `data.concentration[box_index]` | Select a box. |
| `get_partitioning()` | `data.partitioning` | Boolean mask by species. |

!!! note
    Vapor pressure calculations remain on `GasSpecies`. Use the facade when
    you need strategy-driven behavior, and pass `GasData` where only data is
    required.

### `GasData` ↔ `WarpGasData` field authority

Use this table as the migration-facing summary for what each container owns and
what survives the explicit CPU↔GPU helper boundary.

| Field | CPU `GasData` | GPU `WarpGasData` | Round-trip contract |
| --- | --- | --- | --- |
| `name` | Authoritative ordered species metadata as `list[str]` with `len == n_species`. | Not stored. `WarpGasData` has no string field. | `to_warp_gas_data()` drops names. `from_warp_gas_data()` restores caller-supplied ordered names or, when `name` is omitted or `None`, placeholder values such as `species_0`. Current validation checks only list length, so callers must preserve ordering externally. |
| `molar_mass` | Authoritative shared-across-boxes numeric state. Shape `(n_species,)`. | Numeric mirror with shape `(n_species,)`. | Round-trips without value or shape changes. |
| `concentration` | Authoritative per-box gas state. Shape `(n_boxes, n_species)`, including `(1, n_species)` for one-box workflows. | Numeric mirror with shape `(n_boxes, n_species)`. | Round-trips without value or shape changes. |
| `partitioning` | Authoritative per-species boolean mask. Shape `(n_species,)`, dtype `bool`. | Numeric GPU mask with shape `(n_species,)`, dtype `int32`. | Converts `bool → int32 → bool`. GPU restore requires binary `0`/`1` values. |
| `vapor_pressure` | Not owned by `GasData`. Preserve or recompute it separately on the CPU side. | GPU-only helper state with shape `(n_boxes, n_species)`. | Pass it explicitly to `to_warp_gas_data()` when GPU kernels need physical vapor-pressure values. If omitted, the helper allocates zeros with the same shape. `from_warp_gas_data()` always drops this field. |

Migration rules backed by the regression tests:

- Keep the leading box axis explicit. Single-box gas arrays still use
  `(1, n_species)`.
- Treat `name` as caller-owned metadata at the restore boundary. Supplying the
  original ordered names gives a semantic round-trip; omitting `name` or
  passing `name=None` produces placeholders only.
- Treat `partitioning` as a CPU boolean API and a GPU numeric mask. Do not
  depend on non-binary GPU values being coerced.
- Treat `vapor_pressure` as sidecar process state for GPU workflows. Compute it
  on the CPU, pass it to `to_warp_gas_data()`, and preserve it separately if
  you still need it after restoring `GasData`.

Example CPU→GPU→CPU handoff:

```python
import numpy as np

from particula.gpu import from_warp_gas_data, to_warp_gas_data

vapor_pressure = np.array([[2330.0, 120.0]])  # (1, n_species)
gpu_gas = to_warp_gas_data(
    gas_data,
    device="cpu",
    vapor_pressure=vapor_pressure,
)
restored = from_warp_gas_data(gpu_gas, name=gas_data.name)

# Preserve ordered names and any vapor-pressure sidecar outside WarpGasData.
```

## Gradual migration with `.data`

Both legacy facades expose their underlying data containers:

```python
particle_data = rep.data
gas_data = species.data
```

If you need to wrap data without emitting deprecation logs, use the class
methods provided by each facade:

```python
rep = par.particles.ParticleRepresentation.from_data(
    data=particle_data,
    strategy=par.particles.MassBasedMovingBin(),
    activity=par.particles.ActivityIdealMass(),
    surface=par.particles.SurfaceStrategyMass(),
    distribution=particle_data.total_mass[0],
)

species = par.gas.GasSpecies.from_data(
    data=gas_data,
    vapor_pressure_strategy=par.gas.ConstantVaporPressureStrategy(2330.0),
)
```

## Using ParticleData/GasData in dynamics

Condensation and coagulation strategies accept both legacy facades and the new
data containers. The return type matches the input type.

Today, the compatibility boundary is still scalar at many process entry points:
existing dynamics APIs may continue to accept scalar `temperature` and
`pressure`. Only migrated process code should read `EnvironmentData` directly,
and environment fields should be treated as read-only unless the physical model
owns the update and refreshes derived helpers such as `saturation_ratio`.

```python
import particula as par

activity_strategy = par.particles.ActivityIdealMass()
surface_strategy = par.particles.SurfaceStrategyMass()
vapor_pressure_strategy = par.gas.ConstantVaporPressureStrategy(2330.0)

condensation = par.dynamics.CondensationIsothermal(
    molar_mass=0.018,
    activity_strategy=activity_strategy,
    surface_strategy=surface_strategy,
    vapor_pressure_strategy=vapor_pressure_strategy,
)
particle_out, gas_out = condensation.step(
    particle=particle_data,
    gas_species=gas_data,
    temperature=298.15,
    pressure=101325.0,
    time_step=1.0,
)

coagulation = par.dynamics.BrownianCoagulationStrategy(
    distribution_type="discrete"
)
particle_out = coagulation.step(
    particle=particle_out,
    temperature=298.15,
    pressure=101325.0,
    time_step=1.0,
)
```

Use `EnvironmentData` to document and carry per-box thermodynamic state on the
CPU side. `EnvironmentData` owns `temperature`, `pressure`, and
`saturation_ratio`; `GasData` does not. Keep current scalar `temperature` and
`pressure` arguments where the process API has not yet been migrated; migrated
process code may read `EnvironmentData` directly, but only the physical model
that owns the update should mutate it. When a GPU round trip is needed, use
the explicit helper boundary only:
`particula.gpu.WarpEnvironmentData`,
`particula.gpu.to_warp_environment_data()`, and
`particula.gpu.from_warp_environment_data()`. Keep the documented shapes
unchanged across that boundary—`temperature` and `pressure` stay
`(n_boxes,)`, while `saturation_ratio` stays `(n_boxes, n_species)`—and do
not expect kernels or runnables to move environment state for you.

## Conversion helpers

Use the conversion helpers when you need to bridge old and new APIs:

- [`from_representation`](https://github.com/uncscode/particula/blob/main/particula/particles/particle_data.py)
  and [`to_representation`](https://github.com/uncscode/particula/blob/main/particula/particles/particle_data.py)
  for particle data.
- [`from_species`](https://github.com/uncscode/particula/blob/main/particula/gas/gas_data.py)
  and [`to_species`](https://github.com/uncscode/particula/blob/main/particula/gas/gas_data.py)
  for gas data.

```python
from particula.particles.particle_data import from_representation
from particula.gas.gas_data import from_species

particle_data = from_representation(rep, n_boxes=1)
gas_data = from_species(species, n_boxes=1)
```

## Deprecation timeline

- **v0.3.0**: `ParticleRepresentation` and `GasSpecies` are deprecated and
  emit log warnings.
- **v1.0**: planned removal of the legacy facades.

## Troubleshooting

### Shape mismatches when creating data containers

`ParticleData` expects `(n_boxes, n_particles, n_species)` for masses and
`(n_boxes, n_particles)` for concentration/charge. Use `np.newaxis` or
`np.tile` to add the batch dimension.

### Deprecation logs

The facades log at INFO level to avoid `-Werror` failures. To reduce noise,
prefer `ParticleData`/`GasData` directly or wrap with `from_data` methods.

### Single-box vs multi-box data

Legacy facades assume a single box. For data containers, index the first box
when you need legacy-shaped arrays:

```python
radii_single_box = particle_data.radii[0]
concentration_single_box = gas_data.concentration[0]
```

## Related references

- `ParticleData` source:
  [particula/particles/particle_data.py](https://github.com/uncscode/particula/blob/main/particula/particles/particle_data.py)
- `GasData` source:
  [particula/gas/gas_data.py](https://github.com/uncscode/particula/blob/main/particula/gas/gas_data.py)
- Legacy facades:
  [particula/particles/representation.py](https://github.com/uncscode/particula/blob/main/particula/particles/representation.py),
  [particula/gas/species.py](https://github.com/uncscode/particula/blob/main/particula/gas/species.py)
