---
title: ParticleData and GasData Migration
---

# ParticleData and GasData Migration Guide

This guide explains how to migrate from the legacy facades
(`ParticleRepresentation`, `GasSpecies`) to the new data containers
(`ParticleData`, `GasData`). The facades remain available for backward
compatibility, but they are deprecated and emit log warnings to guide you
toward the data-first workflow.

If you arrived here from the legacy path
`docs/Features/particle-data-migration.md`, this document is now the
canonical migration guide.

## Overview

The migration moves **data** into dedicated containers and leaves **behavior**
in strategies and runnables:

- `ParticleData` stores per-particle arrays with an explicit batch dimension.
- `GasData` stores gas species arrays with an explicit box dimension.
- `ParticleRepresentation` and `GasSpecies` remain as facades so existing
  workflows keep working while you migrate.

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
    activity=par.particles.IdealActivity(),
    surface=par.particles.KelvinSurface(),
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
| `get_condensable()` | `data.partitioning` | Boolean mask by species. |

!!! note
    Vapor pressure calculations remain on `GasSpecies`. Use the facade when
    you need strategy-driven behavior, and pass `GasData` where only data is
    required.

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
    activity=par.particles.IdealActivity(),
    surface=par.particles.KelvinSurface(),
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

```python
import particula as par

condensation = par.dynamics.CondensationIsothermal(molar_mass=0.018)
particle_out, gas_out = condensation.step(
    particle=particle_data,
    gas_species=gas_data,
    temperature=298.15,
    pressure=101325.0,
    time_step=1.0,
)

coagulation = par.dynamics.BrownianCoagulationStrategy()
particle_out = coagulation.step(
    particle=particle_out,
    temperature=298.15,
    pressure=101325.0,
    time_step=1.0,
)
```

## Conversion helpers

Use the conversion helpers when you need to bridge old and new APIs:

- [`from_representation`](../../particula/particles/particle_data.py) and
  [`to_representation`](../../particula/particles/particle_data.py) for
  particle data.
- [`from_species`](../../particula/gas/gas_data.py) and
  [`to_species`](../../particula/gas/gas_data.py) for gas data.

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
  [particula/particles/particle_data.py](../../particula/particles/particle_data.py)
- `GasData` source:
  [particula/gas/gas_data.py](../../particula/gas/gas_data.py)
- Legacy facades:
  [particula/particles/representation.py](../../particula/particles/representation.py),
  [particula/gas/species.py](../../particula/gas/species.py)
