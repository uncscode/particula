---
title: Container and Facade Migration
---

# Container and Facade Migration

This page maps legacy particle and gas facades to their data-container
equivalents. See the [migration overview](index.md) for ownership and support
boundaries.

## Quick migration

### Particle data (before to after)

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
from particula.particles import ParticleData

data = ParticleData(
    # (n_boxes, n_particles, n_species)
    masses=rep.get_species_mass()[None, ...],
    concentration=rep.get_concentration()[None, ...],
    charge=rep.get_charge()[None, ...],
    density=rep.get_density(),
    volume=np.array([rep.get_volume()]),
)
```

### Gas data (before to after)

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
from particula.gas import GasData

gas_data = GasData(
    name=["Water"],
    molar_mass=np.array([0.018]),
    concentration=np.array([[1e-6]]),  # (n_boxes, n_species)
    partitioning=np.array([True]),
)
```

## ParticleRepresentation to ParticleData

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

### Property and method mapping

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
    index `data.concentration[0]` or `data.radii[0]` for the legacy shape.

## GasSpecies to GasData

### Constructor mapping

| Legacy input | GasData field | Notes |
| --- | --- | --- |
| `name` | `name` | List of species names. |
| `molar_mass` | `molar_mass` | 1D array of molar masses. |
| `concentration` | `concentration` | Shape `(n_boxes, n_species)`. |
| `partitioning` | `partitioning` | 1D boolean array. |
| `vapor_pressure_strategy` | _behavior_ | Remains on the facade. |

### Property and method mapping

| Legacy API | GasData equivalent | Notes |
| --- | --- | --- |
| `get_name()` | `data.name` | List of names. |
| `get_molar_mass()` | `data.molar_mass` | 1D array. |
| `get_concentration()` | `data.concentration[box_index]` | Select a box. |
| `get_partitioning()` | `data.partitioning` | Boolean mask by species. |

!!! note
    Vapor pressure calculations remain on `GasSpecies`. Use the facade when
    strategy-driven behavior is required, and pass `GasData` where only data
    is required.

### GasData and WarpGasData migration summary

For the canonical transfer boundary and support limits, see
[Data Containers and GPU Foundations](../data-containers-and-gpu-foundations.md).

Migration-focused rules of thumb:

- Keep the leading box axis explicit. Single-box gas arrays still use
  `(1, n_species)`.
- Treat `name` as caller-owned metadata at the restore boundary. Supplying the
  original ordered names gives a semantic round trip; omitting `name` or
  passing `name=None` produces placeholders only.
- Treat `partitioning` as a CPU boolean API and a GPU numeric mask.
- Treat `vapor_pressure` as GPU sidecar process state that must be preserved
  or recomputed outside `GasData` after restore.

Example CPU-to-GPU-to-CPU handoff:

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

# Preserve ordered names outside WarpGasData and keep any vapor-pressure
# sidecar separately (or recompute it) on the CPU side.
```

## Gradual migration with `.data`

Both legacy facades expose their underlying data containers:

```python
particle_data = rep.data
gas_data = species.data
```

To wrap data without emitting deprecation logs, use each facade's class
method:

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

## Conversion helpers

Use conversion helpers when bridging old and new APIs:

- `from_representation` and `to_representation` for particle data. They are
  exported from `particula.particles`.
- `from_species` and `to_species` for gas data. They are exported from
  `particula.gas`.

```python
from particula.gas import from_species
from particula.particles import from_representation

particle_data = from_representation(rep, n_boxes=1)
gas_data = from_species(species, n_boxes=1)
```
