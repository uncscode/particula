---
title: Data Containers and GPU Foundations
---

# Data Containers and GPU Foundations

This page is the canonical reference for Particula's shipped data-container
schemas, leading-axis shape conventions, explicit CPU↔GPU transfer helpers,
and current CPU/GPU support boundaries.

For a runnable walkthrough of the shipped single-box container construction and
optional Warp CPU-backend round trips, see the
[Data Containers example](../Examples/Data_Containers/index.md) and run:

```bash
python docs/Examples/data_containers_and_gpu_foundations.py
```

Use this guide when you need the current contract for:

- `ParticleData`, `GasData`, and `EnvironmentData`
- `WarpParticleData`, `WarpGasData`, and `WarpEnvironmentData`
- explicit transfer helpers in `particula.gpu`
- single-box and multi-box shape conventions
- current shipped limitations for CPU and GPU workflows

For migration walkthroughs and before/after examples, see
[Particle & Gas Data Migration](particle-data-migration.md). For future work
and planned expansions, see the
[Data-Oriented Design and GPU Roadmap](Roadmap/data-oriented-gpu.md).

## Public imports

Prefer the currently exported package-level imports:

```python
from particula.gas import EnvironmentData, GasData
from particula.gpu import (
    from_warp_particle_data,
    from_warp_environment_data,
    from_warp_gas_data,
    to_warp_particle_data,
    to_warp_environment_data,
    to_warp_gas_data,
)
from particula.particles import ParticleData
```

`WarpParticleData`, `WarpEnvironmentData`, and `WarpGasData` are exported from
`particula.gpu` only when Warp is available, so import them only behind an
optional Warp guard:

```python
from particula.gpu import WARP_AVAILABLE

if WARP_AVAILABLE:
    from particula.gpu import (
        WarpEnvironmentData,
        WarpGasData,
        WarpParticleData,
    )
```

## Canonical container schemas

### `ParticleData`

`ParticleData` owns particle-side stored state. Per-box arrays always keep a
leading `n_boxes` axis.

| Field | Shape | Meaning |
| --- | --- | --- |
| `masses` | `(n_boxes, n_particles, n_species)` | Authoritative per-particle, per-species masses. |
| `concentration` | `(n_boxes, n_particles)` | Per-box particle concentration or count, depending on workflow. |
| `charge` | `(n_boxes, n_particles)` | Per-box particle charge state. |
| `density` | `(n_species,)` | Shared-across-box species density. |
| `volume` | `(n_boxes,)` | Authoritative per-box simulation volume. |

Derived properties such as `radii`, `total_mass`, and `effective_density` are
computed from stored fields rather than stored separately.

### `GasData`

`GasData` owns CPU-side gas species data and ordered species metadata.

| Field | Shape | Meaning |
| --- | --- | --- |
| `name` | `len == n_species` | Authoritative ordered CPU species metadata. |
| `molar_mass` | `(n_species,)` | Shared-across-box molar masses. |
| `concentration` | `(n_boxes, n_species)` | Authoritative per-box gas concentration. |
| `partitioning` | `(n_species,)` | Authoritative CPU boolean partitioning mask. |

`GasData` does not own per-box thermodynamic state. Temperature, pressure, and
saturation ratio belong to `EnvironmentData`.

### `EnvironmentData` and `WarpEnvironmentData`

`EnvironmentData` is the shipped CPU-side owner of per-box thermodynamic state.
`WarpEnvironmentData` mirrors that schema across the explicit helper boundary.

| Field | CPU owner | CPU shape | GPU shape | Meaning |
| --- | --- | --- | --- | --- |
| `temperature` | `EnvironmentData` | `(n_boxes,)` | `(n_boxes,)` | Per-box temperature in kelvin. |
| `pressure` | `EnvironmentData` | `(n_boxes,)` | `(n_boxes,)` | Per-box pressure in pascals. |
| `saturation_ratio` | `EnvironmentData` | `(n_boxes, n_species)` | `(n_boxes, n_species)` | Per-box, per-species thermodynamic helper state. |

`EnvironmentData` is available from `particula.gas`, while
`WarpEnvironmentData` is available from `particula.gpu` when Warp is installed.

## Shape conventions

Per-box arrays always keep a leading `n_boxes` dimension, even when
`n_boxes == 1`.

- Single-box particle masses: `(1, n_particles, n_species)`
- Single-box particle concentration: `(1, n_particles)`
- Single-box gas concentration: `(1, n_species)`
- Single-box environment temperature: `(1,)`
- Single-box environment saturation ratio: `(1, n_species)`

Shared arrays keep their shared rank and do not gain a box axis:

- `ParticleData.density -> (n_species,)`
- `GasData.molar_mass -> (n_species,)`
- `GasData.partitioning -> (n_species,)`

Example single-box container construction:

```python
import numpy as np

from particula.gas import EnvironmentData, GasData
from particula.particles import ParticleData

particle_data = ParticleData(
    masses=np.array([[[1e-18, 2e-18]]]),
    concentration=np.array([[1.0]]),
    charge=np.array([[0.0]]),
    density=np.array([1000.0, 1200.0]),
    volume=np.array([1e-6]),
)

gas_data = GasData(
    name=["Water", "H2SO4"],
    molar_mass=np.array([0.018, 0.098]),
    concentration=np.array([[1e-6, 2e-10]]),
    partitioning=np.array([True, True]),
)

environment = EnvironmentData(
    temperature=np.array([298.15]),
    pressure=np.array([101325.0]),
    saturation_ratio=np.array([[0.5, 0.2]]),
)
```

## Explicit CPU↔GPU transfer helpers

Particula uses explicit helper calls for CPU↔GPU container movement. Kernels
and runnables do not perform hidden synchronization or hidden container
transfers.

Available public helpers:

- `to_warp_particle_data()`
- `from_warp_particle_data()`
- `to_warp_gas_data()`
- `from_warp_gas_data()`
- `to_warp_environment_data()`
- `from_warp_environment_data()`

Optional Warp availability can be checked with `WARP_AVAILABLE`.

### Particle transfer boundary

`WarpParticleData` is the explicit Warp mirror for `ParticleData`. Use the
helper boundary when particle-resident arrays must move between CPU-owned
containers and Warp-owned storage:

```python
from particula.gpu import from_warp_particle_data, to_warp_particle_data

gpu_particle = to_warp_particle_data(particle_data, device="cpu")
restored_particle = from_warp_particle_data(gpu_particle)
```

Across this boundary, the shipped particle schema stays aligned:

| Field | CPU `ParticleData` | GPU `WarpParticleData` | Contract |
| --- | --- | --- | --- |
| `masses` | `(n_boxes, n_particles, n_species)` | `(n_boxes, n_particles, n_species)` | Authoritative per-particle, per-species masses round-trip without shape drift. |
| `concentration` | `(n_boxes, n_particles)` | `(n_boxes, n_particles)` | Preserves the leading `n_boxes` axis for single-box and multi-box storage. |
| `charge` | `(n_boxes, n_particles)` | `(n_boxes, n_particles)` | Preserves particle charge state without hidden conversion. |
| `density` | `(n_species,)` | `(n_species,)` | Shared species density remains shared; it does not gain a box axis. |
| `volume` | `(n_boxes,)` | `(n_boxes,)` | Per-box simulation volume stays particle-owned across the helper boundary. |

As with the gas and environment helpers, kernels and runnables do not perform
hidden CPU↔GPU synchronization or implicit container transfers for particle
state.

### Environment transfer boundary

Use the environment helpers when thermodynamic state must cross the CPU↔GPU
boundary:

```python
from particula.gpu import (
    from_warp_environment_data,
    to_warp_environment_data,
)

gpu_environment = to_warp_environment_data(environment, device="cpu")
restored_environment = from_warp_environment_data(gpu_environment)
```

Across this boundary, shapes stay fixed:

- `temperature -> (n_boxes,)`
- `pressure -> (n_boxes,)`
- `saturation_ratio -> (n_boxes, n_species)`

### Gas transfer boundary and intentional lossiness

`WarpGasData` is a numeric GPU mirror, not a complete semantic copy of
`GasData`.

| Field | CPU `GasData` | GPU `WarpGasData` | Restore contract |
| --- | --- | --- | --- |
| `name` | Ordered species names | Not stored | Restore requires caller-supplied ordered names, or placeholder names such as `species_0`. |
| `molar_mass` | `(n_species,)` | `(n_species,)` | Round-trips without shape drift. |
| `concentration` | `(n_boxes, n_species)` | `(n_boxes, n_species)` | Round-trips without shape drift. |
| `partitioning` | `bool`, shape `(n_species,)` | `int32`, shape `(n_species,)` | Converts `bool → int32 → bool`; restored GPU values must remain binary. |
| `vapor_pressure` | Not owned by `GasData` | `(n_boxes, n_species)` | GPU-only helper state. Pass it explicitly when needed; CPU restore always drops it. |

This makes GPU→CPU gas restore intentionally lossy unless the caller preserves
ordered names and any vapor-pressure sidecar state outside `WarpGasData`.

Example explicit gas handoff:

```python
import numpy as np

from particula.gpu import from_warp_gas_data, to_warp_gas_data

vapor_pressure = np.array([[2330.0, 120.0]])
gpu_gas = to_warp_gas_data(
    gas_data,
    device="cpu",
    vapor_pressure=vapor_pressure,
)
restored_gas = from_warp_gas_data(gpu_gas, name=gas_data.name)
```

## Current shipped support boundaries

The containers are multi-box capable, but current execution support is narrower
than storage support.

| Area | Shipped support | Notes |
| --- | --- | --- |
| CPU `ParticleData` / `GasData` storage | Multi-box-capable | Leading `n_boxes` axis is part of the stored schema. |
| CPU condensation with data containers | `n_boxes == 1` only | Multi-box CPU execution still requires a caller-managed per-box loop. |
| CPU coagulation with data containers | `n_boxes == 1` only | Multi-box CPU execution is not yet a built-in runtime path. |
| CPU↔GPU transfer | Explicit helper calls only | No hidden container movement or hidden environment synchronization. |
| Warp/CUDA support | Optional | Warp parity tests always cover Warp `cpu`; `cuda` runs only when available. |
| Low-level GPU coagulation direct-kernel path | Accepted with caveats | Appropriate for many independent boxes, especially when CUDA can supply box-level parallel throughput, Warp-backed direct-kernel workflows, and CUDA benchmark/study runs; caveated for large single-box production workloads and does not imply hidden transfer or synchronization behavior. |
| Fixed-shape GPU/runtime roadmap work | Not current runtime behavior | Graph-capture-oriented and fixed-shape runtime constraints remain roadmap handoff material, not shipped behavior. |

Additional shipped boundaries:

- `ParticleData.volume` remains the authoritative per-box simulation-volume
  owner; it does not move into `EnvironmentData`.
- `EnvironmentData` is the shipped CPU thermodynamic owner for `temperature`,
  `pressure`, and `saturation_ratio`.
- `WarpGasData.vapor_pressure` is helper state only and has no CPU `GasData`
  field.
- Coagulation `rng_states` are caller-owned Warp-resident sidecar state only;
  they are not fields on `ParticleData`, `GasData`, `EnvironmentData`, or any
  Warp container schema.
- No hidden CPU↔GPU synchronization occurs inside kernels or runnables.

## Guidance for current users

- Use package-level imports for public containers and helpers.
- Preserve the leading box axis in all per-box arrays, even for single-box
  workflows.
- Preserve ordered gas names outside GPU containers if you need a semantic
  gas round trip.
- Preserve or recompute vapor pressure separately on the CPU side after GPU
  restore.
- Use the current low-level GPU coagulation path when you want throughput from
  many independent boxes, especially on CUDA where box-level parallelism can
  stay busy, a documented direct-kernel workflow on a Warp-supported device,
  or a CUDA-backed benchmark/study run tied to the measured evidence in the
  roadmap.
- Do not treat the current one-thread-per-box coagulation path as the
  recommended production path for large single-box workloads; the shipped
  caution band is documented in the roadmap's measured decision record.
- Do not expect kernels or runnables to perform hidden CPU↔GPU transfers or
  hidden synchronization for particle, gas, or environment state; use the
  explicit helper calls when state must cross the device boundary.
- Treat Warp and CUDA as optional runtime capabilities: without Warp, this
  low-level GPU path is unavailable, and CUDA benchmark conclusions should not
  be assumed to apply unchanged to Warp `cpu` or other hardware.
- Treat roadmap pages as future-work references, not as evidence that broader
  runtime support has already shipped.

## Related references

- [Data Containers example](../Examples/Data_Containers/index.md)
- [Particle & Gas Data Migration](particle-data-migration.md)
- [Data-Oriented Design and GPU Roadmap](Roadmap/data-oriented-gpu.md)
- [Mass Precision Recommendation Report](Roadmap/mass-precision-study.md)
