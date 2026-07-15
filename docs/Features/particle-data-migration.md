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
that page now redirects to this migration guide.

## Overview

Before adding container fields or changing CPU↔GPU conversion behavior, start
with the canonical
[Data Containers and GPU Foundations](data-containers-and-gpu-foundations.md)
guide for the shipped schema, shape, helper, and support-boundary contract.

For roadmap policy and planned follow-on work, review the roadmap's
[authoritative field ownership decisions](Roadmap/data-oriented-gpu.md#authoritative-field-ownership-decisions),
[canonical shape conventions for container workflows](Roadmap/data-oriented-gpu.md#canonical-shape-conventions-for-container-workflows),
and [final downstream handoff map for sibling
features](Roadmap/data-oriented-gpu.md#final-downstream-handoff-map-for-sibling-features).
Treat the [Mass Precision Recommendation
Report](Roadmap/mass-precision-study.md) as the canonical reference before
changing particle mass dtype/schema behavior.

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
    state, not a separate gas facade. In this migration context, the relevant
    shipped environment-state CPU↔GPU transfer boundary is the explicit helper
    trio `particula.gpu.WarpEnvironmentData`,
    `particula.gpu.to_warp_environment_data()`, and
    `particula.gpu.from_warp_environment_data()`. For the authoritative
    container schema, shape contract, and helper/support-boundary details,
    defer to
    [Data Containers and GPU Foundations](data-containers-and-gpu-foundations.md).

!!! warning
    GPU→CPU gas restore is intentionally lossy unless you preserve ordered
    species metadata outside the GPU container. `WarpGasData` excludes string
    fields, and GPU-only helper state such as `vapor_pressure` is dropped on
    CPU restore. Use
    [Data Containers and GPU Foundations](data-containers-and-gpu-foundations.md)
    as the authoritative shipped contract for this restore boundary, and use
    the roadmap's
    [authoritative field ownership decisions](Roadmap/data-oriented-gpu.md#authoritative-field-ownership-decisions),
    [canonical shape conventions for container workflows](Roadmap/data-oriented-gpu.md#canonical-shape-conventions-for-container-workflows),
    and [final downstream handoff map for sibling features](Roadmap/data-oriented-gpu.md#final-downstream-handoff-map-for-sibling-features)
    for deeper policy and future-work context.

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
from particula.gas import GasData

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

### `GasData` ↔ `WarpGasData` migration summary

For the canonical reference page covering this transfer boundary alongside
`ParticleData`, `EnvironmentData`, and current support limits, see
[Data Containers and GPU Foundations](data-containers-and-gpu-foundations.md).

Migration-focused rules of thumb:

- Keep the leading box axis explicit. Single-box gas arrays still use
  `(1, n_species)`.
- Treat `name` as caller-owned metadata at the restore boundary. Supplying the
  original ordered names gives a semantic round-trip; omitting `name` or
  passing `name=None` produces placeholders only.
- Treat `partitioning` as a CPU boolean API and a GPU numeric mask.
- Treat `vapor_pressure` as GPU sidecar process state that must be preserved
  or recomputed outside `GasData` after restore.

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

# Preserve ordered names outside WarpGasData and keep any vapor-pressure
# sidecar separately (or recompute it) on the CPU side.
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

For the canonical support-boundary summary, including the preserved leading
`n_boxes` axis and the explicit particle/gas/environment helper boundaries, see
[Data Containers and GPU Foundations](data-containers-and-gpu-foundations.md).

Condensation and coagulation strategies accept both legacy facades and the new
data containers. The return type matches the input type, but that container
compatibility does **not** mean every CPU dynamics path already supports full
multi-box execution.

| CPU dynamics path | Containers accepted | Current CPU execution support | If you need multi-box CPU execution |
| --- | --- | --- | --- |
| Condensation | Legacy `ParticleRepresentation` + `GasSpecies`, or `ParticleData` + `GasData` | Supported with `n_boxes == 1` only | Run a caller-managed per-box loop in user code and pass one box at a time |
| Coagulation | Legacy `ParticleRepresentation`, or `ParticleData` | Supported with `n_boxes == 1` only | Run a caller-managed per-box loop in user code and pass one box at a time |

The shipped CPU support boundary remains narrower than the storage schema:
`ParticleData` and `GasData` storage can be multi-box, but the current audited
CPU condensation and CPU coagulation execution paths remain single-box
workflows. Use
[Data Containers and GPU Foundations](data-containers-and-gpu-foundations.md)
as the canonical reference for the support contract.

Today, the compatibility boundary is still scalar at many process entry points:
existing dynamics APIs may continue to accept scalar `temperature` and
`pressure`. Only migrated process code should read `EnvironmentData` directly,
and environment fields should be treated as read-only unless the physical model
owns the update and refreshes derived helpers such as `saturation_ratio`.

For the currently audited CPU baseline:

- `CondensationIsothermal` and related public condensation entry points accept
  `ParticleData` and `GasData` only when `n_boxes == 1`.
- CPU coagulation strategy support for `ParticleData` also remains single-box
  only. Supported `n_boxes == 1` calls still work, but multi-box
  `ParticleData` inputs now fail fast with a clear `ValueError` instead of
  silently reading from or mutating box `0`.

Supported single-box CPU usage looks like this when `particle_data` and
`gas_data` each have `n_boxes == 1`:

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

If you need multi-box CPU execution today, manage the box loop in your own
code rather than expecting a built-in CPU strategy loop. The following is
caller-managed pseudocode, not a built-in particula helper:

```python
# Caller-managed user code for multi-box CPU workflows.
for box_index in range(particle_data.n_boxes):
    single_box_particle = build_single_box_particle_data(
        particle_data,
        box_index,
    )
    single_box_gas = build_single_box_gas_data(gas_data, box_index)

    particle_box_out, gas_box_out = condensation.step(
        particle=single_box_particle,
        gas_species=single_box_gas,
        temperature=298.15,
        pressure=101325.0,
        time_step=1.0,
    )

    particle_box_out = coagulation.step(
        particle=particle_box_out,
        temperature=298.15,
        pressure=101325.0,
        time_step=1.0,
    )

    # Reassemble results in caller-owned storage.
```

Treat that pattern as an application-level workaround, not current built-in CPU
multi-box strategy support.

Use `EnvironmentData` to document and carry per-box thermodynamic state on the
CPU side. `EnvironmentData` owns `temperature`, `pressure`, and
`saturation_ratio`; `GasData` does not. Keep current scalar `temperature` and
`pressure` arguments where the process API has not yet been migrated; migrated
process code may read `EnvironmentData` directly, but that does not expand the
current CPU dynamics boundary beyond the audited behavior above. Only the
physical model that owns the update should mutate environment state. When a GPU
round trip is needed, use the explicit helper boundary documented in
[Data Containers and GPU Foundations](data-containers-and-gpu-foundations.md);
do not expect kernels or runnables to move environment state for you.

### `condensation_step_gpu` environment inputs

The bounded direct GPU condensation path is imported with
`from particula.gpu.kernels import condensation_step_gpu`. Explicitly convert
with `to_warp_*` before the call and restore with `from_warp_*` afterward;
preserve ordered species names outside GPU containers. Callers also own required
synchronization and checkpoint/snapshot responsibility. See the authoritative
[Data Containers and GPU Foundations](data-containers-and-gpu-foundations.md)
page for its modes and schema matrix rather than duplicating that matrix here.

The non-executable signature is
`particles_out, mass_transfer = condensation_step_gpu(...,
thermodynamics=thermodynamics)`. `thermodynamics=` is required and is
caller-owned device-local configuration. Particle masses and
`gas.concentration` mutate in place; there is no second gas return value.
Caller-supplied GPU vapor pressure is derived, non-authoritative state that the
step overwrites.

Direct temperature and pressure inputs may be positive-finite scalars,
same-device Warp arrays with shape `(n_boxes,)`, or hybrid scalar/Warp-array
inputs. Alternatively, pass `environment=WarpEnvironmentData` with both direct
inputs omitted; its `(n_boxes,)` values must be positive finite and on the same
device. Temperature and pressure remain environment-owned state, not `GasData`
fields.

The step executes four fixed equal substeps, uses caller-owned reusable scratch
and diagnostic buffers, P2 inventory finalization, and gas coupling for later
proposals. Its transfer is a whole-call total; optional caller-owned energy
output is not a third return value. This is not a high-level `Aerosol` or
`Runnable` API, automatic fallback, or hidden simulation-state transfer.
Callers synchronize before host observation or restoration; CUDA preflight
validation-flag readbacks may synchronize without transferring simulation
state. It also does not add adaptive stepping, new physics or container
support, BAT, or staggered/Gauss-Seidel support.

## Conversion helpers

Use the conversion helpers when you need to bridge old and new APIs:

- `from_representation` and `to_representation` for particle data. Their
  implementation lives in `particula/particles/particle_data.py` and they are
  exported from `particula.particles`.
- `from_species` and `to_species` for gas data. Their implementation lives in
  `particula/gas/gas_data.py` and they are exported from `particula.gas`.

```python
from particula.gas import from_species
from particula.particles import from_representation

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

Legacy facades assume a single box. For data containers, the current audited
CPU baseline is:

- Condensation public `ParticleData`/`GasData` paths accept only
  `n_boxes == 1` and reject `n_boxes != 1`.
- CPU coagulation `ParticleData` paths accept only `n_boxes == 1`; multi-box
  inputs now raise a clear `ValueError` instead of falling back to box `0`.

For the canonical support contract, including the supported single-box example
and caller-managed per-box loop guidance, see
[Using ParticleData/GasData in dynamics](#using-particledatagasdata-in-dynamics).

When you need legacy-shaped arrays, index the first box explicitly:

```python
radii_single_box = particle_data.radii[0]
concentration_single_box = gas_data.concentration[0]
```

### `condensation_step_gpu` rejects my environment inputs

`condensation_step_gpu(...)` now validates environment inputs before Warp
launch. Check the following first:

- Do not mix direct `temperature`/`pressure` arguments with `environment=` in
  the same call.
- If `environment` is omitted, supply both required direct thermodynamic
  inputs, either as scalars, `(n_boxes,)` Warp arrays, or a supported hybrid.
- Ensure direct or environment-owned `temperature` and `pressure` values are
  positive and finite.
- Ensure `(n_boxes,)` direct arrays match the particle/gas box count and live
  on the same Warp device.

If you need a CPU-owned source of truth, keep using `EnvironmentData` and only
convert it at the explicit `to_warp_environment_data()` boundary.

### Direct-condensation troubleshooting and reproduction

Keep restored ordered gas names and thermodynamics-sidecar species order aligned
with `gas.molar_mass`, including a valid water-species index. Particle, gas,
and sidecar layouts retain their leading `(n_boxes, ...)` dimension; supplied
scratch, latent-heat, and energy sidecars must be active-device `wp.float64`.
Use either `environment=` or both direct positive finite temperature/pressure
inputs, with direct arrays on the active device. P2 inventory limiting bounds
applied transfers rather than proving parity. Synchronize explicitly before host
observation of caller-owned energy output. Warp `device="cpu"` is the baseline
when installed; CUDA is optional/local and skips cleanly when CUDA is
unavailable.

For the single canonical command matrix, see [GPU condensation command matrix](data-containers-and-gpu-foundations.md#focused-reproduction-commands).

## Related references

- `ParticleData` implementation: `particula/particles/particle_data.py`
- `GasData` implementation: `particula/gas/gas_data.py`
- Legacy facades: `particula/particles/representation.py`,
  `particula/gas/species.py`
