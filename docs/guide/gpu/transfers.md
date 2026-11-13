# Transfers

## Explicit CPU↔GPU transfer helpers

Particula uses explicit helper calls for CPU↔GPU container movement. Kernels
and runnables do not perform hidden container transfers. Read-only device
validation may synchronize and read back status to surface invalid values.

Available public helpers:

- `to_warp_particle_data()`
- `from_warp_particle_data()`
- `to_warp_gas_data()`
- `from_warp_gas_data()`
- `to_warp_environment_data()`
- `from_warp_environment_data()`

Optional Warp availability can be checked with `WARP_AVAILABLE`.

For the direct-kernel example path, keep the import split explicit:

- Import direct step functions from `particula.gpu.kernels`.
- Import `WARP_AVAILABLE`, `to_warp_*`, and `from_warp_*` helpers from
  `particula.gpu`.
- Do not expect top-level `particula.gpu` to re-export
  `condensation_step_gpu` or `coagulation_step_gpu`.

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
| `charge` | `(n_boxes, n_particles)` | `(n_boxes, n_particles)` | Preserves dimensionless elementary-charge counts without hidden conversion. |
| `density` | `(n_species,)` | `(n_species,)` | Shared species density remains shared; it does not gain a box axis. |
| `volume` | `(n_boxes,)` | `(n_boxes,)` | Per-box simulation volume stays particle-owned across the helper boundary. |

As with the gas and environment helpers, kernels and runnables do not perform
hidden CPU↔GPU synchronization or implicit container transfers for particle
state.

For direct GPU coagulation, `WarpParticleData.charge` is caller-owned,
device-resident particle state, not a sidecar or hidden transfer result. It
must be a same-device `wp.float64` array matching shape
`(n_boxes, n_particles)`. Charged-containing execution scans charge for finite
values before caller-output validation/allocation, RNG setup, or selector/apply
work. Brownian-only execution scans it only with `validate_charge_finite=True`.

When a collision is accepted, the recipient receives the donor charge and the
donor charge is cleared together with donor mass and concentration. Supported
evidence conserves charge independently in each box. This merge bookkeeping
skips a collision whose finite charge sum cannot be represented as finite
`float64`, leaving both particles unchanged rather than producing infinity.
Before recording a selected collision or removing its candidates from the
active set, selection also verifies that every merged species-mass component
and the merged charge remain finite and nonnegative where applicable. The
apply pass repeats those checks defensively; rejected entries are compacted
using only the populated collision count, so unused capacity in an oversized
caller buffer is never interpreted as a collision.

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
| `concentration` | Mass concentration in `kg/m^3`, `(n_boxes, n_species)` | Authoritative active-device `wp.float64` mass concentration in `kg/m^3`, `(n_boxes, n_species)` | Round-trips without shape drift. Direct condensation mutates the GPU field in place. |
| `partitioning` | `bool`, shape `(n_species,)` | Active-device binary `wp.int32`, shape `(n_boxes, n_species)` | CPU masks expand per box as `bool → int32`; kernels index `[box, species]`. CPU restore converts back to `bool` and requires every box to agree. |
| `vapor_pressure` | Not owned by `GasData` | `(n_boxes, n_species)` | GPU-only helper state. Pass it explicitly when needed; CPU restore always drops it. |

This makes GPU→CPU gas restore intentionally lossy unless the caller preserves
ordered names and any vapor-pressure sidecar state outside `WarpGasData`.
`concentration` remains the authoritative gas inventory while a direct kernel
is running. In contrast, `vapor_pressure` is derived GPU-only helper state and
is intentionally dropped by CPU restore; it is not an alternate concentration
field or a source of hidden CPU↔Warp synchronization.

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

