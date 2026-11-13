# Slots

Run the [container construction](../data-model/index.md#shape-conventions)
first; the code below continues with its `particle_data` object and imports.

## CPU slot diagnostics

`get_slot_diagnostics()` is the public CPU-only, read-only discovery API for
fixed particle-resolved `ParticleData` storage:

```python
import numpy as np
from particula.particles import ParticleData, get_slot_diagnostics

particle_data = ParticleData(
    masses=np.array([[[1e-18, 2e-18], [0.0, 0.0]]]),
    concentration=np.array([[1.0, 0.0]]), charge=np.zeros((1, 2)),
    density=np.array([1000.0, 1200.0]), volume=np.array([1e-6]),
    distribution_type="particle_resolved",
)
free_indices, active_counts, free_counts = get_slot_diagnostics(particle_data)
```

CPU and direct-Warp primitives classify slots identically before diagnostics or
activation:

| Classification | Concentration | Mass lanes and total | Charge |
| --- | --- | --- | --- |
| Active | Finite and positive | Every lane finite and nonnegative; finite, positive total | Finite |
| Free | Exactly zero | Every lane exactly zero | Exactly zero |
| Invalid | Any other state | Any other state | Any other state |

Invalid state raises `ValueError("Invalid particle slot state.")`; diagnostics
are not partially returned. `get_slot_diagnostics` neither mutates nor aliases
particle storage, and it does not activate, resize, compact, or transfer slots
to Warp. It returns freshly allocated NumPy `int32` arrays in the order shown:
`free_indices` has shape `(n_boxes, n_particles)` with ascending free indices
and `-1` tails; `active_counts` and `free_counts` have shape `(n_boxes,)`.

### CPU slot activation

`activate_slots` is a CPU-only direct import, not a `particula.particles`
package export:

```python
from particula.particles.slot_management import activate_slots
```

For each box, request rank `r` is copied to the `r`-th ascending free slot from
`get_slot_diagnostics()`. The function performs global read-only schema,
storage-isolation, slot-state, capacity, and selected-record preflight before
any write. A rejected call therefore leaves particle and request storage
unchanged. Successful calls mutate only caller-owned `masses`, `concentration`,
and `charge`; `density`, `volume`, unselected slots, and request arrays are
preserved. It returns a fresh `np.int32` activated-count vector per box.

### Direct GPU slot activation

Use the package-exported direct boundary when a caller-managed
`WarpParticleData` must fill fixed-capacity free slots without moving state to
the CPU:

```python
import warp as wp
from particula.gpu import to_warp_particle_data
from particula.gpu.kernels import activate_slots_gpu

# Continue the fixed-capacity CPU example above with explicit Warp CPU storage.
particles = to_warp_particle_data(particle_data, device="cpu")
request_masses = wp.array([[[2e-18, 3e-18]]], dtype=wp.float64, device="cpu")
request_concentration = wp.array([[1.0]], dtype=wp.float64, device="cpu")
request_charge = wp.zeros((1, 1), dtype=wp.float64, device="cpu")
requested_counts = wp.array([1], dtype=wp.int32, device="cpu")
activated_counts = wp.zeros(1, dtype=wp.int32, device="cpu")
free_indices = wp.zeros((1, 2), dtype=wp.int32, device="cpu")
active_counts = wp.zeros(1, dtype=wp.int32, device="cpu")
free_counts = wp.zeros(1, dtype=wp.int32, device="cpu")

activated_counts, free_indices, active_counts, free_counts = (
    activate_slots_gpu(
        particles,
        request_masses,
        request_concentration,
        request_charge,
        requested_counts,
        activated_counts,
        free_indices,
        active_counts,
        free_counts,
    )
)
```

For each box, selected request-prefix rank `r` is copied to the `r`-th
ascending free slot. This deterministic fixed-capacity boundary does not
resize, compact, or replace caller buffers. The CPU and GPU schemas are:

| Contract item | CPU | Direct Warp |
| --- | --- | --- |
| Request masses | NumPy `float64`, `(n_boxes, request_capacity, n_species)` | Caller-owned, same-device `wp.float64`, same shape |
| Request concentration / charge | NumPy `float64`, `(n_boxes, request_capacity)` | Caller-owned, same-device `wp.float64`, same shape |
| Requested counts | Any one-dimensional NumPy integer dtype, `(n_boxes,)` | Caller-owned, same-device `wp.int32`, `(n_boxes,)` |
| Diagnostics | Fresh `np.int32`: indices `(n_boxes, n_particles)`, counts `(n_boxes,)` | Caller-owned, same-device `wp.int32` sidecars with those shapes |
| Activation result | Fresh `np.int32` activated counts, `(n_boxes,)` | Identical supplied `(activated_counts, free_indices, active_counts, free_counts)` |

Diagnostic APIs emit ascending `free_indices` rows with `-1` tails. CPU
`activate_slots()` returns only a fresh activated-count array, while direct-Warp
activation returns its caller-owned diagnostic sidecars in the documented
order. On success, activated counts equal requested counts, and the other
diagnostics describe post-activation state. Only records within each requested
prefix are validated or read. Both preserve `density`, `volume`, unselected
slots, and request arrays, and mutate only `masses`, `concentration`, and
`charge` in particle storage. CPU preflight additionally reads `density` and
`volume` to reject protected-field storage aliasing; direct-Warp activation
does not observe those fields.

#### Ownership and failure boundary

This is a low-level direct-kernel operation. Callers own CPU↔Warp transfer,
device placement, synchronization, and all particle, request, and output
buffers; there is no hidden transfer, CPU fallback, or storage resizing.
Direct-Warp activation reads and writes only `masses`, `concentration`, and
`charge`; `density` and `volume` are unobserved and preserved.

Before its writer launches, GPU activation validates metadata/schema/device,
sidecar ownership and aliasing, existing slot state, requested counts, free
capacity, and selected request records. Request records outside each declared
prefix are ignored. A rejected preflight preserves every accessible
caller-owned particle, request, count, and output buffer. Once the writer
launches, a later failure does not promise rollback.

`get_slot_diagnostics_gpu` remains a concrete-module-only full-loop validation helper. Import it
only from its implementation module:

```python
from particula.gpu.kernels.slot_management import get_slot_diagnostics_gpu
```

Do not import it from `particula.gpu.kernels` or `particula.gpu`.

Validate the bounded CPU and direct-kernel contracts with:

```bash
pytest particula/particles/tests/slot_management_test.py -q -Werror
pytest particula/gpu/kernels/tests/slot_management_test.py -q -Werror
```

Warp CPU is the baseline when Warp is installed. CUDA is optional evidence and
the focused Warp suite skips cleanly when CUDA is unavailable.

