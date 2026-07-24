---
title: Data Containers and GPU Foundations
---

# Data Containers and GPU Foundations

This page is the canonical reference for Particula's shipped data-container
schemas, leading-axis shape conventions, explicit CPU↔GPU transfer helpers,
the direct-kernel troubleshooting contract, and current CPU/GPU support
boundaries.

For a runnable walkthrough of the shipped single-box container construction and
optional Warp CPU-backend round trips, see the
[Data Containers example](../Examples/Data_Containers/index.md) and run:

```bash
python docs/Examples/data_containers_and_gpu_foundations.py
```

For the low-level direct-kernel path, use the canonical quick-start at
[`docs/Examples/gpu_direct_kernels_quick_start.py`](../Examples/gpu_direct_kernels_quick_start.py).
That example keeps direct step imports explicit through `particula.gpu.kernels`
while top-level `particula.gpu` stays focused on `WARP_AVAILABLE` and the
`to_warp_*` / `from_warp_*` transfer helpers.

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

## CPU slot diagnostics

`get_slot_diagnostics()` is the public CPU-only, read-only discovery API for
fixed particle-resolved `ParticleData` storage:

```python
from particula.particles import get_slot_diagnostics

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
from particula.gpu.kernels import activate_slots_gpu

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

`get_slot_diagnostics_gpu` remains a concrete-module-only P3 helper. Import it
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

## Canonical container schemas

### `ParticleData`

`ParticleData` owns particle-side stored state. Per-box arrays always keep a
leading `n_boxes` axis.

| Field | Shape | Meaning |
| --- | --- | --- |
| `masses` | `(n_boxes, n_particles, n_species)` | Authoritative per-particle, per-species masses. |
| `concentration` | `(n_boxes, n_particles)` | Per-box particle concentration or count, depending on workflow. |
| `charge` | `(n_boxes, n_particles)` | Per-box particle charge state as dimensionless elementary-charge counts. |
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

### Direct GPU dilution

Import the direct dilution operation from the low-level kernel namespace:

```python
from particula.gpu.kernels import dilution_step_gpu
```

This is distinct from the CPU↔Warp transfer helpers in `particula.gpu`.
Callers own conversion, device placement, synchronization and checkpoint
transfer, and the `WarpParticleData` and `WarpGasData` instances. The kernel
does not provide hidden transfer or CPU fallback.

`dilution_step_gpu` is a fixed-shape, caller-owned in-place operation. It
returns the same particle and gas objects and changes only their
`concentration` fields; masses and every other caller-owned particle or gas
field remain unchanged. It applies `alpha = Q / V` in `s^-1` and
`c_new = c * exp(-alpha * time_step)`. The coefficient is either a finite,
nonnegative scalar or a caller-owned, same-device `wp.float64` Warp array with
shape `(n_boxes,)`.

Entry preflight is ordered and read-only. Rejected calls are atomic before
kernel launch. A zero scalar coefficient or zero time step completes full
preflight and is a write-free, no-update-kernel no-op; preflight validation
scans may still allocate or launch. The contract does not promise rollback
after a launched-kernel failure.

E6-F1 supplies the upstream CPU finite-step oracle; E6-F9 is the planned
integrated direct-call consumer. Independent NumPy comparisons of particle and
gas concentrations run on Warp CPU with float64 `rtol=1e-12, atol=0`; CUDA is
optional and skips cleanly when unavailable. This tolerance-based evidence is
not bitwise parity.

### Direct GPU wall loss

The bounded direct wall-loss path is particle-resolved. It supports neutral and
charged execution. Import the step and its concrete-module-only configuration
separately; the configuration is not
re-exported by `particula.gpu.kernels` or `particula.gpu`:

```python
from particula.gpu.kernels import wall_loss_step_gpu
from particula.gpu.kernels.wall_loss import NeutralWallLossConfig

config = NeutralWallLossConfig(
    geometry="spherical",
    wall_eddy_diffusivity=1.0e-4,  # m^2/s
    chamber_radius=0.5,  # m
)
wall_loss_step_gpu(
    particles,
    temperature=298.15,  # K
    pressure=101325.0,  # Pa
    time_step=1.0,  # s
    config=config,
    rng_states=rng_states,
    initialize_rng=True,
)
```

The snippet is illustrative: `particles` and `rng_states` are caller-owned
same-device Warp state. Later calls reuse `rng_states` and omit
`initialize_rng=True`. Direct `temperature` and `pressure` each accept a scalar
or same-device `(n_boxes,)` Warp array. Alternatively, pass
`environment=...` and set both `temperature=None` and `pressure=None`; direct
values and `environment=` are mutually exclusive.

Wall eddy diffusivity is in m²/s, dimensions are in m, temperature is in K,
pressure is in Pa, and time is in s. Spherical configurations require a
positive radius and no dimensions; rectangular configurations require no radius
and exactly three positive dimensions. The neutral coefficients use the
Crump--Seinfeld spherical and rectangular chamber relations, including turbulent
deposition and gravitational settling:

- Crump, J. G., & Seinfeld, J. H. (1981), *Journal of Aerosol Science*, 12(5).
  https://doi.org/10.1016/0021-8502(81)90036-7
- Crump, J. G., Flagan, R. C., & Seinfeld, J. H. (1982), *Aerosol Science and
   Technology*, 2(3), 303--309.
   https://doi.org/10.1080/02786828308958636

`NeutralWallLossConfig` remains the sole concrete configuration type. Its
`mode` is either `"neutral"` or `"charged"`. Charged mode validates a finite,
signed scalar `wall_potential` in V. For spherical geometry,
`wall_electric_field` is a finite, signed scalar in V/m. For rectangular
geometry, it is a caller-owned, same-device `wp.float64` Warp array of shape
`(3,)`, with finite signed components in V/m. The boundary does not replace or
mutate that rectangular field array.

For charged mode, slots with finite nonzero charge use the private
image-charge and electric-field-drift coefficient composition. Image-charge
enhancement is active even when `wall_potential` is zero. Zero-charge slots
retain the exact neutral coefficient and RNG path, including when the charged
configuration supplies nonzero potential or electric field.

After read-only preflight, eligible finite positive-rate fixed slots survive with
`exp(-k * time_step)`. A selected slot has every mass lane, concentration, and
charge cleared. Density, volume, dtype, device, capacity, and unselected
storage are preserved; inactive or unusable slots are neither sampled nor
reactivated. The asynchronous call mutates caller-owned state in place. Callers
own CPU↔Warp transfer, device placement, synchronization, and any checkpoint.
There is no hidden CPU checkpoint transfer or fallback. Rejected pre-launch
calls do not mutate caller-owned state, while rollback is not promised after a
mutation kernel launches. Zero time completes preflight but is write-free.

Omitted RNG state is private to each successful nonzero call. Supplied
same-device `(n_boxes,)` `wp.uint32` state mutates in place;
`initialize_rng=True` is the only reset, so repeating `rng_seed` alone does not
reset persistent state. Consumption is sequential per box over eligible finite
positive-rate slots; positive infinite-rate removal is deterministic and
consumes no draw. Exact CPU/Warp or per-seed RNG replay is not promised.

| Scope | Status |
| --- | --- |
| Neutral and charged particle-resolved spherical/rectangular direct execution | Supported |
| Charged nonzero-charge image/electric-field composition; zero-charge neutral fallback | Supported |
| Charged configuration and rectangular field ownership semantics | Supported |
| Geometry-specific deterministic coefficient tolerances and P5 eight-stratum exact-binomial survival evidence (4,096 observations per stratum; inclusive bounds with per-stratum alpha 1.25e-7) | Supported bounded evidence |
| CUDA validation | Optional additive evidence; guarded skips are expected when unavailable |
| CPU fallback or hidden transfer; high-level runnable, scheduler, or backend integration | Deferred (E6-F9 covers integration/closeout) |
| Dynamic slots, compaction/activation, graph capture, differentiability, performance guarantees, or exact RNG replay | Deferred |

Warp CPU is the baseline when Warp is installed; CUDA is optional and skips
cleanly when unavailable. The evidence is not CPU-strategy parity or per-seed
trajectory replay.

GPU process orchestration, backend selection and scheduling, GPU-resident
timestep integration, resizing, graph capture, autodiff, performance claims,
and nucleation remain deferred.

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

### GPU coagulation configuration and sidecar ownership

Import the supported direct step and its host-side configuration separately:

```python
from particula.gpu.kernels import coagulation_step_gpu
from particula.gpu.kernels.coagulation import CoagulationMechanismConfig
```

Pass immutable `mechanism_config=CoagulationMechanismConfig(...)` as keyword-only
host metadata. The frozen dataclass is not re-exported by
`particula.gpu.kernels`, is not device-resident simulation state, and neither
transfers nor owns Warp state. In contrast,
`WarpParticleData`, supplied `collision_pairs`, supplied `n_collisions`, and
supplied `rng_states` are caller-owned, same-device Warp resources. Omitted
collision outputs and RNG state use call-local convenience allocation. A
supplied RNG sidecar is reused and changes in place; it resets only with
`initialize_rng=True`. Configuration and caller-owned sidecars have no implicit
CPU↔GPU transfer.

Only `distribution_type="particle_resolved"` is accepted. `"discrete"` and
`"continuous_pdf"` raise `ValueError`; they do not fall back or convert.
Malformed configuration and an unsupported distribution fail before particle
access. Mask `7` fails at capability preflight before particle metadata or
enabled-term validation. Masks `11`, `13`, and `14` validate particle metadata
and enabled terms before they fail closed. These read-only preflight boundaries
do not make a later runtime failure transactional: callers retain their mutable
state, and no rollback is guaranteed after such a failure.

| Identifier | Bit | Status |
| --- | --- | --- |
| `brownian` | `1` | Executable singleton |
| `charged_hard_sphere` | `2` | Executable singleton |
| `sedimentation_sp2016` | `4` | Executable singleton |
| `turbulent_shear_st1956` | `8` | Executable singleton |
| `brownian + charged_hard_sphere` | `3` | Executable two-way mask |
| `brownian + sedimentation_sp2016` | `5` | Executable two-way mask |
| `charged_hard_sphere + sedimentation_sp2016` | `6` | Executable two-way mask |
| `brownian + turbulent_shear_st1956` | `9` | Executable two-way mask |
| `charged_hard_sphere + turbulent_shear_st1956` | `10` | Executable two-way mask |
| `sedimentation_sp2016 + turbulent_shear_st1956` | `12` | Executable two-way mask |
| `brownian + charged_hard_sphere + sedimentation_sp2016 + turbulent_shear_st1956` | `15` | Executable four-way mask |
| `brownian + charged_hard_sphere + sedimentation_sp2016` | `7` | Rejected three-way mask |
| `brownian + charged_hard_sphere + turbulent_shear_st1956` | `11` | Rejected three-way mask |
| `brownian + sedimentation_sp2016 + turbulent_shear_st1956` | `13` | Rejected three-way mask |
| `charged_hard_sphere + sedimentation_sp2016 + turbulent_shear_st1956` | `14` | Rejected three-way mask |

The executable singleton masks are `1`, `2`, `4`, and `8`; executable two-term
masks are `3`, `5`, `6`, `9`, `10`, and `12`; and executable four-term mask is
`15`. Deferred masks are exactly `7`, `11`, `13`, and `14`.

Requested tuples normalize to the canonical Brownian, charged, sedimentation,
then turbulent order. Exact SP2016 sedimentation is selected with
`CoagulationMechanismConfig(("sedimentation_sp2016",))`. Its shipped pair rate
is `K = π (r_i + r_j)^2 |v_i - v_j|` in m³/s, where settling velocities use
Stokes settling with Cunningham slip correction (Seinfeld & Pandis, 2016,
Eq. 13A.4). Collision efficiency is fixed at 1; no public efficiency argument
exists. This mode uses fp64 particle mass state, validates finite particle
charge, finite/nonnegative mass and concentration, plus finite/positive density
before mutable work, and requires equal positive particle concentrations within
each box for representable inventory-preserving merges. Non-finite charge is
rejected during host-side preflight before caller-owned outputs, persistent RNG,
or particle state can be mutated.

The exact turbulent-shear singleton is selected with
`CoagulationMechanismConfig(("turbulent_shear_st1956",))`; approved mixed masks
that include this term use the same inputs. Keyword-only `turbulent_dissipation`
is in `m^2/s^3` and `fluid_density` is in `kg/m^3`. Both are positive finite
Python or NumPy floating scalars, or active-device `wp.float64` Warp arrays
shaped `(n_boxes,)`; supplied arrays retain identity and scalars use private
device-local broadcast/property storage. They are required for structurally
valid turbulent requests, including rejected three-way requests during
enabled-term validation, and ignored by non-turbulent masks. They are not
inferred from a container: NumPy arrays, Python lists, and hidden host-to-device
conversion are not supported as array inputs.

Supply temperature and pressure either as direct scalars, supported-float Warp
arrays of shape `(n_boxes,)` on the particle device, or same-device
`WarpEnvironmentData` with both direct inputs set to `None`. These thermodynamic
inputs are not required to be fp64. Caller-owned particle, collision-output, and
RNG resources retain their documented dtypes and identity. Helper-owned,
call-local fp64 work buffers hold per-particle properties as needed; scalar
turbulence inputs use private device-local `(n_boxes,)` broadcast/property
storage, while valid supplied arrays retain identity. Exact SP2016 sedimentation
uses settling storage only for that mechanism; ST1956 has no dedicated
settling-velocity buffer.
It is a direct-kernel path only: it establishes neither CPU-strategy parity nor
general accuracy or performance claims.

Approved masks sanitize and sum fp64 component rates. The safe summed
component-majorant bound satisfies `sum_m K_m(i, j) <= sum_m M_m`; component
maxima can occur on different pairs, so that bound can be conservative.
Production selection obtains its majorant by an exact compact-active scan of the
summed pair rate. One shared active set, candidate stream, acceptance stream,
collision-buffer set, per-box RNG stream, and apply pass serve each call;
mechanisms are not sequential steps. Invalid, nonfinite, or nonpositive terms do
not add to totals, and invalid candidates do not mutate particle or
collision-output state. Persistent RNG state can advance before an invalid-rate
rejection. This documents bounded implementation scope, not throughput or
scaling evidence.
Read-only preflight can use a private device-status buffer and a bounded
synchronization/readback to report invalid caller state without copying,
mutating, or CPU-falling-back caller simulation state.

The return tuple is exactly `(particles, collision_pairs, n_collisions)`; RNG
state is not returned. Accepted collisions mutate caller-owned particle mass,
concentration, and charge in place. Supplied collision buffers are returned by
identity. Supplied `rng_states` are caller-owned same-device Warp state, are
reused and advanced in place, and reset only when `initialize_rng=True`
explicitly opts in; omitted buffers and RNG state are call-local conveniences.
Sedimentation-specific read-only preflight fails before output allocation, RNG
initialization, or particle mutation, but does not make unrelated later runtime
failures atomic. Transfers are explicit. Warp CPU is the baseline when Warp is
installed. CUDA is optional additive evidence and guarded suites skip cleanly
when unavailable. This direct path provides no high-level `Runnable`
integration, automatic transfer, CPU fallback, mandatory CUDA requirement, or
general-turbulence support.

Future mechanisms must provide a stable identifier; required host and device
inputs; property preparation; a sanitized additive pair-rate term; a proven-safe
additive majorant; a capability-table row; shared-dispatcher integration; and
co-located `*_test.py` coverage. They must not add an independent sampling
loop, acceptance pass, collision buffer, or RNG stream. This boundary excludes
binned/discrete/continuous-PDF GPU coagulation, high-level `Runnable`
integration, graph-capture claims, alternate mechanisms, DNS or general
turbulence, CPU fallback, hidden transfer, and performance or broad-accuracy
claims. See the [validation/evidence record](Roadmap/coagulation-validation.md)
for the focused fixed-mask and stochastic command record.

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

### GPU thermodynamics and condensation refresh

Import the supported direct step only with
`from particula.gpu.kernels import condensation_step_gpu`.
Import its required `ThermodynamicsConfig` from the concrete module only:

```python
from particula.gpu.kernels.thermodynamics import ThermodynamicsConfig
```

`ThermodynamicsConfig` is caller-owned, device-local process configuration.
Its Warp arrays have no species names: species identity is positional, and
`molar_mass_reference` must exactly equal ordered `gas.molar_mass` on the
active device. `WarpGasData.vapor_pressure`, in contrast, is a mutable,
derived GPU helper buffer rather than authoritative `GasData` state. It is
omitted by `from_warp_gas_data()`.

The fixed configuration and refresh contracts are:

| Field | Required Warp dtype and shape | Meaning |
| --- | --- | --- |
| `modes` | `wp.int32`, `(n_species,)` | Per-species model code. |
| `parameters` | `wp.float64`, `(n_species, 4)` | Per-species model parameters. |
| `molar_mass_reference` | `wp.float64`, `(n_species,)` | Ordered compatibility fingerprint for `gas.molar_mass`. |
| Refresh temperature | `wp.float64`, `(n_boxes,)` | Device-local temperature at the standalone refresh boundary. |
| `vapor_pressure` output | `wp.float64`, `(n_boxes, n_species)` | Derived pressure matrix in Pa. |

Only two vapor-pressure model codes are shipped: constant `wp.int32(0)`, which reads
`parameters[:, 0]` in Pa, and canonical Buck `wp.int32(1)`, which evaluates
the water/ice equations and ignores all four reserved parameter slots.

`condensation_step_gpu()` requires keyword-only `thermodynamics=` while
preserving the positional `mass_transfer` slot. Its accepted temperature
sources are a direct scalar, a direct same-device Warp array with shape
`(n_boxes,)`, hybrid direct scalar/Warp-array inputs, or keyword-only
`environment=` when both direct values are `None`. NumPy arrays and Python
lists are not accepted as direct arrays, and the step does not perform hidden
transfers. Temperature and pressure must be positive finite values. Direct and
environment temperature/pressure arrays are active-device `(n_boxes,)` inputs;
non-`wp.float64` temperature arrays are normalized into a device-local fp64
buffer. This normalization does not make every environment input fp64-only.

On every successful condensation call, the step validates aggregate caller
state and optional sidecars first. It then casts a non-`wp.float64` direct or
environment temperature array into a device-local float64 buffer when needed
and performs exactly four equal `time_step / 4.0` substeps. Each substep
overwrites `vapor_pressure`, prepares environment properties, gates disabled
partitioning entries and inactive particle slots, and creates a raw proposal.
P2 finalizes that proposal against particle and gas inventory limits, then
derives the matching concentration-weighted opposite gas delta. The step
validates that delta and all pending commit values before mutating particle
masses, accumulating the finalized transfer in the returned total-transfer
buffer, and mutating `gas.concentration`. Raw proposal work storage is
intermediate state, not returned transfer. Later proposals read the coupled gas
concentration; vapor-pressure refresh does not. A failed aggregate preflight
leaves the vapor-pressure output buffer unchanged. Import the
standalone `refresh_vapor_pressure_gpu` only from
`particula.gpu.kernels.thermodynamics`, never `particula.gpu.kernels`.

### Condensation activity and surface sidecars

The optional `CondensationActivitySurfaceConfig` is a direct-kernel-only,
caller-owned configuration sidecar for `condensation_step_gpu()`. Activity mode
`0` is ideal and mode `1` is kappa; surface-tension mode `0` is static and mode
`1` is composition-weighted. Activity applies only to `water_species_index`;
all non-water species use unit activity. Static tension uses the current
species' supplied value, while composition-weighted tension computes one
particle-wide value from the per-species tensions.

`kappas` and `molar_mass_reference` must be same-device `wp.float64` arrays of
shape `(n_species,)`. Kappas must be finite and nonnegative, and ordered molar
masses must exactly match `gas.molar_mass`; this preserves the positional
ordered-molar-mass compatibility contract. The frozen dataclass prevents field
rebinding, not mutation of its arrays. Callers retain those arrays and must not
mutate them concurrently with a launch.

Successful direct calls mutate particle masses and gas concentration in place,
overwrite the derived GPU-only vapor-pressure buffer, and return the accumulated
**P2-finalized, inventory-limited** transfer. A supplied total-transfer buffer
holds that finalized accumulated output and is returned by identity; separate
supplied work storage retains only the final gated raw proposal and is never the
returned transfer. The two-item return assignment is
`particles_out, mass_transfer = condensation_step_gpu(...)`;
`energy_transfer` is caller-owned output, not a third return value. Disabled
`(box, species)` entries in the
per-box `(n_boxes, n_species)` partitioning mask and inactive particle slots
are zeroed before P2 finalization or reductions. CPU↔Warp conversion expands
the CPU shared species mask to this per-box layout. `to_per_box_partitioning()`
is the public migration helper for legacy one-dimensional masks. CPU restoration
rejects masks that differ by box and rejects zero-box mirrors because no box can
authoritatively supply the shared CPU mask.
P2 demand, release, scale, and accumulator sidecars are validated and used for
finalization; aggregate invalid state, metadata, or ownership fails with
`ValueError` before mutable physics or caller-state mutation. Preflight may run
read-only validation kernels. A later failure caused by a fresh raw proposal
does not roll back completed substeps, although it occurs before P2 mutation in
its failing substep. Unsupported activity or surface
strategies are not silently copied or approximated; passing
`activity_surface=None` retains legacy unit activity and static tension.

`CondensationScratchBuffers` is a concrete-module-only optional sidecar at
`particula.gpu.kernels.condensation`; it is not another step entry point.
Particle mass, transfer, and scratch transfer arrays are active-device
`wp.float64` with shape `(n_boxes, n_particles, n_species)`; gas concentration
and energy arrays are active-device `wp.float64` with shape
`(n_boxes, n_species)`; scratch property arrays are active-device `wp.float64`
with shape `(n_boxes,)`; and latent heat and thermal work arrays are
active-device `wp.float64` with shape `(n_species,)`. Supplied
transfer fields must be active-device, stable-shape `wp.float64` arrays with
shape `(n_boxes, n_particles, n_species)`, and supplied property fields must be
active-device, stable-shape `wp.float64` arrays with shape `(n_boxes,)`.
The P2 demand, release, and scale sidecars must likewise be caller-owned,
active-device, stable-shape `wp.float64` arrays with shape
`(n_boxes, n_species)`. Every supplied field is metadata-validated for its
applicable shape, dtype, and active device before launch; each may be omitted
independently, in which case the step allocates only that field's fallback
storage. Supplied fields preserve identity and remain caller-owned, but
successful calls intentionally write their mutable work/output storage. No new
container field, sidecar API, host conversion, synchronization, or transfer is
implied by this ownership contract.

The direct condensation parity suite exercises all four activity/surface pairs
(ideal/static, ideal/composition-weighted, kappa/static, and
kappa/composition-weighted) against an independent NumPy `float64` reference.
It uses deterministic one-box and multi-box inputs, including pure and mixed
particle compositions, a clamp-to-zero evaporation case, and constant plus
both Buck vapor-pressure branches. Parity checks cover raw transfer, final
particle mass, refreshed vapor pressure, finalized totals, and coupled gas
concentration. Tighter conservation checks separately verify the weighted
particle transfer equals the opposing gas change. These checks remain a bounded
direct-step contract, not a general CPU-strategy parity claim.

### Latent-rate correction and energy diagnostic

The direct step accepts optional keyword-only caller-owned active-device
`wp.float64` `latent_heat` with shape `(n_species,)`. For species `i`, its
shipped latent-rate correction is
`dm_i/dt = isothermal_rate_i / correction_i`, where
`isothermal_rate_i = k_i * Delta_p_i * M_i / (R * T)`,
`R_specific_i = R / M_i`, and
`correction_i = thermal_factor_i / (R_specific_i * T)`. The source thermal
factor is
`thermal_factor_i = (D_i * L_i * p_surface_i / (T * k_thermal)) * (L_i / (T * R_specific_i) - 1) + R_specific_i * T`.
Here `p_surface_i` is the activity- and Kelvin-adjusted surface vapor pressure.
An omitted sidecar, or exactly `L_i = 0`, follows the exact isothermal branch.

Optional keyword-only `energy_transfer` is a caller-owned active-device
`wp.float64`, `(n_boxes, n_species)`, write-only output. It requires valid
`latent_heat`, is overwritten only after successful preflight, and is not a
third return value. Its signed whole-call diagnostic, shipped in issue #1272,
is `Q[box, species] = sum_particles(Delta m_applied) * L[species]`: `Delta m`
is applied transfer in kg, `L` is latent heat in J/kg, and `Q` is J. Thus
condensation is positive and evaporation is negative. The strict #1272
energy-regression tolerance is `rtol=1e-12, atol=1e-18`.

`thermal_work` has the same validated sidecar shape but remains deferred and
unused: the step neither reads nor writes it. It does not evolve temperature or
add a `Runnable`, adaptive substeps, graph capture/replay, or autodiff.

Use `to_warp_*` and `from_warp_*` as the sole data CPU↔Warp boundary. There is
no high-level `Aerosol` or `Runnable` GPU path. CUDA preflight reads one-element
device validation flags with `.numpy()`; those status reads synchronize but do
not transfer or mutate caller-owned simulation buffers. Callers requiring retry
or rollback must snapshot and restore particle masses, gas concentration,
derived vapor pressure, and caller-owned output/work buffers. Deterministic
fp64 tests compare an independent NumPy reference with explicit parity
`rtol`/`atol`, then apply tighter separate ownership invariants. Warp `cpu` is
the baseline whenever Warp is installed; CUDA is optional, availability-guarded,
and separately marked. Warp-dependent test imports are lazy, so marker-
deselected collection does not require Warp. Focused direct-kernel and CPU
integration smoke commands are:

```bash
pytest particula/gpu/kernels/tests/condensation_test.py --collect-only -q -m "not warp"
pytest particula/gpu/kernels/tests/condensation_test.py -q -Werror
pytest particula/integration_tests/condensation_particle_resolved_test.py -q -Werror
```

Run CUDA-marked kernel cases separately only on CUDA-capable hosts; unavailable
Warp or CUDA remains an expected guarded skip, not evidence of a fallback path.

## Current shipped support boundaries

The containers are multi-box capable, but current execution support is narrower
than storage support.

| Area | Shipped support | Notes |
| --- | --- | --- |
| CPU `ParticleData` / `GasData` storage | Multi-box-capable | Leading `n_boxes` axis is part of the stored schema. |
| CPU condensation with data containers | `n_boxes == 1` only | Multi-box CPU execution still requires a caller-managed per-box loop. |
| CPU coagulation with data containers | `n_boxes == 1` only | Multi-box CPU execution is not yet a built-in runtime path. |
| CPU↔GPU transfer | Explicit helper calls only | No hidden container movement or hidden environment synchronization. |
| Warp/CUDA support | Optional | Warp `device="cpu"` is the baseline when Warp is installed; CUDA is additive local evidence and unavailable devices skip cleanly. |
| Low-level GPU slot activation | Shipped fixed-capacity direct-kernel contract | `activate_slots_gpu` maps selected request prefixes to ascending free slots with caller-owned same-device sidecars; no hidden transfer, resizing, or high-level runnable is provided. |
| Low-level GPU condensation direct-kernel path | Shipped bounded direct-kernel contract | Executes four fixed coupled substeps with active-device P2 inventory and gas coupling. This is direct-kernel evidence, not broad GPU-condensation support. |
| Low-level GPU coagulation direct-kernel path | Direct, particle-resolved direct-kernel contract | Exact mask and rejection boundaries follow below. This path establishes no Runnable support, CPU parity, or performance claim. |
| Fixed-shape GPU/runtime roadmap work | Not current runtime behavior | Graph-capture-oriented and fixed-shape runtime constraints remain roadmap handoff material, not shipped behavior. |

Additional shipped boundaries:

- Executable coagulation masks are singletons `1`, `2`, `4`, `8`; unordered
  pairs `3`, `5`, `6`, `9`, `10`, `12`; and four-way mask `15`. Tuples
  normalize to canonical order. Non-turbulent three-way mask `7` rejects at
  capability preflight before particle metadata or enabled-term validation.
  Turbulent three-way masks `11`, `13`, `14` proceed through particle metadata
  and enabled-term validation, then reject before downstream normalization,
  allocation, RNG setup, kernel launch, or mutation. Turbulent masks require
  explicit turbulent inputs.
- Supplied coagulation particle state, collision outputs, and persistent RNG are
  caller-owned same-device Warp resources. Persistent RNG reuse has no implicit
  transfer or synchronization; omitted state allocates call-local storage, and
  `initialize_rng=True` explicitly resets the caller-owned buffer.
- `ParticleData.volume` remains the authoritative per-box simulation-volume
  owner; it does not move into `EnvironmentData`.
- `EnvironmentData` is the shipped CPU thermodynamic owner for `temperature`,
  `pressure`, and `saturation_ratio`.
- `WarpGasData.vapor_pressure` is helper state only and has no CPU `GasData`
  field.
- Coagulation `mechanism_config` is host metadata, while supplied
  `collision_pairs`, `n_collisions`, and `rng_states` are caller-owned
  same-device Warp sidecars. None are fields on `ParticleData`, `GasData`,
  `EnvironmentData`, or any Warp container schema.
- Omitted coagulation collision outputs and RNG state are call-local convenience
  allocations. Supplied RNG state advances in place and resets only with
  `initialize_rng=True`.
- Kernels and runnables do not perform hidden simulation-state transfers.
  Explicitly supplied step and output buffers do mutate in place: particle
  masses, gas concentration, scratch transfer fields, total transfer, and
  energy output retain their documented write semantics.
  Callers remain responsible for synchronization before
  host observation or restoration.
  Validation `.numpy()` readbacks at their documented preflight and per-substep
  boundaries may synchronize without transferring simulation state.
  CUDA preflight validation-flag readbacks may
  synchronize without transferring simulation state.
- Direct condensation does not add a high-level `Aerosol`/`Runnable` path,
  automatic backend selection or fallback, implicit transfer or synchronization,
  adaptive stepping, new container fields, kernels, or physics, BAT, or
  staggered/Gauss-Seidel support. It also does not claim general CPU-strategy
  parity or broader unproven physics support.

## Guidance for current users

- Use package-level imports for public containers and helpers.
- Preserve the leading box axis in all per-box arrays, even for single-box
  workflows.
- Preserve ordered gas names outside GPU containers if you need a semantic
  gas round trip.
- Preserve or recompute vapor pressure separately on the CPU side after GPU
  restore.
- Use the current low-level GPU coagulation path only within its documented
  direct-kernel workflow on a Warp-supported device.
- The current low-level GPU coagulation path remains bounded direct-kernel
  implementation scope; it does not establish a general production-support
  claim beyond its documented helper boundary and supported inputs.
- Do not expect kernels or runnables to perform hidden CPU↔GPU transfers for
  particle, gas, or environment state; use the explicit helper calls when state
  must cross the device boundary. Read-only device validation may synchronize
  and read back status to surface invalid values.
- Treat Warp and CUDA as optional runtime capabilities: without Warp, this
  low-level GPU path is unavailable. Warp `device="cpu"` is the baseline;
  CUDA support is optional and tests skip cleanly when CUDA is unavailable.
- Treat roadmap pages as future-work references, not as evidence that broader
  runtime support has already shipped.

## Direct-kernel troubleshooting

Use
[`docs/Examples/gpu_direct_kernels_quick_start.py`](../Examples/gpu_direct_kernels_quick_start.py)
as the canonical runnable reference when troubleshooting the low-level GPU
path.

- **Warp missing (`WARP_AVAILABLE == False`)**
  - The direct-kernel path is unavailable until Warp is installed.
  - The canonical quick-start keeps `particula.gpu.kernels` imports deferred
    and completes in a CPU-only documentation mode without importing kernel
    steps or pretending kernels ran.
- **CUDA unavailable**
  - CUDA is optional.
  - The default supported runnable path is Warp `device="cpu"`; only opt into
    `device="cuda"` when a CUDA device is actually available.
- **Explicit CPU↔GPU transfer boundary**
  - Kernels operate on Warp-backed container mirrors, not CPU `ParticleData`,
    `GasData`, or `EnvironmentData` objects directly.
  - Move simulation state across the boundary explicitly with `to_warp_*` and
    `from_warp_*`; kernels and runnables do not perform hidden simulation-state
    transfers or top-level fallback imports. Read-only device validation may
    synchronize and read back status to surface invalid values, including on
    CUDA.
- **Device mismatch across particle, gas, environment, and sidecar buffers**
  - Keep Warp arrays and sidecar buffers such as `rng_states` on the same
    device as the particle/gas/environment inputs used by the kernel call.
  - Treat device-mismatch `ValueError` failures as input validation, not as a
    signal that Particula will migrate arrays automatically.
- **Condensation scratch buffers**
  - Preserve leading `(n_boxes, ...)` layouts. Keep supplied
    `CondensationScratchBuffers` fields on the active device with their
    required stable `wp.float64` shapes: transfer fields use
    `(n_boxes, n_particles, n_species)` and property fields use `(n_boxes,)`.
    Only omitted fields allocate fallback storage independently.
  - Supplied fields preserve caller ownership and identity but are mutable
    work/output storage; do not expect a successful step to leave them
    unmodified.
- **Mixed `environment=` plus scalar `temperature`/`pressure` inputs**
  - Pass either `environment=` or direct temperature/pressure inputs.
  - Do not mix scalar or Warp-array `temperature` / `pressure` values with
    `environment=`; current kernels reject that combination explicitly.
  - When `environment=` is absent, provide both direct inputs. They must be
    positive finite physical values and any direct Warp arrays must be
    compatible with the active device.
- **Species metadata and thermodynamics sidecars**
  - Preserve ordered CPU gas-name metadata when restoring `GasData`, and keep
    thermodynamics-sidecar species order aligned with `gas.molar_mass`.
  - Confirm the configured water-species index addresses a valid species.
- **P2 transfer and energy diagnostics**
  - P2 inventory-limited applied transfers are expected bounded behavior, not
    CPU-strategy or runnable parity evidence.
  - `energy_transfer` is caller-owned whole-call P2-finalized output. Explicitly
    synchronize before observing it on the host.
- **Gas/environment restore expectations**
  - `from_warp_gas_data()` is intentionally lossy across the helper boundary:
    it restores ordered species names only when you supply them; otherwise it
    generates placeholder names such as `species_0`.
  - GPU-only helper state such as `vapor_pressure` is not restored onto CPU
    `GasData`.
  - `from_warp_environment_data(..., sync=False)` is an explicit expert path;
    manual synchronization remains the caller's responsibility before NumPy
    access.

### Focused reproduction commands

These focused commands provide distinct evidence; none establishes either of
the other evidence classes.

| Command | Evidence |
| --- | --- |
| `python docs/Examples/gpu_direct_kernels_quick_start.py` | Canonical explicit-transfer walkthrough. |
| `pytest particula/gpu/tests/gpu_direct_kernels_example_test.py -q` | Quick-start regression. |
| `pytest particula/gpu/kernels/tests/condensation_test.py -q -Werror` | Primary direct CPU-oracle particle-mass/gas-concentration parity matrix. |
| `pytest particula/gpu/kernels/tests/condensation_stiffness_test.py -q -Werror` | Bounded direct-step stiffness coverage. |
| `pytest particula/gpu/kernels/tests/slot_management_test.py -q -Werror` | Fixed-slot activation mapping, caller-owned sidecars, and preflight state-safety coverage. |
| `pytest particula/gpu/dynamics/tests/coagulation_funcs_test.py -q -Werror` | Deterministic GPU coagulation pair-helper parity. |
| `pytest particula/gpu/kernels/tests/coagulation_test.py -q -Werror` | Direct coagulation coverage, including singleton sedimentation and ST1956 configurations, direct/environment inputs, caller-owned output/RNG behavior, conservation, and rejected-call state safety. |
| `pytest particula/gpu/kernels/tests/coagulation_validation_test.py -q -m "warp and gpu_parity" -Werror` | Fixed-mask deterministic/ownership evidence. |
| `pytest particula/gpu/kernels/tests/coagulation_stochastic_validation_test.py -q -m "warp and stochastic and not cuda" -Werror` | Stochastic evidence. |
| `pytest particula/tests/gpu_coagulation_docs_test.py -q -Werror` | Hardware-free documentation-contract coverage. |
| `pytest particula/integration_tests/condensation_latent_heat_conservation_test.py -q` | CPU integration/inventory-conservation evidence (separate particle-plus-gas inventory conservation checks); not direct-GPU validation. |
| `pytest particula/integration_tests/condensation_particle_resolved_test.py -q` | CPU integration evidence for particle-resolved condensation; not direct-GPU validation. |
| `pytest particula/tests/condensation_latent_heat_docs_test.py -q -Werror` | Latent-heat energy/bookkeeping documentation checks. |

The required baseline is Warp `device="cpu"` when Warp is installed. The
parity matrix, inventory conservation checks, and latent-heat energy/bookkeeping
checks are separate: no one class proves either of the others.

**Optional/local CUDA evidence:**

```bash
pytest particula/gpu/kernels/tests/condensation_test.py -q -m "warp and cuda" -Werror
pytest particula/gpu/kernels/tests/coagulation_stochastic_validation_test.py -q -m "warp and cuda" -Werror
```

This is additive to the required Warp `device="cpu"` baseline and skips
cleanly when CUDA is unavailable.

The [GPU condensation parity walkthrough](../Examples/gpu_condensation_parity_walkthrough.py)
and its [condensation parity walkthrough ownership record](Roadmap/condensation-parity-walkthrough.md)
document bounded physics, conservation, and energy evidence only. They retain
the fixed-four-substep low-level direct-kernel boundary: Warp CPU is the
installed-Warp baseline and CUDA is optional additive evidence, not required
support. `energy_transfer` remains caller-owned, write-only diagnostic output,
not a return value or temperature feedback mechanism (`kg * J/kg = J`).

## Related references

The fixed-four-substep low-level direct-kernel walkthrough can be run with
`python docs/Examples/gpu_condensation_parity_walkthrough.py`. Its focused
regressions are
`pytest particula/gpu/tests/gpu_condensation_parity_walkthrough_test.py -q -Werror`
and
`pytest particula/tests/condensation_parity_walkthrough_docs_test.py -q -Werror`.

- [Data Containers example](../Examples/Data_Containers/index.md)
- [Particle & Gas Data Migration](particle-data-migration.md)
- [Data-Oriented Design and GPU Roadmap](Roadmap/data-oriented-gpu.md)
- [Mass Precision Recommendation Report](Roadmap/mass-precision-study.md)
