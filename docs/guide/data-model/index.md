---
title: Data model
---

# Data model

This page is the canonical reference for the v0.3.0 development checkout's data-container
schemas, leading-axis shape conventions, explicit CPU↔GPU transfer helpers,
the direct-kernel troubleshooting contract, and current CPU/GPU support
boundaries.

For a runnable walkthrough of native single-box container construction and
optional Warp CPU-backend round trips, see the
[Data Containers example](../../examples/fundamentals/data-containers.md) and run:

```bash
python docs/examples/fundamentals/data_containers.py
```

For the low-level direct-kernel path, use the canonical quick-start at
[`docs/examples/gpu/direct/gpu_direct_kernels_quick_start.py`](https://github.com/uncscode/particula/blob/main/docs/examples/gpu/direct/gpu_direct_kernels_quick_start.py).
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
[Particle & Gas Data Migration](../migration/v0.3.0/index.md). For future work
and planned expansions, see the
[Data-Oriented Design and GPU Roadmap](../../Features/Roadmap/data-oriented-gpu.md).

The Python snippets in each walkthrough share imports and state in page order.
Slot examples continue the container construction shown here; follow the
[slot guide](../gpu/slots.md) after creating that state. Warp examples
require Warp installed and select its CPU device explicitly; no CUDA evidence
is implied. Native transfers and caller-owned scratch remain visible.

## Public imports

### Concrete CPU structural alignment (API validation)

Import `validate_aerosol_structure` directly from
`particula.aerosol_validation`, not from the top-level package. It checks
current raw array shapes, a positive shared box count, nonempty unique
nonblank ordered gas names, and a saturation-ratio lane for **every** gas
species. It reads only metadata: no copies, value or representation checks,
mutation, mask filtering, or process admission. Particle and gas species
widths may differ; empty particle axes and all-false partitioning masks are
allowed. Caller-managed full gas and ratio lane order remains unchanged.

This structural check cannot certify a same-width permutation of ratio lanes.
Expected ordered names and ratio-producer provenance, particle/gas mapping,
mask eligibility, physical validity, and distribution/process admission belong
to process bindings, not to this structural helper.

### CPU particle population interpretation

For new builder-created `ParticleData`, explicitly call
`set_distribution_type("discrete" | "continuous_pdf" | "particle_resolved")`
before `build()`. Discrete/PMF bin concentrations are already in m⁻³ and
continuous radius PDF values are `dn/dr` in m⁻⁴; both require a fixed 1 m³
volume. A PDF additionally needs `set_radius_grid(...)`: positive, finite,
strictly increasing radius nodes in metres, at least two, matching its slot
width. `number_density` and `species_mass_density` integrate PDFs with
trapezoids (mass times PDF is integrated together) and sum PMF bins.
Particle-resolved slots instead hold represented counts and are normalized
once by their positive physical volume. `slot_concentration_density` returns
fresh per-slot density; `species_mass_inventory` returns kg per box and
`total_mass` remains kg per particle. An empty PDF retains a valid grid with
zero PDF and zero population mass.

Call `validate_representation()` after directly constructing or mutating
writable particle arrays; new population helpers call it on each read and
reject invalid interpretation without repairing input. The caller owns
representation-consistent changes. Native Warp transfers preserve the explicit
`discrete` or `particle_resolved` kind and its volume interpretation. Continuous
PDF upload is unsupported. Missing or untagged population metadata rejects
before transfer setup or readback; clearing a tag cannot bypass admission.
The former `from_representation()` and `to_representation()` bridges have
been removed. Construct native data with an explicit distribution kind;
use `copy()` to retain that provenance in detached storage.

Prefer the currently exported package-level imports. The low-level
`particula.gpu` container, conversion-helper, explicit-transfer, and
direct-kernel workflow is experimental, but these import paths and its
caller-owned transfer model are supported:

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

## Native data contract

Native `Aerosol` holds `ParticleData`, `GasData` and `EnvironmentData` by
identity. Scientific processes own their physics and validate their supported
single-box inputs. Incomplete CPU material scratch storage may omit a tag,
but it is not an executable or transferable population. This guide describes
the selected checkout; release publication remains a separate decision.

### Stored units and physical inventories

Let `c[b,n]` be stored concentration, `m[b,n,s]` per-particle species mass
in kg, and `V[b]` physical volume in m³. The declared kind is authoritative;
array shape, a legacy strategy, or fractional weights do not infer it.

| Declared `distribution_type` | Raw `c` units | Volume rule | Number density (m⁻³) | Species mass density (kg/m³) |
| --- | --- | --- | --- | --- |
| `discrete` (PMF) | Number per bin per m³ | Exactly 1 m³ | Sum `c` over bins | Sum `c*m` over bins |
| `continuous_pdf` | Radius PDF `dn/dr`, m⁻⁴ | Exactly 1 m³ | Trapezoidal integral of `c` over radius | Trapezoidal integral of `c*m` over radius |
| `particle_resolved` | Represented particle counts | Finite, strictly positive V | Sum `c` / V | Sum `c*m` / V |

PDF quadrature uses the explicitly supplied positive, finite, strictly
increasing float64 radius grid in metres, with at least two nodes and one
node per slot. Integrate the product of varying species mass and PDF on that
interval, without extrapolation; summing PDF nodes or multiplying separate
integrals is incorrect. Zero PDF and mass on a valid grid are allowed.
Density-to-storage conversion would multiply by V only for resolved counts;
Reading these properties performs no inverse conversion or automatic rescaling.

`species_mass_inventory` is species mass density times V, in kg per box.
`total_mass` is instead mass per particle, summed over species. Gas
`concentration` is always kg/m³, so extensive gas inventory is concentration
multiplied by the same physical box volume. Gas molar mass is kg/mol,
particle material density is kg/m³, and saturation ratio is dimensionless.

### Delivered helper inventory

These are existing signatures/properties, not additional aggregate APIs:

| Owner and access | Result and contract |
| --- | --- |
| `ParticleData.validate_representation()` | Returns `None`; read-only kind, float64 schema, physical-value, volume and grid admission; rejects without repair. |
| `ParticleData.slot_concentration_density` | Fresh `(B,N)` array; m⁻³ for PMF/resolved, m⁻⁴ for PDF. |
| `ParticleData.number_density` | Fresh `(B,)` array, m⁻³. |
| `ParticleData.species_mass_density` | Fresh `(B,Sp)` array, kg/m³. |
| `ParticleData.species_mass_inventory` | Fresh `(B,Sp)` array, kg. |
| `ParticleData.copy()` | Detached arrays and PDF grid, preserved kind. |
| `GasData.copy()` | Detached names list, molar masses, concentrations and partitioning flags. |
| `EnvironmentData.copy()` | Detached temperature, pressure and saturation-ratio arrays. |
| `validate_aerosol_structure(particles, gas, environment)` | Concrete import from `particula.aerosol_validation`; returns `None`, checks raw dimensions and ordered-name metadata without mutation. |

The four population properties call representation validation on every read.
Builder creation enforces explicit kind and its volume/grid/units rules;
direct `ParticleData(...)` construction checks raw shapes but does not invoke
representation validation. After raw edits, callers must explicitly validate
before interpreting data. The structural checker does not call representation
validation, inspect physical values, certify chemistry, or admit a process.

Raw fields are writable. `ParticleData` retains supplied arrays except for
constructor density broadcasting/empty-density normalization;
`GasData` retains its supplied name list and may share already-compatible
NumPy arrays after dtype normalization. `EnvironmentData` copies constructor
arrays. None of these mutable objects is a frozen snapshot: use `copy()` for
isolation. Derived population results are detached values, not writable views
into the source, and a subsequent read reflects current source arrays.

### Ordered lanes and process-owned mapping

For `Sp` particle species and `Sg` gas species, `Sp` need not equal `Sg`.
Gas names, molar masses, partitioning flags and concentration columns retain
one full order. Environment saturation ratios use all `Sg` lanes in that same
order, including nonpartitioning species; a mask never compresses storage.
The structural checker requires nonblank unique names and matching dimensions,
but matching widths cannot prove the ratio producer used the right chemistry.

A process binding must compare expected full ordered gas names and
ratio-producer provenance and supply explicit unique, in-range, one-to-one
`(gas_index, particle_index)` pairs. It must check mapped mask eligibility,
classify unmatched species, and admit physical values and distribution
capabilities before mutation. See the [native scientific process contract](../processes/index.md).
An all-false mask preserves all gas/ratio lanes and admits no partitioning
transfer; it is not an empty gas container.

The data-only integration regression uses particle order `[solute_a, solute_b]`
and gas order `[solute_b, inert, solute_a]`, with explicit pairs `(0,1)` and
`(2,0)` for the mixed mask. At V=0.25 m³, particle inventories are
`[5, 13.5] * 1e-18` kg and gas inventories are
`[0.5, 1.75, 0.75] * 1e-18` kg: mapped totals are therefore
`[14, 5.75] * 1e-18` kg in map order, with inert gas accounted for separately.
It also checks V=4 m³, all-false masks, current reads and detached copies.
This is inventory accounting evidence, not a physics execution example.

### Explicit transfer and ownership boundary

The supported raw transfer calls remain
`to_warp_particle_data(data, device="cuda", copy=True)`,
`to_warp_gas_data(data, device="cuda", copy=True, vapor_pressure=None)`, and
`to_warp_environment_data(data, device="cuda", copy=True)`.
They do not convert count storage to density. `copy=False` requests the
compatible zero-copy path and must not be treated as an isolation guarantee.
Callers own device placement and synchronization.

Downloads are `from_warp_particle_data(gpu_data, sync=True)`,
`from_warp_gas_data(gpu_data, name=None, sync=True)`, and
`from_warp_environment_data(gpu_data, sync=True)`; `sync=False` requires
caller synchronization before host access. Native particle restore preserves
its population kind and volume but must not be assumed detached from Warp CPU
storage; explicitly copy when
independent CPU ownership is needed. Preserve gas names externally and pass
`name=` to restore their meaning; otherwise names are placeholders. GPU vapor
pressure is omitted on CPU restore. Restored gas numeric storage can also
share Warp CPU arrays; use `copy()` when isolation is required. Environment
construction detaches its
restored arrays. The detailed raw transfer tables below remain authoritative.

Native direct condensation and nucleation require explicit species bindings
prepared against the transferred containers. Resident restart needs fresh
bindings for the fresh containers; CPU object replacement does not rebind GPU
state. See the [migration dynamics guide](../migration/v0.3.0/dynamics.md)
and the runnable direct and resident examples. GPU PDF execution and
cross-device continuation are unsupported; no performance evidence follows
from successful transfers.

### Aggregate access and validated replacement

The native interface is keyword-only
`Aerosol(particles=..., gas=..., environment=...)`, identity-returning properties
with validated setters, and keyword-only
`replace_data(particles=..., gas=..., environment=...)` requiring all three.

The aggregate validates the complete proposed triple before publishing any
reference. Same-object assignment preserves identity; a compatible distinct
candidate is retained by identity. An incompatible single setter rejects
without changing held or candidate state. A compatible all-three layout change
publishes the three supplied identities together; one invalid candidate rejects
the entire operation. Representation validation and structural alignment are
separate checks; process mapping and physics admission remain process gates.
Acceptance implies no hidden copy, transfer, GPU/resident rebinding, inventory
conservation or automatic interception of subsequent raw edits. See
[access and replacement](ownership.md) for executable
examples and rejection checks.

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
| `volume` | `(n_boxes,)` | Authoritative per-box simulation volume in m³. |
| `distribution_type` | Scalar string or `None` | Explicit CPU interpretation; `None` is incomplete material scratch storage, rejected by native execution and Warp transfer. |
| `radius_grid` | `(n_particles,)` or `None` | CPU-only PDF radius nodes in metres; required only for `continuous_pdf`. |

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
    distribution_type="particle_resolved",  # raw concentration stores counts
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

## Related references

The fixed-four-substep low-level direct-kernel walkthrough can be run with
`python docs/examples/gpu/direct/gpu_condensation_parity_walkthrough.py`. Its focused
runtime regression command is
`pytest examples_tests/gpu_condensation_parity_walkthrough_test.py -q -Werror`.

- [Data Containers example](../../examples/fundamentals/data-containers.md)
- [Particle & Gas Data Migration](../migration/v0.3.0/index.md)
- [Data-Oriented Design and GPU Roadmap](../../Features/Roadmap/data-oriented-gpu.md)
- [Mass Precision Recommendation Report](../../contribute/validation/numerics/mass-precision.md)

## Related execution guides

- [Transfers](../gpu/transfers.md)
- [Slots](../gpu/slots.md)
- [Execution and resident scheduling](../gpu/execution.md)
- [Support boundaries](../gpu/support.md)
- [Troubleshooting](../gpu/troubleshooting.md)
