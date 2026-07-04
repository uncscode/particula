# Data-Oriented Design and GPU Roadmap

This page tracks the status of the data-oriented container migration and the
Warp-backed GPU work for particle-resolved aerosol simulations.

The remaining work is organized into epics ordered by dependency: Epic A
(foundations) unblocks Epic B (physics), which unblocks Epics C-E. Epic F
(docs, API stability, validation infrastructure) runs alongside all of them.

- [Epic A: Data-Model and Numerical Foundations](#epic-a-data-model-and-numerical-foundations)
- [Epic B: GPU Physics Coverage and Parity](#epic-b-gpu-physics-coverage-and-parity)
- [Epic C: High-Level Integration and GPU-Resident Simulation](#epic-c-high-level-integration-and-gpu-resident-simulation)
- [Epic D: Graph Capture and Performance](#epic-d-graph-capture-and-performance)
- [Epic E: Differentiability and Global Optimization](#epic-e-differentiability-and-global-optimization)
- [Epic F: API Stability, Validation Infrastructure, and Documentation](#epic-f-api-stability-validation-infrastructure-and-documentation)

Quick links:

- [Current container schema inventory](#current-container-schema-inventory)
- [Authoritative field ownership decisions](#authoritative-field-ownership-decisions)

## Motivation and Target Workloads

The GPU and data-oriented work exists to enable two driving use cases.

- **Large multi-box scaling.** The primary performance target is many
  simulation boxes evolved together on the GPU, not just large particle counts
  in a single box. Multi-box throughput is the main scaling axis, and benchmarks
  should prioritize scaling in the number of boxes.
- **Global optimization against observations.** A longer-term goal is to use
  Warp automatic differentiation to compute gradients through the simulation so
  model parameters can be fit to experiments or observations. Large multi-box
  ensembles support this by evaluating many conditions per optimization step.

These two goals reinforce each other: multi-box execution provides both the
scaling target and the batch of conditions used during global optimization.

The intended physical scope spans new particle formation through cloud droplet
sizes. This wide dynamic range drives the precision and mass-resolution
decisions in
[Numerical Precision and Mass Resolution](#numerical-precision-and-mass-resolution)
and the time-integration decisions in
[Time-Scale Stiffness](#time-scale-stiffness). New particle formation is both
a size-range driver and a planned process: a nucleation/particle-source
process does not exist in particula today and is added as an Epic B work item
so freshly formed particles can enter GPU-resident simulations through slot
activation.

## Non-Goals

The following are explicitly out of near-term scope.

- No multi-GPU or distributed-memory execution yet. Single-device GPU-resident
  simulation comes first.
- No full CFD coupling between boxes. Box communication is limited to prescribed
  advection, dilution, expansion, and simple mixing maps.
- No dynamic in-kernel array resizing. Particle-count changes are represented as
  active/inactive fixed-shape slots.
- DNS turbulence is deferred until the simpler GPU coagulation kernels are
  stable and validated.
- No GPU staggered (Gauss-Seidel) condensation. The staggered strategy is
  inherently sequential per-particle and remains CPU-only; GPU condensation
  covers the isothermal and latent-heat variants.
- No automatic silent CPU/GPU data movement inside long simulation loops.
  Fallback boundaries are explicit.

## Current Status

### Data-Oriented Design

The core data-oriented model is implemented.

- `ParticleData` stores particle masses, concentrations, charges, densities,
  and box volumes with explicit batch dimensions.
- `GasData` stores gas species names, molar masses, concentrations, and
  partitioning flags with explicit box dimensions.
- Conversion helpers bridge legacy facades and new containers.
- Condensation and coagulation strategies accept the new data containers in
  addition to legacy facades.
- Migration documentation exists in
  [ParticleData and GasData Migration](../particle-data-migration.md).

The work is still a migration, not a full replacement. `ParticleRepresentation`
and `GasSpecies` remain available for compatibility.

Known data-model gaps:

- There is no container for per-box thermodynamic state. Temperature,
  pressure, and humidity are passed as scalars into kernels and have no home
  in `ParticleData` or `GasData`, which blocks parcel/expansion workflows and
  latent-heat temperature feedback (see
  [EnvironmentData Container](#environmentdata-container)).
- `ParticleData.density` is shaped `(n_species,)` and shared across boxes, so
  boxes at different temperatures cannot carry per-box densities.
- Dilution exists only as free functions (`get_volume_dilution_coefficient`,
  `get_dilution_rate`); there is no `Dilution` strategy or runnable even on
  the CPU side, so GPU dilution has no process-level CPU reference yet.

### Warp GPU Backend

The lower-level Warp backend is implemented and covered by targeted tests.

- Warp-side particle and gas containers mirror the CPU-side data containers.
- CPU/GPU transfer helpers support long GPU-resident workflows, including a
  `gpu_context` context manager and a `WARP_AVAILABLE` feature flag.
- GPU condensation kernels provide a tested `condensation_step_gpu` API.
- GPU Brownian coagulation kernels provide a tested `coagulation_step_gpu` API.
- GPU benchmark scaffolding exists for CUDA-enabled environments.
- Parity tests parametrize over available Warp devices, running on Warp CPU
  everywhere and additionally on CUDA when a device is present.

The GPU backend is currently a directly callable lower-level API. It is not yet
integrated as an automatic backend in the main `Aerosol`, `Runnable`, or
high-level dynamics workflows.

Known GPU physics gaps remain:

- Latent-heat condensation is implemented on the CPU side, but there is no
  Warp-backed latent-heat condensation kernel yet.
- Particle charge is present in the GPU data container, but charged particle
  coagulation kernels are not implemented yet.
- The current GPU coagulation path is Brownian-focused; charged, turbulent,
  sedimentation, and combined particle-resolved coagulation workflows still
  need GPU coverage decisions and implementations.
- CPU sedimentation coagulation and simple turbulent shear coagulation exist,
  but neither has a GPU kernel yet. DNS turbulence is intentionally deferred
  from the near-term GPU scope.
- Wall loss, dilution, and other dynamics processes are CPU-only today. They
  need GPU implementations before a full simulation can remain GPU-resident for
  every timestep.
- GPU condensation ignores activity and surface strategies. The kernel uses a
  hardcoded surface-tension default (0.072 N/m) with raw Kelvin and vapor
  pressure terms, so composition-dependent water uptake, kappa-hygroscopicity,
  and activity effects are not represented on the GPU.
- Gas vapor pressure is a GPU-only field: `WarpGasData.vapor_pressure`
  defaults to zeros when not supplied and is dropped when converting back to
  `GasData`. There is no on-device recomputation of temperature-dependent
  vapor pressures between timesteps.
- No nucleation/particle-source process exists on CPU or GPU.

Known GPU kernel defects and design limits (see
[Known Kernel Issues](#known-kernel-issues)):

- `coagulation_step_gpu` re-initializes RNG states on every call, so identical
  seeds across timesteps produce correlated draws unless the caller manually
  varies the seed.
- The Brownian coagulation kernel launches one thread per box with sequential
  pair selection inside the thread, which limits single-box scaling at large
  particle counts.
- Rejection sampling uses a single `k_max` bound from the min/max radius pair,
  which degrades acceptance rates when small particles and large droplets
  coexist in one box.

## Epic A: Data-Model and Numerical Foundations

Foundation work that other epics depend on: container schemas, per-box
thermodynamic state, precision, mass representation, and time integration.

### Remaining Data-Oriented Work

- Finish reducing dependence on legacy facade objects in new examples and
  documentation.
- Decide when `ParticleRepresentation` and `GasSpecies` should move from
  compatibility facades to deprecated APIs.
- Keep strategy APIs consistent when accepting either legacy facades or data
  containers.
- Expand examples that start directly from `ParticleData` and `GasData` instead
  of converting from legacy objects.
- Document shape conventions for single-box, multi-box, binned, and
  particle-resolved simulations in one place.

#### Current container schema inventory

The table below inventories the current public stored schema for
`ParticleData`, `GasData`, `WarpParticleData`, and `WarpGasData`, including
shared-across-box fields, box-batched fields, validation hooks, and current
CPU↔GPU round-trip behavior. Evidence entries point to the current enforcing
constructor or conversion helper and to representative existing regression
tests. When a behavior is source-only, that is called out explicitly rather than
implying broader test coverage.

| Container | Field | Canonical owner | Shape | Dtype | Storage / mutability note | Validation / coercion hook | CPU↔GPU transfer behavior | Evidence |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `ParticleData` | `masses` | `ParticleData` | `(n_boxes, n_particles, n_species)` | `float64` | Stored, mutable, box-batched particle/species mass array | `ParticleData.__post_init__()` requires 3D masses and uses it as the shape anchor for all other particle fields | `to_warp_particle_data()` copies or zero-copies to `WarpParticleData.masses`; `from_warp_particle_data()` restores the same shape and values | Source: `particula/particles/particle_data.py:76-98`, `particula/gpu/conversion.py:72-139,244-287`; tests: `particula/particles/tests/particle_data_test.py:124-136`, `particula/gpu/tests/conversion_test.py:512-548` |
| `ParticleData` | `concentration` | `ParticleData` | `(n_boxes, n_particles)` | `float64` | Stored, mutable, box-batched number concentration/count | `ParticleData.__post_init__()` enforces `(n_boxes, n_particles)` | Round-trips through `to_warp_particle_data()` / `from_warp_particle_data()` without schema change | Source: `particula/particles/particle_data.py:77,96-103`, `particula/gpu/conversion.py:113-115,283`; tests: `particula/particles/tests/particle_data_test.py:138-164`, `particula/gpu/tests/conversion_test.py:521-545` |
| `ParticleData` | `charge` | `ParticleData` | `(n_boxes, n_particles)` | `float64` | Stored, mutable, box-batched particle charge array | `ParticleData.__post_init__()` enforces `(n_boxes, n_particles)` | Round-trips through `to_warp_particle_data()` / `from_warp_particle_data()` without dtype conversion | Source: `particula/particles/particle_data.py:78,104-109`, `particula/gpu/conversion.py:116,284`; tests: `particula/particles/tests/particle_data_test.py:166-192`, `particula/gpu/tests/conversion_test.py:524-526,546-548` |
| `ParticleData` | `density` | `ParticleData` | `(n_species,)` | `float64` | Stored, mutable, shared across boxes | `ParticleData.__post_init__()` requires 1D density, broadcasts scalar density to `n_species`, and fills empty density with zeros when `n_species > 0` | `to_warp_particle_data()` mirrors the shared 1D array to `WarpParticleData.density`; `from_warp_particle_data()` restores it to CPU storage | Source: `particula/particles/particle_data.py:79,119-137`, `particula/gpu/conversion.py:117-119,285`; tests: `particula/particles/tests/particle_data_test.py:208-234`, `particula/gpu/tests/warp_types_test.py:38-58` |
| `ParticleData` | `volume` | `ParticleData` | `(n_boxes,)` | `float64` | Stored, mutable, per-box simulation volume | `ParticleData.__post_init__()` enforces `(n_boxes,)` | Round-trips through `to_warp_particle_data()` / `from_warp_particle_data()` without schema change | Source: `particula/particles/particle_data.py:80,111-117`, `particula/gpu/conversion.py:120,286`; tests: `particula/particles/tests/particle_data_test.py:194-206`, `particula/gpu/tests/conversion_test.py:530-548` |
| `GasData` | `name` | `GasData` | `len == n_species` | `list[str]` | Stored, mutable CPU-only species metadata | `GasData.__post_init__()` rejects an empty species list; it does not enforce uniqueness or survive GPU transfer on its own | Not transferred by `to_warp_gas_data()` because `WarpGasData` has no string field; `from_warp_gas_data()` restores caller-supplied names or generates placeholders such as `species_0` | Source: `GasData.__post_init__()` and `from_warp_gas_data()` in `particula/gas/gas_data.py` and `particula/gpu/conversion.py`; tests: `particula/gas/tests/gas_data_test.py:119-127`, `particula/gpu/tests/conversion_test.py:600-640` |
| `GasData` | `molar_mass` | `GasData` | `(n_species,)` | `float64` | Stored, mutable, shared across boxes | `GasData.__post_init__()` coerces to `np.float64` and enforces `(n_species,)` | `to_warp_gas_data()` mirrors it to `WarpGasData.molar_mass`; `from_warp_gas_data()` restores the same numeric field | Source: `particula/gas/gas_data.py:65,75-77,103-108`, `particula/gpu/conversion.py:214-216,354`; tests: `particula/gpu/tests/warp_types_test.py:168-184,218-232`, `particula/gpu/tests/conversion_test.py:583-595` |
| `GasData` | `concentration` | `GasData` | `(n_boxes, n_species)` | `float64` | Stored, mutable, box-batched gas mass concentration | `GasData.__post_init__()` coerces to `np.float64`, requires 2D, and checks width against `n_species` | `to_warp_gas_data()` mirrors it to `WarpGasData.concentration`; `from_warp_gas_data()` restores the same shape | Source: `particula/gas/gas_data.py:66,76-78,89-101`, `particula/gpu/conversion.py:217-219,355`; tests: `particula/gas/tests/gas_data_test.py:99-117`, `particula/gpu/tests/conversion_test.py:593-595` |
| `GasData` | `partitioning` | `GasData` | `(n_species,)` | `bool` | Stored, mutable, shared-across-boxes partitioning mask | `GasData.__post_init__()` coerces with `np.asarray(..., dtype=np.bool_)` and enforces `(n_species,)`; this is dtype coercion, not a stricter semantic validation layer beyond NumPy truthiness | `to_warp_gas_data()` converts `bool → int32`; `from_warp_gas_data()` converts `int32 → bool` on restore | Source: `GasData.__post_init__()`, `to_warp_gas_data()`, and `from_warp_gas_data()` in `particula/gas/gas_data.py` and `particula/gpu/conversion.py`; tests: `particula/gas/tests/gas_data_test.py:77-97`, `particula/gpu/tests/conversion_test.py:189-199,609-619` |
| `WarpParticleData` | `masses` | `WarpParticleData` | `(n_boxes, n_particles, n_species)` | `wp.float64` | Stored, mutable GPU mirror of `ParticleData.masses` | Declared as `wp.array3d(dtype=wp.float64)`; populated by `to_warp_particle_data()` | Receives CPU particle masses on transfer to GPU; `from_warp_particle_data()` returns the same values to `ParticleData` | Source: `particula/gpu/warp_types.py:73`, `particula/gpu/conversion.py:111-125,282`; tests: `particula/gpu/tests/warp_types_test.py:38-58,100-118`, `particula/gpu/tests/conversion_test.py:512-548` |
| `WarpParticleData` | `concentration` | `WarpParticleData` | `(n_boxes, n_particles)` | `wp.float64` | Stored, mutable GPU mirror of `ParticleData.concentration` | Declared as `wp.array2d(dtype=wp.float64)`; populated by `to_warp_particle_data()` | Receives CPU particle concentration/count state and round-trips back unchanged | Source: `particula/gpu/warp_types.py:74`, `particula/gpu/conversion.py:113-128,283`; tests: `particula/gpu/tests/warp_types_test.py:54-58,113-118`, `particula/gpu/tests/conversion_test.py:521-545` |
| `WarpParticleData` | `charge` | `WarpParticleData` | `(n_boxes, n_particles)` | `wp.float64` | Stored, mutable GPU mirror of `ParticleData.charge` | Declared as `wp.array2d(dtype=wp.float64)`; populated by `to_warp_particle_data()` | Receives CPU charge data and round-trips back unchanged | Source: `particula/gpu/warp_types.py:75`, `particula/gpu/conversion.py:116,129-131,284`; tests: `particula/gpu/tests/warp_types_test.py:55-58,114-118`, `particula/gpu/tests/conversion_test.py:524-526,546-548` |
| `WarpParticleData` | `density` | `WarpParticleData` | `(n_species,)` | `wp.float64` | Stored, mutable GPU mirror of the shared-across-boxes `ParticleData.density` array | Declared as `wp.array(dtype=wp.float64)`; populated by `to_warp_particle_data()` | Receives shared CPU density array and restores it on `from_warp_particle_data()` | Source: `particula/gpu/warp_types.py:76`, `particula/gpu/conversion.py:117-119,132-134,285`; tests: `particula/gpu/tests/warp_types_test.py:57-58,113-118,302-320`, `particula/gpu/tests/conversion_test.py:527-529` |
| `WarpParticleData` | `volume` | `WarpParticleData` | `(n_boxes,)` | `wp.float64` | Stored, mutable GPU mirror of per-box `ParticleData.volume` | Declared as `wp.array(dtype=wp.float64)`; populated by `to_warp_particle_data()` | Receives per-box CPU volume array and restores it on `from_warp_particle_data()` | Source: `particula/gpu/warp_types.py:77`, `particula/gpu/conversion.py:120,135-137,286`; tests: `particula/gpu/tests/warp_types_test.py:57-58,113-118,302-320`, `particula/gpu/tests/conversion_test.py:530-548` |
| `WarpGasData` | `molar_mass` | `WarpGasData` | `(n_species,)` | `wp.float64` | Stored, mutable GPU mirror of `GasData.molar_mass` shared across boxes | Declared as `wp.array(dtype=wp.float64)`; populated by `to_warp_gas_data()` from CPU gas state | Round-trips back through `from_warp_gas_data()` without schema drift | Source: `particula/gpu/warp_types.py:131`, `particula/gpu/conversion.py:214-216,228-230,354`; tests: `particula/gpu/tests/warp_types_test.py:168-184,218-232`, `particula/gpu/tests/conversion_test.py:583-595` |
| `WarpGasData` | `concentration` | `WarpGasData` | `(n_boxes, n_species)` | `wp.float64` | Stored, mutable GPU mirror of `GasData.concentration` | Declared as `wp.array2d(dtype=wp.float64)`; populated by `to_warp_gas_data()` from CPU gas state | Round-trips back through `from_warp_gas_data()` without schema drift | Source: `particula/gpu/warp_types.py:132`, `particula/gpu/conversion.py:217-219,231-233,355`; tests: `particula/gpu/tests/warp_types_test.py:180-184,229-231`, `particula/gpu/tests/conversion_test.py:593-595` |
| `WarpGasData` | `vapor_pressure` | `WarpGasData` | `(n_boxes, n_species)` | `wp.float64` | Stored, mutable GPU-only helper state for condensation kernels | `to_warp_gas_data()` validates an optional provided shape or creates a zero-filled default when `vapor_pressure` is `None` | Present only on the GPU container; `from_warp_gas_data()` always drops it because `GasData` has no matching field, so CPU restore is intentionally lossy | Source: `WarpGasData` plus `to_warp_gas_data()` / `from_warp_gas_data()` in `particula/gpu/warp_types.py` and `particula/gpu/conversion.py`; tests: `particula/gpu/tests/warp_types_test.py:177-183,225-231`, `particula/gpu/tests/conversion_test.py:200-229,484-496,588-595` |
| `WarpGasData` | `partitioning` | `WarpGasData` | `(n_species,)` | `wp.int32` | Stored, mutable GPU mirror/helper mask; shared across boxes | Declared as `wp.array(dtype=wp.int32)`; `to_warp_gas_data()` converts `GasData.partitioning` from `bool` to `int32` | `from_warp_gas_data()` restores `int32 → bool` into `GasData.partitioning` | Source: `particula/gpu/warp_types.py:134`, `particula/gpu/conversion.py:159-160,208-209,223-239,347-356`; tests: `particula/gpu/tests/warp_types_test.py:178-184,218-232,322-338`, `particula/gpu/tests/conversion_test.py:189-199,609-619` |

Derived `ParticleData` accessors are intentionally separate from the stored
schema above: `radii`, `total_mass`, `effective_density`, and
`mass_fractions` are computed properties, not constructor-owned fields
(`particula/particles/particle_data.py:166-218`; tests:
`particula/particles/tests/particle_data_test.py:237-360`).

Current gas round-trip behavior is intentionally lossy in two places:

- `from_warp_gas_data()` restores `GasData.name` from caller input or
  placeholder values such as `species_0`; the GPU container itself never
  preserves string species names.
- `WarpGasData.vapor_pressure` is GPU-only helper state. It defaults to zeros
  in `to_warp_gas_data()` when omitted and is dropped when restoring CPU
  `GasData`.

Keep testing round-trips for the current CPU/GPU schema drift explicitly:
`WarpGasData` drops `name`, stores `partitioning` as `int32` instead of
`bool`, and adds `vapor_pressure` that is not restored to the CPU gas
container.

### Authoritative field ownership decisions

This section is the canonical ownership and CPU↔GPU round-trip contract for
follow-on E2 work. The inventory table above remains the shipped current-state
evidence record; use the decision table below for policy when adding fields,
conversion behavior, or future environment state.

| Field / group | Authoritative owner | CPU shape | GPU shape | Dtype | Mutability | Round-trip behavior | Downstream consumers | Evidence |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `ParticleData.masses` | Owned by `ParticleData` as the authoritative particle/species mass state | `(n_boxes, n_particles, n_species)` | `(n_boxes, n_particles, n_species)` via `WarpParticleData.masses` | `float64` on CPU / `wp.float64` on GPU | Mutable stored state on both containers | Must round-trip without schema drift through `to_warp_particle_data()` and `from_warp_particle_data()` | Particle property accessors, condensation, coagulation, and future GPU-resident timestep loops | `particula/particles/particle_data.py:76-98`; `particula/gpu/conversion.py:72-139,244-287`; `particula/particles/tests/particle_data_test.py:124-136`; `particula/gpu/tests/conversion_test.py:512-548` |
| `ParticleData.concentration` | Owned by `ParticleData` as authoritative per-box particle concentration/count state | `(n_boxes, n_particles)` | `(n_boxes, n_particles)` via `WarpParticleData.concentration` | `float64` / `wp.float64` | Mutable stored state on both containers | Must round-trip without schema change | Particle dynamics kernels and any box-batched particle workflow | `particula/particles/particle_data.py:76-109`; `particula/gpu/conversion.py:113-115,283`; `particula/particles/tests/particle_data_test.py:138-164`; `particula/gpu/tests/conversion_test.py:521-545` |
| `ParticleData.charge` | Owned by `ParticleData` as authoritative particle charge state | `(n_boxes, n_particles)` | `(n_boxes, n_particles)` via `WarpParticleData.charge` | `float64` / `wp.float64` | Mutable stored state on both containers | Must round-trip without dtype drift | Charged coagulation follow-on work and GPU parity paths that consume particle charge | `particula/particles/particle_data.py:76-109`; `particula/gpu/conversion.py:116,284`; `particula/particles/tests/particle_data_test.py:166-192`; `particula/gpu/tests/conversion_test.py:524-526,546-548` |
| `ParticleData.density` | Owned by `ParticleData` and must remain shared-across-boxes material state, not per-box environment state | `(n_species,)` | `(n_species,)` via `WarpParticleData.density` | `float64` / `wp.float64` | Mutable stored state, but shared across boxes | Must round-trip as shared 1D species density state | Radius, effective-density, and mass-fraction calculations plus future GPU particle property parity | `particula/particles/particle_data.py:67-68,119-137`; `particula/gpu/conversion.py:117-119,285`; `particula/particles/tests/particle_data_test.py:208-234`; `particula/gpu/tests/warp_types_test.py:38-58` |
| `ParticleData.volume` | Owned by `ParticleData` as the authoritative per-box simulation-volume carrier | `(n_boxes,)` | `(n_boxes,)` via `WarpParticleData.volume` | `float64` / `wp.float64` | Mutable stored state on both containers | Must round-trip without schema change | Per-box particle workflows, dilution-style process work, and future environment/process coordination | `particula/particles/particle_data.py:69-70,111-117`; `particula/gpu/conversion.py:120,286`; `particula/particles/tests/particle_data_test.py:194-206`; `particula/gpu/tests/conversion_test.py:530-548` |
| `GasData.name` | Owned by CPU `GasData` as authoritative species-name metadata | `len == n_species` | Not owned on GPU | `list[str]` | Mutable CPU metadata only | Does not survive transfer on its own; CPU restore requires caller-supplied names or generated placeholders from external metadata | CPU-facing reporting, facade compatibility, and any restore path back to `GasData` | `particula/gas/gas_data.py:52-73`; `particula/gpu/warp_types.py:82-99`; `particula/gpu/conversion.py:155-157,301-345`; `particula/gas/tests/gas_data_test.py:119-127`; `particula/gpu/tests/conversion_test.py:600-640` |
| `GasData.molar_mass` | Owned by `GasData` as authoritative gas species molar-mass state | `(n_species,)` | `(n_species,)` via `WarpGasData.molar_mass` | `float64` / `wp.float64` | Mutable stored state on both containers | Must round-trip without schema drift | Gas property calculations, condensation, and GPU gas kernels | `particula/gas/gas_data.py:53-67,75-108`; `particula/gpu/warp_types.py:100-111,131`; `particula/gpu/conversion.py:214-216,354`; `particula/gpu/tests/warp_types_test.py:168-184,218-232`; `particula/gpu/tests/conversion_test.py:583-595` |
| `GasData.concentration` | Owned by `GasData` as authoritative per-box gas concentration state | `(n_boxes, n_species)` | `(n_boxes, n_species)` via `WarpGasData.concentration` | `float64` / `wp.float64` | Mutable stored state on both containers | Must round-trip without shape drift | Condensation, gas-phase workflows, and future GPU-resident gas updates | `particula/gas/gas_data.py:55-57,76-101`; `particula/gpu/warp_types.py:104-105,132`; `particula/gpu/conversion.py:217-219,355`; `particula/gas/tests/gas_data_test.py:99-117`; `particula/gpu/tests/conversion_test.py:593-595` |
| `GasData.partitioning` | Owned by `GasData` as authoritative shared-across-boxes partitioning eligibility state | `(n_species,)` | `(n_species,)` via `WarpGasData.partitioning` | `bool` on CPU / `wp.int32` on GPU | Mutable stored state on both containers | Must round-trip with explicit `bool → int32 → bool` conversion | Condensation partitioning decisions and GPU kernels that require a numeric mask | `particula/gas/gas_data.py:57-58,79-115`; `particula/gpu/warp_types.py:96-111,134`; `particula/gpu/conversion.py:159-160,208-209,223-239,347-356`; `particula/gas/tests/gas_data_test.py:77-97`; `particula/gpu/tests/conversion_test.py:189-199,609-619` |
| `WarpParticleData` numeric mirrors | Owned on GPU only as mirrors of authoritative `ParticleData` fields, not as a separate source of truth | Mirrors CPU particle shapes | Stored on GPU as declared in `WarpParticleData` | `wp.float64` | Mutable GPU working state | Must restore to the corresponding `ParticleData` fields without adding or dropping particle schema | GPU-resident particle workflows and parity tests | `particula/gpu/warp_types.py:73-77`; `particula/gpu/conversion.py:111-137,281-287`; `particula/gpu/tests/warp_types_test.py:38-58,100-118,302-320`; `particula/gpu/tests/conversion_test.py:512-548` |
| `WarpGasData.molar_mass` / `concentration` / `partitioning` | Owned on GPU only as numeric mirrors/helpers of authoritative CPU `GasData` state | `(n_species,)`, `(n_boxes, n_species)`, `(n_species,)` | Same declared GPU shapes | `wp.float64` / `wp.float64` / `wp.int32` | Mutable GPU working state | Must restore to `GasData` numeric state, with explicit `int32 → bool` recovery for `partitioning` | GPU condensation and other gas-kernel workflows | `particula/gpu/warp_types.py:82-99,100-111,131-134`; `particula/gpu/conversion.py:142-241,290-357`; `particula/gpu/tests/warp_types_test.py:168-184,218-232,322-338`; `particula/gpu/tests/conversion_test.py:189-229,583-619` |
| `WarpGasData.vapor_pressure` | Owned by no CPU container; treated as GPU-helper/process state rather than authoritative `GasData` or future `EnvironmentData` state | Not owned on CPU containers | `(n_boxes, n_species)` via `WarpGasData.vapor_pressure` | `wp.float64` | Mutable GPU helper state | Must be recomputed, explicitly provided, or carried as sidecar state; CPU restore from `WarpGasData` is intentionally lossy because `from_warp_gas_data()` drops it | GPU condensation kernels and future on-device thermodynamic updates | `particula/gpu/warp_types.py:98-108,133`; `particula/gpu/conversion.py:162-206,220-241,301-304`; `particula/gpu/tests/warp_types_test.py:177-183,225-231`; `particula/gpu/tests/conversion_test.py:200-229,484-496,588-595` |
| Future `EnvironmentData.temperature` | Must be owned by future `EnvironmentData`, not by `ParticleData` or `GasData` | `(n_boxes,)` | `(n_boxes,)` via future `WarpEnvironmentData` | `float64` on CPU / planned GPU numeric mirror | Mutable per-box thermodynamic state | Must round-trip through future environment conversion helpers once implemented; this phase records policy only | Parcel/expansion workflows, latent-heat condensation, and other per-box thermodynamic updates | Direction recorded in [EnvironmentData Container](#environmentdata-container); `docs/Features/Roadmap/data-oriented-gpu.md`; `.opencode/plans/sections/features/E2-F1/architecture_design.md:31-46` |
| Future `EnvironmentData.pressure` | Must be owned by future `EnvironmentData`, not by `ParticleData` or `GasData` | `(n_boxes,)` | `(n_boxes,)` via future `WarpEnvironmentData` | `float64` on CPU / planned GPU numeric mirror | Mutable per-box thermodynamic state | Must round-trip through future environment conversion helpers once implemented; this phase records policy only | Parcel/expansion workflows, kernel inputs, and per-box forcing profiles | Direction recorded in [EnvironmentData Container](#environmentdata-container); `docs/Features/Roadmap/data-oriented-gpu.md`; `.opencode/plans/sections/features/E2-F1/architecture_design.md:31-46` |
| Future `EnvironmentData.saturation_ratio` | Must be owned by future `EnvironmentData` as per-box, per-species thermodynamic state | `(n_boxes, n_species)` | `(n_boxes, n_species)` via future `WarpEnvironmentData` or equivalent GPU environment state | `float64` on CPU / planned GPU numeric mirror | Mutable derived-or-updated thermodynamic state | Must round-trip through future environment conversion helpers once implemented; this phase records policy only | Latent-heat condensation, parcel expansion, and humidity-coupled follow-on work | Direction recorded in [EnvironmentData Container](#environmentdata-container); `docs/Features/Roadmap/data-oriented-gpu.md`; `.opencode/plans/sections/features/E2-F1/architecture_design.md:31-46` |
| Simulation volume ownership | Not owned by future `EnvironmentData`; must remain owned by `ParticleData.volume` | `(n_boxes,)` on `ParticleData` | `(n_boxes,)` on `WarpParticleData` | `float64` / `wp.float64` | Mutable per-box simulation state under particle container ownership | Must continue to round-trip only with particle container conversion helpers | Per-box particle state, dilution-style workflows, and timestep orchestration that needs simulation volume | `particula/particles/particle_data.py:69-70,111-117`; `particula/gpu/conversion.py:120,286`; `particula/particles/tests/particle_data_test.py:194-206`; `.opencode/plans/sections/features/E2-F1/architecture_design.md:31-46` |

#### Rationale for issue-critical ownership decisions

- `ParticleData.density` remains shared-across-boxes state with CPU/GPU shape
  `(n_species,)`; the constructor enforces 1D species density semantics rather
  than per-box thermodynamic ownership
  (`particula/particles/particle_data.py:67-68,119-137`; tests:
  `particula/particles/tests/particle_data_test.py:208-234`,
  `particula/gpu/tests/warp_types_test.py:38-58`).
- `ParticleData.volume` is the authoritative per-box simulation-volume carrier
  with shape `(n_boxes,)`, and future `EnvironmentData` must not own or mutate
  simulation volume
  (`particula/particles/particle_data.py:69-70,111-117`; tests:
  `particula/particles/tests/particle_data_test.py:194-206`;
  `.opencode/plans/sections/features/E2-F1/architecture_design.md:31-46`).
- `vapor_pressure` is process/GPU-helper state rather than owned CPU
  `GasData` or future `EnvironmentData` state; CPU-facing workflows must
  recompute it or pass it as sidecar state, and CPU restore from
  `WarpGasData` is intentionally lossy because `to_warp_gas_data()` injects
  zeros when absent and `from_warp_gas_data()` drops the field
  (`particula/gpu/conversion.py:162-206,301-304`; tests:
  `particula/gpu/tests/conversion_test.py:200-229,484-496,588-595`).
- `WarpGasData` is numeric-only; restoring CPU gas names requires
  caller-supplied names or equivalent external metadata because the GPU
  container excludes string fields and `from_warp_gas_data()` otherwise
  generates placeholder names such as `species_0`
  (`particula/gpu/warp_types.py:82-99`;
  `particula/gpu/conversion.py:305-345`; tests:
  `particula/gpu/tests/conversion_test.py:600-640`).
- Future environment ownership is `temperature: (n_boxes,)`, `pressure:
  (n_boxes,)`, and `saturation_ratio: (n_boxes, n_species)`; this phase records
  those ownership decisions for downstream work without moving simulation volume
  out of `ParticleData.volume`
  ([EnvironmentData Container](#environmentdata-container);
  `.opencode/plans/sections/features/E2-F1/architecture_design.md:31-46`).

- Decide the particle mass storage representation given the wide dynamic range
  (see [Numerical Precision and Mass Resolution](#numerical-precision-and-mass-resolution)).
  Options include per-species absolute mass, log-mass, or bin/section reference
  masses. This choice affects both accuracy and differentiability.

### EnvironmentData Container

Parcel/expansion workflows and latent-heat condensation require per-box
thermodynamic state that has no home in the current containers. The direction
is a third container rather than extending `GasData`.

- Add an `EnvironmentData` container with per-box temperature, pressure, and
  derived humidity/saturation state, shaped `(n_boxes,)`, mirrored by a
  `WarpEnvironmentData` struct on the GPU.
- Latent-heat condensation must read and update per-box temperature; rising
  parcels, expansion, and combustion boxes prescribe temperature, pressure,
  and volume evolution per box.
- Define ownership and update ordering: which processes read environment
  state, which mutate it, and where prescribed (user-supplied) profiles are
  applied within a timestep.
- Add round-trip conversion helpers and tests matching the existing
  `ParticleData`/`GasData` conversion patterns.
- Decide how existing kernel APIs that accept scalar temperature and pressure
  migrate to per-box arrays without breaking the current low-level API.

### Numerical Precision and Mass Resolution

The simulation must span new particle formation clusters through cloud droplets.
That is roughly fifteen orders of magnitude in particle mass, which sits near
the limit of double precision (~15-16 significant digits). This dynamic range,
not raw speed, is the main precision driver.

- **Keep fp64 as the reference precision** until a resolution study shows that a
  lower precision or mixed-precision path preserves acceptable accuracy. fp64 is
  justified by the physical dynamic range, not by preference.
- **Target mass resolution is on the order of 0.1 ng per tracked quantity** so
  that both small freshly formed particles and large droplets remain
  representable in the same simulation.
- **The open question is representation, not just dtype.** Storing per-particle
  absolute mass in a single fp64 array can lose small-mass resolution when large
  droplets coexist with tiny particles in the same box. Evaluate per-species
  mass, log-mass storage, or binned reference masses as alternatives.
- **fp64 doubles memory** relative to fp32, which directly taxes the large
  multi-box goal. Precision and memory budget must be evaluated together (see
  [Performance and Memory](#performance-and-memory)).
- **fp64 throughput is heavily reduced on consumer CUDA hardware.** Record which
  target devices matter, and whether a validated fp32 or mixed-precision path is
  needed for those devices.
- Add a resolution/accuracy study that checks conservation and small-particle
  fidelity across the full NPF-to-droplet range before changing precision.

### Time-Scale Stiffness

The same NPF-to-droplet range that stresses mass precision also stresses time
integration: nanometer particles equilibrate with vapor in microseconds while
cloud droplets evolve over seconds. The current GPU condensation path is an
explicit fixed-step update with non-negative clamping, which will be either
unstable or wastefully slow when both size extremes share a timestep.

- Characterize condensation stiffness across the target size range and record
  the stable explicit step size as a function of particle size.
- Evaluate integration options: sub-stepping, semi-implicit or asymptotic
  updates (for example exponential integrators for first-order mass
  transfer), and per-box or per-size-class adaptive stepping.
- Any chosen scheme must remain graph-capture friendly (fixed iteration
  counts, fixed shapes) and differentiable for the optimization path (see
  [Epic E](#epic-e-differentiability-and-global-optimization)).
- Water condensation near cloud activation is the hardest case: high vapor
  concentration, tight supersaturation coupling, and latent-heat temperature
  feedback. Use it as the stiffness stress test.

## Epic B: GPU Physics Coverage and Parity

Extend GPU support beyond condensation and Brownian coagulation where it is
scientifically useful, fix known kernel defects, and validate CPU/GPU parity.

### Physics Coverage

- Add a Warp-backed latent-heat condensation path that matches the CPU
  `CondensationLatentHeat` behavior, including latent-heat-corrected mass
  transfer and per-step latent heat energy bookkeeping. See the
  [condensation equations](../../Theory/Technical/Dynamics/Condensation_Equations.md#condensation-with-latent-heat)
  for the governing equation. This depends on the
  [EnvironmentData Container](#environmentdata-container) for per-box
  temperature feedback.
- Add on-device vapor pressure recomputation. Temperature-dependent saturation
  vapor pressures must be recomputed on the GPU each timestep for parcels,
  expansion, and latent-heat workflows; today `vapor_pressure` is set once at
  transfer time and defaults to zeros.
- Add GPU activity and surface-tension support needed by the condensation
  targets: replace the hardcoded surface-tension default with per-species or
  per-particle values, and decide which activity models (water activity,
  kappa-hygroscopicity) get GPU kernels. This is a prerequisite for the
  hygroscopicity and mixing-state optimization targets in
  [Epic E](#epic-e-differentiability-and-global-optimization).
- Add GPU support for charged particle-resolved coagulation, using particle
  charge from `WarpParticleData.charge` rather than treating charge as stored
  but inactive metadata.
- Add GPU sedimentation coagulation matching the CPU Seinfeld-Pandis 2016
  sedimentation kernel.
- Add GPU simple turbulent shear coagulation matching the CPU Saffman-Turner
  1956 turbulent shear kernel.
- Defer GPU DNS turbulence until the simpler coagulation kernels are stable;
  DNS turbulence is not part of the near-term GPU target.
- Add a CPU `Dilution` strategy/runnable as the reference implementation, then
  add the GPU dilution kernel and parity tests against it.
- Add a nucleation/particle-source process. No nucleation code exists in
  particula today. Implement a CPU reference process first, then a GPU version
  that activates inactive particle slots (see
  [Particle Slot Management](#particle-slot-management)). See the
  [nucleation equations](../../Theory/Technical/Dynamics/Nucleation_Equations.md)
  for the governing theory, rate expressions, and the mass-conserving source
  term.
- Define which wall loss, charged coagulation, turbulent coagulation, and
  coupled-process workflows should receive GPU kernels.
- Validate expansion to the other dynamical processes needed for complete
  aerosol simulations, including wall loss, dilution, gas updates, and future
  process modules.
- Clarify which particle distributions are supported by GPU kernels and which
  remain CPU-only. The current GPU path implicitly assumes particle-resolved
  semantics (per-slot merges, concentration zeroing); binned/moving-bin
  strategies have no GPU counterpart.
- Staggered (Gauss-Seidel) condensation stays CPU-only (see
  [Non-Goals](#non-goals)).

Priority GPU physics candidates:

- Latent-heat condensation parity with `CondensationLatentHeat` (requires
  `EnvironmentData` and on-device vapor pressure).
- Charged Brownian coagulation for particle-resolved simulations.
- Combined Brownian plus charged coagulation when both mechanisms are active.
- Sedimentation coagulation for gravitational settling collisions.
- Simple turbulent shear coagulation.
- Combined Brownian, charged, sedimentation, and turbulent shear kernels when
  multiple mechanisms are active.
- DNS turbulence only after the simpler GPU coagulation kernels are validated,
  and only if the expected simulation workloads justify the added complexity.
- Neutral wall loss and dilution kernels so complete simulations can stay on
  the GPU between output checkpoints.
- Charged wall loss after the neutral wall loss and core condensation and
  coagulation GPU paths are stable.
- Nucleation/particle-source via slot activation, after the CPU reference
  process exists.

### Known Kernel Issues

Concrete defects and design limits in the existing kernels that should be
fixed or explicitly accepted during this epic.

- **RNG re-initialization per call.** `coagulation_step_gpu` re-launches the
  RNG initialization kernel on every call, overwriting persisted state.
  Identical seeds across timesteps produce correlated draws, and callers must
  manually increment seeds per step. Fix by seeding once and persisting
  per-box RNG state in a reusable buffer across timesteps (see
  [Random Number Strategy](#random-number-strategy)); this also matters for
  graph capture, where re-seeding inside a captured graph would freeze the
  seed.
- **Rejection-sampling acceptance collapse.** The Brownian kernel bounds
  acceptance with a single `k_max` computed from the min/max radius pair. With
  NPF clusters and droplets in the same box, `k_max` far exceeds typical pair
  kernels, acceptance rates collapse, and trial counts explode. Evaluate
  size-binned majorant kernels or stratified pair sampling for wide size
  distributions.
- **One-thread-per-box coagulation.** The kernel launches one GPU thread per
  box with sequential pair selection inside the thread. This matches the
  multi-box scaling priority but serializes large single-box workloads. Record
  this as a deliberate design decision with its measured single-box limit, or
  plan a parallel-within-box variant.

### Validation and Parity

- Maintain CPU/GPU numerical parity tests for representative aerosol states.
- Standardize device-aware test execution: tests detect available Warp devices
  and run parity on Warp CPU everywhere, adding CUDA automatically when a
  device is present. This keeps local `pytest` runs meaningful on GPU
  workstations and will light up CUDA coverage automatically if CI gains a GPU
  runner. Formalize the existing device parametrization helper as the
  project-wide policy (a pytest flag/config plus marker), and document that
  CUDA-device validation is currently local/manual.
- Add CPU/GPU parity tests for latent-heat condensation, including the stored
  latent heat energy diagnostic described in the
  [condensation equations](../../Theory/Technical/Dynamics/Condensation_Equations.md#condensation-with-latent-heat).
- Add CPU/GPU parity or statistically bounded tests for charged
  particle-resolved coagulation.
- Add CPU/GPU parity tests for sedimentation and simple turbulent shear
  coagulation kernels.
- Add CPU/GPU parity tests for wall loss and dilution once GPU kernels exist.
- Add larger multi-box and particle-resolved regression cases that exercise
  realistic simulation loops.
- Add full-loop validation tests that compare CPU, uncaptured GPU, and
  graph-captured GPU execution for the same process sequence.
- Record acceptable tolerances for stochastic coagulation and floating-point
  differences across CPU, Warp CPU, and CUDA devices.
- Add multi-box validation tests for independent boxes, prescribed advection,
  dilution, and expansion.
- Add particle-slot validation tests for inactive slots, activation, slot
  exhaustion handling, and conservation across resampling or volume scaling.

## Epic C: High-Level Integration and GPU-Resident Simulation

Make GPU execution reachable from user-facing APIs and keep full simulations
resident on the device between checkpoints.

### High-Level Integration

- Add user-facing APIs that can choose CPU or GPU execution without requiring
  users to call kernel modules directly.
- Decide whether backend selection belongs on strategies, runnables, builders,
  or a separate execution context.
- Preserve CPU behavior as the reference path and make GPU fallback behavior
  explicit when Warp or CUDA is unavailable.
- Add a GPU simulation loop abstraction that keeps `WarpParticleData`,
  `WarpGasData`, and `WarpEnvironmentData` resident on the selected device
  across all enabled dynamics.
- Support process ordering for full aerosol simulations, for example
  condensation, coagulation, wall loss, dilution, environment updates, and any
  gas updates that must occur between those steps.

### Full GPU-Resident Simulation

- Define the minimum dynamics set required for a complete GPU-only aerosol
  simulation loop.
- Keep particles, gases, environment state, temporary work buffers, and
  random-number state on the GPU between timesteps.
- Transfer data back to CPU only for checkpoints, diagnostics, visualization, or
  final results.
- Add a process graph or scheduler that can execute supported GPU processes in
  a deterministic order.
- Provide explicit errors or CPU fallback boundaries when a requested dynamics
  process has no GPU implementation.
- Treat multiple independent boxes as a first-class simulation mode. Each box
  should keep its own particle, gas, environment, volume, and diagnostic state
  while sharing process configuration when appropriate.
- Support simple box communication through prescribed advection or mixing maps.
  This should cover early 1D parcel workflows such as rising parcels, expanding
  parcels, and flame/combustion-style expansion where box volumes and gas states
  evolve in a prescribed way.

Candidate GPU process coverage:

- Condensation: isothermal and latent-heat variants (staggered stays
  CPU-only).
- Coagulation: Brownian, charged particle-resolved, combined kernels, turbulent
  shear, and sedimentation. DNS turbulence is deferred from the near-term GPU
  scope.
- Wall loss: neutral spherical/rectangular first, then charged wall loss.
- Dilution: particle and gas concentration dilution on GPU-resident arrays.
- Nucleation: particle-source process activating inactive slots, after the CPU
  reference exists.
- Gas updates: partitioning-related gas concentration changes needed by coupled
  condensation workflows, plus on-device vapor pressure recomputation when
  temperature changes.
- Environment updates: prescribed or process-driven per-box temperature,
  pressure, and volume evolution.
- Diagnostics: optional GPU-side reductions for total mass, number
  concentration, latent heat energy, and conservation checks.
- Box transport: prescribed advection, dilution, expansion, and simple mixing
  between neighboring boxes or user-defined box pairs.

### Multi-Box Communication

- Start with prescribed communication, not full CFD coupling.
- Represent box-to-box transport with fixed-shape maps or sparse edge lists so
  graph-captured GPU loops can reuse the same allocation layout.
- Support independent boxes by default; communication should be opt-in per
  simulation loop.
- Include per-box volume changes for rising parcel, expansion, and combustion
  use cases.
- Keep particle and gas transport rules explicit. Gas concentrations, particle
  concentrations, and particle slot contents may need different update kernels.
- Validate simple 1D advection and expansion cases against CPU references before
  adding more complex coupling.

### Particle Slot Management

- Use fixed particle slot counts per box for GPU-resident simulations.
- Represent inactive particle slots as particles with zero mass, zero radius,
  and zero concentration or count.
- Avoid dynamic allocation inside timestep kernels. Processes that create new
  particles, including the planned nucleation process, should activate
  inactive slots when available.
- Add a resampling or volume-scaling policy for cases that would exceed the
  available particle slots in a box.
- Track per-box active particle counts as diagnostics, but keep the underlying
  arrays fixed-shape for GPU kernels and graph capture.
- Define compaction rules only if needed; inactive zero-mass slots are simpler
  and graph-friendly.

### Random Number Strategy

- Define deterministic RNG seeding for stochastic coagulation on GPU.
- Fix the current per-call re-initialization: seed once at loop setup and
  persist per-box RNG state between timesteps instead of re-launching the
  initialization kernel on every `coagulation_step_gpu` call (see
  [Known Kernel Issues](#known-kernel-issues)).
- Support per-box RNG streams so independent boxes remain reproducible when the
  number of boxes changes or when selected boxes are disabled.
- Track RNG state on the GPU between timesteps and include it in graph-captured
  execution tests.
- Document expected reproducibility limits across CPU, Warp CPU, and CUDA.

## Epic D: Graph Capture and Performance

Reduce launch overhead with graph capture and establish performance and memory
targets aligned with the multi-box scaling goal.

### Warp Graph Capture

- Investigate Warp graph capture for repeated timestep execution with a fixed
  process order and stable array shapes.
- Separate graph-capturable kernels from setup work such as allocation,
  validation, and host-side scheduling.
- Reuse preallocated buffers for mass transfer, collision pairs, wall-loss
  rates, dilution factors, diagnostics, and RNG state before capture begins.
- Add tests that compare captured-graph execution against uncaptured GPU
  execution and CPU reference results.
- Document graph-capture limitations, including shape changes, dynamic process
  selection, stochastic coagulation state, and device availability.
- Require fixed array shapes during graph capture. Changing `n_boxes`,
  `n_particles`, or `n_species` should invalidate the captured graph and require
  a new setup/capture step.
- Handle changing active particle counts through inactive slots rather than
  resizing arrays.
- Use resampling, merging, or volume scaling before a box exhausts inactive
  particle slots.
- Keep graph-captured loops focused on repeated timesteps with stable process
  order, stable buffer shapes, and stable communication maps.

For graph capture, particle count changes should be represented as changes in
active slots, not changes in array shape. If a simulation would create more
particles than the fixed slots allow, the GPU loop should trigger a documented
resampling or volume-scaling policy before slot exhaustion.

### Performance and Memory

- Establish benchmark targets with box count as the primary axis (for example
  1, 10, 100, and 1000 boxes), varying particles per box at each box count.
- Establish secondary benchmark targets for GPU-resident simulations at 1k,
  10k, 100k, and larger particle counts per box.
- Record the single-box scaling limit of the one-thread-per-box coagulation
  kernel and decide whether a parallel-within-box variant is needed (see
  [Known Kernel Issues](#known-kernel-issues)).
- Minimize CPU/GPU round trips in example workflows and high-level APIs.
- Reuse temporary buffers for repeated timesteps to avoid repeated allocation.
- Profile CUDA kernels for occupancy, memory access patterns, and scaling across
  boxes and particle counts.
- Benchmark graph-captured GPU loops against uncaptured GPU loops to quantify
  launch-overhead savings for small and medium timestep workloads.
- Track end-to-end performance using GPU-resident loops, not only isolated
  single-kernel timings.
- Add memory-budget estimates for `n_boxes × n_particles × n_species` state,
  inactive slots, temporary buffers, collision-pair buffers, diagnostics,
  communication maps, and autodiff tape storage for multi-step gradient runs
  (see [Epic E](#epic-e-differentiability-and-global-optimization)).
- Include benchmark cases that vary boxes, particles per box, species count,
  active-slot fraction, and process combinations.

## Epic E: Differentiability and Global Optimization

A longer-term goal is gradient-based global optimization: using Warp automatic
differentiation to fit model parameters to experiments or observations.
Differentiability constrains how kernels are written, so it must be considered
while Epic B kernels are authored, not added afterward.

See [Warp Autodiff: Limitations and Stochastic Process Handling](warp-autodiff-limitations.md)
for the detailed autodiff mechanics, kernel-authoring constraints, offline code
patterns, and the options for differentiating stochastic coagulation.

**The optimization targets are initial state, not process parameters.** Physical
coagulation and condensation parameters (accommodation coefficient, diffusivity,
and similar) are prescribed and held fixed. The inverse problem fits the initial
aerosol state to a measured final state. Examples: initial versus final size
distribution, initial versus final hygroscopicity, or initial versus final
mixing state. Coagulation and condensation are the size-dependent forward
operators that connect initial to final state; their parameters are not tuned.

The consequence for differentiability is important: because the loss is on the
final state and the unknowns are the initial state, gradients must flow
**through** each process operator with respect to the state it acts on (per-bin
or per-particle masses, concentrations, and composition), not with respect to
its parameters. Prescribed parameters mean no parameter adjoints are needed, but
the operators themselves must still be differentiable in state.

- **Author optimization-path kernels against Warp's `wp.Tape` requirements.**
  In-place mutation and some control-flow patterns can break or zero gradients.
  The current `apply_mass_transfer_kernel` and `apply_coagulation_kernel` use
  in-place updates and would need differentiable-friendly variants for the
  optimization path.
- **Start with deterministic condensation.** Isothermal and latent-heat
  condensation are natural first differentiable targets because mass transfer is
  deterministic. Prove an end-to-end gradient and optimization loop here before
  extending to other processes.
- **Budget tape memory and plan checkpointing.** Backpropagating through a
  multi-step loop stores intermediates per step, so tape memory scales with
  timesteps times `n_boxes × n_particles × n_species` fp64 state and can
  dominate the memory budget. Evaluate gradient checkpointing (recompute
  segments of the forward pass during the backward pass) before committing to
  long differentiable loops, and include tape storage in the Epic D
  memory-budget model.
- **Stochastic coagulation blocks state gradients as currently implemented.** The
  GPU Brownian coagulation kernel uses stochastic acceptance-rejection and
  discrete pair selection, so gradients do not flow through it back to the
  initial state, even though its parameters are fixed. How coagulation
  participates in the initial-state inversion is an **open decision**. Candidate
  approaches:
  - A deterministic binned or sectional coagulation operator (Smoluchowski) for
    the gradient path, distinct from the particle-resolved stochastic forward
    model, validated to agree in the mean.
  - A relaxed or differentiable particle-resolved formulation if per-particle
    fidelity is required.
- **Mixing state and hygroscopicity favor composition-resolved state.** Fitting
  initial versus final mixing state or hygroscopicity requires tracking
  per-species composition, which coagulation redistributes when particles of
  different composition merge. Decide the representation that is both
  physically adequate and differentiable, for example composition-resolved
  sectional bins (size by composition) versus particle-resolved with a
  differentiable surrogate. Note that hygroscopicity targets also require the
  GPU activity/kappa support planned in
  [Epic B](#epic-b-gpu-physics-coverage-and-parity); a condensation operator
  without water-activity modeling cannot fit hygroscopicity.
- **Autodiff, graph capture, and RNG state interact.** Differentiable execution,
  graph-captured timestep loops, and per-box RNG streams must coexist for
  gradient-based fitting. Validate them together, not only in isolation.
- **Multi-box supports optimization directly.** Evaluating many boxes per
  optimization step provides the batch of conditions used to match experiments
  or observations.
- Define the loss functions on state, for example distance between initial and
  final size distributions, hygroscopicity distributions, or mixing-state
  metrics, and confirm each has a differentiable path back to the initial state.

## Epic F: API Stability, Validation Infrastructure, and Documentation

Cross-cutting work that runs alongside the other epics.

### CPU Fallback and API Stability

- Mark low-level `particula.gpu.*` APIs as experimental until high-level backend
  selection and full-loop validation are in place.
- Decide whether missing GPU processes should raise a clear error, fall back to
  CPU, or require an explicit CPU/GPU synchronization boundary.
- Prefer explicit fallback boundaries for scientific reproducibility; avoid
  silently moving data between CPU and GPU in long simulations.

### Validation Infrastructure

- Adopt device-aware pytest execution as project policy: detect CUDA at test
  time and run GPU parity tests on `cpu` plus `cuda` devices when available,
  falling back to Warp CPU otherwise. This supports local validation on GPU
  workstations today and enables CI coverage automatically if a GPU runner is
  added later.
- Keep benchmarks opt-in (`--benchmark` style gating) and CUDA-gated, separate
  from the default parity suite.
- Document the current CUDA validation cadence (local/manual before releases)
  until GPU CI exists.

### Documentation and Examples

- Add a GPU quick-start example using `ParticleData`, `GasData`, and Warp
  transfer helpers, including `gpu_context` and the `WARP_AVAILABLE` flag for
  environments without Warp or CUDA.
- Add an end-to-end GPU-resident simulation example that runs multiple
  timesteps before transferring data back to CPU.
- Add a full GPU simulation example once condensation, coagulation, wall loss,
  and dilution have GPU implementations.
- Add a Warp graph-capture example for a fixed process sequence and explain when
  graph capture is useful.
- Document installation and environment expectations for `warp-lang`, Warp CPU,
  and CUDA devices.
- Explain current limitations clearly so users know when to choose CPU versus
  GPU execution.

## Risks and Key Decisions

| Risk / Decision | Impact | Direction |
| --- | --- | --- |
| fp64 throughput on consumer CUDA | Slower GPU, weaker speedups | Keep fp64 as reference; evaluate mixed precision only after a resolution study |
| NPF-to-droplet dynamic range near fp64 limits | Lost small-mass resolution | Study mass storage representation (per-species, log-mass, or binned) |
| fp64 memory doubling vs large multi-box | Fewer boxes fit in memory | Build a memory-budget model; treat precision and box count together |
| Time-scale stiffness across NPF-to-droplet range | Unstable or wastefully slow explicit stepping | Characterize stiffness; evaluate sub-stepping and semi-implicit schemes that stay capture- and autodiff-friendly |
| No per-box thermodynamic state container | Blocks parcels, expansion, latent-heat feedback | Add `EnvironmentData`/`WarpEnvironmentData` with per-box T, P, humidity |
| Rejection-sampling acceptance collapse for wide size ranges | Coagulation trial counts explode in mixed NPF/droplet boxes | Evaluate binned majorant kernels or stratified pair sampling |
| One-thread-per-box coagulation | Serializes large single-box workloads | Record as deliberate multi-box tradeoff or add parallel-within-box variant |
| RNG state re-initialized every coagulation step | Correlated draws across steps; frozen seed under graph capture | Seed once, persist per-box RNG state between timesteps |
| Autodiff through stochastic coagulation | Blocks global optimization if coag must be fit | Open decision; start with deterministic condensation, defer coag choice |
| In-place kernels break gradients | Autodiff path unusable | Author differentiable-friendly kernel variants for the optimization path |
| Tape memory for multi-step differentiable loops | Gradient runs exhaust GPU memory | Budget tape storage; evaluate gradient checkpointing |
| RNG reproducibility across CPU/Warp CPU/CUDA | Non-reproducible stochastic results | Per-box RNG streams; document tolerances and reproducibility limits |
| `WarpGasData` schema drift from `GasData` | Silent field mismatches; placeholder names break name-keyed logic | Define authoritative container per field; test round-trips |
| Graph capture fragility (shape/process changes) | Invalid captured graphs | Fixed shapes, inactive slots, documented re-capture triggers |
| GPU condensation lacks activity/kappa physics | Hygroscopicity and mixing-state targets unfittable | Plan GPU activity/surface-tension support in Epic B before Epic E targets |

## Suggested Milestones

Each milestone lists an exit bar so "done" is measurable, and maps to the
epics above.

### Milestone 1: Documented Low-Level GPU API (Epic F)

- Publish examples for direct use of `to_warp_particle_data`,
  `to_warp_gas_data`, `condensation_step_gpu`, and `coagulation_step_gpu`,
  including `gpu_context` and `WARP_AVAILABLE` for environments without CUDA.
- Add troubleshooting notes for missing Warp, missing CUDA, and device mismatch
  errors.
- Resolve whether `condensation_step_gpu` and `coagulation_step_gpu` are exported
  from top-level `particula.gpu` or documented under `particula.gpu.kernels`
  (today they are only importable from `particula.gpu.kernels`).
- **Exit bar:** A new user can run both kernels from the docs on a CUDA device
  and on Warp CPU without reading source.

### Milestone 2: Foundations and Backend Selection (Epics A, C)

- Add the `EnvironmentData`/`WarpEnvironmentData` containers with round-trip
  conversion and tests.
- Add an API design for selecting CPU or GPU execution from user-facing
  simulation code.
- Implement backend selection for at least one condensation workflow and one
  Brownian coagulation workflow.
- Keep return types and mutation semantics explicit.
- Adopt the device-aware pytest policy for GPU parity tests.
- **Exit bar:** At least one condensation and one coagulation workflow run on GPU
  through the selection API, match CPU within recorded tolerance, and are used in
  a documented example; `EnvironmentData` round-trips are tested.

### Milestone 2.5: Missing GPU Physics (Epic B)

- Fix RNG state persistence in `coagulation_step_gpu` (seed once, persist
  per-box state between steps).
- Implement and validate Warp-backed latent-heat condensation, including
  per-box temperature feedback through `EnvironmentData` and on-device vapor
  pressure recomputation.
- Implement and validate charged particle-resolved coagulation on the GPU.
- Implement and validate GPU sedimentation coagulation.
- Implement and validate GPU simple turbulent shear coagulation.
- Add the CPU `Dilution` strategy/runnable reference, then implement and
  validate GPU wall loss and dilution kernels needed for complete GPU-resident
  aerosol simulations.
- Decide the nucleation/particle-source design and land the CPU reference
  implementation following the
  [nucleation equations](../../Theory/Technical/Dynamics/Nucleation_Equations.md)
  (GPU slot-activation version may land in Milestone 3).
- Decide the GPU activity/surface-tension scope needed for water uptake and
  kappa-hygroscopicity.
- Document which CPU strategy options are still unsupported on GPU (including
  staggered condensation, which stays CPU-only).
- **Exit bar:** Each new kernel has CPU/GPU parity (or statistically bounded)
  tests, and a complete GPU-resident timestep can run condensation, coagulation,
  wall loss, and dilution together with persistent RNG state.

### Milestone 2.75: Differentiable Condensation and Global Optimization (Epic E)

- Author differentiable-friendly condensation kernels compatible with `wp.Tape`.
- Demonstrate an end-to-end gradient from a final-state loss back to the initial
  state through a multi-step condensation loop, recording tape memory usage and
  the checkpointing approach if needed.
- Implement one global-optimization example that recovers an initial size
  distribution from a synthetic final distribution across multiple boxes, with
  process parameters held fixed.
- Record the open decision for differentiable coagulation and the state
  representation for mixing-state and hygroscopicity targets.
- **Exit bar:** A gradient-based optimizer recovers a known initial state from
  synthetic multi-box final-state data within recorded tolerance, using GPU
  autodiff, without fitting any process parameters.

### Milestone 3: Production GPU Workflows (Epics C, D)

- Add multi-step GPU-resident examples and benchmark reports.
- Expand parity coverage to realistic aerosol scenarios.
- Add full-loop CPU, GPU, and graph-captured GPU validation tests.
- Add Warp graph-capture support for stable, repeated timestep loops.
- Publish multi-box scaling benchmarks that vary box count as the primary axis.
- Land the GPU nucleation/particle-source process with slot-activation and
  conservation tests, if the Milestone 2.5 decision calls for it.
- Decide which GPU APIs are stable enough to document as supported public APIs.
- **Exit bar:** A multi-box GPU-resident simulation runs graph-captured
  timesteps, matches CPU and uncaptured GPU references, and has published
  scaling numbers across box counts.
