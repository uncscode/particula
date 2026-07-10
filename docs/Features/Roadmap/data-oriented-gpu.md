# Data-Oriented Design and GPU Roadmap

This page tracks the status of the data-oriented container migration and the
Warp-backed GPU work for particle-resolved aerosol simulations.

The roadmap is a single ordered sequence of epics with explicit boundaries.
Epics are worked in the order below: when an epic meets its exit bar, the
next pending epic in the sequence becomes the active one. Cross-cutting
documentation, validation-infrastructure, and API-stability work is folded
into each epic's feature list rather than running as a separate parallel
epic.

Sizing convention: each epic targets roughly 5-10 features, each feature
roughly 5-15 phases, and each phase roughly 100 lines of new source code plus
its tests and documentation (canonical rules:
`.opencode/guides/phase-sizing-rules.md`).

## Epic Sequence and Status

| Order | Epic | Status | ADW plan |
| --- | --- | --- | --- |
| 1 | [Epic A: Data-Model and Numerical Foundations](#epic-a-data-model-and-numerical-foundations) | Shipped | E2 |
| 2 | [Epic B: Non-Isothermal Condensation Public API (CPU)](#epic-b-non-isothermal-condensation-public-api-cpu) | Shipped | E1 |
| 3 | [Epic C: GPU Kernel Correctness and Low-Level API Hardening](#epic-c-gpu-kernel-correctness-and-low-level-api-hardening) | Next up | not scheduled |
| 4 | [Epic D: GPU Condensation Physics Parity](#epic-d-gpu-condensation-physics-parity) | Pending | not scheduled |
| 5 | [Epic E: GPU Coagulation Physics Coverage](#epic-e-gpu-coagulation-physics-coverage) | Pending | not scheduled |
| 6 | [Epic F: GPU Process Completeness](#epic-f-gpu-process-completeness) | Pending | not scheduled |
| 7 | [Epic G: Backend Selection and GPU-Resident Simulation](#epic-g-backend-selection-and-gpu-resident-simulation) | Pending | not scheduled |
| 8 | [Epic H: Graph Capture and Performance](#epic-h-graph-capture-and-performance) | Pending | not scheduled |
| 9 | [Epic I: Differentiability and Global Optimization](#epic-i-differentiability-and-global-optimization) | Pending | not scheduled |

The former suggested milestones are absorbed into the per-epic exit bars:
Milestone 1 (documented low-level GPU API) is Epic C, Milestone 2 (backend
selection) is Epic G, Milestone 2.5 (missing GPU physics) is Epics D-F,
Milestone 2.75 (differentiable condensation) is Epic I, and Milestone 3
(production GPU workflows) is the combined Epic G and Epic H exit bars.

Quick links:

- [Current container schema inventory](#current-container-schema-inventory)
- [Shipped E2 foundation baseline](#shipped-e2-foundation-baseline)
- [Authoritative field ownership decisions](#authoritative-field-ownership-decisions)
- [Shipped foundation guide](../data-containers-and-gpu-foundations.md)
- [Runnable Data Containers example](../../Examples/Data_Containers/index.md)
- [Final downstream handoff map for sibling features](#final-downstream-handoff-map-for-sibling-features)

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
process does not exist in particula today and is added as an
[Epic F](#epic-f-gpu-process-completeness) work item so freshly formed
particles can enter GPU-resident simulations through slot activation.

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
  addition to legacy facades, but that compatibility is a container-boundary
  statement and not proof of shipped CPU multi-box execution across every box.
- Migration documentation exists in
  [ParticleData and GasData Migration](../particle-data-migration.md).

For the canonical user-facing CPU support boundary, including the current
`n_boxes == 1` limitation for audited CPU condensation and CPU coagulation
container workflows, refer to
[Data Containers and GPU Foundations](../data-containers-and-gpu-foundations.md).
Use
[ParticleData and GasData Migration](../particle-data-migration.md)
as the implementation walkthrough companion.

The work is still a migration, not a full replacement. `ParticleRepresentation`
and `GasSpecies` remain available for compatibility.

Known data-model gaps:

- `EnvironmentData` now provides a CPU-side home for per-box thermodynamic
  state, with `temperature -> (n_boxes,)`, `pressure -> (n_boxes,)`, and
  `saturation_ratio -> (n_boxes, n_species)`. The shipped baseline is
  available from `particula.gas.environment_data`, exported from
  `particula.gas`, supports `copy()` for CPU-side state handling, and only
  crosses the CPU↔GPU boundary through public `particula.gpu` helpers:
  `WarpEnvironmentData`, `to_warp_environment_data()`, and
  `from_warp_environment_data()`. Those helpers are the explicit transfer
  boundary: kernels and runnables do not perform hidden CPU↔GPU environment
  synchronization or movement. Broad runtime integration still remains
  downstream work (see [EnvironmentData Container](#environmentdata-container)).
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
- Parity tests parametrize over available Warp devices, always running on Warp
  CPU and additionally on Warp CUDA when a device is present. CUDA is optional,
  not a hard runtime or test requirement, and that parity coverage documents
  the explicit helper surface rather than a shipped automatic GPU runtime
  integration.

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

- `coagulation_step_gpu` supports a convenience allocate-and-seed path when
  `rng_states` are omitted, while caller-owned `rng_states` persist across
  repeated calls until explicitly reset with `initialize_rng=True`.
- The Brownian coagulation kernel launches one thread per box with sequential
  pair selection inside the thread, which limits single-box scaling at large
  particle counts.
- Rejection sampling uses a single `k_max` bound from the min/max radius pair,
  which degrades acceptance rates when small particles and large droplets
  coexist in one box.

## Epic A: Data-Model and Numerical Foundations

Status: shipped as plan E2. This foundation work now defines the container
schemas, per-box thermodynamic state, precision baseline, mass representation
policy, and time-integration recommendation that downstream GPU roadmap work
must build on.

### Shipped E2 Foundation Baseline

E2 closed the foundation scope needed by later GPU roadmap epics:

- `ParticleData`, `GasData`, `WarpParticleData`, and `WarpGasData` have a
  documented field-ownership and shape contract, including leading `n_boxes`
  dimensions for per-box state and species-only shapes for shared material or
  gas metadata.
- `EnvironmentData` is the CPU owner for per-box `temperature`, `pressure`, and
  per-box/per-species `saturation_ratio`; simulation volume remains owned by
  `ParticleData.volume`.
- `WarpEnvironmentData`, `to_warp_environment_data()`, and
  `from_warp_environment_data()` provide the explicit CPU↔GPU environment
  transfer boundary. Kernels and runnables still do not perform hidden
  CPU/GPU synchronization or movement.
- Gas CPU/GPU restore semantics are locked down: `GasData.name` is CPU-only
  ordered metadata, `partitioning` converts `bool` on CPU to `int32` on GPU,
  and `WarpGasData.vapor_pressure` is GPU helper state dropped on CPU restore.
- The low-level GPU condensation and coagulation environment inputs now share
  a scalar-or-per-box normalization contract, including explicit
  `WarpEnvironmentData` inputs.
- The mass-precision report keeps absolute per-species `np.float64` /
  `wp.float64` particle mass storage as the production baseline and defers any
  dtype or schema migration until stronger evidence exists.
- The condensation-stiffness study recommends `fixed_count_substeps_4` as the
  fixed-shape foundation for later GPU condensation integration, while keeping
  gas-coupled production support deferred.
- The foundation guide, migration guide, data-container example, and support
  boundary docs now distinguish multi-box-capable storage from still-limited
  CPU process execution.

### Post-E2 Data-Oriented Work

- Finish reducing dependence on legacy facade objects in new examples and
  documentation.
- Decide when `ParticleRepresentation` and `GasSpecies` should move from
  compatibility facades to deprecated APIs.
- Keep strategy APIs consistent when accepting either legacy facades or data
  containers.
- Expand examples that start directly from `ParticleData` and `GasData` instead
  of converting from legacy objects.
- Expand implementation-facing examples and migration notes that point back to
  the canonical shape-conventions section for single-box, multi-box, binned,
  and particle-resolved simulations.

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
| `ParticleData` | `density` | `ParticleData` | `(n_species,)` | `float64` | Stored, mutable, shared across boxes | `ParticleData.__post_init__()` requires a 1D density array, expands a length-1 density array to `n_species`, and fills empty density with zeros when `n_species > 0` | `to_warp_particle_data()` mirrors the shared 1D array to `WarpParticleData.density`; `from_warp_particle_data()` restores it to CPU storage | Source: `particula/particles/particle_data.py:79,119-137`, `particula/gpu/conversion.py:117-119,285`; tests: `particula/particles/tests/particle_data_test.py:208-234`, `particula/gpu/tests/warp_types_test.py:38-58` |
| `ParticleData` | `volume` | `ParticleData` | `(n_boxes,)` | `float64` | Stored, mutable, per-box simulation volume | `ParticleData.__post_init__()` enforces `(n_boxes,)` | Round-trips through `to_warp_particle_data()` / `from_warp_particle_data()` without schema change | Source: `particula/particles/particle_data.py:80,111-117`, `particula/gpu/conversion.py:120,286`; tests: `particula/particles/tests/particle_data_test.py:194-206`, `particula/gpu/tests/conversion_test.py:530-548` |
| `GasData` | `name` | `GasData` | `len == n_species` | `list[str]` | Stored, mutable CPU-only species metadata | `GasData.__post_init__()` rejects an empty species list; it does not enforce uniqueness or survive GPU transfer on its own | Not transferred by `to_warp_gas_data()` because `WarpGasData` has no string field; `from_warp_gas_data()` restores caller-supplied names or generates placeholders such as `species_0`, and current restore validation checks only the supplied name-list length | Source: `GasData.__post_init__()` and `from_warp_gas_data()` in `particula/gas/gas_data.py` and `particula/gpu/conversion.py`; tests: `particula/gas/tests/gas_data_test.py:119-127`, `particula/gpu/tests/conversion_test.py:600-640` |
| `GasData` | `molar_mass` | `GasData` | `(n_species,)` | `float64` | Stored, mutable, shared across boxes | `GasData.__post_init__()` coerces to `np.float64` and enforces `(n_species,)` | `to_warp_gas_data()` mirrors it to `WarpGasData.molar_mass`; `from_warp_gas_data()` restores the same numeric field | Source: `particula/gas/gas_data.py:65,75-77,103-108`, `particula/gpu/conversion.py:214-216,354`; tests: `particula/gpu/tests/warp_types_test.py:168-184,218-232`, `particula/gpu/tests/conversion_test.py:583-595` |
| `GasData` | `concentration` | `GasData` | `(n_boxes, n_species)` | `float64` | Stored, mutable, box-batched gas mass concentration | `GasData.__post_init__()` coerces to `np.float64`, requires 2D, and checks width against `n_species` | `to_warp_gas_data()` mirrors it to `WarpGasData.concentration`; `from_warp_gas_data()` restores the same shape | Source: `particula/gas/gas_data.py:66,76-78,89-101`, `particula/gpu/conversion.py:217-219,355`; tests: `particula/gas/tests/gas_data_test.py:99-117`, `particula/gpu/tests/conversion_test.py:593-595` |
| `GasData` | `partitioning` | `GasData` | `(n_species,)` | `bool` | Stored, mutable, shared-across-boxes partitioning mask | `GasData.__post_init__()` coerces with `np.asarray(..., dtype=np.bool_)` and enforces `(n_species,)`; this is dtype coercion, not a stricter semantic validation layer beyond NumPy truthiness | `to_warp_gas_data()` converts `bool → int32`; `from_warp_gas_data()` converts `int32 → bool` on restore | Source: `GasData.__post_init__()`, `to_warp_gas_data()`, and `from_warp_gas_data()` in `particula/gas/gas_data.py` and `particula/gpu/conversion.py`; tests: `particula/gas/tests/gas_data_test.py:77-97`, `particula/gpu/tests/conversion_test.py:189-199,609-619` |
| `WarpParticleData` | `masses` | `WarpParticleData` | `(n_boxes, n_particles, n_species)` | `wp.float64` | Stored, mutable GPU mirror of `ParticleData.masses` | Declared as `wp.array3d(dtype=wp.float64)`; populated by `to_warp_particle_data()` | Receives CPU particle masses on transfer to GPU; `from_warp_particle_data()` returns the same values to `ParticleData` | Source: `particula/gpu/warp_types.py:73`, `particula/gpu/conversion.py:111-125,282`; tests: `particula/gpu/tests/warp_types_test.py:38-58,100-118`, `particula/gpu/tests/conversion_test.py:512-548` |
| `WarpParticleData` | `concentration` | `WarpParticleData` | `(n_boxes, n_particles)` | `wp.float64` | Stored, mutable GPU mirror of `ParticleData.concentration` | Declared as `wp.array2d(dtype=wp.float64)`; populated by `to_warp_particle_data()` | Receives CPU particle concentration/count state and round-trips back unchanged | Source: `particula/gpu/warp_types.py:74`, `particula/gpu/conversion.py:113-128,283`; tests: `particula/gpu/tests/warp_types_test.py:54-58,113-118`, `particula/gpu/tests/conversion_test.py:521-545` |
| `WarpParticleData` | `charge` | `WarpParticleData` | `(n_boxes, n_particles)` | `wp.float64` | Stored, mutable GPU mirror of `ParticleData.charge` | Declared as `wp.array2d(dtype=wp.float64)`; populated by `to_warp_particle_data()` | Receives CPU charge data and round-trips back unchanged | Source: `particula/gpu/warp_types.py:75`, `particula/gpu/conversion.py:116,129-131,284`; tests: `particula/gpu/tests/warp_types_test.py:55-58,114-118`, `particula/gpu/tests/conversion_test.py:524-526,546-548` |
| `WarpParticleData` | `density` | `WarpParticleData` | `(n_species,)` | `wp.float64` | Stored, mutable GPU mirror of the shared-across-boxes `ParticleData.density` array | Declared as `wp.array(dtype=wp.float64)`; populated by `to_warp_particle_data()` | Receives shared CPU density array and restores it on `from_warp_particle_data()` | Source: `particula/gpu/warp_types.py:76`, `particula/gpu/conversion.py:117-119,132-134,285`; tests: `particula/gpu/tests/warp_types_test.py:57-58,113-118,302-320`, `particula/gpu/tests/conversion_test.py:527-529` |
| `WarpParticleData` | `volume` | `WarpParticleData` | `(n_boxes,)` | `wp.float64` | Stored, mutable GPU mirror of per-box `ParticleData.volume` | Declared as `wp.array(dtype=wp.float64)`; populated by `to_warp_particle_data()` | Receives per-box CPU volume array and restores it on `from_warp_particle_data()` | Source: `particula/gpu/warp_types.py:77`, `particula/gpu/conversion.py:120,135-137,286`; tests: `particula/gpu/tests/warp_types_test.py:57-58,113-118,302-320`, `particula/gpu/tests/conversion_test.py:530-548` |
| `WarpGasData` | `molar_mass` | `WarpGasData` | `(n_species,)` | `wp.float64` | Stored, mutable GPU mirror of `GasData.molar_mass` shared across boxes | Declared as `wp.array(dtype=wp.float64)`; populated by `to_warp_gas_data()` from CPU gas state | Restores the same numeric field through `from_warp_gas_data()` within the intentionally lossy gas boundary (`name` remains CPU-only and `vapor_pressure` is still dropped on CPU restore) | Source: `particula/gpu/warp_types.py:131`, `particula/gpu/conversion.py:214-216,228-230,354`; tests: `particula/gpu/tests/warp_types_test.py:168-184,218-232`, `particula/gpu/tests/conversion_test.py:583-595` |
| `WarpGasData` | `concentration` | `WarpGasData` | `(n_boxes, n_species)` | `wp.float64` | Stored, mutable GPU mirror of `GasData.concentration` | Declared as `wp.array2d(dtype=wp.float64)`; populated by `to_warp_gas_data()` from CPU gas state | Restores the same box-batched numeric field through `from_warp_gas_data()` within the intentionally lossy gas boundary (`name` remains caller-managed CPU metadata and `vapor_pressure` stays GPU-only) | Source: `particula/gpu/warp_types.py:132`, `particula/gpu/conversion.py:217-219,231-233,355`; tests: `particula/gpu/tests/warp_types_test.py:180-184,229-231`, `particula/gpu/tests/conversion_test.py:593-595` |
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
  preserves string species names, so callers must preserve ordered species
  metadata externally if they need a lossless semantic restore.
- `WarpGasData.vapor_pressure` is GPU-only helper state. It defaults to zeros
  in `to_warp_gas_data()` when omitted and is dropped when restoring CPU
  `GasData`.

#### CPU↔GPU restore boundary for ordered gas metadata

Keep testing round-trips for the current intentional CPU/GPU contract
explicitly:
`WarpGasData` drops `name`, stores `partitioning` as `int32` instead of
`bool`, and adds `vapor_pressure` that is not restored to the CPU gas
container.

### Authoritative field ownership decisions

This section is the canonical ownership and CPU↔GPU round-trip contract shipped
by E2. The inventory table above remains the current-state evidence record; use
the decision table below for policy when adding fields, conversion behavior, or
future GPU environment state.

| Field / group | Authoritative owner | CPU shape | GPU shape | Dtype | Mutability | Round-trip behavior | Downstream consumers | Evidence |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `ParticleData.masses` | Owned by `ParticleData` as the authoritative particle/species mass state | `(n_boxes, n_particles, n_species)` | `(n_boxes, n_particles, n_species)` via `WarpParticleData.masses` | `float64` on CPU / `wp.float64` on GPU | Mutable stored state on both containers | Must round-trip without schema drift through `to_warp_particle_data()` and `from_warp_particle_data()` | Particle property accessors, condensation, coagulation, and future GPU-resident timestep loops | `particula/particles/particle_data.py:76-98`; `particula/gpu/conversion.py:72-139,244-287`; `particula/particles/tests/particle_data_test.py:124-136`; `particula/gpu/tests/conversion_test.py:512-548` |
| `ParticleData.concentration` | Owned by `ParticleData` as authoritative per-box particle concentration/count state | `(n_boxes, n_particles)` | `(n_boxes, n_particles)` via `WarpParticleData.concentration` | `float64` / `wp.float64` | Mutable stored state on both containers | Must round-trip without schema change | Particle dynamics kernels and any box-batched particle workflow | `particula/particles/particle_data.py:76-109`; `particula/gpu/conversion.py:113-115,283`; `particula/particles/tests/particle_data_test.py:138-164`; `particula/gpu/tests/conversion_test.py:521-545` |
| `ParticleData.charge` | Owned by `ParticleData` as authoritative particle charge state | `(n_boxes, n_particles)` | `(n_boxes, n_particles)` via `WarpParticleData.charge` | `float64` / `wp.float64` | Mutable stored state on both containers | Must round-trip without dtype drift | Charged coagulation follow-on work and GPU parity paths that consume particle charge | `particula/particles/particle_data.py:76-109`; `particula/gpu/conversion.py:116,284`; `particula/particles/tests/particle_data_test.py:166-192`; `particula/gpu/tests/conversion_test.py:524-526,546-548` |
| `ParticleData.density` | Owned by `ParticleData` and must remain shared-across-boxes material state, not per-box environment state | `(n_species,)` | `(n_species,)` via `WarpParticleData.density` | `float64` / `wp.float64` | Mutable stored state, but shared across boxes | Must round-trip as shared 1D species density state | Radius, effective-density, and mass-fraction calculations plus future GPU particle property parity | `particula/particles/particle_data.py:67-68,119-137`; `particula/gpu/conversion.py:117-119,285`; `particula/particles/tests/particle_data_test.py:208-234`; `particula/gpu/tests/warp_types_test.py:38-58` |
| `ParticleData.volume` | Owned by `ParticleData` as the authoritative per-box simulation-volume carrier | `(n_boxes,)` | `(n_boxes,)` via `WarpParticleData.volume` | `float64` / `wp.float64` | Mutable stored state on both containers | Must round-trip without schema change | Per-box particle workflows, dilution-style process work, and future environment/process coordination | `particula/particles/particle_data.py:69-70,111-117`; `particula/gpu/conversion.py:120,286`; `particula/particles/tests/particle_data_test.py:194-206`; `particula/gpu/tests/conversion_test.py:530-548` |
| `GasData.name` | Owned by CPU `GasData` as authoritative species-name metadata | `len == n_species` | Not owned on GPU | `list[str]` | Mutable CPU metadata only | Does not survive transfer on its own; CPU restore requires caller-supplied ordered species names from external metadata, but the current `from_warp_gas_data()` implementation validates only name-list length before reconstructing `GasData`, so ordering must be preserved by the caller rather than inferred or checked during restore | CPU-facing reporting, facade compatibility, and any restore path back to `GasData` | `particula/gas/gas_data.py:52-73`; `particula/gpu/warp_types.py:82-99`; `particula/gpu/conversion.py:155-157,301-345`; `particula/gas/tests/gas_data_test.py:119-127`; `particula/gpu/tests/conversion_test.py:600-640` |
| `GasData.molar_mass` | Owned by `GasData` as authoritative gas species molar-mass state | `(n_species,)` | `(n_species,)` via `WarpGasData.molar_mass` | `float64` / `wp.float64` | Mutable stored state on both containers | Must round-trip with the same numeric values inside the documented gas restore boundary (`name` still requires caller-managed ordered metadata and `vapor_pressure` still has no CPU field) | Gas property calculations, condensation, and GPU gas kernels | `particula/gas/gas_data.py:53-67,75-108`; `particula/gpu/warp_types.py:100-111,131`; `particula/gpu/conversion.py:214-216,354`; `particula/gpu/tests/warp_types_test.py:168-184,218-232`; `particula/gpu/tests/conversion_test.py:583-595` |
| `GasData.concentration` | Owned by `GasData` as authoritative per-box gas concentration state | `(n_boxes, n_species)` | `(n_boxes, n_species)` via `WarpGasData.concentration` | `float64` / `wp.float64` | Mutable stored state on both containers | Must round-trip without shape drift | Condensation, gas-phase workflows, and future GPU-resident gas updates | `particula/gas/gas_data.py:55-57,76-101`; `particula/gpu/warp_types.py:104-105,132`; `particula/gpu/conversion.py:217-219,355`; `particula/gas/tests/gas_data_test.py:99-117`; `particula/gpu/tests/conversion_test.py:593-595` |
| `GasData.partitioning` | Owned by `GasData` as authoritative shared-across-boxes partitioning eligibility state | `(n_species,)` | `(n_species,)` via `WarpGasData.partitioning` | `bool` on CPU / `wp.int32` on GPU | Mutable stored state on both containers | Must round-trip with explicit `bool → int32 → bool` conversion | Condensation partitioning decisions and GPU kernels that require a numeric mask | `particula/gas/gas_data.py:57-58,79-115`; `particula/gpu/warp_types.py:96-111,134`; `particula/gpu/conversion.py:159-160,208-209,223-239,347-356`; `particula/gas/tests/gas_data_test.py:77-97`; `particula/gpu/tests/conversion_test.py:189-199,609-619` |
| `WarpParticleData` numeric mirrors | Owned on GPU only as mirrors of authoritative `ParticleData` fields, not as a separate source of truth | Mirrors CPU particle shapes | Stored on GPU as declared in `WarpParticleData` | `wp.float64` | Mutable GPU working state | Must restore to the corresponding `ParticleData` fields without adding or dropping particle schema | GPU-resident particle workflows and parity tests | `particula/gpu/warp_types.py:73-77`; `particula/gpu/conversion.py:111-137,281-287`; `particula/gpu/tests/warp_types_test.py:38-58,100-118,302-320`; `particula/gpu/tests/conversion_test.py:512-548` |
| `WarpGasData.molar_mass` / `concentration` / `partitioning` | Owned on GPU only as numeric mirrors/helpers of authoritative CPU `GasData` state | `(n_species,)`, `(n_boxes, n_species)`, `(n_species,)` | Same declared GPU shapes | `wp.float64` / `wp.float64` / `wp.int32` | Mutable GPU working state | Must restore to `GasData` numeric state, with explicit `int32 → bool` recovery for `partitioning` | GPU condensation and other gas-kernel workflows | `particula/gpu/warp_types.py:82-99,100-111,131-134`; `particula/gpu/conversion.py:142-241,290-357`; `particula/gpu/tests/warp_types_test.py:168-184,218-232,322-338`; `particula/gpu/tests/conversion_test.py:189-229,583-619` |
| `WarpGasData.vapor_pressure` | Owned by no CPU container; treated as GPU-helper/process state rather than authoritative `GasData` or shipped CPU `EnvironmentData` state | Not owned on CPU containers | `(n_boxes, n_species)` via `WarpGasData.vapor_pressure` | `wp.float64` | Mutable GPU helper state | Must be recomputed from the current authoritative thermodynamic inputs, explicitly provided, or carried as sidecar state; update order must prevent mixed stale/new state, and CPU restore from `WarpGasData` remains intentionally lossy because `from_warp_gas_data()` drops it | GPU condensation kernels and future on-device thermodynamic updates | `particula/gpu/warp_types.py:98-108,133`; `particula/gpu/conversion.py:162-206,220-241,301-304`; `particula/gpu/tests/warp_types_test.py:177-183,225-231`; `particula/gpu/tests/conversion_test.py:200-229,484-496,588-595` |
| EnvironmentData.temperature | Owned by shipped CPU `EnvironmentData`, not by `ParticleData` or `GasData` | `(n_boxes,)` | `(n_boxes,)` via shipped `WarpEnvironmentData` | `float64` on CPU / `wp.float64` on GPU | Mutable per-box thermodynamic state | Round-trips through `to_warp_environment_data()` / `from_warp_environment_data()`; broader GPU-resident runtime ownership remains future work | Parcel/expansion workflows, latent-heat condensation, and other per-box thermodynamic updates | `particula/gas/environment_data.py`; `particula/gpu/warp_types.py`; `particula/gpu/conversion.py`; `particula/gpu/tests/conversion_test.py` |
| EnvironmentData.pressure | Owned by shipped CPU `EnvironmentData`, not by `ParticleData` or `GasData` | `(n_boxes,)` | `(n_boxes,)` via shipped `WarpEnvironmentData` | `float64` on CPU / `wp.float64` on GPU | Mutable per-box thermodynamic state | Round-trips through `to_warp_environment_data()` / `from_warp_environment_data()`; broader GPU-resident runtime ownership remains future work | Parcel/expansion workflows, kernel inputs, and per-box forcing profiles | `particula/gas/environment_data.py`; `particula/gpu/warp_types.py`; `particula/gpu/conversion.py`; `particula/gpu/tests/conversion_test.py` |
| EnvironmentData.saturation_ratio | Shipped CPU `EnvironmentData` helper state; not an independent source of truth separate from current environment and gas state | `(n_boxes, n_species)` | `(n_boxes, n_species)` via shipped `WarpEnvironmentData` or equivalent GPU environment state | `float64` on CPU / `wp.float64` on GPU | Mutable derived/cache state that must be refreshed after invalidating updates | Round-trips through `to_warp_environment_data()` / `from_warp_environment_data()` and still must be invalidated/recomputed after upstream state changes; high-level process integration remains future work | Latent-heat condensation, parcel expansion, and humidity-coupled follow-on work | `particula/gas/environment_data.py`; `particula/gpu/warp_types.py`; `particula/gpu/conversion.py`; `particula/gpu/tests/conversion_test.py` |
| Simulation volume ownership | Not owned by `EnvironmentData`; must remain owned by `ParticleData.volume` | `(n_boxes,)` on `ParticleData` | `(n_boxes,)` on `WarpParticleData` | `float64` / `wp.float64` | Mutable per-box simulation state under particle container ownership | Must continue to round-trip only with particle container conversion helpers | Per-box particle state, dilution-style workflows, and timestep orchestration that needs simulation volume | `particula/particles/particle_data.py:69-70,111-117`; `particula/gpu/conversion.py:120,286`; `particula/particles/tests/particle_data_test.py:194-206`; `.opencode/plans/sections/features/E2-F1/architecture_design.md:31-46` |

#### Canonical shape conventions for container workflows

Use this subsection as the single shape contract for container-first workflow
documentation. Per-box arrays always keep a leading `n_boxes` dimension, even
when `n_boxes == 1`. Shared arrays keep their species-only or metadata-only
shape and do not gain a box dimension just because the workflow is batched.

Workflow rules:

- Single-box workflows still store every per-box array with a leading `1`.
- Multi-box workflows store every per-box array with a leading `n_boxes`.
- Particle-resolved workflows use `ParticleData.masses`
  `(n_boxes, n_particles, n_species)` and `ParticleData.concentration`
  `(n_boxes, n_particles)`, where concentration carries count semantics.
- Binned workflows keep the same container ranks, but particle concentration is
  interpreted as number per m^3 rather than count.
- Shared arrays such as `ParticleData.density`, `GasData.molar_mass`, and
  `GasData.partitioning` remain species-only arrays with no box axis.
- Storage shape and interpretation semantics are separate concerns: a
  particle-resolved workflow and a binned workflow may use the same stored
  ranks while assigning different physical meaning to the values.

Concrete example shapes backed by current constructors and tests:

- `ParticleData.masses -> (1, n_particles, n_species)`
- `ParticleData.concentration -> (1, n_particles)`
- `GasData.concentration -> (1, n_species)`
- `ParticleData.density -> (n_species,)`
- `WarpGasData.vapor_pressure -> (n_boxes, n_species)`

Verification evidence for the shape claims in this subsection:

- `particula/particles/tests/particle_data_test.py` for single-box, multi-box,
  shared-density, and derived-shape behavior.
- `particula/gas/tests/gas_data_test.py` and
  `particula/gas/tests/gas_data_builder_test.py` for gas constructor and
  builder-added leading box dimensions.
- `particula/gpu/tests/warp_types_test.py` and
  `particula/gpu/tests/conversion_test.py` for Warp container parity and
  CPU↔GPU shape preservation.
- `particula/dynamics/condensation/tests/condensation_strategies_test.py` and
  `particula/dynamics/coagulation/coagulation_strategy/tests/coagulation_strategy_abc_test.py`,
  plus the current helper entry points they exercise, for the current
  single-box CPU execution boundary documented below.

`ParticleData`

Per-box fields:

| Field | Shape | Notes |
| --- | --- | --- |
| `masses` | `(n_boxes, n_particles, n_species)` | Same stored rank for single-box, multi-box, particle-resolved, and binned workflows. |
| `concentration` | `(n_boxes, n_particles)` | Count semantics for particle-resolved workflows; number per m^3 semantics for binned workflows. |
| `charge` | `(n_boxes, n_particles)` | Per-box particle charge state. |
| `volume` | `(n_boxes,)` | Per-box simulation volume remains owned by `ParticleData`. |

Shared-across-box fields:

| Field | Shape | Notes |
| --- | --- | --- |
| `density` | `(n_species,)` | Shared material density; never add a box dimension. |

`GasData`

Per-box fields:

| Field | Shape | Notes |
| --- | --- | --- |
| `concentration` | `(n_boxes, n_species)` | Single-box gas workflows still use `(1, n_species)`. |

Shared-across-box and metadata fields:

| Field | Shape | Notes |
| --- | --- | --- |
| `name` | `len == n_species` | CPU-only ordered species metadata. |
| `molar_mass` | `(n_species,)` | Shared across boxes. |
| `partitioning` | `(n_species,)` | Shared across boxes; stored as `bool` on CPU. |

Shipped CPU `EnvironmentData` fields:

Per-box fields:

| Field | Shape | Notes |
| --- | --- | --- |
| `temperature` | `(n_boxes,)` | Shipped per-box thermodynamic state owned by `EnvironmentData` on CPU. |
| `pressure` | `(n_boxes,)` | Shipped per-box thermodynamic state owned by `EnvironmentData` on CPU. |
| `saturation_ratio` | `(n_boxes, n_species)` | Shipped per-box, per-species thermodynamic helper state; refresh it after invalidating updates. |

`WarpParticleData`

Per-box fields:

| Field | Shape | Notes |
| --- | --- | --- |
| `masses` | `(n_boxes, n_particles, n_species)` | GPU mirror of `ParticleData.masses`. |
| `concentration` | `(n_boxes, n_particles)` | GPU mirror of particle concentration/count state. |
| `charge` | `(n_boxes, n_particles)` | GPU mirror of particle charge state. |
| `volume` | `(n_boxes,)` | GPU mirror of per-box simulation volume. |

Shared-across-box fields:

| Field | Shape | Notes |
| --- | --- | --- |
| `density` | `(n_species,)` | GPU mirror of shared particle density. |

`WarpGasData`

Per-box fields:

| Field | Shape | Notes |
| --- | --- | --- |
| `concentration` | `(n_boxes, n_species)` | GPU mirror of gas concentration. |
| `vapor_pressure` | `(n_boxes, n_species)` | GPU-only helper state; not restored to CPU `GasData`. |

Shared-across-box fields:

| Field | Shape | Notes |
| --- | --- | --- |
| `molar_mass` | `(n_species,)` | GPU mirror of shared gas molar mass. |
| `partitioning` | `(n_species,)` | GPU mirror/helper mask; stored as `int32` on GPU. |

#### Current CPU execution limits for multi-box-ready containers

!!! caution
    Container storage is already multi-box-capable, but shipped CPU execution
    remains single-box for the currently audited condensation and coagulation
    workflows. CPU condensation explicitly enforces `n_boxes == 1`, and CPU
    coagulation support for `ParticleData` is also documented and validated as
    `n_boxes == 1` only. For the canonical user-facing support contract, see
    [ParticleData and GasData Migration](../particle-data-migration.md).

#### Rationale for issue-critical ownership decisions

- `ParticleData.density` remains shared-across-boxes state with CPU/GPU shape
  `(n_species,)`; the constructor enforces 1D species density semantics rather
  than per-box thermodynamic ownership
  (`particula/particles/particle_data.py:67-68,119-137`; tests:
  `particula/particles/tests/particle_data_test.py:208-234`,
  `particula/gpu/tests/warp_types_test.py:38-58`).
- `ParticleData.volume` is the authoritative per-box simulation-volume carrier
  with shape `(n_boxes,)`, and shipped CPU `EnvironmentData` plus any future GPU
  mirrors must not own or mutate simulation volume
  (`particula/particles/particle_data.py:69-70,111-117`; tests:
  `particula/particles/tests/particle_data_test.py:194-206`;
  `.opencode/plans/sections/features/E2-F1/architecture_design.md:31-46`).
- `vapor_pressure` is process/GPU-helper state rather than owned CPU
  `GasData` or shipped CPU `EnvironmentData` state; CPU-facing workflows must
  recompute it or pass it as ordered sidecar state after any temperature or
  concentration update, and CPU restore from `WarpGasData` is intentionally
  lossy because `to_warp_gas_data()` injects zeros when absent and
  `from_warp_gas_data()` drops the field
  (`particula/gpu/conversion.py:162-206,301-304`; tests:
  `particula/gpu/tests/conversion_test.py:200-229,484-496,588-595`).
- `WarpGasData` is numeric-only; restoring CPU gas names requires
  caller-supplied ordered names or equivalent external metadata because the
  GPU container excludes string fields and `from_warp_gas_data()` otherwise
  generates placeholder names such as `species_0`. Current restore logic checks
  only metadata length against `n_species`, so preserving the intended species
  ordering is a caller responsibility rather than a restore-time guarantee
  (`particula/gpu/warp_types.py:82-99`;
  `particula/gpu/conversion.py:305-345`; tests:
  `particula/gpu/tests/conversion_test.py:600-640`).
- Shipped CPU environment ownership is `temperature: (n_boxes,)`, `pressure:
  (n_boxes,)`, and `saturation_ratio: (n_boxes, n_species)` under
  `EnvironmentData`, without moving simulation volume out of
  `ParticleData.volume`. `saturation_ratio` should be treated as a derived
  thermodynamic helper tied to the current environment and gas state, not as an
  independent source of truth. Any step that changes the upstream environment or
  gas state must invalidate and recompute it before downstream kernels consume
  it, so GPU/CPU paths never mix stale helpers with freshly updated particle or
  gas fields
  ([EnvironmentData Container](#environmentdata-container);
  `.opencode/plans/sections/features/E2-F1/architecture_design.md:31-46`).

#### Final downstream handoff map for sibling features

This phase publishes the finalized P2/P3 ownership and shape contract for
downstream implementers; it does not add new schema semantics.

- `E2-F2`: inherit shipped CPU `EnvironmentData.temperature -> (n_boxes,)`,
  `pressure -> (n_boxes,)`, and `saturation_ratio -> (n_boxes, n_species)`;
  keep `ParticleData.volume -> (n_boxes,)` as the authoritative simulation
  volume owner rather than moving volume into `EnvironmentData`.
- `E2-F3`: inherit the now-shipped exact CPU↔GPU schema mirroring between
  `EnvironmentData` and `WarpEnvironmentData`, with
  `temperature -> (n_boxes,)`, `pressure -> (n_boxes,)`, and
  `saturation_ratio -> (n_boxes, n_species)`; do not add an extra volume field
  or an implicit transfer path.
- `E2-F4`: inherit that `GasData.name` remains CPU-only ordered metadata,
  `GasData.partitioning` stays `bool` on CPU and `int32` on GPU, and
  `WarpGasData.vapor_pressure -> (n_boxes, n_species)` remains explicit
  GPU-helper state dropped on CPU restore; callers must preserve ordered
  species names outside `WarpGasData` and pass them back on restore because
  GPU→CPU restore is intentionally lossy for gas metadata and helper state.
- `E2-F5`: inherit the single-box leading-dimension contract for per-box arrays
  such as `ParticleData.masses -> (1, n_particles, n_species)`,
  `ParticleData.concentration -> (1, n_particles)`, and
  `GasData.concentration -> (1, n_species)` without changing the canonical
  storage ranks for multi-box-ready containers.
- `E2-F6`: inherit the current `float64` / `wp.float64` schema baseline for
  stored particle and gas arrays. Treat the
  [Mass Precision Recommendation Report](mass-precision-study.md) as the
  canonical downstream reference before any production dtype-or-schema
  migration proceeds.
- `E2-F7`: inherit the existing environment and gas ownership boundaries,
  including `ParticleData.volume` ownership and `WarpGasData.vapor_pressure` as
  lossy helper state, then build later condensation integration work on the P4
  recommendation in
  [`condensation-stiffness-study.md`](condensation-stiffness-study.md):
  `fixed_count_substeps_4` as the preferred fixed-shape foundation, bounded by
  the E2-F2 environment-shape contract, the E2-F6 `float64` evidence envelope,
  and the still-deferred gas-coupled production gate.
- `E2-F8`: inherit that container schemas are already multi-box capable through
  leading `n_boxes` dimensions, while current CPU condensation runtime support
  remains single-box and current CPU coagulation paths are still documented as
  single-box or explicitly transitional execution boundaries; do not infer
  broader multi-box CPU execution support from storage shape alone.
- `E2-F9`: start user-facing docs and examples from the shipped
  [Data Containers and GPU Foundations](../data-containers-and-gpu-foundations.md)
  guide and the
  [Data Containers example](../../Examples/Data_Containers/index.md), then
  link back to this roadmap page only for implementation-planning detail such
  as field ownership, shape tables, CPU↔GPU transfer caveats, validation
  evidence, and downstream handoff anchors.

- Keep the shipped particle mass storage representation unless a future
  production migration proposal satisfies the evidence requirements in the
  [Mass Precision Recommendation Report](mass-precision-study.md).

### EnvironmentData Container

Parcel/expansion workflows and latent-heat condensation require per-box
thermodynamic state. That baseline now exists as
`particula.gas.environment_data.EnvironmentData`, which validates per-box
`temperature`, `pressure`, and per-box/per-species `saturation_ratio`.
Current shipped scope is still intentionally narrow: it is a constructor-
validated CPU container, requires at least one box, is exported from
`particula.gas`, supports `copy()` for CPU-side state management, and now has
public CPU↔GPU round-trip helpers only through
`particula.gpu.WarpEnvironmentData`, `to_warp_environment_data()`, and
`from_warp_environment_data()`. What remains future work is direct process
integration and a broader GPU-resident runtime. Those named helpers are also
the only shipped environment transfer boundary; kernels and runnables do not
perform hidden CPU↔GPU environment movement or implicit synchronization.

- Preserve the current `EnvironmentData` shape contract: `temperature` and
  `pressure` are `(n_boxes,)`, and `saturation_ratio` is
  `(n_boxes, n_species)`.
- Build on the now-shipped `WarpEnvironmentData` mirror and CPU↔GPU conversion
  helpers when broader GPU-side thermodynamic state work begins.
- Existing process APIs may still accept scalar `temperature` and `pressure`
  until later migration work lands; only migrated process code should read
  `EnvironmentData` directly.
- Latent-heat condensation must read and update per-box temperature; rising
  parcels, expansion, and combustion boxes prescribe temperature and pressure
  per box, while any simulation-volume evolution continues through
  `ParticleData.volume` rather than `EnvironmentData` ownership.
- Define ownership and update ordering: which processes read environment
  state, which mutate it, where prescribed (user-supplied) profiles are
  applied within a timestep, and when derived thermodynamic helpers such as
  `saturation_ratio` and GPU-side `vapor_pressure` must be invalidated and
  recomputed. Treat environment state as read-only unless the physical model
  owns the update. `saturation_ratio` should remain a derived/cache field tied
  to current thermodynamic inputs rather than a separately authoritative state.
- Keep round-trip conversion coverage aligned with the existing
  `ParticleData`/`GasData` conversion patterns, including one-box and multi-box
  cases plus explicit synchronization behavior, with device-parametrized parity
  checks that always run on Warp `cpu` and add Warp `cuda` only when
  available. That test coverage confirms the helper surface only; it does not
  mean automatic environment transfers inside broader runtimes have shipped.
- Decide how existing kernel APIs that accept scalar temperature and pressure
  migrate to per-box arrays without breaking the current low-level API.

### Numerical Precision and Mass Resolution

The simulation must span new particle formation clusters through cloud droplets.
That is roughly fifteen orders of magnitude in particle mass, which sits near
the limit of double precision (~15-16 significant digits). This dynamic range,
not raw speed, is the main precision driver.

- **Keep fp64 as the reference and production baseline.** The E2 precision study
  found the current absolute per-species `np.float64` / `wp.float64` schema is
  the accepted baseline for shipped particle mass storage. Lower-precision or
  mixed-precision paths remain future proposals until they meet the report's
  conservation and small-particle fidelity evidence requirements.
- **Target mass resolution is on the order of 0.1 ng per tracked quantity** so
  that both small freshly formed particles and large droplets remain
  representable in the same simulation.
- **Representation changes are deferred, not part of the shipped baseline.**
  Per-species absolute mass remains the production representation. Log-mass,
  binned reference masses, and other alternatives require a future scoped
  migration proposal with reproducible evidence across the NPF-to-droplet cases.
- **fp64 doubles memory** relative to fp32, which directly taxes the large
  multi-box goal. Precision and memory budget must be evaluated together (see
  [Performance and Memory](#performance-and-memory)).
- **fp64 throughput is heavily reduced on consumer CUDA hardware.** Record which
  target devices matter, and whether a validated fp32 or mixed-precision path is
  needed for those devices.
- Use the shipped
  [Mass Precision Recommendation Report](mass-precision-study.md) as the
  acceptance gate before changing precision, dtype, or mass schema.

### Time-Scale Stiffness

The same NPF-to-droplet range that stresses mass precision also stresses time
integration: nanometer particles equilibrate with vapor in microseconds while
cloud droplets evolve over seconds. P2/P3 recorded the current particle-only
explicit timestep grid and compared two deterministic fixed-shape candidates;
P4 turns that evidence into a development recommendation without expanding the
shipped runtime boundary.

- Detailed evidence lives in
  [`condensation-stiffness-study.md`](condensation-stiffness-study.md),
  including the recorded explicit grid, the candidate comparison, and the
  particle-only gas-update caveat.
- Later GPU condensation work should build on fixed-count explicit
  sub-stepping, specifically `fixed_count_substeps_4`, because it has the best
  documented agreement with the current CPU/explicit reference while preserving
  deterministic fixed-shape buffer reuse.
- Keep the implementation graph-capture-friendly and autodiff-compatible:
  fixed iteration counts, stable allocation layouts, and no adaptive or
  data-dependent loop bounds on the recommended path.
- Do not claim gas-coupled GPU production support until the production hook and
  same-issue conservation regression in
  `particula/integration_tests/condensation_particle_resolved_test.py` land.
- Downstream implementers must stay inside the E2-F2 environment-shape
  contract and the E2-F6 `float64` evidence envelope when applying this
  recommendation.
- Water condensation near cloud activation remains the hardest stress case for
  future follow-up work because of high vapor concentration, tight
  supersaturation coupling, and latent-heat temperature feedback.

## Epic B: Non-Isothermal Condensation Public API (CPU)

Status: shipped as ADW plan E1. This epic completed the CPU-side public API,
validation, documentation, and acceleration readiness for non-isothermal
(latent-heat) condensation, giving the GPU parity work in
[Epic D](#epic-d-gpu-condensation-physics-parity) a stable,
builder-accessible CPU reference to match.

Shipped scope (tracked in plan E1):

- Public builder and factory support for latent-heat condensation (E1-F1),
  exported through `particula.dynamics` and covered by export regression
  tests.
- Regression coverage for the latent-heat CPU path at the unit level,
  including mass-transfer conservation and per-step
  `last_latent_heat_energy` bookkeeping checks.
- User-facing feature documentation with worked code snippets in the
  [condensation strategy system guide](../condensation_strategy_system.md)
  and the latent-heat section of the
  [condensation equations theory page](../../Theory/Technical/Dynamics/Condensation_Equations.md#condensation-with-latent-heat).
- CPU API decisions recorded as the reference contract for GPU latent-heat
  parity (fixed-shape state, explicit environment inputs, no hidden CPU↔GPU
  movement).

Two follow-ups were deferred out of E1 (a runnable latent-heat example and
an integration-level conservation case); they are scheduled as features 8
and 9 in
[Epic C](#epic-c-gpu-kernel-correctness-and-low-level-api-hardening).

**Exit bar (met):** Latent-heat condensation is a documented, builder/factory
accessible CPU workflow with validation coverage, and its API decisions are
recorded as the reference contract for GPU latent-heat parity in Epic D.

## Epic C: GPU Kernel Correctness and Low-Level API Hardening

Make the existing low-level GPU kernels correct, reproducible, and usable
from documentation alone before adding new physics. This epic absorbs the
former "documented low-level GPU API" milestone, the cross-cutting
validation-infrastructure work, and the two documentation/validation
follow-ups deferred from Epic B (plan E1).

For the current shipped baseline on explicit CPU↔GPU transfer helpers,
`EnvironmentData` ownership, current CPU/GPU support boundaries, and the
canonical runnable entrypoint, start from the
[Data Containers and GPU Foundations](../data-containers-and-gpu-foundations.md)
guide and the
[Data Containers example](../../Examples/Data_Containers/index.md); this
epic and every later epic extend those contracts rather than redefining
them.

Planned features:

1. Persist coagulation RNG state: seed once at loop setup and keep per-box
   RNG state between `coagulation_step_gpu` calls instead of re-launching the
   initialization kernel on every call.
2. Improve rejection sampling for wide size ranges: evaluate size-binned
   majorant kernels or stratified pair sampling so mixed NPF/droplet boxes do
   not collapse acceptance rates.
3. Record the one-thread-per-box coagulation design decision with a measured
   single-box scaling limit, or scope a parallel-within-box variant.
4. Resolve kernel entry-point exports: decide whether `condensation_step_gpu`
   and `coagulation_step_gpu` are exported from top-level `particula.gpu` or
   documented under `particula.gpu.kernels` (today they are only importable
   from `particula.gpu.kernels`).
5. Formalize device-aware pytest execution as project policy (a pytest
   flag/config plus marker): parity tests always run on Warp CPU and add
   CUDA automatically when a device is present; document that CUDA-device
   validation is currently local/manual before releases.
6. Record acceptable parity tolerances for stochastic coagulation and
   floating-point differences across CPU, Warp CPU, and CUDA devices.
7. Publish a GPU quick-start example for direct kernel use with
   `ParticleData`, `GasData`, the Warp transfer helpers, `gpu_context`, and
   `WARP_AVAILABLE`, plus troubleshooting notes for missing Warp, missing
   CUDA, and device-mismatch errors, and installation/environment
   expectations for `warp-lang`.
8. Add a runnable `docs/Examples/` entry for a `CondensationLatentHeat`
   workflow (deferred from E1; current tutorials only set `latent_heat` as a
   vapor property), following the paired `.py`/`.ipynb` example conventions.
9. Add a `particula/integration_tests/` latent-heat conservation case
   (deferred from E1; today the conservation and energy regressions are
   unit-level only), establishing the CPU integration baseline that the
   Epic D latent-heat parity tests will reuse.

**Exit bar:** A new user can run both kernels from the docs on a CUDA device
and on Warp CPU without reading source, repeated coagulation steps draw
uncorrelated samples from a caller-initialized persistent RNG buffer, the
device-aware test policy plus parity tolerances are recorded, and the deferred
E1 latent-heat example and integration-conservation case have landed.

### Known Kernel Issues

Concrete defects and design limits in the existing kernels that should be
fixed or explicitly accepted during this epic.

- **RNG state ownership.** `coagulation_step_gpu` seeds once and reuses a
  caller-owned per-box RNG buffer across timesteps. Repeated calls should keep
  the seed fixed unless an explicit reset is requested, and graph-captured
  loops should initialize or reset the buffer before capture or before the
  repeated-step loop (see [Random Number Strategy](#random-number-strategy)).
- **Rejection-sampling acceptance collapse.** The shipped `E3-F2-P2` outcome
  was a bounded selector hardening improvement inside the existing Brownian
  sampler, not a new public API, not a container or transfer-path change, and
  not a full mixed-scale acceptance-rate fix. The remaining limitation is still
  the same kernel shape: one global `k_max` computed from the min/max radius
  pair plus one-thread-per-box execution inside the existing sampler. Wide
  mixed NPF/droplet size distributions can therefore still suffer low
  acceptance and high trial counts even after the selector hardening landed.

  The shipped mixed-scale evidence comes from the private test-only fixture and
  diagnostics in `particula/gpu/kernels/tests/coagulation_test.py`:
  `_make_mixed_npf_droplet_particle_data()`,
  `_brownian_coagulation_attempt_diagnostic_kernel(...)`, and
  `_collect_test_local_attempt_diagnostics(...)`. These helpers stay test-local
  and do not imply implicit CPU/GPU synchronization or runtime-integrated
  diagnostics in production paths. The diagnostic interpretation follows the
  test terminology directly: scheduled trials are the bounded integer-like
  trials requested by the mirrored sampler, executed trials are the trials that
  actually run before early exit, accepted collisions are the realized
  coagulation events, and the Brownian reference uses only active upper-triangle
  particle pairs to compute `active_pair_count`, expected mean, and the
  Poisson-style sigma `sqrt(expected_mean)`.

  The measured evidence recorded in E3-F2-P3 is 139 accepted collisions across
  the fixed seeded replay range `range(101, 201)`, compared with the Brownian
  expected mean `143.846`, Poisson-style sigma `11.994`, and documented
  `3-sigma` tolerance `35.981`. That tolerance is based on repeated fresh
  seeded runs against the same mixed-scale fixture, with the expected mean
  computed from active upper-triangle Brownian pairs only and sigma taken as
  `sqrt(expected_mean)`.

  The shipped mixed-scale test surface currently includes
  `test_mixed_scale_diagnostic_reports_attempted_and_accepted_counts(device)`,
  `test_mixed_scale_brownian_collision_totals_match_expected_mean_within_sigma_tolerance(device)`,
  `test_mixed_scale_expected_collision_statistics_use_active_pairs_only()`,
  `test_mixed_scale_acceptance_fraction_is_finite_and_nonnegative(device)`,
  `test_mixed_scale_selector_only_emits_sorted_active_in_bounds_pairs(device)`,
  `test_mixed_scale_sparse_or_degenerate_active_sets_return_zero_collisions(device, active_indices)`,
  `test_mixed_scale_diagnostic_tracks_executed_trials_under_early_exit(device)`,
  `test_mixed_scale_two_active_particles_accept_the_only_valid_pair(device)`,
  `test_mixed_scale_diagnostic_clamps_scheduled_trials_to_int32_limit(device)`,
  `test_mixed_scale_coagulation_conserves_total_mass(device)`,
  `test_mixed_scale_repeated_seeded_runs_conserve_total_mass_even_with_zero_acceptance_trials(device)`,
  `test_mixed_scale_caller_owned_rng_states_advance_without_hidden_reseed(device)`,
  and
  `test_mixed_scale_initialize_rng_true_replays_seeded_state_and_outcome(device)`.
  The Brownian repeated-run evidence is the seeded `range(101, 201)` check in
  `test_mixed_scale_brownian_collision_totals_match_expected_mean_within_sigma_tolerance(device)`:
  the shipped bounded-selector path is accepted only when the observed accepted
  collision total stays within a `3 * sigma` tolerance of the active-pairs-only
  Brownian expected mean computed by
  `_get_expected_collision_statistics(...)` for the documented `time_step=0.5`,
  `max_collisions=8`, and `volume=1.0e-10` replay setup. The same test-local
  coverage also verifies active-pairs-only reference statistics, sorted
  in-bounds accepted pairs, finite zero-acceptance behavior, early-exit
  accounting for executed versus scheduled trials, bounded trial clamping,
  total-mass conservation, and caller-owned `rng_states` reuse/reset semantics.
  Treat this as bounded evidence that the selector hardening preserved expected
  Brownian behavior within stochastic tolerance inside the existing sampler,
  not as proof that the global-majorant acceptance collapse is solved for every
  mixed-scale distribution.

  Reproduce the seeded checks with:

  ```bash
  pytest particula/gpu/kernels/tests/coagulation_test.py -q -k mixed_scale
  pytest particula/gpu/kernels/tests/coagulation_test.py -q -k "mixed_scale or sparse or degenerate or conservation" -Werror
  ```

  Documentation validation for this note is limited to markdown/path readback
  plus those focused `pytest` checks; no separate repo-specific docs formatter
  or link checker is claimed here.
- **One-thread-per-box coagulation.** The kernel launches one GPU thread per
  box with sequential pair selection inside the thread. This matches the
  multi-box scaling priority but serializes large single-box workloads. Record
  this as a deliberate design decision with its measured single-box limit, or
  plan a parallel-within-box variant.

## Epic D: GPU Condensation Physics Parity

Bring GPU condensation to parity with the CPU reference paths from Epics A
and B: latent-heat physics, on-device thermodynamics, and the activity and
surface-tension support required by the later optimization targets in
[Epic I](#epic-i-differentiability-and-global-optimization). Staggered
(Gauss-Seidel) condensation stays CPU-only (see [Non-Goals](#non-goals)).
Every kernel in this epic ships with CPU/GPU parity tests under the Epic C
device-aware test policy.

Planned features:

1. On-device vapor pressure recomputation: temperature-dependent saturation
   vapor pressures recomputed on the GPU each timestep for parcel,
   expansion, and latent-heat workflows; today `vapor_pressure` is set once
   at transfer time and defaults to zeros.
2. Warp-backed latent-heat condensation kernel matching CPU
   `CondensationLatentHeat`, including latent-heat-corrected mass transfer
   and per-step latent heat energy bookkeeping (see the
   [condensation equations](../../Theory/Technical/Dynamics/Condensation_Equations.md#condensation-with-latent-heat)),
   with per-box temperature feedback through the
   [EnvironmentData Container](#environmentdata-container).
3. Fixed-count sub-stepping integration: build GPU condensation on the
   `fixed_count_substeps_4` recommendation from the
   [condensation stiffness study](condensation-stiffness-study.md), keeping
   the implementation graph-capture-friendly and autodiff-compatible (fixed
   iteration counts, stable allocation layouts, no data-dependent loop
   bounds).
4. Per-species or per-particle surface tension, replacing the hardcoded
   0.072 N/m kernel default.
5. GPU activity support: decide which activity models (water activity,
   kappa-hygroscopicity) get GPU kernels and implement the selected scope; a
   prerequisite for the hygroscopicity and mixing-state targets in
   [Epic I](#epic-i-differentiability-and-global-optimization).
6. CPU/GPU parity tests for latent-heat condensation, including the stored
   latent heat energy diagnostic.
7. Gas-coupled support gating: do not claim gas-coupled GPU production
   support until the production hook and same-issue conservation regression
   in `particula/integration_tests/condensation_particle_resolved_test.py`
   land.

**Exit bar:** GPU isothermal and latent-heat condensation match the CPU
reference within recorded tolerances on Warp CPU (and CUDA when available),
with on-device vapor pressure recomputation and configurable surface
tension, while staying inside the E2 environment-shape contract and
`float64` evidence envelope.

## Epic E: GPU Coagulation Physics Coverage

Extend GPU coagulation beyond the Brownian kernel to the collision
mechanisms already available on CPU. DNS turbulence remains deferred (see
[Non-Goals](#non-goals)). Every kernel in this epic ships with CPU/GPU
parity or statistically bounded tests under the Epic C device-aware test
policy.

Planned features:

1. Charged particle-resolved coagulation using `WarpParticleData.charge`
   rather than treating charge as stored but inactive metadata.
2. Combined Brownian plus charged coagulation when both mechanisms are
   active.
3. Sedimentation coagulation matching the CPU Seinfeld-Pandis 2016
   sedimentation kernel.
4. Simple turbulent shear coagulation matching the CPU Saffman-Turner 1956
   kernel.
5. Combined Brownian, charged, sedimentation, and turbulent shear kernels
   when multiple mechanisms are active.
6. Parity and statistical validation: CPU/GPU parity or statistically
   bounded tests for each new kernel, with recorded tolerances.
7. Distribution-support decision record: the current GPU path implicitly
   assumes particle-resolved semantics (per-slot merges, concentration
   zeroing); document which binned/moving-bin strategies remain CPU-only.

**Exit bar:** Each in-scope CPU coagulation mechanism has a GPU kernel with
parity or statistically bounded tests, combined-mechanism kernels are
validated, and unsupported distribution types are documented as CPU-only.

## Epic F: GPU Process Completeness

Add the remaining processes a complete aerosol simulation needs to stay
GPU-resident between checkpoints — dilution, wall loss, and nucleation —
plus the fixed-slot particle management that nucleation and graph capture
depend on.

Planned features:

1. CPU `Dilution` strategy/runnable reference implementation (today dilution
   exists only as free functions, so GPU dilution has no process-level CPU
   reference).
2. GPU dilution kernel with parity tests against the CPU reference.
3. GPU neutral wall loss (spherical/rectangular) with parity tests.
4. GPU charged wall loss, after neutral wall loss and the core condensation
   and coagulation GPU paths are stable.
5. Nucleation/particle-source CPU reference process following the
   [nucleation equations](../../Theory/Technical/Dynamics/Nucleation_Equations.md)
   (no nucleation code exists in particula today).
6. GPU nucleation via slot activation (see
   [Particle Slot Management](#particle-slot-management)).
7. Particle slot management: inactive zero-mass slots, activation,
   per-box active-count diagnostics, and an exhaustion policy (resampling or
   volume scaling).
8. Slot and conservation validation: tests for inactive slots, activation,
   slot exhaustion handling, and conservation across resampling or volume
   scaling.

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

**Exit bar:** A complete GPU-resident timestep can run condensation,
coagulation, wall loss, and dilution together with persistent RNG state, and
new-particle creation works through validated, conservation-checked slot
activation.

## Epic G: Backend Selection and GPU-Resident Simulation

Make GPU execution reachable from user-facing APIs and keep full simulations
resident on the device between checkpoints. This epic also owns the
cross-cutting CPU-fallback and API-stability decisions formerly tracked as a
separate epic.

Planned features:

1. Backend-selection API design: decide whether selection belongs on
   strategies, runnables, builders, or a separate execution context.
2. Backend selection for at least one condensation workflow, with explicit
   return types and mutation semantics.
3. Backend selection for at least one Brownian coagulation workflow.
4. GPU simulation loop abstraction that keeps `WarpParticleData`,
   `WarpGasData`, and `WarpEnvironmentData` resident on the selected device
   across all enabled dynamics.
5. Deterministic process ordering/scheduler for full aerosol simulations
   (condensation, coagulation, wall loss, dilution, environment updates, and
   gas updates between steps).
6. CPU fallback and API-stability policy: mark low-level `particula.gpu.*`
   APIs experimental until selection and full-loop validation land; missing
   GPU processes raise clear errors or use explicit fallback boundaries; no
   silent CPU/GPU data movement in long simulations.
7. Multi-box communication: prescribed advection, dilution, expansion, and
   simple mixing maps, including per-box volume changes for parcel and
   combustion cases.
8. Per-box RNG streams so independent boxes remain reproducible, with
   GPU-resident RNG state exercised in loop tests.
9. Documentation and regression coverage: a GPU-resident multi-timestep
   example that transfers back only at checkpoints, plus larger multi-box and
   particle-resolved regression cases for independent boxes, prescribed
   advection, dilution, and expansion.

### High-Level Integration

- Use the shipped
  [Data Containers and GPU Foundations](../data-containers-and-gpu-foundations.md)
  guide and
  [Data Containers example](../../Examples/Data_Containers/index.md) as the
  current baseline for explicit transfer helpers, `EnvironmentData` ownership,
  support boundaries, and runnable entrypoint wording before layering new
  high-level GPU execution APIs on top.

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
- Environment updates: prescribed or process-driven per-box temperature and
  pressure updates, plus recomputation of derived thermodynamic fields;
  simulation-volume evolution remains owned by `ParticleData.volume`.
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

Fixed-slot particle management for these loops is defined in
[Particle Slot Management](#particle-slot-management) under Epic F.

### Random Number Strategy

- Define deterministic RNG seeding for stochastic coagulation on GPU. The
  shipped baseline is seed-once initialization for caller-owned persistent
  `rng_states`, while omitted `rng_states` still use a convenience
  allocate-and-seed path per call.
- Support per-box RNG streams so independent boxes remain reproducible when the
  number of boxes changes or when selected boxes are disabled.
- Track caller-owned RNG state on the GPU between timesteps and include it in
  graph-captured execution tests.
- Document expected reproducibility limits across CPU, Warp CPU, and CUDA.

**Exit bar:** At least one condensation and one coagulation workflow run on
GPU through the selection API, match CPU within recorded tolerance, and are
used in a documented example, and a multi-box GPU-resident loop runs all
supported processes between checkpoints with per-box RNG streams and no
hidden CPU transfers.

## Epic H: Graph Capture and Performance

Reduce launch overhead with graph capture and establish performance and memory
targets aligned with the multi-box scaling goal.

Planned features:

1. Warp graph capture for repeated timestep execution with fixed process
   order and stable array shapes.
2. Separation of graph-capturable kernels from setup work (allocation,
   validation, host-side scheduling).
3. Preallocated buffer reuse for mass transfer, collision pairs, wall-loss
   rates, dilution factors, diagnostics, and RNG state.
4. Full-loop validation tests comparing CPU, uncaptured GPU, and
   graph-captured GPU execution for the same process sequence.
5. Multi-box scaling benchmarks with box count as the primary axis, plus
   secondary particles-per-box scaling; benchmarks stay opt-in
   (`--benchmark`-style gating) and CUDA-gated, separate from the default
   parity suite.
6. Memory-budget model covering state arrays, inactive slots, temporary
   buffers, diagnostics, communication maps, and autodiff tape storage.
7. Graph-capture example plus documented limitations and re-capture
   triggers.
8. CUDA kernel profiling (occupancy, memory access patterns) and
   captured-vs-uncaptured launch-overhead benchmarks.

**Exit bar:** A multi-box GPU-resident simulation runs graph-captured
timesteps, matches CPU and uncaptured GPU references, and has published
scaling numbers across box counts plus a recorded memory-budget model.

### Warp Graph Capture

- Preserve the shipped explicit CPU↔GPU helper boundary and documented
  container ownership model from the
  [Data Containers and GPU Foundations](../data-containers-and-gpu-foundations.md)
  guide while designing graph-captured GPU loops; performance work should not
  reintroduce implicit synchronization or move the canonical example entrypoint
  away from the
  [Data Containers example](../../Examples/Data_Containers/index.md).

- Investigate Warp graph capture for repeated timestep execution with a fixed
  process order and stable array shapes.
- Separate graph-capturable kernels from setup work such as allocation,
  validation, and host-side scheduling.
- Reuse preallocated buffers for mass transfer, collision pairs, wall-loss
  rates, dilution factors, diagnostics, and RNG state before capture begins.
  For coagulation, seed or explicitly reset persistent `rng_states` before the
  repeated-step loop or before graph capture, not through hidden reseeding
  inside the captured path.
- Add tests that compare captured-graph execution against uncaptured GPU
  execution and CPU reference results.
- Document graph-capture limitations, including shape changes, dynamic process
  selection, stochastic coagulation state ownership, and device availability.
- Require fixed array shapes during graph capture. Changing `n_boxes`,
  `n_particles`, or `n_species` should invalidate the captured graph and require
  a new setup/capture step.
- Handle changing active particle counts through inactive slots rather than
  resizing arrays.
- Use resampling, merging, or volume scaling before a box exhausts inactive
  particle slots.
- Keep graph-captured loops focused on repeated timesteps with stable process
  order, stable buffer shapes, and stable communication maps.

For shipped coagulation behavior, `rng_states` are Warp-resident sidecar state
owned by the caller when persistence is needed across timesteps. Passing a
fixed `rng_seed` with a reused `rng_states` buffer does not trigger hidden
re-seeding; explicit initialization happens only during omitted-buffer
convenience allocation or when `initialize_rng=True` is requested before the
captured loop or repeated-step run.

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
  (see [Epic I](#epic-i-differentiability-and-global-optimization)).
- Include benchmark cases that vary boxes, particles per box, species count,
  active-slot fraction, and process combinations.

#### Shipped coagulation benchmark evidence (2026-07-10 UTC)

- Command evidence: `.artifacts/benchmarks/gpu_benchmark_results.json`
  now records the exact executed command in
  `benchmark_metadata.command`. The 2026-07-10 UTC capture was produced by
  `/home/kyle/Code/particula/.venv/bin/pytest -v --tb=short --cov`
  `--cov-report=term-missing particula/gpu/tests/benchmark_test.py`
  `--benchmark -v -s`, with the user-facing opt-in portion preserved as
  `particula/gpu/tests/benchmark_test.py --benchmark -v -s`.
- Hardware/context: the same artifact records `warp_version=1.15.0` and
  `device.alias=cuda:0`, `device.name=NVIDIA GeForce RTX 5060`,
  `device.total_memory_bytes=8082096128` (about 8.08 GiB), with
  `started_at=2026-07-10T01:59:34.215800+00:00` and
  `updated_at=2026-07-10T02:00:43.918911+00:00`.
- Artifact output: `.artifacts/benchmarks/gpu_benchmark_results.json`
- Mixed-scale fixture note: the coagulation benchmark path now uses a dedicated
  deterministic mixed NPF/droplet fixture aligned with the shipped E3-F2
  baseline, while condensation benchmarks keep the generic helper.
- Timing summary: single-box coagulation remains the limiting case for the
  one-thread-per-box kernel (`1x500` GPU `0.0361s`, `1x2k` `0.1366s`, `1x5k`
  `0.3411s`, `1x10k` `0.6809s`, `1x20k` `1.3689s`, `1x50k` `3.3984s`), while
  equivalent or larger total-particle multi-box runs scale much better across
  independent boxes (`10x500` `0.0359s`, `10x1k` `0.0699s`, `50x1k` `0.0714s`,
  `10x5k` `0.3437s`, `50x5k` `0.3528s`, `100x1k` `0.0766s`, `10x10k`
  `0.6938s`).
- Interpretation boundary: record this as current benchmark evidence for the
  existing one-thread-per-box design, not as a final acceptance decision on the
  long-term scaling strategy.

## Epic I: Differentiability and Global Optimization

A longer-term goal is gradient-based global optimization: using Warp automatic
differentiation to fit model parameters to experiments or observations.
Differentiability constrains how kernels are written, so it must be considered
while the Epic D and Epic E kernels are authored, not added afterward. For
the shipped
container, transfer, and runnable-example baseline that this differentiability
work builds on, start from the
[Data Containers and GPU Foundations](../data-containers-and-gpu-foundations.md)
guide and the
[Data Containers example](../../Examples/Data_Containers/index.md).

See [Warp Autodiff: Limitations and Stochastic Process Handling](warp-autodiff-limitations.md)
for the detailed implementation-planning companion on autodiff mechanics,
kernel-authoring constraints, offline code patterns, and the options for
differentiating stochastic coagulation.

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

Planned features:

1. Differentiable-friendly variants of `apply_mass_transfer_kernel` and
   `apply_coagulation_kernel` compatible with `wp.Tape` (no gradient-breaking
   in-place mutation or control flow on the optimization path).
2. End-to-end gradient from a final-state loss back to the initial state
   through a multi-step deterministic condensation loop, with recorded tape
   memory usage.
3. Tape memory budgeting and gradient-checkpointing evaluation for long
   differentiable loops, feeding the Epic H memory-budget model.
4. Global-optimization example that recovers an initial size distribution
   from a synthetic final distribution across multiple boxes, with process
   parameters held fixed.
5. Decision record for differentiable coagulation (deterministic
   binned/sectional operator versus relaxed particle-resolved formulation).
6. Decision record for the mixing-state and hygroscopicity state
   representation (composition-resolved sectional versus particle-resolved
   with a differentiable surrogate).
7. Combined validation of autodiff, graph capture, and per-box RNG streams
   operating together.
8. Loss-function definitions on state (size distribution, hygroscopicity,
   mixing-state metrics), each with a confirmed differentiable path back to
   the initial state.

Design constraints and open decisions:

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
  long differentiable loops, and include tape storage in the Epic H
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
  [Epic D](#epic-d-gpu-condensation-physics-parity); a condensation operator
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

**Exit bar:** A gradient-based optimizer recovers a known initial state from
synthetic multi-box final-state data within recorded tolerance, using GPU
autodiff, without fitting any process parameters, and the differentiable
coagulation and state-representation decisions are recorded.

## Risks and Key Decisions

| Risk / Decision | Impact | Direction |
| --- | --- | --- |
| fp64 throughput on consumer CUDA | Slower GPU, weaker speedups | Keep fp64 as the shipped reference and production baseline; evaluate mixed precision only through a future evidence-backed migration proposal |
| NPF-to-droplet dynamic range near fp64 limits | Lost small-mass resolution | Use the shipped mass-precision report as the baseline; keep per-species absolute mass until a future proposal proves an alternative across the E2 cases |
| fp64 memory doubling vs large multi-box | Fewer boxes fit in memory | Build a memory-budget model; treat precision and box count together |
| Time-scale stiffness across NPF-to-droplet range | Unstable or wastefully slow explicit stepping | Build later GPU condensation integration from the shipped `fixed_count_substeps_4` recommendation while keeping gas-coupled production support gated |
| No fully integrated GPU-ready per-box thermodynamic runtime path | Blocks parcels, expansion, latent-heat feedback from running end to end on GPU | Keep shipped CPU `EnvironmentData` as the authoritative CPU owner, build on the shipped `WarpEnvironmentData` CPU↔GPU round-trip helpers, and add integration work |
| Rejection-sampling acceptance collapse for wide size ranges | Coagulation trial counts explode in mixed NPF/droplet boxes | Evaluate binned majorant kernels or stratified pair sampling |
| One-thread-per-box coagulation | Serializes large single-box workloads | Record as deliberate multi-box tradeoff or add parallel-within-box variant |
| Caller must seed or explicitly reset persistent coagulation RNG state before repeated loops or graph capture | Reused buffers continue their stream unless intentionally reset, which can surprise callers expecting hidden reseeding | Keep the shipped seed-once contract, document caller-owned sidecar setup, and reset only via explicit initialization before the loop or capture |
| Autodiff through stochastic coagulation | Blocks global optimization if coag must be fit | Open decision; start with deterministic condensation, defer coag choice |
| In-place kernels break gradients | Autodiff path unusable | Author differentiable-friendly kernel variants for the optimization path |
| Tape memory for multi-step differentiable loops | Gradient runs exhaust GPU memory | Budget tape storage; evaluate gradient checkpointing |
| RNG reproducibility across CPU/Warp CPU/CUDA | Non-reproducible stochastic results | Per-box RNG streams; document tolerances and reproducibility limits |
| Caller drops ordered gas metadata or vapor-pressure sidecar across `WarpGasData` restore | Placeholder names or missing GPU-helper state break name-keyed or condensation follow-up logic | Treat the current field split as the resolved contract, preserve ordered names externally, and recompute or carry `vapor_pressure` sidecar data when needed |
| Graph capture fragility (shape/process changes) | Invalid captured graphs | Fixed shapes, inactive slots, documented re-capture triggers |
| GPU condensation lacks activity/kappa physics | Hygroscopicity and mixing-state targets unfittable | Plan GPU activity/surface-tension support in Epic D before Epic I targets |
