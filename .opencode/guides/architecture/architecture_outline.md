# Architecture Outline

## Execution Capability Vocabulary

`particula.execution` is a dependency-neutral, standard-library-only,
explicit-selection seam. Its package-level public APIs are exactly `Backend`,
`Device`, `Process`, `Capability`, `CapabilityRequirements`,
`CapabilityDeclaration`, `CapabilityMatrix`, `ExecutionRequest`,
`ExecutionAdapter`, and `ExecutionContext`, plus the public
`ExecutionContext.register_adapter()` method. The backing per-context registry
remains private.

Declarations and requests are immutable exact metadata. Nonempty requirements
must match a complete declaration exactly; empty requirements are accepted when
the matrix contains a declaration for the same `Device` and `Process`. CPU
metadata uses exactly `Device(Backend.CPU, "cpu")`. Registration and resolution
are context-local, selection-only operations: complete exact matrix validation
of nonempty requirements occurs before one exact `(Process, Backend)` lookup.
A resolved adapter retains identity and is not executed by selection. Selection
does not import, probe, resolve, or check availability of a backend; transfer or
synchronize data; execute an adapter; retry; or fallback. Generic execution
argument, result, state, mutation, and runtime-error semantics are not public.
P3/P4 state, result, mutation, and concrete CPU execution-adapter types remain
direct-module-only.

Strategy physics, builder configuration, and existing CPU `RunnableABC`
behavior remain separate from E7-F1 typed selection and downstream process
adapter/session layers. The exact ordering is
`E7-F1 -> E7-F6 -> {E7-F2, E7-F3, E7-F4} -> E7-F5`; E7-F6 owns availability,
fallback, error taxonomy, API stability, and export policy. GPU adapters,
resident sessions or loops, schedulers, implicit transfer/synchronization,
retry, fallback, and replacement of direct GPU APIs remain deferred.

### particula/execution/

**Key Components:**
- `__init__.py` - Dependency-neutral selection seam. It preserves the exact
  ten-name public selection export surface and does not import Warp, the GPU
  package, or concrete adapters.
- `Backend`, `Device`, `Process`, and `Capability` - Immutable typed metadata;
  `Device.native` is an opaque native identifier.
- `CapabilityRequirements` and `CapabilityDeclaration` - Immutable exact
  capability-support declarations.
- `CapabilityMatrix` - Pure, immutable exact-match lookup. Nonempty
  requirements must match a complete declaration exactly; empty requirements
  are accepted when the matrix contains a declaration for the same `Device` and
  `Process`.
- `ExecutionRequest` and `ExecutionContext` - Package-level public selection
  values and context-local coordinator. Requests must pair matching
  backends/devices; CPU selection accepts only `Device(Backend.CPU, "cpu")`.
- `_AdapterRegistry` - Private, per-context exact-key backing registry.
  `ExecutionContext.register_adapter()` is public; registration and resolution
  select adapters by identity after matrix validation and never execute them.
  P3/P4 state, result, mutation, and concrete CPU execution-adapter types stay
  direct-module-only.
- `gpu_session.py` - Concrete-only, unexported resident-session boundary. P1
  `ResidentSession` validates and retains already-resident caller-owned Warp
  particle, gas, and environment containers, immutable dimensions, `Device`
  metadata, a CPU gas-name tuple, and lifecycle state by identity. P2's
  direct-import-only `setup_resident_session` performs local CPU-only carrier,
  shape, name, and exact-Warp-`Device` preflight, then performs exactly one
  particle/gas/environment upload in that order through
  `particula.gpu.conversion`, preserving ordered CPU gas names solely as
  metadata and publishing only a complete `ACTIVE` session. It relies on the
  upstream E7-F6 native-availability precondition and neither probes nor
  substitutes devices. The boundary has no fallback, synchronization,
  restoration, sidecars, lifecycle transition, finalization, close, scheduler,
  or migration behavior, and remains absent from package exports. See
  [ADR-004](decisions/ADR-004-concrete-gpu-resident-session-boundary.md) and
   [ADR-005](decisions/ADR-005-one-time-gpu-resident-session-setup.md).
- `gpu_resources.py` - Direct-import-only, Warp-dependent concrete registry for
  complete reusable native process sidecars. It accepts only an exact `ACTIVE`
  `ResidentSession`, pins its lifecycle, dimensions, device, and all primary
  array identities, and rejects session drift before every acquisition. Typed
  concrete manifests drive fixed schema/device/contiguity checks, checked
  allocation sizes, and metadata-only primary/sidecar nonaliasing. A role pins
  one caller- or registry-allocated identity; compatible repeats preserve the
  exact view, records, and arrays. This validates pinned ownership rather than
  unverifiable allocator provenance. It creates no public package export and
  has no execution/selection, transfer/sync/restore, lifecycle, transport,
  process-configuration/physics, or RNG reset/advance/initialization behavior.
  Condensation thermodynamic roles are derived scratch/property storage only.
- `adapters/condensation.py` - Concrete-only P2 condensation configuration and
  CPU/Warp state carriers plus selected P3 CPU/Warp execution carriers and
  adapters. P2 construction retains caller-owned resources by identity and
  performs read-only validation. After local P3 preflight, each adapter makes
  exactly one selected native call: CPU dispatches to the caller-owned
  `MassCondensation` runnable and Warp lazily resolves and dispatches to
  `condensation_step_gpu`. Neither path transfers, allocates, restores,
  synchronizes, retries, falls back, or recovers after native dispatch; native
  exceptions and post-launch limits remain authoritative. CPU dispatch remains
  isothermal. Warp profile preflight occurs before lazy kernel resolution and
  forwards caller-owned `latent_heat`, `energy_transfer`, and deferred
  `thermal_work` sidecars by identity; the direct kernel owns thermal validation
  and execution. Warp does not transfer, allocate, synchronize, restore,
  reconstruct results, fall back, or recover after native dispatch. All carriers
  and adapters remain direct-module-only and Warp is imported only for Warp P2
  validation or after P3 Warp preflight.
- `adapters/coagulation.py` - Concrete-only P2 Brownian coagulation state and
  result carriers plus selected P3 CPU and resident-Warp execution states and
  adapters. Construction retains caller-owned CPU or resident-Warp resources,
  diagnostic sidecars, and persistent RNG intent by identity while applying
  only selection-owned kind/form and metadata-detectable ownership checks.
  After local control/state preflight, each adapter makes exactly one selected
  native call: CPU dispatches the caller-owned `Coagulation` runnable and Warp
  lazily resolves and calls `coagulation_step_gpu`. Neither path selects another
  backend; transfers, converts, restores, synchronizes, allocates, retries,
  falls back, or recovers. The resident-Warp adapter forwards the caller-owned
  RNG sidecar and its explicit initialization intent unchanged; the direct
  kernel owns native physical/schema validation, RNG advancement/reset, and
  post-launch behavior. CPU and Warp stochastic trajectories are independent.
  All carriers and adapters are absent from `particula.execution`, the adapters
  package, and top-level exports.
- `tests/` - Test coverage

## Particle Package

`particula/particles/` contains particle-data representations, distribution
strategies, and focused particle-domain helpers.

## Direct GPU Nucleation

`particula/gpu/kernels/nucleation.py` implements package-exported direct-Warp
`nucleation_step_gpu`: P1 preflight, P2 admission, P3 fixed-slot staging, P4
resampling-first/scaling-fallback, and fused P5 selected-slot/gas transfer.
Only the step is exported; configuration, records, sidecars, and helpers remain
concrete-only. It has no hidden transfer, CPU fallback, resize/compaction, GPU
Runnable, or E6-F9 integration.

### particula/particles/

**Key Components:**
- `exhaustion.py` - Concrete, deliberately unexported CPU P1 read-only
  fixed-shape capacity exhaustion planning boundary, P2 validated resampling
  apply commit, and P4 direct all-box-preflighted representative-volume scaling
  commit with caller-owned sidecars and float64 weighted
  inventory accounting. P1 validates every box before resolution, applies
  resampling-first deferred-policy selection, and returns immutable plans
  without mutating state. P2 and P4 each own their separate CPU commit
  boundaries; the module owns no GPU work or container schema.
- `distribution_strategies/` - Particle distribution implementations
- `particle_data.py` - Fixed-shape CPU particle-data container and conversion
  helpers
- `slot_management.py` - CPU-only fixed-slot classification, discovery, and
  direct-import activation; exports only `get_slot_diagnostics` through
  `particula.particles`. Activation preserves fixed capacity and excludes
  `ParticleData` API changes, GPU support, and a top-level particles export
- `properties/` - Particle property calculations
- `tests/` - Test coverage

## Dynamics Package

`particula/dynamics/` contains physics-domain calculations and narrowly scoped
implementation boundaries.

### particula/dynamics/

**Key Components:**
- `particle_process.py` - Public CPU-only, single-box `Nucleation` runnable
  boundary and immutable `NucleationCommitConfig`. `Nucleation` adapts legacy
  `Aerosol` particle and partitioning-gas backing data by identity, accepts P4
  source configuration plus `EnvironmentData`, and runs equal, gas-coupled
  substeps. Each attempted substep has P3 transaction atomicity; prior
  successful substeps remain applied if a later substep fails. This boundary
  introduces no GPU or broader runnable orchestration.

### particula/dynamics/nucleation/

CPU-only P4 construction boundary for nucleation potential-rate
parameterizations, consumed by the supported P5 `Nucleation` process boundary
in `particula.dynamics`. P2 source-demand planning and P3 transaction helpers
remain concrete-module-only.

**Key Components:**
- `nucleation_strategies.py` - Immutable scalar configuration records and
  activation/kinetic potential-rate algorithms. The P4 records, builders, and
  factory are deliberately exported through `particula.dynamics.nucleation`
  and `particula.dynamics`; strategies still return rates only and own no
  source admission, state mutation, runnable, or GPU integration. The P5
  runnable composes them without changing their ownership.
- `nucleation_configuration.py`, `nucleation_builders.py`, and
  `nucleation_factories.py` - `NucleationSourceConfig`, its builder, and the
  activation/kinetic builders and factory form the strict, fresh,
  unit-normalizing P4 construction API. They do not import or expose P2/P3
  particle-source records or transaction helpers.
- `particle_source.py` - Concrete-module-only P2 source-demand planning and P3
  atomic particle-source transaction records and helpers. It is intentionally
  absent from both `particula.dynamics.nucleation` and `particula.dynamics`
  exports.
- `tests/` - Test coverage

## GPU Package

`particula/gpu/` contains Warp-backed data containers, explicit CPU↔GPU
transfer helpers, device-side physics helpers, and kernel entry points.

### particula/gpu/

**Key Components:**
- `__init__.py` - Public GPU exports
- `conversion.py` - Explicit CPU↔GPU transfer helpers only
- `warp_types.py` - Warp container schemas only
- `dynamics/` - GPU physics helper functions
- `properties/` - GPU property helper functions
- `kernels/` - GPU kernel entry points and private kernel support helpers
- `tests/` - Test coverage

### particula/gpu/kernels/

GPU kernel entry points own launch-time orchestration and may depend on shared
private helpers for cross-kernel setup.

**Key Components:**
- `condensation.py` - Condensation GPU entry points and kernels
- `coagulation.py` - Coagulation GPU entry points and kernels
- `dilution.py` - Concrete P1 GPU dilution input boundary; validation scans may
  allocate or launch, but rejected calls have no update-kernel launch or caller
  mutation
- `exhaustion.py` - Direct Warp fixed-capacity equal-weight resampling boundary;
  `resampling_step_gpu` remains the only exhaustion package export. Resampling
  consumes explicit per-box release counts, uses caller-owned planning and
  diagnostic buffers, and atomically commits only after all boxes pass
  diagnostics. The concrete-only P4 representative-volume scaling helper uses
  caller-owned sidecars and a separate all-box-preflighted scaling commit; it
  adds no policy, transfer, resizing, or runnable behavior. Only
  `resampling_step_gpu` is exported; `ResamplingBuffers`, P4 sidecars, status
  codes, and kernels remain concrete-module-only. Neither boundary provides a
  runnable, policy resolution, CPU fallback or transfer, or resizing.
- `nucleation.py` - Concrete E6-F8 implementation of package-exported
  `nucleation_step_gpu` for fixed-capacity GPU nucleation. P1 preflights; P2
  calculates and inventory-admits `E_pot=J*dt`, with survival already in `J`;
  P3 stages demand and fixed slots; P4 uses resampling before
  representative-volume-scaling fallback; and fused P5 commits selected slots
  and matching gas transfer. P3 retains counts beyond free capacity and emits
  ascending free-slot prefixes with `-1` tails. Only the step is exported;
  configuration, records, sidecars, and helpers remain concrete-only. Public
  rejection before P4 primitive entry preserves particle and gas state, though
  P2--P4 may have written documented sidecars before a later rejection. No
  rollback is promised after an E6-F6 primitive entry or P5 writer launch. The
  step provides no hidden transfer, CPU fallback, resize/compaction, GPU
  `Runnable`, or E6-F9 integration.
- `wall_loss.py` - Concrete fixed-slot neutral/charged GPU wall-loss boundary;
  owns immutable host configuration, frozen preflight, bounded fixed-slot
  removal, and the external caller-owned per-box RNG sidecar lifecycle. Charged
  mode composes private image-charge and field-drift helpers from
  `particula.gpu.dynamics.wall_loss_funcs` only for nonzero-charge slots;
  zero-charge slots retain the neutral path. The sidecar is not added to Warp
  particle schemas or package exports, and sequential per-box ownership
  advances it only for eligible slots.
- `slot_management.py` - Concrete-only P3 read-only direct-Warp diagnostics
  classify particle mass, concentration, and charge into caller-owned `int32`
  sidecars without accessing density or volume. Package-exported P4
  `activate_slots_gpu` maps selected request prefixes to ascending
  fixed-capacity free slots. It reads and writes only caller-owned mass,
  concentration, and charge storage; its activation and diagnostics sidecars
  are caller-owned device `int32` arrays. P4 completes preflight before its
  writer launches, makes no hidden transfers, and does not promise rollback
  after a launched writer.
- `environment.py` - Shared private normalization and validation for kernel
  environment inputs
- `tests/` - Test coverage
