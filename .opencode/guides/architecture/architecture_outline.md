# Architecture Outline

## Execution Capability Vocabulary

`particula.execution` is a dependency-neutral, standard-library-only,
explicit-selection seam. Its frozen ordered 26-name package-level public APIs
contain the 10 selection declarations (`Backend` through `ExecutionContext`),
the 13-name capability-error taxonomy (`ExecutionCapabilityReason` through
`FallbackDisallowedError`), and `FallbackPolicy`, `FallbackBoundary`, and
`CPUStateAuthority`, plus the public `ExecutionContext.register_adapter()`
method. The concrete `errors` and `fallback` modules stay direct-import-only
for their mechanics, but their public values are re-exported. The backing
per-context registry remains private.

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
adapter/session layers. E7-F4 P1--P7 ships the bounded, concrete
GPU-resident lifecycle below; it does not ship a resident loop or scheduler.
The exact downstream ordering remains
  `E7-F1 -> E7-F6 -> {E7-F2, E7-F3, E7-F4} -> E7-F5`; E7-F6 owns availability,
  fallback, error taxonomy, API stability, and export policy. The
   dependency-neutral `scheduler` remains declaration-only, while E7-F5 P6 adds
    a bounded concrete resident complete-loop composer. E7-F7 P1 remains the
    concrete communication-map declaration and read-only validation boundary.
    E7-F7 P2 separately ships the concrete-only direct-Warp final-volume writer
    at `particula.gpu.kernels.communication`; it neither binds nor changes P1.
     P3 retains gas inventory/time-step transfer admission. E7-F7 P4 ships a
     separate concrete-only direct-Warp particle-transport seam and owns its
     immutable particle-plan admission; P5 retains
      resident binding. E7-F8 P1 supplies only the isolated RNG stream-identity
     and explicit caller-buffer initialization boundary below; integration and
     remaining RNG policy remain deferred, along with implicit
     transfer/synchronization, retry, broad fallback, and
    replacement of direct GPU APIs. The sole shipped fallback seam is the
    explicit, CPU-authoritative,
  direct-import-only boundary described below.

### particula/execution/

**Key Components:**
- `__init__.py` - Dependency-neutral seam with a frozen ordered 26-name public
   surface: the 10 selection declarations, the 13-name capability-error
   taxonomy, and three fallback policy enums. It does not import Warp, the GPU
   package, or concrete adapters and fallback mechanics. See
   [ADR-015](decisions/ADR-015-execution-public-surface-and-experimental-gpu-policy.md).
- `errors.py` - Standard-library execution-capability error taxonomy. Its public
  values are re-exported by the package and top level, while this concrete module
  remains absent from their export lists.
- `availability.py` - Concrete, direct-import-only E7-F6 P2 availability
  resolver. `resolve_availability()` consumes already-validated P1 request and
  capability metadata, first fail-closes an exact CPU/Warp provider registry,
  then short-circuits through device recognition, structural process and exact
  capability declarations, lazy runtime status, device status, and injected
  request-associated state validation. Its frozen decision retains only the
  exact request; it selects no adapter and performs no execution, transfer,
  synchronization, allocation, or mutation. CPU recognizes only
  `Device(Backend.CPU, "cpu")`; Warp accepts every validated Warp declaration
   for recognition and passes its opaque native string unchanged to its lazy
   runtime device check. This module is not package- or top-level-exported. See
    [ADR-013](decisions/ADR-013-pre-execution-availability-resolution.md).
- `rng.py` - Concrete, direct-import-only E7-F8 P1 deterministic RNG
  stream-identity boundary. It validates immutable host stream metadata,
  derives per-process/per-logical-box initial `uint32` words without optional
  GPU imports, and explicitly initializes only validated caller-owned Warp
  state arrays. Initialization allocates temporary NumPy/Warp copy sources, then
  deterministically overwrites the retained arrays without acquiring,
  replacing, or rebinding them. It does not advance or reset those arrays and
  has no package/top-level export, resource-registry, resident-session,
  scheduler, or checkpoint integration.
- `fallback.py` - Concrete, direct-import-only E7-F6 P3 opt-in CPU fallback
  policy boundary. It defaults to re-raising the exact eligible typed
  availability/support error and selects the already-registered canonical CPU
  adapter only when callers explicitly choose CPU policy and attest to a
  CPU-authoritative `PRE_UPLOAD` or `RESTORED` boundary. It accepts no resident,
  uploaded, or mutated state; performs one CPU selection and one adapter dispatch
  at most; and exposes requested backend, selected backend, and original reason
  without changing native result metadata. It does not transfer, synchronize,
   manage lifecycle, restore, retry, or roll back. Its concrete module,
   operations, and carriers are not package- or top-level-exported; its three
   policy enums are intentionally re-exported. See
  [ADR-014](decisions/ADR-014-opt-in-cpu-fallback-boundary.md).
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
- `gpu_session.py` - Concrete-only, unexported resident-session boundary. The
  shipped E7-F4 P1--P7 lifecycle is available only by direct import from
  `particula.execution.gpu_session`, `particula.execution.gpu_resources`, and
  `particula.execution.checkpoint`; none of its names are package or top-level
  exports. P1
  `ResidentSession` validates and retains already-resident caller-owned Warp
   particle, gas, and environment containers, immutable dimensions, `Device`
   metadata, a CPU gas-name tuple, validated immutable resident stream metadata,
   and lifecycle state by identity. P2's
  direct-import-only `setup_resident_session` performs local CPU-only carrier,
  shape, name, and exact-Warp-`Device` preflight, then performs exactly one
  particle/gas/environment upload in that order through
  `particula.gpu.conversion`, preserving ordered CPU gas names solely as
  metadata and publishing only a complete `ACTIVE` session. It relies on the
  upstream E7-F6 native-availability precondition and neither probes nor
   substitutes devices. Session setup validates stream metadata before uploads
   but does not allocate a native RNG sidecar. The boundary has no fallback,
   synchronization, restoration, sidecars, lifecycle transition, finalization, close, scheduler,
   or migration behavior, and remains absent from package exports. P4 adds
   direct-import-only `ResidentStepGuard` and identity-only
   `ResidentStepToken`: one exact active session/registry binding has at most
   one open token, and count/time bookkeeping advances only on matching
   completion. It does not execute adapters, transfer, synchronize, allocate,
   resize, restore, or fall back. Future checkpoint, restore, finalize, close,
   fault, conversion, and resize/rebind boundaries must call
  `assert_step_closed()` before their own work. P5 adds direct-import-only
  `ResidentSession.checkpoint()` and
   `ResidentSession.finalize()`, delegated to one exact-identity-bound
   controller with its active session, pinned `GPUResourceRegistry`, and closed
   `ResidentStepGuard`. A checkpoint is a fresh immutable host snapshot and is
   nonterminal; finalization caches its first complete snapshot, then makes the
   session terminal `FINALIZED`, and later calls return that exact cached object.
  The controller/records/restart helper remain absent from all package exports.
  P6 adds private direct-owner failure classification and exact-token abort:
  read-only failures release the token without advancing guard counters/time
  and leave the session `ACTIVE`; a writer that may have launched releases the
  exact token and faults that session with no rollback. It preserves the
  original operational exception and does not intercept raw helpers. P6
  `close()`/`discard()` are identity-bound terminal lifecycle operations only:
  active close validates the pinned binding and a closed guard, faulted close
  uses identity-only binding checks, and repeated `CLOSED` or `FINALIZED`
  closes are write-free no-ops. They never checkpoint, restore, synchronize,
  transfer, allocate, or perform implicit runtime work. P5 checkpoint,
  finalize, and restart behavior remains unchanged. See
   [ADR-004](decisions/ADR-004-concrete-gpu-resident-session-boundary.md),
   [ADR-005](decisions/ADR-005-one-time-gpu-resident-session-setup.md), and
   [ADR-006](decisions/ADR-006-resident-gpu-step-lifecycle-guard.md),
  [ADR-007](decisions/ADR-007-resident-session-checkpoint-finalize-restart.md),
  and [ADR-008](decisions/ADR-008-resident-session-failure-close-semantics.md).
- `checkpoint.py` - Concrete-only P5/P7 in-memory checkpoint/restart boundary.
   `ResidentCheckpoint` stores versioned immutable canonical bytes for all
   primary arrays, including GPU-only gas vapor pressure, and acquired registry
   sidecars, plus detached CPU inspection carriers. Inspection data is
   non-authoritative and intentionally lossy for vapor pressure; restart uses
   canonical payload bytes. Direct-import-only `restart_resident_session()`
   first fully validates the record, then creates fresh session, registry,
   guard, containers, primary arrays, and sidecars only on an explicitly exact
    compatible device. Restart compatibility fails closed: it accepts only an
     `ACTIVE` `ResidentSession` carrier with schema-v1 noncommunication or
     schema-v2 optional-communication payloads, complete valid descriptors and
     bytes, and an exactly equal target `Device`. A v2 communication checkpoint
     retains one complete GAS or PARTICLES family plus matching metadata and
     restores fresh identities. It does not select or migrate devices, automatically
   restart normal session use, fall back to CPU, serialize to disk/remote, or
   guarantee rollback after an asynchronous device writer launches. Snapshotting
    requires roughly one additional host copy of resident payload bytes plus
    detached inspection copies. A published resident coagulation RNG stream
    fail-closes checkpoint and finalize before device or payload work; stream
    metadata and RNG words are not serialized, so restart continuation is not
    available. See
    [ADR-007](decisions/ADR-007-resident-session-checkpoint-finalize-restart.md),
    [ADR-008](decisions/ADR-008-resident-session-failure-close-semantics.md),
    and [ADR-018](decisions/ADR-018-resident-communication-integration.md).
- `gpu_resources.py` - Direct-import-only, Warp-dependent concrete registry for
   complete reusable native process sidecars, including one optional closed-map
   communication family. Each registry accepts exactly one
  exact `ACTIVE` `ResidentSession`, pins its lifecycle, dimensions, device, and
  all primary-array identities, and rejects session drift before every
  acquisition. Typed concrete manifests drive complete fixed
  dtype/shape/device/contiguity metadata checks, checked allocation sizes, and
  metadata-only nonaliasing among sidecars and with protected primaries. A role
  pins one caller- or registry-allocated identity; compatible repeats preserve
  the exact view, records, and arrays. This validates pinned ownership rather
  than unverifiable allocator provenance. It creates no public package export
  and has no execution/selection, transfer/sync/restore, lifecycle, transport,
   process-configuration/physics, or general RNG reset/advance behavior. On
   first `acquire_coagulation()` or `acquire_wall_loss()`, it creates and
   initializes that process's distinct P1-derived `wp.uint32` sidecar from
   immutable resident stream metadata, then retains its registry, binding, and
   view by identity. Compatible repeats neither allocate nor reseed it. There is
   no reset/inspection API, hidden transfer/synchronization, package export, or
   checkpoint continuation.
   `validate_pinned_session()` is the metadata-only integration seam: it
     requires exact retained-session identity and reuses active
     lifecycle/signature/schema validation without acquisition or allocation.
   `validate_diagnostic_outputs()` similarly validates only separately owned
   contiguous float64 `(B, S)` diagnostic outputs against pinned primaries and
   established sidecars; it neither publishes nor allocates those outputs.
    `validate_wall_loss_resources()`, `validate_nucleation_resources()`, and
    `validate_communication_resources()` first
   validate that pinned session, then require the exact already-published view
   and its pinned sidecar bindings for the corresponding family. These
    established-view seams neither acquire resources nor inspect payloads,
   mutate registry state, transfer, synchronize, or execute physics.
   Condensation thermodynamic roles are derived scratch/property storage only.
- `diagnostics.py` - Concrete direct-import-only E7-F5 P6 closed resident
  snapshot protocol. It supports only ordered gas-concentration and
  saturation-ratio snapshots into separately caller-owned contiguous float64
  `(B, S)` outputs, which are validated against resident primaries, published
  sidecars, and each other. Canonical empty outputs are no-dispatch no-ops; it
  exposes neither callbacks nor arbitrary resident inspection.
  - `resident_communication.py` - Concrete direct-import-only E7-F7 P5 barrier
    executor. It validates an already acquired complete closed GAS or PARTICLES
    map by identity, dispatches communication using pre-update volumes before
    optional prescribed volume evolution, and has no P1 validation, acquisition,
    transfer, synchronization, fallback, retry, or rollback behavior. Each
    barrier invalidates saturation ratio only; vapor pressure remains fresh.
    Standalone direct-kernel empty/disabled-map and unchanged-volume no-op
    behavior does not extend to resident barrier composition.
 - `resident_scheduler.py` - Concrete direct-import-only E7-F5 P6 composition
  boundary. It requires the exact active session/registry/closed-guard binding,
    matching request carriers, and exactly the twelve resolver-produced canonical
    nodes. It opens one token after complete preflight, dispatches the canonical
    communication then optional volume-evolution barrier (both before the ten
    ordinary nodes), then resolver order,
  consumes virtual thermodynamic refreshes only through condensation and
  diagnostics consumer windows, and completes the token only after the full
  loop succeeds. It has no package export, transfer, synchronization, fallback,
   resource replacement, or rollback; a possible post-launch failure faults the
    resident session. Resident Brownian dispatch uses the exact published
    coagulation sidecar by identity and forces `initialize_rng=False`; resident
    wall loss does the same with its independent wall-loss namespace. The
    resolved wall-loss selection is authoritative: disabled, prelaunch-skipped,
    zero-time, and valid no-work lanes retain their supplied RNG state, while
    only selected lanes whose work launches may consume it. The scheduler does
    not acquire, replace, inspect, synchronize, or reseed either stream. See
   [ADR-012](decisions/ADR-012-resident-complete-loop-and-diagnostics.md) and
   [ADR-018](decisions/ADR-018-resident-communication-integration.md).
- `process_adapters.py` - Concrete-only, direct-import resident delegation
   boundary for dilution, wall loss, and nucleation. Frozen request carriers
   retain the exact active `ResidentSession`, its pinned
    `GPUResourceRegistry`, and (for wall loss/nucleation) an exact established
    published resource view by identity. Wall-loss requests also retain the
    scheduler-resolved ascending logical-box selection. An empty selection is a
    prelaunch skip; a partial selection dispatches one-box aliases of selected
    state and RNG lanes, so disabled lanes cannot be written. After metadata-only
    session/view/selection validation, each adapter lazily resolves and invokes
    the supported direct GPU kernel, forwarding resident containers, sidecars,
    controls, and persistent RNG state unchanged and returning the native result.
    It never transfers, synchronizes, acquires or replaces resources, retries,
    rolls back, falls back, or performs physics; direct-kernel validation,
     mutation, and post-launch failure semantics remain authoritative. No name
    is exported through `particula.execution`, its adapters package, or top-level
    `particula`. See
    [ADR-009](decisions/ADR-009-resident-process-delegation-adapters.md).
- `state_updates.py` - Concrete-only, direct-import Warp-resident state-update
  boundary beside the session, registry, and process adapters. Frozen immutable
  environment and gas request carriers retain the exact active
  `ResidentSession`, pinned `GPUResourceRegistry`, `ResolvedProcessGraph`, and
  canonical `environment_update` or `gas_update` `ProcessNode` by identity.
  After deterministic read-only registry, graph-role, schema, alias, and payload
  validation, the executor copies only temperature/pressure or gas concentration
  into the existing resident arrays. Canonical zero-box and zero-species schemas
  are accepted write-free no-ops. It preserves all resident/container/primary
  identities, leaves `particles.volume` and every untargeted array unchanged,
  and does not schedule, refresh vapor pressure or saturation ratio, acquire
  registry resources, transfer host data, synchronize, change lifecycle,
  transport, fall back, or export names through `particula.execution` or the
   top-level package. See
   [ADR-010](decisions/ADR-010-resident-state-update-boundary.md).
- `communication.py` - Concrete-only, direct-import, Warp-dependent E7-F7 P1
  declaration and read-only validation boundary for fixed-shape communication
  maps. Its frozen carriers retain caller-owned map arrays by identity without
  binding a resident session or registry. Validation accepts only valid map
  topology, enabled-edge flags, nonnegative finite rates in 1/s, optional final
  volumes in m³, and nonaliasing fixed schemas; it writes neither caller arrays
  nor resident state and copies no payload. Empty and all-disabled maps still
  receive complete applicable preflight and are write-free on success. P1 has no
  source-inventory or time-step input, so P3—not this module—must atomically
   reject population-dependent outbound overdraw before writers launch. The
   separate GPU-kernel P2 owns final-volume writes; the GPU-kernel P4 owns
   particle transport, and P5 owns exact resident
  primary/sidecar binding and alias checks. This module has no transfer,
  synchronization, fallback, scheduling, or package/top-level export.
- `thermodynamic_updates.py` - Concrete-only, direct-import Warp-resident
  freshness coordinator. Its immutable request retains exact session, pinned
  registry, resolver-produced graph, resolved schedule, and thermodynamic
  configuration identities. Callers report each successful ordinary node; the
  coordinator owns cursor and stale markers and, before one explicit
  condensation or diagnostics callback, consumes immediately preceding virtual
  refresh IDs. It delegates vapor-pressure writes to the authoritative concrete
  primitive and launches the on-device SI saturation calculation without host
  payload reads, transfer, synchronization, or CPU fallback. Writer failures do
  not advance the cursor: a failed vapor writer keeps both fields stale, while a
   subsequent saturation failure retains fresh vapor pressure and stale
   saturation. It has no lifecycle, resource acquisition, full scheduler, or
   general process-dispatch behavior and is not package- or top-level-exported.
   See [ADR-011](decisions/ADR-011-resident-thermodynamic-freshness-coordinator.md).
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
   RNG sidecar and its explicit initialization intent unchanged. The distinct
   resident-Warp carrier instead requires the registry-published coagulation
   stream by identity and always forwards `initialize_rng=False`; the direct
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

`particula/gpu/` contains experimental Warp-backed data containers, explicit
CPU↔GPU transfer helpers, device-side physics helpers, and kernel entry points.
Its current import paths and caller-owned explicit-transfer/direct-kernel model
remain supported without a semantic change. See
[ADR-015](decisions/ADR-015-execution-public-surface-and-experimental-gpu-policy.md).

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
- `communication.py` - Concrete-only, direct-import, Warp-dependent E7-F7 P2
   final-volume evolution writer, P3 gas-communication seam, and P4
   particle-transport seam.
   `volume_evolution_step_gpu` accepts only a
  caller-owned active-device contiguous `wp.float64` final-volume array of
  shape `(B,)` in m³. After complete schema, domain, nonaliasing, factor, and
  scaled-concentration safety preflight, it mutates only `particles.volume` and
  particle/gas concentration by `old_volume / final_volume`, preserving
  extensive inventory and all container/array identities. Equal final volumes
  are a write-free no-op; rejection before an apply writer leaves caller state
  unchanged, while rollback is not promised after an asynchronous writer
  launches. It has no package export, hidden transfer or synchronization, CPU
   fallback, map declaration, transport, scheduler/session binding, resizing,
   or `Runnable` behavior. `GasCommunicationBuffers` and
   `gas_communication_step_gpu` are concrete-only direct imports. The gas seam
   stages immutable pre-step `amount = concentration * volume` ledgers and
   makes one primary commit,
   `gas.concentration = (amounts + amount_deltas) / volume`; it does
   not accept or fuse a final-volume update. It accepts prescribed GAS maps,
   including enabled declared `-1` open source/sink endpoints only when their
   matching caller-owned `(B, S)` accounting ledgers are supplied. Its caller-owned
   `(B, S)` amount, delta, and outbound ledgers make closed-map extensive
   conservation and open-boundary inventory changes explicit. Host/schema
   preflight preserves primaries, although documented ledgers may change during
   device planning; no rollback is promised after a writer launches. Direct
   callers retain same-device ownership and synchronize explicitly. It has no
   hidden transfer, CPU fallback, volume evolution, particle transport,
   scheduler/session binding, resizing, or package export.
   `ParticleCommunicationBuffers` and `particle_communication_step_gpu` are
   likewise concrete-only and accept
   prescribed closed, in-domain `PARTICLES` maps only. P4 plans immutable
   pre-call state, reuses exact destination populations or reserves ascending
   pre-step free slots, then performs one gated commit. It preserves weighted
   particle number, every mass lane, and signed charge without mutating gas or
   volume. No hidden transfer, synchronization, fallback, resizing, compaction,
   scheduling/resident integration, or RNG is provided. P1 remains separately
   owned by `particula.execution.communication`; P3 retains gas-transfer
   admission, P4 retains particle-plan admission, and P5 retains resident
   binding. Resident composition separately pins only one complete closed GAS
   or PARTICLES family, dispatches communication using old volumes before
   optional volume evolution, and rejects direct-only open GAS endpoints. See
   [ADR-016](decisions/ADR-016-direct-gpu-volume-evolution-boundary.md) and
   [ADR-017](decisions/ADR-017-direct-gpu-particle-transport-boundary.md), and
   [ADR-018](decisions/ADR-018-resident-communication-integration.md).
- `environment.py` - Shared private normalization and validation for kernel
  environment inputs
- `tests/` - Test coverage
