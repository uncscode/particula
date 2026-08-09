# Architecture Guide

## Execution Capability and Selection Boundary

- `particula.execution` is a dependency-neutral, standard-library-only seam.
  Its frozen ordered 26-name package-level public APIs consist of `Backend`, `Device`,
  `Process`, `Capability`, `CapabilityRequirements`, `CapabilityDeclaration`,
  `CapabilityMatrix`, `ExecutionRequest`, `ExecutionAdapter`,
  `ExecutionContext`; `ExecutionCapabilityReason`, `ExecutionCapabilityError`,
  `UnknownExecutionTargetError`, `UnavailableExecutionTargetError`,
  `UnsupportedExecutionRequestError`, `UnknownBackendError`,
  `UnknownDeviceError`, `UnavailableRuntimeError`, `UnavailableDeviceError`,
  `UnsupportedProcessError`, `UnsupportedCapabilityError`,
  `InvalidExecutionStateError`, `FallbackDisallowedError`; and
  `FallbackPolicy`, `FallbackBoundary`, `CPUStateAuthority`. These names are
  top-level exports by identity; the concrete `errors` and `fallback` modules
  remain direct-import-only for their mechanics, while their public values are
   re-exported. The per-context backing registry remains private.
   [ADR-015](decisions/ADR-015-execution-public-surface-and-experimental-gpu-policy.md)
   records this value-versus-mechanics export policy.
- Declarations and requests are immutable exact metadata. Nonempty requirements
  must match a complete declaration exactly; empty requirements are accepted
  when the matrix contains a declaration for the same `Device` and `Process`.
  CPU metadata is spelled exactly `Device(Backend.CPU, "cpu")`; native
  identifiers for other backends remain opaque. Registration and resolution are
  context-local, selection-only operations: complete exact matrix validation of
  nonempty requirements occurs before one exact `(Process, Backend)` registry
  lookup. A resolved adapter retains
  identity, and the adapter is not executed during selection.
- This seam does not import, probe, or resolve a backend; perform availability
  detection; transfer or synchronize state; execute an adapter; retry; or
  fallback. In particular, generic `ExecutionAdapter` argument, result, state,
  mutation, and runtime-error semantics are not public. P3/P4 state, result,
  mutation, and concrete CPU execution-adapter types remain direct-module-only.
- `particula.execution.availability` is the separate concrete, direct-import-only
  E7-F6 P2 availability boundary. Its `resolve_availability()` first validates
  an exact CPU/Warp provider registry without optional-runtime work, then
  short-circuits in this order: pure recognition, structural process
  declaration, exact capability declaration, lazy runtime status, runtime
  device status, and request-associated state validation. The frozen decision
  retains only the exact P1 request. It does not select an adapter or own a
  runtime handle, device object, execution payload, or state; it never
  executes, transfers, synchronizes, allocates, or mutates. CPU recognizes only
  `Device(Backend.CPU, "cpu")`; Warp native strings are opaque and are passed
  unchanged to the lazy runtime device check. This concrete module remains
  absent from package and top-level exports. See
  [ADR-013](decisions/ADR-013-pre-execution-availability-resolution.md).
- Strategy physics, builder configuration, and existing CPU `RunnableABC`
  behavior are separate from E7-F1 typed selection and downstream process
  adapter/session layers. E7-F4 P1--P7 ships the concrete lifecycle described
  below, not a resident loop or scheduler. The exact downstream ordering is
  `E7-F1 -> E7-F6 -> {E7-F2, E7-F3, E7-F4} -> E7-F5`: E7-F6 owns availability,
  fallback, error taxonomy, API stability, and export policy. The
   dependency-neutral `scheduler` remains declaration-only; E7-F5 P6 separately
    provides a bounded concrete resident complete-loop composer. E7-F7 P1
    remains the concrete communication-map declaration/read-only validation
    seam, while the separate direct-Warp P2 final-volume writer is now shipped
     at `particula.gpu.kernels.communication`. P3 retains transfer admission;
     the separate concrete-only P4 direct-Warp seam owns particle transport; and
      P5 retains resident binding. E7-F8 P2 integrates one narrow resident
     Brownian stream: immutable metadata is retained by the session, first
     coagulation-resource acquisition initializes one P1-derived sidecar, and
     resident dispatch retains it by identity with `initialize_rng=False`.
     Stream reset/inspection, wall-loss integration, hidden
     transfer/synchronization, retry, broad fallback, and replacement of direct
     GPU APIs remain deferred.
- `particula.execution.fallback` is the sole concrete, direct-import-only E7-F6
  P3 opt-in CPU fallback boundary. Its default `RAISE` policy re-raises the
  exact eligible typed availability/support failure. Explicit CPU policy may
  select the already-registered canonical CPU adapter only for
  CPU-authoritative `PRE_UPLOAD` or caller-asserted `RESTORED` state; resident,
  uploaded, and mutated state claims fail closed. It records requested backend,
  selected backend, and capability reason outside the unchanged native result
  metadata. The module performs neither transfer, conversion, synchronization,
  lifecycle work, restoration, retry, nor rollback. Its concrete module,
  operations, and carriers remain absent from package and top-level exports,
  while its three policy enums are intentionally public. It does not add
  implicit fallback to resident or direct GPU boundaries. See
  [ADR-014](decisions/ADR-014-opt-in-cpu-fallback-boundary.md).

## Concrete GPU-Resident Session Boundary

- E7-F4 P1--P7 is a bounded concrete lifecycle/checkpoint architecture. Its
  direct import seams are `particula.execution.gpu_session` (setup, session,
  and guard), `particula.execution.gpu_resources` (registry), and
  `particula.execution.checkpoint` (records, controller, and explicit restart).
  These names are deliberately absent from `particula.execution` and top-level
  exports; the boundary is not a public resident-process API.
- `particula.execution.gpu_session` is an intentionally concrete-only,
  unexported P1/P2 boundary. `ResidentSession` retains valid caller-owned Warp
   particle, gas, and environment containers, immutable dimensions, a Warp
   `Device`, a CPU-owned gas-name tuple, validated immutable resident stream
   metadata, and one declared lifecycle value by identity.
   `setup_resident_session` is direct-import-only and must not be
  promoted through `particula.execution`, `particula.execution.adapters`, or
  top-level `particula`.
- Construction performs only fixed-cost, read-only carrier and generated Warp
  container/primary-array metadata validation: type, dtype, shape, and same
  device agreement. Warp and generated types are lazy imports; CPU-only carrier
  validation does not import or probe Warp. Construction neither reads payloads
  nor synchronizes, transfers, converts, allocates, launches kernels, or
  schedules work.
- P2 setup first performs CPU-only local preflight of an exact Warp `Device`;
  `ParticleData`, `GasData`, and `EnvironmentData`; their cross-container
  shapes; and exact-string ordered gas names. Native availability is an upstream
  E7-F6 precondition: this boundary does not probe, normalize, select, or
  substitute a device. Only after preflight, it calls each established
  `particula.gpu.conversion` upload helper exactly once in particle, gas, and
   environment order with the unchanged native identifier. It retains
   `tuple(gas.name)` and validated stream metadata as CPU metadata; neither is
   uploaded to `WarpGasData`. Setup does not allocate an RNG sidecar.
  A conversion or final session-validation error propagates with no partial
  session publication. P2 has no fallback, synchronization, restoration,
  sidecars, retry, or cleanup behavior.
- The lifecycle vocabulary (`ACTIVE`, `FAULTED`, `FINALIZED`, and `CLOSED`)
  begins as immutable P1 state; P5 owns checkpoint/finalize and P6 owns explicit
  failure and terminal-close transitions. Existing direct GPU kernels and
  adapter-local physical validation remain authoritative. See
   [ADR-004](decisions/ADR-004-concrete-gpu-resident-session-boundary.md) and
   [ADR-005](decisions/ADR-005-one-time-gpu-resident-session-setup.md).
- P4 adds direct-import-only `ResidentStepGuard` and identity-only,
  frozen `ResidentStepToken` beside the immutable P1 carrier. One exact active
  `ResidentSession`/`GPUResourceRegistry` binding permits one open token only;
  completed-step count and simulated time advance only after matching token
  completion. The guard does not execute adapters, transfer or restore state,
  synchronize, acquire or allocate sidecars, resize, evolve the environment, or
  fall back. P6 permits only an explicit direct operation owner to classify a
  failure: a read-only failure aborts and releases the matching open token with
  no counter/time advance and leaves the session reusable; a writer that may
  have launched releases that token, faults the exact active session, preserves
  observable device mutation, and provides no rollback. The seam never infers
  outcomes from exception types, intercepts raw helpers, or replaces the
  original operational exception.
- `GPUResourceRegistry.validate_pinned_session()` is the direct-module-only P4
  binding seam: it requires exact retained-session identity and reuses active
  lifecycle, pinned-signature, and schema validation without payload inspection,
  sidecar acquisition, allocation, or mutation. Future checkpoint, restore,
  finalize, close, fault, conversion, resize, and rebind boundaries must call
  `assert_step_closed()` before their own work; P5/P6 retain those operations
    and their policy. The gate does not globally intercept raw low-level helpers.
     See [ADR-006](decisions/ADR-006-resident-gpu-step-lifecycle-guard.md).
- `GPUResourceRegistry.acquire_coagulation()` is the sole resident stream
  acquisition point. For an exact active session it validates a supplied RNG
  sidecar before publication, or allocates one, then initializes exactly one
  P1-derived coagulation-only `wp.uint32` stream from the session metadata.
  Compatible repeats retain the exact stream and resource view by identity and
  neither allocate nor reseed. Resident Brownian dispatch requires that exact
  stream and passes literal `initialize_rng=False`; no wall-loss stream,
  reset/inspection API, hidden transfer/synchronization, or package/top-level
  export is introduced.
- `GPUResourceRegistry.validate_wall_loss_resources()` and
  `.validate_nucleation_resources()` are direct-module-only, metadata-only
  established-view seams. After validating the exact pinned active session,
  they require the exact already-published resource view and retain its pinned
  sidecar bindings by identity. They neither acquire or replace sidecars,
  inspect payloads, mutate registry state, transfer, synchronize, nor execute
  a process.
- `particula.execution.process_adapters` is a concrete-only, direct-import
  delegation boundary for resident dilution, wall loss, and nucleation. Its
  frozen request carriers retain exact session/registry references and, where
  required, exact established wall-loss or nucleation views. Each matching
  adapter completes metadata-only preflight, lazily resolves one supported
  direct kernel, invokes it exactly once, and returns its native result
  unchanged. It forwards resident containers, published sidecars, controls, and
  persistent RNG state by identity. It does not acquire resources, transfer,
  synchronize, retry, roll back, fall back, inspect physics, or recover direct
  writer failures. These names remain absent from `particula.execution`, its
  adapters package, and top-level `particula`; the direct kernel retains
  numerical validation and post-launch semantics. See
   [ADR-009](decisions/ADR-009-resident-process-delegation-adapters.md).
- `particula.execution.state_updates` is a separate concrete-only,
  direct-import Warp-resident update boundary. Its frozen request carriers retain
  exact identities for an active `ResidentSession`, its pinned
  `GPUResourceRegistry`, a `ResolvedProcessGraph`, and a canonical
  `environment_update` or `gas_update` `ProcessNode`. The executor first
  validates that registry binding, graph membership and canonical role, and
  complete input schemas, ownership, and payload values without a writer. It
  then copies only temperature and pressure or gas concentration into the
  existing resident storage. It preserves resident/container/primary-array
  identity and leaves particle volume and all untargeted arrays unchanged.
  Canonical zero-box and `(n_boxes, 0)` gas schemas are accepted write-free
  no-ops. The boundary neither schedules nodes nor refreshes vapor pressure or
  saturation ratio; it does not acquire resources, transport or transfer host
  data, synchronize, alter lifecycle state, fall back, or gain package/top-level
   exports. See [ADR-010](decisions/ADR-010-resident-state-update-boundary.md).
- `particula.execution.communication` is a concrete-only, direct-import,
  Warp-dependent E7-F7 P1 declaration and read-only validation boundary for
  fixed-shape communication maps. Its frozen configuration preserves
  caller-owned map arrays by identity without session or registry binding.
  Validation checks only map topology, enabled-edge flags, nonnegative finite
  per-edge rates in 1/s, optional finite positive final volumes in m³, and
  fixed-schema nonaliasing. It writes no caller-owned or resident array and
  copies no payload; empty and all-disabled maps remain successful write-free
  cases after complete applicable preflight. Because P1 has neither source
   inventory nor time-step input, it cannot validate population-dependent
   outbound overdraw: P3 owns that atomic pre-writer check. The separate GPU
   kernel P2 owns final-volume writes, the GPU-kernel P4 owns particle
   transport, and P5 owns exact resident
  primary/sidecar alias checks. The module provides no transfer,
   synchronization, fallback, or scheduling behavior and is not exported through
   `particula.execution` or top-level `particula`.
- `particula.execution.resident_communication` is the separate concrete-only,
  direct-import P5 composition seam. `GPUResourceRegistry.acquire_communication`
  is the sole P1 payload-validation and allocation point: it pins one exact
  closed-map `GAS` or `PARTICLES` configuration, its maps, native work record,
  and optional final-volume sidecar by identity. Normal execution uses only
  metadata/identity validation; it neither reacquires nor scans payloads.
  Combined maps and open `-1` endpoints are not resident forms. The executor
  dispatches gas or particle communication using pre-update volumes, then calls
  prescribed volume evolution only when final volumes are pinned. The standalone
  `volume_evolution_step_gpu` is independently callable; this resident use is
  an optional scheduled barrier with separate composition rules. Both barriers
   invalidate `SATURATION_RATIO` only, leaving vapor pressure fresh. This seam
   has no package or top-level export and no transfer, synchronization, fallback,
   retry, or rollback behavior. See
   [ADR-018](decisions/ADR-018-resident-communication-integration.md).
- `particula.execution.thermodynamic_updates` is a concrete-only,
  direct-import Warp-dependent freshness coordinator. Its exact request binds
  an active `ResidentSession`, pinned `GPUResourceRegistry`, resolver-produced
  graph, resolved schedule, and `ThermodynamicsConfig`. Callers report only
  successful ordinary nodes; coordinator-owned cursor and stale markers then
  consume immediately preceding virtual refresh IDs before one explicit
  condensation or diagnostics callback. It delegates vapor-pressure writes to
  the authoritative primitive (including its documented configuration
  fingerprint reads) and calculates saturation on resident device arrays. A
  failed vapor writer leaves both fields stale; if vapor succeeds and saturation
  fails, vapor remains fresh and saturation stale, with no cursor advance. The
  coordinator preserves resident identities and does not acquire resources, own
  lifecycle, transfer, synchronize, fall back, run a full scheduler, or provide
    general process dispatch or package/top-level exports. See
   [ADR-011](decisions/ADR-011-resident-thermodynamic-freshness-coordinator.md).
- E7-F5 P6 adds `particula.execution.diagnostics` and
  `particula.execution.resident_scheduler` as concrete direct-import-only
  resident composition seams. Diagnostics is a closed, ordered protocol with
  exactly `GAS_CONCENTRATION_SNAPSHOT` and `SATURATION_RATIO_SNAPSHOT`; it copies
  the current resident `(B, S)` fields into separately caller-owned contiguous
  float64 outputs after validating exact plan provenance and nonaliasing against
  primaries, published sidecars, and other diagnostic outputs. Canonical empty
  schemas are successful write-free no-ops. It does not expose callbacks,
  arbitrary inspection, registration, or output allocation.
- `ResidentSimulationScheduler` accepts only the exact resolver-produced
  twelve-node schedule: communication, optional volume evolution, environment
  update, gas update, vapor-pressure refresh, saturation refresh, condensation,
  Brownian coagulation, dilution, wall loss, nucleation, and diagnostics. The
  closed communication then volume-evolution barrier precedes every ordinary
  node; communication uses pre-volume-update state. With exact active
  session/registry/closed-guard and request bindings, it fully preflights before
  opening one token, dispatches resolver order, and routes condensation and
  diagnostics through thermodynamic consumer windows. A complete success calls
  `complete_step()` once. A failure before writer-capable dispatch leaves the
  session active; after a writer may launch it closes the token, faults the
  session, and offers no rollback. Neither seam is package- or top-level-exported
  and neither transfers, restores, synchronizes, checkpoints, finalizes,
   acquires/replaces resources, resizes, compacts, or falls back. See
   [ADR-012](decisions/ADR-012-resident-complete-loop-and-diagnostics.md) and
   [ADR-018](decisions/ADR-018-resident-communication-integration.md).
- P5 adds a concrete-only in-memory checkpoint boundary in
  `particula.execution.checkpoint`. `ResidentSession.checkpoint(registry, guard)`
  and `.finalize(registry, guard)` bind one controller by exact identity to that
  session, its pinned `GPUResourceRegistry`, and its `ResidentStepGuard`; both
  require the active pinned binding and a closed step before readback. The
  controller, checkpoint records, and restart helper are direct imports only and
  are not exported through `particula.execution`, its adapters package, or
  top-level `particula`.
- `checkpoint()` leaves the session active and returns a fresh immutable host
  snapshot. The first successful `finalize()` creates and caches the complete
  snapshot before transitioning the session to terminal `FINALIZED`; every later
  call returns the exact cached snapshot without new validation, synchronization,
  conversion, allocation, or upload. Snapshots include canonical immutable bytes
  for primaries and acquired sidecars and detached CPU inspection carriers. The
   inspection `GasData` intentionally omits GPU-only vapor pressure and is not
   authoritative; restart recovers vapor pressure from canonical bytes. A
   published resident coagulation RNG stream causes checkpoint and finalize to
   reject before device synchronization, payload conversion, or sidecar
   enumeration. Stream metadata and words are intentionally not serialized, so
   checkpoint restart never continues a resident RNG stream.
- `restart_resident_session(checkpoint, device)` is explicit and same-device
  only. Its preflight fails closed: it accepts an `ACTIVE` `ResidentSession`
  checkpoint with complete valid descriptors and bytes, an exactly equal target
  `Device`, and either schema-v1 without communication or schema-v2 with no
  communication family or one complete matching closed-map GAS/PARTICLES family
  and metadata. Restart creates fresh session, registry, guard, container,
  primary-array, sidecar, and (when present) communication resource identities;
  it never reuses the source binding. It neither chooses nor migrates a device,
  falls back to CPU, restarts automatically during normal use, provides
  disk/remote/delta storage, nor promises rollback after an asynchronous device
   writer launches. Snapshotting requires roughly one additional host copy of
   resident payload bytes plus detached inspection copies. See
   [ADR-007](decisions/ADR-007-resident-session-checkpoint-finalize-restart.md)
   and [ADR-018](decisions/ADR-018-resident-communication-integration.md).
- P6 `ResidentSession.close(registry, guard)` and its `discard()` spelling are
  concrete-only terminal lifecycle operations, not recovery actions. From
  `ACTIVE`, close validates the exact pinned binding once and requires a closed
  guard; from `FAULTED`, it uses only exact local binding checks and a closed
  guard. Both transitions end at `CLOSED`. Repeated close on `CLOSED`, and close
  on `FINALIZED`, are write-free no-ops retaining existing identities and the
  P5 cached checkpoint. Close/discard never synchronizes, checkpoints,
  finalizes, restores, restarts, converts, allocates, transfers, retries,
  migrates, falls back, or performs other implicit runtime work. See
  [ADR-008](decisions/ADR-008-resident-session-failure-close-semantics.md).

## Concrete Condensation Execution Boundary

- `particula.execution.adapters.condensation` remains concrete-only. It retains
  P2 CPU/Warp resource carriers and selected P3 CPU/Warp execution-state and
  adapter types; none are promoted through `particula.execution`, its adapters
  package, or top-level `particula`.
- P3 carriers retain their exact P2 state, controls, and CPU runnable by
  identity. They are frozen against field rebinding, but retained caller-owned
  resources remain mutable. Construction is side-effect-free and P2 retains its
  read-only ownership and metadata validation boundary.
- The selected CPU adapter completes local validation, requires the selected
  isothermal CPU profile, then calls the supplied `MassCondensation.execute()`
  once with the original aerosol, `time_step`, and `sub_steps`. The runnable
  must return that same aerosol. The adapter neither splits controls nor catches
  delegate exceptions.
- The selected Warp adapter completes local validation and profile preflight
  before lazily resolving `condensation_step_gpu`; it then makes one direct
  native call with the retained resources and forwards its native result
  unchanged. It imports neither Warp nor `particula.gpu` on the CPU path, and
  does not resolve the direct kernel before successful Warp preflight.
- Both adapters report state mutation while retaining state and backend results
  by identity. They perform no conversion, allocation, transfer, restoration,
  synchronization, retry, fallback, or post-launch recovery. Backend exceptions
  propagate unchanged; a launched Warp kernel retains its native rollback
  limits.
- The CPU selected boundary remains isothermal and rejects semantic latent heat.
  The selected Warp boundary forwards caller-owned `latent_heat`,
  `energy_transfer`, and deferred `thermal_work` sidecars by identity. The
  direct kernel retains thermal-sidecar validation, execution, exception, and
  post-launch authority; the adapter adds no transfer, allocation,
  synchronization, fallback, result reconstruction, or rollback behavior.

## CPU Nucleation Boundaries

- `particula.dynamics.nucleation` provides the bounded CPU-only P4 construction
  API: immutable activation/kinetic potential-rate strategies,
  `NucleationSourceConfig`, their builders, and `NucleationFactory`. These
  names are deliberately re-exported through `particula.dynamics`.
- P4 strategies calculate potential formation-event rates only. They do not
  create particles, admit inventory or slots, mutate gas or particle state, or
  provide GPU integration.
- `particula.dynamics.Nucleation` and `NucleationCommitConfig` are the
  supported CPU-only, single-box P5 process boundary. The runnable takes public
  P4 source configuration and `EnvironmentData`, adapts the legacy `Aerosol`'s
  existing particle and partitioning-gas backing data by identity, and returns
  that same aerosol.
- P5 splits a positive duration into equal sequential substeps. Every substep
  reads the gas state produced by prior successful substeps before evaluating
  the potential rate, then delegates source-demand finalization and commit to
  the concrete P2/P3 boundary. Atomicity is per attempted P3 substep, not for
  the complete `Nucleation.execute` call; successful earlier substeps persist
  if a later substep fails.
- P2 source-demand and P3 transaction records/helpers remain deliberately
  concrete-only in `particula.dynamics.nucleation.particle_source`; they are
  not package exports and P4 construction types do not import them. In
  particular, public construction does not expose particle-source
  finalization or commit helpers.
- P5 does not introduce GPU execution, hidden data transfer, or additional
  runnable/scheduler orchestration.

## CPU Particle Slot Management Boundary

- `particula.particles.slot_management` owns CPU-only fixed-slot classification,
  discovery, and direct-import activation for `ParticleData`.
- `get_slot_diagnostics` is its sole package-level export through
  `particula.particles`; `activate_slots` remains a direct import from
  `particula.particles.slot_management`, and validation helpers remain
  module-private.
- Discovery preserves all `ParticleData` storage and returns newly allocated
  fixed-shape `int32` free-index and count sidecars. Activation maps request
  prefixes to ascending free slots after complete read-only preflight, then
  mutates only mass, concentration, and charge storage in place.
- Storage resize or compaction, `ParticleData` mutation API changes, CPU↔GPU
  transfer, GPU execution, and a top-level particles activation export remain
  outside this boundary. Its fixed-shape behavior provides a deterministic CPU
  reference for later parity work.

## GPU Module Boundaries

The GPU package keeps a strict separation between transfer, schema, and
kernel-entry responsibilities.

### Transfer boundary

- `particula/gpu/conversion.py` owns explicit CPU↔GPU transfer helpers only.
- It should not absorb launch-time kernel validation or normalization logic.

### Schema boundary

- `particula/gpu/warp_types.py` defines Warp-backed container schemas only.
- It should remain a passive data-shape layer rather than a behavior layer.

### Kernel normalization boundary

- `particula/gpu/kernels/environment.py` owns shared private normalization and
  validation for GPU kernel entry points.
- This module is the common boundary for accepting legacy scalars, direct
  `(n_boxes,)` Warp arrays, or `WarpEnvironmentData` inputs before launch-time
  work.
- Condensation and coagulation should reuse this boundary rather than
  re-implementing environment validation independently.

### GPU package export boundary

- `particula.gpu` remains the experimental public home for Warp availability,
  context, low-level containers, and explicit CPU↔GPU transfer helpers. Its
  current import paths and caller-owned explicit-transfer/direct-kernel model
  remain supported without an import-time warning or semantic change. See
  [ADR-015](decisions/ADR-015-execution-public-surface-and-experimental-gpu-policy.md).
- Direct GPU step entry points should be imported from
  `particula.gpu.kernels`, not re-exported from top-level `particula.gpu`.
- Lower-level kernel helpers should stay module-local to
  `particula.gpu.kernels.condensation` and
  `particula.gpu.kernels.coagulation` unless a broader public contract is
  intentionally documented.
- Import the supported low-level dilution entry point with
  `from particula.gpu.kernels import dilution_step_gpu`.
- `dilution_step_gpu` completes deterministic, read-only validation before
  allocating private storage, launching a kernel, or mutating caller-owned
  state. Successful calls update particle and gas concentrations in place as
  `c_new = c * exp(-alpha * time_step)` and return the identical containers.
- The preflight guarantee ends at launch: post-launch rollback is not
  provided. This direct entry point does not imply CPU fallback or runnable
  support.
- Import the supported fixed-slot activation boundary with
  `from particula.gpu.kernels import activate_slots_gpu`. Its P3
  `get_slot_diagnostics_gpu` helper remains concrete-module-only at
  `particula.gpu.kernels.slot_management` and must not be re-exported.
- `activate_slots_gpu` maps selected request-prefix ranks to ascending free
  slots in caller-owned, fixed-capacity Warp storage. It reads and writes only
  particle mass, concentration, and charge; density and volume are
  intentionally unobserved. Requests and all activation/diagnostic `int32`
  sidecars are caller-owned, same-device inputs and outputs.
- P4 validates schema, ownership, current slot state, selected requests, and
  capacity before launching its writer. Rejected calls make no caller mutation
  or hidden CPU↔GPU transfer; after a writer launches, rollback is not
  promised. This direct boundary does not establish resizing, compaction,
  hidden fallback, or a higher-level runnable API. See
   [ADR-002](decisions/ADR-002-gpu-fixed-slot-activation-boundary.md).

### Direct GPU volume-evolution boundary

- `particula.gpu.kernels.communication` provides the concrete-only,
  direct-import, Warp-dependent E7-F7 P2 final-volume evolution writer:
  `volume_evolution_step_gpu`. It is deliberately absent from
  `particula.gpu.kernels`, `particula.gpu`, and top-level exports.
- It accepts caller-owned, active-device contiguous `wp.float64` final volumes
  shaped `(B,)` in m³. Complete read-only preflight validates the primary
  container schemas, domains, device ownership, byte-range nonaliasing, volume
  factors, and proposed concentration scaling before an apply writer launches.
- On success it returns the identical particle and gas containers, updates only
  `particles.volume` and particle/gas concentrations by
  `old_volume / final_volume`, and preserves extensive particle-number, mass,
  charge, and gas inventories. It leaves masses, charge, density, and gas
  metadata untouched. At this standalone direct boundary, equal final volumes
  are write-free no-ops; rejected calls leave caller-owned state unchanged, but
  rollback is not promised after an asynchronous writer launches.
- This isolated writer does not create or consume communication maps, perform
  transfer admission or particle transport, bind a resident session or scheduler,
  transfer or synchronize host data, fall back to CPU, resize/compact storage, or
  provide a `Runnable`. P1 map declaration and read-only validation remain
  exclusively in `particula.execution.communication`; P3+ own transport and
   other communication phases. See
   [ADR-016](decisions/ADR-016-direct-gpu-volume-evolution-boundary.md).

### Direct GPU gas-communication boundary

- Import this concrete-only direct boundary only from
  `particula.gpu.kernels.communication.gas_communication_step_gpu`. It is not
  exported through `particula.gpu.kernels`, `particula.gpu`, or top-level
  `particula`.
- It stages immutable pre-step `amount = concentration * volume` ledgers, then
  makes one gas commit using `new_concentration = final_amount / new_volume`.
  At this direct boundary, `new_volume` is the unchanged current volume; there
  is no fused direct-gas `new_volume` argument or volume-evolution operation.
- Caller-owned `(B, S)` amount, delta, and outbound work arrays are required;
  each enabled open source/sink endpoint additionally requires its matching
  `(B, S)` accounting ledger. Closed maps conserve extensive amounts within
  floating-point tolerance, while open ledger entries make inventory changes
  explicit.
- Callers own same-device storage and explicit synchronization. The boundary
  has no hidden transfer, synchronization, or CPU fallback. Host/schema
  preflight preserves primaries; documented work ledgers can change during
  device planning, and no rollback is promised after a writer launches.
- This is independent of the closed-only particle transport and optional volume
  evolution boundaries. Validated empty or disabled maps are write-free no-ops;
  invalid schema, alias, domain, device, overdraw, or required-ledger inputs
  gate the sole primary gas commit.

### Direct GPU particle-transport boundary

- `particula.gpu.kernels.communication` also provides the E7-F7 P4
  concrete-only, direct-import `ParticleCommunicationBuffers` carrier and
  `particle_communication_step_gpu`. Neither name is exported through
  `particula.gpu.kernels`, `particula.gpu`, or top-level `particula`.
- The seam accepts only prescribed closed, in-domain `PARTICLES` maps. It plans
  requests from immutable pre-call particle state, preserves complete mass
  vectors and signed charge, uses exact destination-population matches or
  ascending pre-step free slots, and performs one gated primary commit.
- A successful call preserves concentration-weighted particle number, every
  species-mass lane, and signed charge. It returns the identical particle
  container; gas, volume, maps, and primary/buffer identities remain
  caller-owned. Valid zero-demand cases are write-free after applicable
  validation. Pre-launch plan failures preserve primaries, while rollback is not
  promised after the commit writer launches.
- This direct seam does not transfer or synchronize host data, fall back to CPU,
  resize or compact slots, implicitly activate slots, use RNG, or bind a
  scheduler or resident session. P1 remains the declaration/read-only validation
  owner, P3 owns transfer admission, and P5 owns resident binding. See
  [ADR-017](decisions/ADR-017-direct-gpu-particle-transport-boundary.md).

### Direct GPU nucleation boundary

- `particula.gpu.kernels.nucleation` concretely implements the supported
  low-level import `from particula.gpu.kernels import nucleation_step_gpu`.
  P1 preflights fixed-capacity state and caller-owned sidecars; P2 admits shared
  inventory-safe source demand; P3 stages fixed slots; P4 resolves exhaustion
  with resampling before scaling fallback; and fused P5 activates selected slots
  and commits the matching gas transfer.
- Only `nucleation_step_gpu` is package-exported. `NucleationConfig`, P2/P3
  records, exhaustion controls, sidecars, and helpers are concrete-only in
  `particula.gpu.kernels.nucleation`; they are not promoted through either GPU
  package namespace.
- The direct caller owns CPU↔Warp conversion, same-device contiguous fixed-shape
  sidecars, device placement, and synchronization before inspecting successful
  asynchronous results. The step returns the identical particle and gas
  containers and provides no hidden transfer or CPU fallback.
- P3 retains demand beyond free capacity and uses ascending free-slot prefixes
  with `-1` tails. P4 selects resampling only when it fully covers a deficit,
  then uses representative-volume scaling as the configured fallback. P5 owns
  the selected-slot activation and matching gas mutation.
- Public rejection before P4 primitive entry preserves particle and gas state;
  P2--P4 may have written their documented sidecars before a later rejection.
  No rollback is promised after an E6-F6 primitive has been entered or a P5
  writer has launched.
- This boundary has no resize/compaction, GPU `Runnable`, scheduler/backend
  integration, or E6-F9 integration. E6-F9 remains a downstream
  explicit-transfer consumer.
- Import the supported fixed-slot wall-loss boundary with
  `from particula.gpu.kernels import wall_loss_step_gpu`. Its
  `NeutralWallLossConfig` is deliberately concrete-module-only at
  `particula.gpu.kernels.wall_loss`; do not re-export it through
  `particula.gpu.kernels` or `particula.gpu`.
- `wall_loss_step_gpu` owns immutable host configuration and frozen preflight
  for particle-resolved neutral and charged inputs. It dispatches the
  unchanged neutral kernel for neutral mode; charged kernels compose the private
  image-charge and electric-field-drift helpers in
  `particula.gpu.dynamics.wall_loss_funcs` only for nonzero-charge slots.
  Image-charge enhancement remains active for nonzero charge at zero wall
  potential, while charged zero-charge slots retain the exact neutral
  coefficient and RNG path. Spherical charged execution preserves a signed
  scalar field before adding the signed potential-derived contribution.
  Rectangular execution reduces caller-owned `(3,)` `float64` Warp field
  storage to its Euclidean magnitude before adding the signed
  potential-derived contribution; component signs do not individually select
  drift direction. The rectangular field is passed only to the charged
  rectangular kernel.
- After successful preflight, a nonzero call stochastically clears eligible
  fixed slots in place and returns the identical particle object. Removed slots
  have every mass lane, concentration, and caller-owned `charge` cleared;
  capacity and unselected storage are preserved. Zero time is write-free;
  pre-launch failures are atomic; rollback after a mutation launch is not
  promised.
- Its caller-owned `WarpParticleData.charge` field and optional `(n_boxes,)`
  `uint32` RNG sidecar remain external state rather than hidden transfer
  results. Successful positive-time calls advance each box sequentially only
  for eligible slots, while omitted RNG state is private to the call. Explicit
  `initialize_rng=True` is the only supplied-state reset path. Zero-time,
  preflight failures, and positive-time inputs with no usable slots leave a
  supplied sidecar unchanged. This serial per-box ownership is a correctness
  constraint, not a throughput claim. Runnables, hidden transfers/fallbacks,
  and CPU/Warp stochastic parity remain deferred. See
  [ADR-001](decisions/ADR-001-neutral-gpu-wall-loss-boundary.md).

## Design Intent

- Keep CPU↔GPU transfers explicit.
- Keep Warp container definitions stable and behavior-free.
- Keep cross-entry-point normalization private to `particula/gpu/kernels/`.
- Share validation at kernel boundaries when multiple GPU entry points consume
  the same environment contract.
- Keep GPU exports deliberate: top-level helpers in `particula.gpu`, direct
  step entry points in `particula.gpu.kernels`.
