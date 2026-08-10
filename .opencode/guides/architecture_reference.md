# Architecture Reference

**Project:** particula  
**Last Updated:** 2026-08-09

This reference summarizes the particula package structure and key architectural
conventions migrated from the legacy guide set.

## Package Map

```text
particula/
├── activity/          # Activity coefficients and phase separation
├── dynamics/          # Coagulation, condensation, wall loss
├── equilibria/        # Partitioning calculations
├── execution/         # Dependency-neutral execution selection
├── gas/               # Gas phase, species, vapor pressure
├── particles/         # Particle distributions and representations
├── util/              # Constants, validation, chemistry utilities
└── integration_tests/ # Cross-module integration tests
```

## Architectural Patterns

- Keep physics calculations in focused modules with clear units and citations.
- Use strategy, builder, and factory patterns where the package already uses
  them, especially in wall loss, vapor pressure, and representation code.
- Keep tests co-located with modules in `tests/` directories.
- Export public APIs deliberately through package `__init__.py` files.
- Keep validation close to public function boundaries.

## Execution Selection and Condensation State

`particula.execution` has a frozen ordered 26-name public surface: the 10
selection declarations (`Backend` through `ExecutionContext`), the closed
13-name capability-error taxonomy (`ExecutionCapabilityReason` through
`FallbackDisallowedError`), and `FallbackPolicy`, `FallbackBoundary`, and
`CPUStateAuthority`. These values are also top-level `particula` exports by
identity. The concrete `errors` and `fallback` modules remain direct-import-
only for mechanics and carriers, but their public values are re-exported.
Concrete adapter modules, state/result carriers, availability, fallback
operations and carriers, lifecycle seams, and GPU boundaries remain
direct-module-only. This frozen value-versus-mechanics policy is recorded in
[ADR-015](architecture/decisions/ADR-015-execution-public-surface-and-experimental-gpu-policy.md).

`particula.gpu` retains its existing low-level containers, explicit CPU↔GPU
transfer helpers, and direct-kernel workflow as experimental APIs. Their current
import paths and caller-owned transfer model remain supported without an
import-time warning or semantic change.

`particula.execution.availability` is the concrete, direct-import-only P2
pre-execution resolver. `resolve_availability()` consumes already-valid P1
request and capability metadata, validates an exact complete CPU/Warp provider
registry before provider calls, and short-circuits through recognition,
structural process declaration, exact capability declaration, lazy runtime
status, device status, and request-associated state validation. Its immutable
decision retains only the exact request; it neither selects an adapter nor
executes, transfers, synchronizes, allocates, or mutates state. CPU recognizes
only canonical `Device(Backend.CPU, "cpu")`; Warp recognizes validated Warp
metadata without parsing or normalizing its native string, which remains opaque
until the lazy runtime device check. This boundary remains absent from
`particula.execution` and top-level exports. See
[ADR-013](architecture/decisions/ADR-013-pre-execution-availability-resolution.md).

`particula.execution.rng` is the concrete, direct-import-only E7-F8 P1 RNG
stream-identity boundary. It owns immutable host metadata and deterministic
per-process/per-logical-box initial-word derivation, then explicitly initializes
validated caller-owned Warp state buffers. Initialization allocates temporary
NumPy/Warp copy sources only and deterministically overwrites retained buffers
without acquiring, replacing, or rebinding them. Resident Brownian coagulation
and wall loss each retain an independent registry-owned sidecar initialized
from the canonical process manifest. Scheduled dispatch retains each sidecar by
identity with `initialize_rng=False`. Explicit direct-module lifecycle calls
alone may inspect frozen host metadata or reset all published streams or valid
selected lanes under an exact ACTIVE session/registry/closed-guard binding.
They never read back, synchronize, or affect normal dispatch. Schema-v3
 checkpoints require continuation metadata and may preserve an empty or
 nonempty published-current-word collection for exact-device
continuation; only explicit resets rederive words from the root seed.
Wall-loss dispatch receives its scheduler-resolved ascending logical-box set.
An empty set is a write-free prelaunch skip; a partial set delegates one-box
resident aliases so disabled lanes cannot be written. Thus disabled,
prelaunch-skipped, zero-time, and no-work lanes retain their RNG words. The
direct kernel signature and physics remain unchanged, and no rollback is
promised after a selected writer launches.

`particula.execution.adapters.condensation` contains concrete-only P2
condensation state carriers and shipped P3/P4 selected CPU/Warp adapters. The
carriers retain caller-owned CPU or resident Warp resources by identity and
perform construction-time, read-only metadata and ownership checks. The
adapters select no profile themselves: after exact profile and sidecar
preflight, CPU calls the supplied isothermal runnable once and Warp calls the
direct kernel once. They do not transfer, restore, synchronize, fall back, or
recover failures. Omitted direct-kernel property arrays retain the native
step's documented step-local fallback allocation behavior; callers own reusable
native sidecars, resource lifetime, synchronization, concurrency, and any
post-launch mutation or rollback semantics. These concrete names remain absent
from the package selection surface and top-level `particula`.

`particula.execution.gpu_session` is a concrete-only direct-import boundary:
import `setup_resident_session`, `ResidentSession`, and `ResidentStepGuard` from
that module. Setup preflights CPU data then converts particle, gas, and
environment carriers once in that order. It retains immutable dimensions,
resident identities, Warp `Device` metadata, and CPU-only ordered gas names.
One exact active session/registry binding permits one open guard token; normal
bookkeeping neither executes adapters nor bulk transfers or synchronizes.
Checkpoint, finalization, close, and discard require that exact binding and a
closed guard. Read-only failures may leave ACTIVE state reusable after token
release, while a possible post-launch writer failure faults without rollback;
close and discard never checkpoint, synchronize, restore, or mutate payloads.
These names are absent from `particula.execution` and top-level exports. See
[ADR-004](architecture/decisions/ADR-004-concrete-gpu-resident-session-boundary.md).

P5 adds the concrete-only `particula.execution.checkpoint` boundary. It is not
exported through `particula.execution`, its adapters package, or top-level
`particula`. `ResidentSession.checkpoint(registry, guard)` and
`ResidentSession.finalize(registry, guard)` use one controller bound by exact
identity to that active session, its pinned `GPUResourceRegistry`, and its
closed `ResidentStepGuard`. A checkpoint is a fresh immutable host snapshot;
the first successful finalization caches the complete snapshot and transitions
the session to terminal `FINALIZED`, while later finalization calls return that
same cached object.

Checkpoint records retain canonical immutable bytes for every primary array,
including GPU-only gas vapor pressure, and acquired sidecars, as well as
detached CPU inspection `ParticleData`, `GasData`, and `EnvironmentData`.
Inspection gas intentionally has no vapor-pressure field and is non-authoritative;
restart recovers canonical vapor-pressure bytes rather than inspection data.
Snapshots require approximately one additional host copy of resident payload
bytes plus detached inspection copies.

`restart_resident_session(checkpoint, device)` is an explicit direct import and
only accepts an exactly compatible target device. After complete host preflight,
it creates fresh session, registry, guard, containers, primaries, and sidecars;
it never reuses source identities. It does not select or migrate a device, fall
back to CPU, restart automatically during normal session use, provide disk,
remote, or delta persistence, or guarantee rollback after an asynchronous device
writer launches. See
[ADR-007](architecture/decisions/ADR-007-resident-session-checkpoint-finalize-restart.md).

Restart compatibility is exact and fail-closed: schema-v1 records remain
noncommunication, schema-v2 permits no communication family or exactly
one complete matching closed-map GAS or PARTICLES family and metadata. Both
require carrier type `"ResidentSession"`, ACTIVE records, complete valid payload
schemas, and an exactly equal `Device`; restart reconstructs fresh communication
resources rather than reusing source identities. Other versions, partial,
mixed, malformed, or non-ACTIVE records and device mismatches reject.
Finalization makes its source session terminal but returns an ACTIVE checkpoint
eligible for explicit restart. Schema-v3 requires canonical published-stream
continuation metadata, although its current-word payload collection may be
empty. Immutable current RNG words are recovery authority and ordinary payloads
exclude RNG sidecars. Before setup, acquired coagulation and wall-loss process
families must pair bidirectionally with continuation payloads; legacy records
with acquired RNG process resources reject rather than reseed. E7-F5 P2 supplies declaration-only scheduling;
E7-F7 transport is shipped, while E7-F8 integration and remaining RNG-stream
policy remain future work. E7-F7 P4 particle
transport is shipped as the concrete-only
``particula.gpu.kernels.communication.particle_communication_step_gpu`` seam.
P4 owns immutable pre-step planning, admission, and one gated particle commit;
it accepts prescribed closed in-domain maps only and remains absent from package
and top-level exports.

E7-F5 P2 adds `particula.execution.scheduler`, a concrete direct-import-only
declaration-only scheduling boundary. It first resolves the complete P1 graph,
then applies enabled-node selection, one reviewed nucleation/condensation
direction, required freshness closure, and deterministic lexical topology. Its
immutable result is metadata only: it does not load GPU/Warp, lifecycle, or
resource boundaries; allocate, transfer, mutate, refresh, synchronize, or
launch processes. Scheduler names remain absent from `particula.execution` and
top-level exports.

`particula.execution.process_graph` remains the P1 owner of graph declaration
validation and normalization. Its canonical topology helper is a read-only,
lexical tie-breaking utility used by the scheduler; it does not add an execution
order field, scheduling policy, lifecycle behavior, or backend work to the
resolved P1 graph.

`particula.execution.thermodynamic_updates` is the P5 direct-import-only,
Warp-dependent resident freshness boundary. Its request binds one exact active
session, pinned registry, resolver-produced graph, resolved schedule, and
thermodynamic configuration by identity. Callers explicitly report only
successful non-refresh nodes. The coordinator owns its cursor and stale markers,
then immediately brackets a scheduled condensation or diagnostics callback with
the required virtual vapor-pressure and saturation writers. Vapor pressure is
delegated to the concrete GPU primitive (including its documented configuration
fingerprint reads); saturation is an on-device resident calculation. A failed
vapor writer leaves both fields stale; after a successful vapor writer, a failed
saturation writer leaves only saturation stale, and neither failure advances the
cursor. This concrete boundary preserves resident identities and provides no
lifecycle work, transfer, synchronization, fallback, full scheduler, or general
process dispatch. It is absent from package and top-level exports.

`particula.execution.gpu_resources` is a separate direct-import-only,
Warp-dependent concrete boundary beside `gpu_session`. Each `GPUResourceRegistry`
accepts exactly one exact `ACTIVE` `ResidentSession`, pins its dimensions,
device, lifecycle, and all primary-array identities, and rejects any later drift
before acquisition. It builds complete reusable native condensation,
coagulation, wall-loss, and nucleation sidecars; validates their complete fixed
dtype/shape/device/contiguity metadata; checks allocation sizes; and rejects
sidecar-to-sidecar and primary-to-sidecar aliasing. Compatible repeats return the
same views, records, and Warp arrays by identity. Ownership means pinned role
identity and nonaliasing, not allocator-provenance inference. Its typed
 manifests and views remain concrete-only and absent from package exports. The
 boundary performs no execution or selection, transfer/synchronization/
 restoration, lifecycle change, process physics/configuration, or RNG
  initialization, advancement, or reset, except that first coagulation or wall
  loss acquisition creates and initializes its distinct P1-derived,
  registry-retained ``wp.uint32`` stream from immutable resident stream metadata.
  Resident dispatch always supplies the exact stream with
  ``initialize_rng=False``. Its direct-only published-stream inspection and
  explicit selected reset APIs expose frozen metadata only and require a closed
  ACTIVE binding; they have no hidden transfer/synchronization, package export,
  or arbitrary checkpoint continuation. Schema-v3 capture privately enumerates
  published streams, reads their current words after its one synchronization,
  and restart rebinds fresh arrays without reseeding. Its communication acquisition seam is
 the sole exception for fixed-shape transport work storage: it validates one
 exact closed-map GAS or PARTICLES configuration and pins its native record,
 map arrays, and optional final-volume sidecar. Omitted required work arrays may
 be allocated at acquisition; normal resident steps perform metadata-only
 validation and neither reacquire nor inspect communication payloads.
Condensation thermodynamic entries are derived scratch storage only, never
thermodynamic configuration.
Its narrow `validate_pinned_session()` seam first requires exact session
identity, then reuses the existing active lifecycle/signature/schema validation
without acquisition, allocation, payload inspection, or mutation.

`GPUResourceRegistry.validate_wall_loss_resources()` and
`.validate_nucleation_resources()` are metadata-only direct-module seams for
resident delegation. Each first validates the exact pinned active session, then
requires the exact already-published wall-loss or nucleation view and its pinned
sidecar bindings by identity. Neither seam acquires or replaces resources,
inspects payloads, mutates registry state, transfers, synchronizes, nor invokes
physics.

`particula.execution.resident_communication` is the concrete-only composition
seam for those pinned closed-map resources. It dispatches communication with
pre-update volumes before optional prescribed volume evolution, with resident
containers and records by identity. This canonical barrier precedes all ten
ordinary nodes in the closed twelve-node schedule. The barriers invalidate
saturation ratio only, retain fresh vapor pressure, and provide no hidden
transfer, synchronization, fallback, retry, rollback, or public API. Combined
communication maps and open endpoints are not resident forms. See
[ADR-018](architecture/decisions/ADR-018-resident-communication-integration.md).
Acquisition pins maps, native work, and an optional final-volume sidecar once.
Normal steps validate only that identity and metadata: they do not inspect map
payloads, allocate, transfer, synchronize, restore, or replace resources.
Standalone direct-kernel empty or disabled maps and unchanged final volumes are
write-free no-ops; resident barriers instead follow their own composition and
validation rules. Schema-v1 restart remains noncommunication; schema-v2 permits no family
or one complete closed-map family. Both require an ACTIVE valid record and an
exact device; explicit restart creates fresh identities and is never automatic,
migratory, or rollback-capable.

`particula.execution.process_adapters` is a concrete-only direct-import
delegation boundary for resident dilution, wall loss, and nucleation. Its frozen
request carriers retain exact active session/registry bindings and, for wall
loss and nucleation, exact established published views. After metadata-only
preflight, each adapter lazily resolves and calls exactly one supported direct
GPU kernel, forwarding resident containers, sidecars, controls, and persistent
RNG state by identity and returning the kernel's native result unchanged. It
does not acquire resources, transfer, synchronize, retry, roll back, fall back,
or perform physics; direct-kernel validation, mutation, and post-launch failure
semantics remain authoritative. No process-adapter name is exported through
`particula.execution`, its adapters package, or top-level `particula`. See
[ADR-009](architecture/decisions/ADR-009-resident-process-delegation-adapters.md).

E7-F5 P6 adds two further direct-import-only resident boundaries:
`particula.execution.diagnostics` and
`particula.execution.resident_scheduler`. Diagnostics is a closed two-operation
protocol (`GAS_CONCENTRATION_SNAPSHOT` and `SATURATION_RATIO_SNAPSHOT`), not a
callback API. Its separately caller-owned contiguous float64 `(B, S)` outputs
are checked against primaries, published sidecars, and one another; canonical
empty shapes are successful write-free no-ops. The scheduler accepts only the
complete twelve-node resolver-produced schedule and one exact active
session/registry/closed-guard binding. Communication and optional prescribed
volume evolution are closed-map barriers that run first and invalidate only
saturation ratio; it dispatches the remaining resolved order using the
thermodynamic consumer windows for condensation and diagnostics. Neither
boundary is package-exported and neither uploads, restores, synchronizes,
checkpoints, acquires/replaces storage, resizes, or falls back. After a writer
may launch, a failure closes the token and faults the session without rollback.
See [ADR-012](architecture/decisions/ADR-012-resident-complete-loop-and-diagnostics.md).
The resident communication and checkpoint integration is recorded in
[ADR-018](architecture/decisions/ADR-018-resident-communication-integration.md).

## Wall Loss

Wall loss strategies live in `particula.dynamics.wall_loss` and are exported
through `particula.dynamics`.

Key concepts:

- `WallLossStrategy` is the abstract base class.
- `WallLoss` wraps a strategy, splits `time_step` across `sub_steps`, clamps
  concentrations nonnegative, and composes with other runnables via `|`.
- `ChargedWallLossStrategy` adds image-charge enhancement, optional electric
  field drift, and neutral fallback when particle charge and field are zero.
- Supported distribution types are `"discrete"`, `"continuous_pdf"`, and
  `"particle_resolved"`.

## Nucleation

The public CPU-only nucleation API is exported through `particula.dynamics`.
P4 provides immutable activation/kinetic potential-rate strategies,
source-selection metadata, builders, and a factory. P5 provides the one-box
`Nucleation` runnable and `NucleationCommitConfig`.

`Nucleation` preserves the legacy `Aerosol` and backing-data identities. It
uses equal, gas-coupled substeps and is atomic per attempted substep, not across
the complete call. P2 source-demand planning and P3 particle-source
transactions remain concrete-only in `nucleation.particle_source`.

The concrete `particula.gpu.kernels.nucleation` module implements the
package-exported direct `nucleation_step_gpu`: P1 preflight, P2 admission, P3
fixed-slot staging, P4 resampling-first/scaling-fallback, and fused P5
selected-slot/gas-transfer commit. Only the step is exported; configuration,
records, sidecars, and helpers remain concrete-only. Callers own conversion,
same-device fixed-shape sidecars, device placement, and synchronization; a
successful call returns the identical particle and gas containers. P3 retains
demand beyond free capacity and reports ascending free-slot prefixes with `-1`
tails. Public rejection before P4 primitive entry preserves particle and gas
state, while documented P2--P4 sidecars may already have been written; no
rollback is promised after E6-F6 primitive entry or P5 writer launch. The direct
step has no hidden transfer, CPU fallback, resize/compaction, GPU `Runnable`, or
E6-F9 integration.

## Scientific Utilities

- Physical constants belong in `particula.util.constants`.
- Public numerical validation uses `particula.util.validate_inputs`.
- Chemical helper code lives under `particula/util/chemical/`.

## Documentation

Architecture guides and ADRs are under `.opencode/guides/architecture/`. Update
them when module boundaries, exported APIs, or major design patterns change.
