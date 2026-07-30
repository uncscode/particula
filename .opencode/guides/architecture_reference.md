# Architecture Reference

**Project:** particula  
**Last Updated:** 2026-07-30

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

`particula.execution` is a package. Its supported selection surface remains
the existing exact ten-name public export list; do not promote adapter modules
or state carriers through it or through top-level `particula`.

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

Restart compatibility is exact and fail-closed: only schema version `1`, carrier
type `"ResidentSession"`, ACTIVE records, complete valid payload schemas, and an
exactly equal `Device` are accepted. Other versions, schemas, malformed records,
non-ACTIVE checkpoint records, and device mismatches reject. Finalization makes
its source session terminal but returns an ACTIVE checkpoint eligible for explicit
restart. E7-F5 P2 supplies declaration-only scheduling; E7-F7 transport and
E7-F8 detailed RNG-stream policy remain future work.

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
restoration, lifecycle change, transport allocation, process
physics/configuration, or RNG initialization, advancement, or reset.
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
complete ten-node resolver-produced schedule and one exact active
session/registry/closed-guard binding. It dispatches that resolved order, using
the thermodynamic consumer windows for condensation and diagnostics. Neither
boundary is package-exported and neither uploads, restores, synchronizes,
checkpoints, acquires/replaces storage, resizes, or falls back. After a writer
may launch, a failure closes the token and faults the session without rollback.
See [ADR-012](architecture/decisions/ADR-012-resident-complete-loop-and-diagnostics.md).

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
