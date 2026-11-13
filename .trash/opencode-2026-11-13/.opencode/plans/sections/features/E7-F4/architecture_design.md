# Architecture Design

## High-Level Design

The resident session is a resource/lifecycle object in the E7-F1 execution
layer. It owns references to mutable Warp state and process resources while
keeping configuration and CPU-only metadata explicit. It is not a process
scheduler and never invokes physics by itself.

```text
CPU ParticleData + GasData + EnvironmentData + ordered metadata
                           |
          E7-F1 request + E7-F6 capability/availability validation
                           |
                           v
                 ResidentSession.setup(device)
                    |      |       |
                    |      |       +-- allocate/validate SidecarRegistry
                    |      +---------- one to_warp_* call per container
                    +----------------- freeze B/N/S/device metadata
                           |
                 ACTIVE resident state
                           |
             E7-F5 scheduler-facing lifecycle hooks
             (no restore, no bulk transfer, no implicit sync)
                           |
              +------------+-------------+
              |                          |
       checkpoint()                 finalize()
       sync once                    sync once
       restore x3                   restore x3
       ACTIVE remains               FINALIZED terminal
              |
       Checkpoint(state + metadata + opaque mutable resources)
              |
       explicit restart -> new setup/upload boundary
```

## Data / API / Workflow Changes

- **Data model:** Add a typed `ResidentSession`, immutable
  `ResidentDimensions`, lifecycle enum, `SidecarRegistry`, and immutable
  checkpoint record. The session references existing `WarpParticleData`,
  `WarpGasData`, and `WarpEnvironmentData`; no container schema changes are
  required.
- **Metadata:** Retain ordered gas names, backend/device identity, shape tuple,
  schema version, and process-resource manifest. P4 step/time counters are
  mutable `ResidentStepGuard` state, not fields on immutable `ResidentSession`.
  GPU-only derived fields and mutable sidecars are represented explicitly rather
  than silently dropped by CPU `GasData` restore.
- **Setup API:** Accept CPU containers plus E7-F1 context and E7-F6-validated
  device request. Validate compatible box/species/particle dimensions before
  conversion. Setup either succeeds with a complete active session or exposes
  no partially usable session.
- **Resource API:** Allocate known fixed-shape resources once or validate
  caller-supplied arrays. Registry keys are typed/internal; concrete kernel
  records remain in their owner modules. Process adapters receive only their
  required view.
- **Lifecycle API:** Concrete-only `ResidentStepGuard.begin_step()` returns one
  opaque identity token for an exact active session/registry binding;
  `complete_step()` advances count/time only for that same token. It neither
  schedules nor invokes processes. `assert_step_closed()` is the required
  preflight gate for future checkpoint/finalize/close/conversion/resize/rebind/
  fault boundaries; it has no transfer, synchronization, allocation, or global
  interception behavior. P5/P6 own those boundaries and their policy.
- **Restart:** Reconstruct CPU containers and upload into a fresh session.
  E7-F4 preserves opaque mutable RNG/resource state; E7-F8 defines stream
  semantics and exact stochastic restart guarantees.
- **Workflow hooks:** E7-F2/E7-F3 can acquire resource views; E7-F5 consumes the
  lifecycle hooks; E7-F7 extends resources for transport; E7-F8 specializes RNG
  state; E7-F9 validates the complete loop.
- **Compatibility:** Existing direct kernels, conversions, `gpu_context()`, CPU
  runnables, and exports remain unchanged. No hidden transfer or fallback is
  introduced.

## P1 Implementation

Issue #1484 implemented the construction-only foundation in
`particula/execution/gpu_session.py`. `ResidentDimensions` validates positive
box and nonnegative particle/species counts; `ResidentMetadata` retains an
exact Warp `Device` and ordered exact-string gas-name tuple; and
`ResidentLifecycle` declares `ACTIVE`, `FAULTED`, `FINALIZED`, and `CLOSED`
without transitions. Frozen, `eq=False` `ResidentSession` validates exact
carrier types, gas-name count, non-identical top-level containers, generated
Warp struct forms, twelve primary-array metadata declarations, and the declared
native device string.

The ordered preflight validates CPU-only carriers before attempting the optional
Warp import. It reads only fixed metadata (`dtype`, tuple `shape`, and device),
so validation is O(1) in `(B, N, S)` and has no payload read, synchronization,
transfer, launch, or data-sized allocation. P4 retains ownership of lifecycle
transition guards; P2, P3, P5, and P6 retain conversion, resources,
checkpoint/finalization, and failure/close behavior.

## P2 Implementation

Issue #1485 added direct-import-only `setup_resident_session()` to
`particula/execution/gpu_session.py`. Local preflight requires an exact
`Device` with `Backend.WARP`, validates concrete CPU carriers and their shared
`(B, N, S)` schema without importing Warp or conversion helpers, then derives
dimensions and validates the ordered exact-string CPU gas-name tuple.

Only after preflight, the factory function-locally imports established
conversion helpers and calls `to_warp_particle_data`, `to_warp_gas_data`, and
`to_warp_environment_data` once each, in order, using the unmodified native
device identifier. `ResidentSession` remains the final generated-schema and
shared-device gate; errors propagate with no partial-session publication. P2
does not query availability, select or normalize devices, use fallback,
synchronize, restore, create sidecars, or add exports. Availability approval is
E7-F6's explicit upstream responsibility.

## P4 Implementation

Issue #1487 (commit `61f101de1`) added `ResidentStepGuard` and
`ResidentStepToken` in `particula/execution/gpu_session.py`. The frozen token
uses identity equality and guard-private origin/duration fields, so only the
single outstanding token can complete its originating guard. Duration accepts
finite nonnegative non-boolean `Real` values without coercion; zero-duration
cycles are valid. Binding validation occurs before token publication or metadata
updates, preserving guard state on invalid duration, inactive/drifted binding,
nested begin, and mismatched/repeated completion.

The same change added direct-module-only
`GPUResourceRegistry.validate_pinned_session(session)`. It first checks
`session is self._session` and then calls the existing pinned-signature
validation path, retaining its ACTIVE lifecycle, primary-identity, and schema
checks without allocating resources or examining payloads. The guard and token
remain absent from `particula.execution`, adapter exports, and top-level exports.

## P5 Implementation

Issue #1488 implemented the explicit checkpoint boundary in
`particula/execution/checkpoint.py`. `ResidentCheckpointController` is bound by
identity to one `ResidentSession`, `GPUResourceRegistry`, and
`ResidentStepGuard`. Before any readback it requires ACTIVE lifecycle, the exact
pinned binding, and a closed step. It synchronizes exactly once, converts
particle/gas/environment in that order with `sync=False`, then captures
immutable canonical bytes for the 12 primary arrays and each acquired registry
sidecar. Detached inspection carriers are deliberately non-authoritative:
their gas carrier loses vapor pressure, while the primary payload restores it
exactly.

`checkpoint()` returns a new equivalent ACTIVE snapshot. `finalize()` performs
that operation once, changes the session from ACTIVE to FINALIZED only after a
successful capture, and caches the exact record for O(1) idempotent later
calls. Restart accepts an exact nonterminal checkpoint and explicit matching
`Device`, validates every record and descriptor before setup, uploads fresh CPU
primaries, restores vapor pressure in place, acquires independently materialized
sidecars through the registry APIs, and restores guard counters last. It never
reuses source identities, selects/migrates devices, or offers rollback after a
launched device operation.

## Failure and Atomicity

- Validation and capability failures occur before conversion/allocation where
  possible and do not mutate caller-owned CPU state.
- A failure before any process launch leaves an active session reusable when
  all invariants still hold.
- A failure after launched work may leave resident arrays partially changed;
  mark the session faulted, preserve the original exception, and require
  explicit discard/close. Do not promise rollback or checkpoint uncertain state.
- `close()` is idempotent and never implies restore. `finalize()` is the
  terminal snapshot/finalization operation. `restart_resident_session()` is the
  explicit restoration operation and creates a fresh compatible session. A
   failed finalization remains observable and may be retried only if E7-F6 policy
   classifies it as pre-transfer/recoverable.

## P6 Implementation

Issue #1489 implemented direct-owner failure and disposal semantics in
`particula/execution/gpu_session.py`. Private
`_ResidentOperationOutcome` has only `READ_ONLY` and
`WRITER_MAY_HAVE_LAUNCHED`; the owner supplies that classification explicitly.
After exact-type, identity, active-lifecycle, pinned-session, and open-token
validation, `_handle_failed_resident_operation()` calls
`ResidentStepGuard._abort_step()`. Abort clears/releases the same token from the
guard and registry but deliberately leaves step count and simulated time
unchanged. A read-only result remains `ACTIVE`; a writer-uncertain result faults
the session only after confirming the guard is closed. Direct owners bare-reraise
the caught operational exception, so cleanup errors cannot replace it.

`ResidentSession.close(registry, guard)` and `discard()` are concrete-only
terminal lifecycle operations. Active close validates the pinned binding exactly
once and a closed guard before `ACTIVE -> CLOSED`; faulted close uses a private
identity-only binding check because active-only validation is invalid after a
fault. `CLOSED -> CLOSED` and `FINALIZED -> FINALIZED` do no validation or
runtime work. Close/discard never checkpoint, synchronize, restore, allocate,
migrate, retry, or roll back resident payloads.

## Security & Compliance

No network, credential, or regulated-data behavior is added. Reject untrusted
dynamic registry keys, cross-device arrays, aliases that violate kernel
contracts, malformed checkpoint versions, and incompatible dimensions before
mutation. Do not deserialize arbitrary Python objects or import process types
from checkpoint strings. Resource sizes derive from validated fixed dimensions,
and allocation errors propagate without fallback. Public exports follow E7-F6;
concrete scratch internals remain non-public.
