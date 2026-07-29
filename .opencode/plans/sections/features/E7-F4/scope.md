# Scope

E7-F4 creates the resident-state and transfer lifecycle used by later E7
schedulers. It packages existing fixed-shape Warp containers, immutable session
metadata, process resources, and explicit restore boundaries without changing
the behavior or signatures of shipped direct process kernels.

## In Scope

- A typed GPU resident-session model under the E7-F1 execution layer.
- Explicit setup from validated CPU `ParticleData`, `GasData`, and
  `EnvironmentData` using one conversion call per container.
- Preservation of ordered gas names and other CPU-only metadata outside Warp
  structs.
- A fixed-shape sidecar registry for condensation scratch and thermodynamics,
  coagulation output/RNG arrays, wall-loss RNG arrays, and nucleation planning,
  diagnostics, and exhaustion resources.
- Device, dtype, shape, ownership, identity, and lifecycle validation before a
  session becomes active.
- Normal-step lifecycle hooks that neither synchronize nor restore bulk state.
- Explicit checkpoint and finalization operations with one synchronization
  followed by `sync=False` particle, gas, and environment restores.
- Restart setup from a checkpoint, including opaque mutable sidecar/RNG payloads
  needed by downstream process adapters.
- Clear preflight, launched-work failure, faulted-session, close, and
  idempotency semantics.
- Warp CPU tests, no-intermediate-transfer spies, identity/shape tests,
  checkpoint/restart tests, and optional CUDA rows.

## Implemented in P1 (Issue #1484)

- `particula/execution/gpu_session.py` defines concrete-only frozen
  `ResidentDimensions`, `ResidentMetadata`, and `ResidentSession`, plus the
  four-value `ResidentLifecycle` vocabulary.
- Resident construction retains the three supplied generated Warp containers,
  dimensions, metadata, and lifecycle value by identity after constant-cost
  dtype, shape, and shared-device metadata checks.
- Warp and generated container types are imported only during resident
  validation. P1 performs no payload access, transfer, synchronization, kernel
  launch, allocation, conversion, fallback, device migration, export change, or
  lifecycle operation.
- Co-located tests are in `particula/execution/tests/gpu_session_test.py`.

## Implemented in P2 (Issue #1485)

- Concrete-only `setup_resident_session()` locally validates an exact Warp
  `Device`, CPU carrier types, rank-3 particle masses, and required `(B, N, S)`
  cross-container shapes before importing conversion helpers.
- It snapshots ordered CPU gas names as an exact-string tuple, converts particle,
  gas, and environment carriers exactly once in that order using unchanged
  `device.native`, then constructs the sole published `ACTIVE` session.
- Tests cover import-free local failures, conversion ordering/failures, final
  session-schema rejection, identity retention, and input immutability.
- E7-F6 device availability is not probed or emulated: selected native-device
  availability remains an upstream precondition.

## Implemented in P4 (Issue #1487)

- `particula/execution/gpu_session.py` defines direct-import-only
  `ResidentStepGuard` and frozen, identity-equality `ResidentStepToken`.
  The guard retains one exact session/registry pair, permits one outstanding
  token, and updates completed-step/time bookkeeping only after matching token
  completion.
- `GPUResourceRegistry.validate_pinned_session()` is the metadata-only binding
  seam: it first requires the exact retained session and then reuses the
  registry's active signature/schema validation. It does not acquire sidecars,
  allocate, inspect payloads, or mutate registry state.
- `assert_step_closed()` rejects future guarded checkpoint, finalize, close,
  conversion/restore, resize/rebind, and fault entries while a token is open.
  It does not globally intercept raw low-level helpers. P4 adds no adapter
  ordering/execution, synchronization, conversion, restore, resizing, or CPU
  fallback.

## Out of Scope

- Process ordering, environment evolution, or derived thermodynamic refresh
  scheduling (E7-F5).
- Condensation or Brownian coagulation adapter physics (E7-F2 and E7-F3).
- Per-box RNG stream identity, split/reset policy, and stochastic invariance
  across box reorderings (E7-F8); E7-F4 only stores and restores opaque state.
- Multi-box transport, communication, mixing, or volume evolution (E7-F7).
- Silent CPU fallback, hidden transfers, automatic device migration, or runtime
  retry on another backend.
- Disk/file serialization formats, remote checkpoints, or incremental/delta
  checkpoints.
- Dynamic resizing, compaction, multi-GPU/distributed execution, graph capture,
  profiling, performance claims, autodiff, or unsupported physics expansion.
