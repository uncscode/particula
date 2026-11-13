# Implementation Tasks

## Runtime / Backend

- [x] Add the direct-import-only P1 qualification controller and exact immutable
  native-callable/qualification records in
  `particula/execution/graph_capture.py` (issue #1567).
- [x] Resolve runtime, device, and capture-API capability lazily through an
  injected adapter; distinguish unqualified capability, malformed adapter
  metadata, and adapter exceptions without invoking native callables.
- [x] Validate the exact E8-F1 READY signature, ACTIVE session, pinned registry,
  closed guard, E8-F2 prepared simulation, E8-F3 resource set, device, and
  duration before adapter lookup; preserve READY on every result.
- [x] Execute one E8-F2 prepared enqueue between `capture_begin` and
  `capture_end`; publish `CapturedResidentGraph` only after post-end
  revalidation and the CAPTURED transition succeed.
- [x] Implement exact-once capture cleanup and preserve operation plus cleanup
  failures without attempting a second `capture_end`; release a successful-end
  opaque handle by identity after post-end rejection.
- [x] Implement replay preflight and one-token/one-`capture_launch` execution
  with no process dispatch, allocation, validation scans, readback, transfer,
  synchronization, RNG reset, fallback, or recapture.
- [x] Compare E8-F1 signatures and exact nested resource identities before every
  launch; map each structural mismatch to the canonical invalidation reason.
- [x] Couple launch failure to resident and capture faulting through existing
  `_handle_failed_resident_operation` writer semantics.
- [x] Centralize explicit invalidation and idempotent teardown in the graph
  owner; stale records unregister before one release and recapture requires a
  fresh READY record (issue #1570).
- [x] Keep graph handles absent from checkpoint/finalization payloads and all
  package export tables.

## Tooling / Tests

- [x] Extend `particula/execution/tests/graph_capture_test.py` with host-only
  adapter-order, exact identity, lifecycle, malformed metadata, no-handle/
  no-cleanup, and no-native-call tests for P1.
- [x] Add forbidden-operation spies proving prepared capture invokes no normal
  scheduler/token work, allocator, host readback, validation scan,
  synchronization, transfer, or resource work in the native window.
- [x] Add a CUDA-gated native capture smoke test with twelve device no-ops that
  skips only for unavailable Warp/CUDA/capture APIs and never falls back to CPU.
- [x] Add test-only P5 coverage in `captured_full_loop_test.py` for zero-duration
  preservation, CUDA availability, lifecycle, stale rejection, cleanup, and
  forbidden host work; add matching documentation-contract assertions.
- [x] Add `captured_full_loop_test.py` with independent NumPy, uncaptured Warp,
  and captured CUDA fixtures over multiple genuine native timesteps (#1571).
- [x] Compare every meaningful particle, gas, environment, diagnostics, and RNG
  outcome separately; retain tight conservation checks and aggregate stochastic
  bounds.
- [ ] Add structural drift cases for dimensions, devices, arrays, prepared plan,
  graph/schedule, process configurations, communication maps, diagnostics, RNG
  sidecars, resident finalize/fault/close, teardown, and restart.
- [x] Extend `particula/execution/tests/exports_test.py` to prove P1 and P2
   graph-capture names remain concrete-only and are denied from public exports.
- [x] Extend `particula/execution/tests/graph_capture_test.py` for P3 authentic
  opaque-handle provenance, exact replay validation, one-token/one-launch
   behavior, and writer-capable launch/completion failures (issue #1569).
- [x] Add P4 lifecycle regressions across `graph_capture_test.py`,
  `gpu_session_test.py`, `checkpoint_test.py`, and `gpu_resources_test.py` for
  exact-once release, stale provenance, exact notification context, terminals,
  recapture, and guarded stream initialization (issue #1570).
- [ ] Run focused execution assertions without coverage, then the untargeted
  repository coverage runner and strict documentation build.
