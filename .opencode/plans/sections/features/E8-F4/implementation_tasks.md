# Implementation Tasks

## Runtime / Backend

- [ ] Create `particula/execution/graph_capture.py` with exact concrete-only
  capture adapter and captured-plan types.
- [ ] Resolve Warp capture APIs lazily and distinguish unsupported capability
  from invalid binding and unexpected native failures.
- [ ] Validate E8-F1 READY signature, exact ACTIVE session/registry/closed guard,
  E8-F2 prepared plan, E8-F3 resource set, CUDA device, and duration before
  `capture_begin`.
- [ ] Execute one E8-F2 prepared enqueue between `capture_begin` and
  `capture_end`; publish no handle until both enqueue and capture end succeed.
- [ ] Implement exact-once capture cleanup and preserve operation plus cleanup
  failures without attempting a second `capture_end`.
- [ ] Implement replay preflight and one-token/one-`capture_launch` execution
  with no process dispatch, allocation, validation scans, readback, transfer,
  synchronization, RNG reset, fallback, or recapture.
- [ ] Compare E8-F1 signatures and exact nested resource identities before every
  launch; map each structural mismatch to the canonical invalidation reason.
- [ ] Couple launch failure to resident and capture faulting through existing
  `_handle_failed_resident_operation` writer semantics.
- [ ] Implement explicit invalidation and idempotent teardown, and require a
  fresh READY record for recapture after retiring the old graph.
- [ ] Keep graph handles absent from checkpoint/finalization payloads and all
  package export tables.

## Tooling / Tests

- [ ] Add `particula/execution/tests/graph_capture_test.py` with fake-runtime
  call traces, exact-type checks, capability errors, cleanup, compatibility,
  lifecycle, token, and failure tests.
- [ ] Add forbidden-operation spies proving prepared capture/replay invokes no
  allocator, host readback, validation scan, synchronization, or process
  scheduler loop.
- [ ] Add CUDA-gated capture and repeated-replay smoke tests that skip only for
  explicit unavailable device/API conditions and never fall back to CPU.
- [ ] Add `captured_full_loop_test.py` with identical CPU, uncaptured Warp, and
  captured CUDA fixtures over multiple timesteps.
- [ ] Compare every meaningful particle, gas, environment, diagnostics, and RNG
  outcome separately; retain tight conservation checks and aggregate stochastic
  bounds.
- [ ] Add structural drift cases for dimensions, devices, arrays, prepared plan,
  graph/schedule, process configurations, communication maps, diagnostics, RNG
  sidecars, resident finalize/fault/close, teardown, and restart.
- [ ] Extend export tests to prove all graph lifecycle names remain concrete-only.
- [ ] Run focused execution assertions without coverage, then the untargeted
  repository coverage runner and strict documentation build.
