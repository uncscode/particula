# Implementation Tasks

## Execution Layer

- [x] Add concrete-only `ResidentDimensions`, immutable lifecycle vocabulary,
  `ResidentMetadata`, and `ResidentSession` types in
  `particula/execution/gpu_session.py` (P1, issue #1484).
- [x] Validate declared `(n_boxes, n_particles, n_species)`, gas-name count,
  generated Warp container form, fixed primary dtype/shape/shared-device
  metadata, and declared native device without payload access (P1, #1484).
- [x] Add co-located metadata-only construction and rejection coverage in
  `particula/execution/tests/gpu_session_test.py` (P1, #1484).
- [x] Implement all-or-nothing publication setup using the three existing
  `to_warp_*` helpers exactly once each, in particle/gas/environment order (P2,
  issue #1485).
- [x] Preserve ordered CPU gas names outside `WarpGasData` in validated
  `ResidentMetadata` (P2, issue #1485).
- [x] Add concrete-only scheduler-facing `ResidentStepGuard`/
  `ResidentStepToken` entry/exit guards without process ordering,
  synchronization, transfer, allocation, or fallback (P4, issue #1487).
- [x] Track guard-owned step/time metadata only after matching successful token
  completion (P4, issue #1487).
- [x] Validate the exact active registry/session binding before begin and
  completion through `GPUResourceRegistry.validate_pinned_session()`; terminal
  lifecycle states and identity drift reject without guard mutation (P4,
  issue #1487).

## Resource Registry

- [ ] Add `particula/execution/gpu_resources.py` with typed resource manifests
  and fixed-shape allocation helpers.
- [ ] Allocate condensation scratch/thermodynamics, coagulation output/RNG,
  wall-loss RNG, and nucleation scratch/diagnostic/exhaustion resources using
  the canonical `(B, N, S)` dimensions.
- [ ] Validate supplied arrays for exact dtype, shape, contiguity, active device,
  and prohibited aliasing before registration.
- [ ] Return process-specific views while preserving caller/session-owned array
  identity and keeping concrete kernel records out of broad public exports.
- [ ] Provide extension seams for E7-F7 transport buffers and E7-F8 RNG
  metadata without implementing either feature.

## Checkpoint and Finalization

- [x] Add the versioned immutable concrete-only checkpoint model and controller
  in `particula/execution/checkpoint.py` (P5, issue #1488).
- [x] Synchronize once and perform ordered particle/gas/environment
  `from_warp_*` conversions with `sync=False` (P5, #1488).
- [x] Capture detached inspection containers plus canonical primary/sidecar
  bytes, ordered names, dimensions/device, and step/time metadata (P5, #1488).
- [x] Keep checkpoint nonterminal; cache and idempotently finalize the terminal
  checkpoint without duplicate device activity (P5, #1488).
- [x] Restart explicitly into a fresh, same-device-compatible session after full
  schema/descriptor preflight (P5, #1488).
- [x] Define concrete close/discard behavior that never restores implicitly;
  active/faulted sessions close terminally and closed/finalized calls are
  write-free idempotent cases (P6, issue #1489).
- [x] Classify failed direct operations explicitly, release matching failed-step
  tokens without advancing accounting, retain read-only session reuse, and fault
  uncertain writer sessions without rollback (P6, issue #1489).

## Tooling / Tests

- [x] Add P2 setup preflight, conversion-spy, identity, and failure coverage to
  `particula/execution/tests/gpu_session_test.py` (issue #1485). Availability
   rejection remains pending the E7-F6 public seam.
- [x] Add P4 session-guard, closed-step-gate, no-transfer/no-allocation, and
  pinned-session validation regression coverage in
  `particula/execution/tests/gpu_session_test.py` and
  `particula/execution/tests/gpu_resources_test.py` (issue #1487).
- [ ] Add `particula/execution/tests/gpu_resources_test.py` for the complete
  sidecar shape/dtype/device/alias matrix.
- [x] Add `particula/execution/tests/checkpoint_test.py` for immutable payload,
  restore, resource-family restart, malformed-descriptor, and terminal behavior
   coverage (P5, #1488).
- [x] Add P6 failure-outcome, abort, fault/close/discard, original-error, and
  no-runtime-work regression coverage in
  `particula/execution/tests/gpu_session_test.py` (issue #1489).
- [ ] Reuse canonical fixtures from
  `particula/gpu/tests/process_sequence_test.py` rather than inventing divergent
  physics fixtures.
- [ ] Extend `particula/gpu/tests/kernel_exports_test.py` or execution export
  tests to prove concrete scratch types remain non-public.
- [ ] Run Warp CPU routinely; add CUDA rows that skip cleanly when unavailable.
