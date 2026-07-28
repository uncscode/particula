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
- [ ] Implement all-or-nothing setup using the three existing `to_warp_*`
  helpers exactly once each.
- [ ] Preserve ordered gas names and schema/version metadata outside
  `WarpGasData`.
- [ ] Add scheduler-facing step entry/exit guards without process ordering,
  synchronization, transfer, or fallback.
- [ ] Track step/time metadata only after successful step completion.
- [ ] Add active, faulted, finalized, and closed guards with deterministic
  exception types supplied by E7-F6.

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

- [ ] Add a versioned immutable checkpoint model in
  `particula/execution/checkpoint.py`.
- [ ] Implement one explicit `wp.synchronize()` followed by all three
  `from_warp_*` calls with `sync=False`.
- [ ] Capture CPU containers, ordered names, dimensions, device/backend,
  step/time counters, and restart-required mutable resource payloads.
- [ ] Keep `checkpoint()` nonterminal and identity-preserving for the live
  resident session; make `finalize()` terminal and idempotent.
- [ ] Implement explicit restart into a fresh session and reject incompatible
  checkpoint versions or requested dimensions/devices.
- [ ] Define close/discard behavior that never restores implicitly.

## Tooling / Tests

- [ ] Add `particula/execution/tests/gpu_session_test.py` for setup, lifecycle,
  identity, transfer-spy, availability, and failure semantics.
- [ ] Add `particula/execution/tests/gpu_resources_test.py` for the complete
  sidecar shape/dtype/device/alias matrix.
- [ ] Add `particula/execution/tests/checkpoint_test.py` for synchronized
  restore, metadata, restart equivalence, and terminal behavior.
- [ ] Reuse canonical fixtures from
  `particula/gpu/tests/process_sequence_test.py` rather than inventing divergent
  physics fixtures.
- [ ] Extend `particula/gpu/tests/kernel_exports_test.py` or execution export
  tests to prove concrete scratch types remain non-public.
- [ ] Run Warp CPU routinely; add CUDA rows that skip cleanly when unavailable.
