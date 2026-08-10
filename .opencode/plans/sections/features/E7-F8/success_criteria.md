# Success Criteria

- [x] Every supported stochastic process/logical-box pair has a unique,
  versioned stream descriptor and deterministic initial state.
- [x] Resident Brownian coagulation retains one P1-derived `(n_boxes,)`
  `wp.uint32` sidecar by identity, initializes it once on acquisition, and always
  dispatches with `initialize_rng=False`.
- [x] Brownian coagulation and wall loss use separate resident `(n_boxes,)`
  `wp.uint32` state arrays and preserve their identities across normal steps.
- [x] Normal scheduler steps never implicitly allocate, initialize, reset,
  synchronize, restore, or read back persistent streams.
- [x] Repeating a root seed does not reset state; only an explicit valid reset
  operation does so for published streams or selected registered lanes.
- [x] Inspection returns only frozen host metadata and explicit reset requires an
  exact ACTIVE session/registry/closed-guard binding before selector handling or
  writer work.
- [x] Adding, removing, disabling/no-work, or physically reordering unrelated
  boxes leaves the covered enabled logical box's same-backend stream and outputs
  unchanged for resident Brownian coagulation and selected neutral wall loss.
- [x] Disabled, prelaunch-skipped, zero-time, and valid no-work wall-loss boxes
  do not advance their supplied stream lanes.
- [x] A compatible checkpoint/restart split exactly matches uninterrupted RNG
  state and stochastic outputs on Warp CPU for covered configurations.
- [x] Schema-v3 optionally captures canonical published-stream metadata and
  current words after one synchronization, restores fresh exact-device state
  without reseeding, and preserves schema-v1/v2 compatibility.
- [x] Malformed v3 records reject before setup/allocation; capture/readback and
  fresh-restart failures expose no partial checkpoint/session and retain the
  established source/fresh-resource lifecycle boundaries.
- [x] Malformed P1 IDs, seeds, arrays, dimensions, devices, or process manifests
  fail before caller-buffer mutation.
- [x] Post-launch uncertainty faults the session and is never advertised as an
  atomic or checkpointable state.
- [x] Existing direct wall-loss kernel API/physics, stochastic validation,
  narrow exports, and issue #1451 exclusions remain intact.

## Metrics

| Metric | Baseline | Target | Source |
|---|---:|---:|---|
| Implicit reseeds during normal resident steps | Caller-dependent | 0 | Initialization spies |
| Normal-step bulk transfers/synchronizations | No integrated contract | 0 | Conversion/sync spies |
| Enabled streams perturbed by unrelated box add/disable/reorder | Not guaranteed | 0 | RNG invariance matrix |
| Same-backend restart mismatches | Not guaranteed | 0 | Checkpoint split-run regressions |
| Supported stochastic process namespaces | Ad hoc sidecars | 2 (coagulation, wall loss) | Registry tests |
| Changed-module coverage | Repository threshold | At least 80% | pytest-cov |
