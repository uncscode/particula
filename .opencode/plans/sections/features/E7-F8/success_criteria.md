# Success Criteria

- [x] Every supported stochastic process/logical-box pair has a unique,
  versioned stream descriptor and deterministic initial state.
- [x] Resident Brownian coagulation retains one P1-derived `(n_boxes,)`
  `wp.uint32` sidecar by identity, initializes it once on acquisition, and always
  dispatches with `initialize_rng=False`.
- [ ] Brownian coagulation and wall loss use separate resident `(n_boxes,)`
  `wp.uint32` state arrays and preserve their identities across normal steps.
- [ ] Normal scheduler steps never implicitly allocate, initialize, reset,
  synchronize, restore, or read back persistent streams.
- [ ] Repeating a root seed does not reset state; only an explicit valid reset
  operation does so.
- [ ] Adding, removing, disabling, or reordering unrelated boxes leaves each
  enabled logical box's same-backend stream and outputs unchanged.
- [ ] Disabled process/box work does not advance its stream.
- [ ] A compatible checkpoint/restart split exactly matches uninterrupted RNG
  state and stochastic outputs on Warp CPU for covered configurations.
- [x] Checkpoint and finalize fail closed before payload work after resident
  coagulation-sidecar publication; no RNG persistence/restart continuation is
  advertised.
- [x] Malformed P1 IDs, seeds, arrays, dimensions, devices, or process manifests
  fail before caller-buffer mutation.
- [ ] Post-launch uncertainty faults the session and is never advertised as an
  atomic or checkpointable state.
- [ ] Existing direct kernel APIs, stochastic validation, conservation tests,
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
