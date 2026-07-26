# Success Criteria

- [x] Plan metadata retains mandatory dependencies on E6-F1 through E6-F8.
- [x] P1 private fixture evidence exists at
  `particula/gpu/tests/process_sequence_test.py` with deterministic fp64
  one-/multi-box fixtures, independent invariants, ownership snapshots, and an
  optional Warp mirror, without production or public API changes.
- [x] Required Warp CPU coverage executes condensation, coagulation, dilution,
  wall loss, and nucleation on shared fixed-shape device state without an
  intermediate CPU restore.
- [x] Optional CUDA coverage runs when available and otherwise skips cleanly.
- [x] Process-specific CPU/Warp parity, statistical bounds, conservation/loss
  budgets, diagnostics, shape/device/dtype, identity, and failure-before-
  mutation contracts pass at recorded tolerances.
- [x] Persistent RNG is reused without hidden reseeding, and caller-owned
  sidecars retain identity.
- [x] The runnable example performs exactly one conversion of each CPU container,
  five direct calls in fixed illustrative order, one synchronization, and one
  final restore of each container, with no hidden transfer or CPU fallback.
- [x] The focused P3 regression suite covers deterministic no-Warp behavior,
  caller-owned sidecar/RNG identities, transfer/order constraints, visible
  failures, and a guarded real Warp CPU route. (#1448)
- [x] Both roadmap documents cross-link E6 and E6-F1 through E6-F9 plus the
  integrated validation and example artifacts.
- [x] Every E6 child plan is shipped, the parent exit bar is verified, and E6
  is Shipped/completed.
- [x] Documentation states that backend selection, high-level GPU runnables,
  process scheduling, and resident multi-step simulation remain Epic G scope.
- [x] `pytest particula/tests/gpu_complete_process_sequence_docs_test.py -q
  -Werror` and plan validation pass for #1449; this documentation-only change
  adds hardware-free regression coverage without changing runtime behavior.

## Metrics

| Metric | Baseline | Target | Source |
|--------|----------|--------|--------|
| E6 upstream dependencies recorded | 0 in initial E6-F9 metadata | 8 of 8 | `adw plans show E6-F9` |
| Direct processes in integrated Warp sequence | 0 | 5 | `process_sequence_test.py` |
| Intermediate host state restores | No integrated sequence | 0 | Example/test transfer instrumentation |
| CPU-to-Warp conversions per example container | No public example | 1 | `gpu_complete_process_sequence_example_test.py` |
| Required installed-Warp backend coverage | Separate process tests | Warp CPU passes | Pytest marker results |
| Per-box/species accounting failures | Not measured together | 0 | Integrated assertions |
| Roadmap plan IDs cross-linked | E6 unscheduled | E6 plus E6-F1-F9 | Roadmap link validation |
| Coverage threshold | 80% configured minimum | At least 80% | `pytest --cov=particula` |
