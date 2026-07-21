# Success Criteria

- [ ] Plan metadata retains mandatory dependencies on E6-F1 through E6-F8.
- [ ] Required Warp CPU coverage executes condensation, coagulation, dilution,
  wall loss, and nucleation on shared fixed-shape device state without an
  intermediate CPU restore.
- [ ] Optional CUDA coverage runs when available and otherwise skips cleanly.
- [ ] Process-specific CPU/Warp parity, statistical bounds, conservation/loss
  budgets, diagnostics, shape/device/dtype, identity, and failure-before-
  mutation contracts pass at recorded tolerances.
- [ ] Persistent RNG is reused without hidden reseeding, and caller-owned
  sidecars retain identity.
- [ ] The runnable example performs explicit initial conversions and explicit
  final checkpoint restores, with no hidden transfer or CPU fallback.
- [ ] Both roadmap documents cross-link E6 and E6-F1 through E6-F9 plus the
  integrated validation and example artifacts.
- [ ] E6 closes only after every child plan is shipped and the parent exit bar
  is verified.
- [ ] Documentation states that backend selection, high-level GPU runnables,
  process scheduling, and resident multi-step simulation remain Epic G scope.
- [ ] Coverage remains at or above 80%, documentation checks pass, and all plan
  files validate.

## Metrics

| Metric | Baseline | Target | Source |
|--------|----------|--------|--------|
| E6 upstream dependencies recorded | 0 in initial E6-F9 metadata | 8 of 8 | `adw plans show E6-F9` |
| Direct processes in integrated Warp sequence | 0 | 5 | `process_sequence_test.py` |
| Intermediate host state restores | No integrated sequence | 0 | Example/test transfer instrumentation |
| Required installed-Warp backend coverage | Separate process tests | Warp CPU passes | Pytest marker results |
| Per-box/species accounting failures | Not measured together | 0 | Integrated assertions |
| Roadmap plan IDs cross-linked | E6 unscheduled | E6 plus E6-F1-F9 | Roadmap link validation |
| Coverage threshold | 80% configured minimum | At least 80% | `pytest --cov=particula` |
