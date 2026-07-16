# Risk Register

| Risk | Likelihood | Impact | Mitigation | Owner |
|------|------------|--------|------------|-------|
| Tests reuse production Warp formulas and create a circular oracle | Medium | High | Build expected values from public CPU APIs or direct NumPy equations; review imports and perturb one expected term in test-support unit tests | E5-F7-P1 owner |
| Stochastic bounds are flaky or so broad they hide bias | Medium | High | Predeclare sample count and analytic/empirical variance rule; use fixed seed sets, report observed margin, and keep deterministic invariants separate | E5-F7-P3 owner |
| Tiny-particle mass errors are masked by large-particle totals | Medium | High | Assert per-box/per-species balances with relative tolerance plus physical absolute floor; include mixed NPF/droplet fixtures | E5-F7-P2 owner |
| Aggregate charge appears conserved while donor charge is not cleared | Low | High | Assert recipient transfer, donor clearing, inactive preservation, and total charge separately | E5-F7-P2 owner |
| A supported E5-F6 combination is omitted as sibling plans evolve | Medium | High | Derive/freeze the canonical approved-row table from the shipped configuration and fail on uncovered executable rows | E5-F7-P1 owner |
| CUDA-only behavior is mistaken for the release baseline | Low | Medium | Make Warp CPU mandatory in device enumeration and documentation; label CUDA additive and cleanly skippable | E5-F7-P3 owner |
| Expanded matrix makes normal tests too slow | Medium | Medium | Keep deterministic fixtures small, bound stochastic sample counts analytically, avoid performance markers, and expose focused marker commands | E5-F7-P3 owner |
| Validation work changes production behavior to make tests pass | Low | High | Treat E5-F3-F6 APIs/physics as upstream; route discovered defects to owning feature and limit E5-F7 production changes to verified bug fixes with co-located tests | Feature lead |
| Warp absence causes collection errors rather than supported skips | Low | Medium | Reuse existing import guards and test collection patterns before adding new modules | E5-F7-P1 owner |
