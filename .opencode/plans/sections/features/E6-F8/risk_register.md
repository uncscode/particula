# Risk Register

| Risk | Likelihood | Impact | Mitigation | Owner |
|------|------------|--------|------------|-------|
| GPU equations drift from the bounded E6-F7 CPU contract | Medium | High | Freeze equations/units/domains in P1; compare every deterministic diagnostic to an independent float64 oracle. | Physics owner |
| Per-species admission is replaced by aggregate limiting and creates/skews inventory | Medium | High | Use one shared per-box admission factor over participating species; assert per-species conservation and nonnegative gas. | Physics owner |
| Full slots silently truncate represented source demand | Medium | High | Require complete E6-F6 planning, finalize scale explicitly, reject final-domain residual before commit, and test every policy combination. | Particle owner |
| Slot predicates or ordering diverge from E6-F5 | Medium | High | Call E6-F5 services directly; test strict truth table, ascending indices, exact counts, and `-1` tails. | GPU owner |
| Multi-box execution partially commits before a later failure | Low | High | Separate read-only preflight/planning from commit; snapshot all caller-owned arrays on every rejection path. | GPU owner |
| Scratch aliasing or wrong device corrupts caller state | Medium | High | Validate fixed shapes, dtypes, devices, counts, and disallowed overlap before any output clear or launch. | GPU owner |
| Convenience work introduces hidden host transfer or fallback | Medium | High | Keep a direct-kernel-only API; prohibit conversion helpers/`.numpy()` physics in the step and add source/API regression checks. | API owner |
| CUDA numerical behavior exceeds Warp CPU tolerances | Medium | Medium | Make Warp CPU required, CUDA optional, record per-fixture tolerances with justification, and avoid exact backend-identity claims. | Test owner |
| Scope expands into scheduling or unsupported nucleation physics | Medium | Medium | Preserve Epic G and E6-F7 boundaries in API, docs, tests, and review checklists. | E6 owner |
