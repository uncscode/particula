# Milestones and Timeline

Issue #1451 sets no fixed deadline. Dates remain uncommitted until feature
owners estimate child plans; dependency gates, not calendar promises, control
readiness.

| Milestone | Planned Date | Actual Date | Status | Notes |
|-----------|--------------|-------------|--------|-------|
| M1: Selection foundation accepted | TBD | 2026-07-27 | Shipped | E7-F1 capability matrix, typed API, CPU adapter, and tests |
| M2: Explicit boundary policy frozen | TBD | 2026-07-30 | Shipped | E7-F6 errors, fallback, exports, and negative tests |
| M3: Process adapters and resident state ready | TBD | 2026-08-30 | Shipped | E7-F2 through E7-F4 process adapters, resident lifecycle, checkpoints, tests, and documentation |
| M4: Deterministic full loop operational | TBD | 2026-08-30 | Shipped | E7-F5 canonical process and thermodynamic ordering |
| M5: Multi-box and restart contracts ready | TBD | 2026-08-09 | Shipped | E7-F7 communication and E7-F8 persistent RNG/restart semantics |
| M6: Epic G exit bar satisfied | TBD | 2026-08-30 | Shipped | E7-F9 diagnostics, regressions, documentation, and closeout evidence |

Each implementation milestone includes its own unit tests in the same change.
M6 adds integration and documentation validation; it is not a standalone unit
testing phase.
