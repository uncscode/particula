# Milestones and Timeline

Issue #1451 sets no fixed deadline. Dates remain uncommitted until feature
owners estimate child plans; dependency gates, not calendar promises, control
readiness.

| Milestone | Planned Date | Actual Date | Status | Notes |
|-----------|--------------|-------------|--------|-------|
| M1: Selection foundation accepted | TBD | - | Not Started | E7-F1 capability matrix, typed API, CPU adapter, and tests |
| M2: Explicit boundary policy frozen | TBD | - | Not Started | E7-F6 errors, fallback, exports, and negative tests |
| M3: Process adapters and resident state ready | TBD | 2026-07-29 (partial) | In Progress | E7-F4 P1--P7 resident lifecycle/checkpoint foundation, lazy example, regressions, and strict docs validation shipped; E7-F2/F3 remain |
| M4: Deterministic full loop operational | TBD | - | Not Started | E7-F5 canonical process and thermodynamic ordering |
| M5: Multi-box and restart contracts ready | TBD | - | Not Started | E7-F7 and E7-F8; parallel after M4 |
| M6: Epic G exit bar satisfied | TBD | - | Not Started | E7-F9 regressions, example, docs, and closeout evidence |

Each implementation milestone includes its own unit tests in the same change.
M6 adds integration and documentation validation; it is not a standalone unit
testing phase.
