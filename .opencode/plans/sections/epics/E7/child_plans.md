# Child Plans

The following order preserves issue #1451's nine authoritative feature tracks.
Execution order is governed by the dependency column and dependency map.

### Feature Tracks

| ID | Feature Plan | Size | Status | Dependencies |
|----|--------------|------|--------|--------------|
| E7-F1 | Backend-selection and execution-context API | M | Shipped | None |
| E7-F2 | Backend-selected condensation with explicit semantics and parity | M | Shipped | E7-F1, E7-F6 |
| E7-F3 | Backend-selected Brownian coagulation with persistent RNG | M | Shipped | E7-F1, E7-F6 |
| E7-F4 | GPU-resident session state, reusable sidecars, and checkpoints | L | Shipped | E7-F1, E7-F6 |
| E7-F5 | Deterministic full-process scheduling with environment and gas updates | L | Shipped | E7-F2, E7-F3, E7-F4 |
| E7-F6 | CPU fallback, capability errors, exports, and API-stability policy | M | Shipped | E7-F1 |
| E7-F7 | Prescribed multi-box communication, mixing, and volume evolution | L | Shipped | E7-F4, E7-F5, E7-F6 |
| E7-F8 | Persistent per-box RNG streams and restart semantics | M | Shipped | E7-F3, E7-F4, E7-F5 |
| E7-F9 | Diagnostics, full-loop regressions, documentation, and closeout evidence | L | Shipped | E7-F1 through E7-F8 |

### Maintenance Tracks

Maintenance Tracks: none

All nine feature tracks are shipped. E7-F9 closed the epic after the scheduler,
communication, diagnostics, checkpoint/restart, and persistent RNG contracts
from E7-F1 through E7-F8 were complete.
