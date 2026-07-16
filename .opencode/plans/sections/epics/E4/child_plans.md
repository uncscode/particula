# Child Plans

## Feature Tracks

| Order | ID | Feature Plan | Status | Dependencies |
|---:|---|---|---|---|
| 1 | E4-F1 | Thermodynamic configuration and vapor-pressure refresh | Shipped (completed) | E1/E2/E3 foundations |
| 2 | E4-F2 | Activity and effective surface-tension physics | Shipped (completed) | E4-F1 |
| 3 | E4-F3 | Fixed four-substep integration and reusable scratch | Shipped (completed) | E4-F1 |
| 4 | E4-F4 | Latent-heat correction and energy diagnostics | Shipped (completed) | E4-F1, E4-F2, E4-F3 |
| 5 | E4-F5 | Gas coupling, inventory limits, and conservation | Shipped (completed) | E4-F3, E4-F4 |
| 6 | E4-F6 | Device-aware parity and readiness evidence | Shipped (completed) | E4-F1 through E4-F5 |
| 7 | E4-F7 | Support matrix, examples, and troubleshooting | Shipped (completed) | E4-F1 through E4-F6 |

The seven tracks and their order are authoritative. E4-F2 and E4-F3 may run
in parallel after E4-F1; all other dependency edges are gates.

## Status Reconciliation (2026-07-15)

The plan records are the status authority. E4 and all seven child features are
**Shipped** with a **completed** lifecycle. E4-F1-P2 is reconciled to its
Issue #1282 completion record, and every phase in E4-F1 through E4-F7 is
Shipped.

The independent CPU/Warp parity walkthrough omitted from E4-F7 is a deferred
follow-up, not an incomplete E4 phase. Its disposition is tracked in the Epic E
roadmap description with the other explicit Epic D boundaries.

## Maintenance Tracks

Maintenance Tracks: none
