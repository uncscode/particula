# Child Plans

## Feature Tracks

| Order | ID | Feature Plan | Status | Dependencies |
|---:|---|---|---|---|
| 1 | E4-F1 | Thermodynamic configuration and vapor-pressure refresh | Draft (active) | E1/E2/E3 foundations |
| 2 | E4-F2 | Activity and effective surface-tension physics | Shipped (completed) | E4-F1 |
| 3 | E4-F3 | Fixed four-substep integration and reusable scratch | Shipped (completed) | E4-F1 |
| 4 | E4-F4 | Latent-heat correction and energy diagnostics | Shipped (completed) | E4-F1, E4-F2, E4-F3 |
| 5 | E4-F5 | Gas coupling, inventory limits, and conservation | Draft (active) | E4-F3, E4-F4 |
| 6 | E4-F6 | Device-aware parity and readiness evidence | Draft (active) | E4-F1 through E4-F5 |
| 7 | E4-F7 | Support matrix, examples, and troubleshooting | Draft (active) | E4-F1 through E4-F6 |

The seven tracks and their order are authoritative. E4-F2 and E4-F3 may run
in parallel after E4-F1; all other dependency edges are gates.

## Status Reconciliation (2026-07-14)

The plan records are the status authority. E4 itself remains **Draft** with an
**active** lifecycle. E4-F2 and E4-F3 are **Shipped** with a **completed**
lifecycle. E4-F1 remains **Draft** with an **active** lifecycle and has
shipped P1 and P3--P5 while P2 is Not Started. E4-F4 is **Shipped** with a
**completed** lifecycle; E4-F4-P4 (issue #1300) is Shipped. E4-F5 through
E4-F7 remain Draft/active with all recorded phases Not Started.

Issue #1300 validation and roadmap wording support the shipped E4-F4-P4
record. E4-F1 remains unshipped until its feature record is updated.

## Maintenance Tracks

Maintenance Tracks: none
