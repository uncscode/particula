# Open Questions and Review Decisions

Authority: [E9 review decisions](../../epics/E9/appendix.md#review-decisions-2026-09-27).

- [x] **Q1 entry contracts:** D1–D4 fix normalization, radius-based distribution
  metadata, species mapping and aggregate API. M1/M2 implementation revisions
  and Kyle's acceptance must be supplied at entry, not fabricated during planning.
- [x] **Q2 supported matrix:** preserve currently supported strategy/configuration
  combinations, including latent heat, staggered condensation, turbulent-DNS
  coagulation and both neutral/charged wall-loss suites. No automatic Cartesian
  product or newly supported weighted/PDF selector follows from shared metadata.
- [x] **Q3 temporary seams:** consume M2's isolated legacy paths and deletion
  ledger; no duplicate arrays or new permanent public compatibility facade.
- [x] **Q4 normalization disagreement:** fix against D1's physical contract,
  not whichever legacy/native path happens to pass at V=1. D2 authorizes direct
  and prepared GPU corrections for these process families. Record reproductions,
  independent conserved quantities, rate units and intentional result changes.
- [x] **Q5 review:** Kyle/Gorkowski accepts scientific completion and M4 entry.

## Evidence still required in M3

Freeze per-family executable support rows before changing that family. Cover
V=0.25/1/4, radius-PDF quadrature, mapped species, supported unequal weights,
rate/step consistency and concentration-weighted conservation. Unsupported
weight/representation combinations reject before mutation. GPU raw values retain
count interpretation across transfer; unsupported PDFs reject explicitly.
Run direct/prepared regressions and both CPU wall-loss suites. No scientific
correction, compatibility bridge or phase is declared implemented by this review.
