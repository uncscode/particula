# E2-F8 Change Log

## 2026-07-03

- Created first-pass feature plan for issue #1172 feature E2-F8.
- Added three issue-sized phases:
  - E2-F8-P1: Audit CPU dynamics container boundaries and baseline docs.
  - E2-F8-P2: Clarify single-box and box-0 behavior with focused tests.
  - E2-F8-P3: Improve unsupported multi-box errors and user documentation.
- Incorporated parent epic E2 context and sibling feature dependency on E2-F1.
- Incorporated codebase research findings for `ParticleData`, `GasData`, CPU
  condensation single-box validation, CPU coagulation box-0 adapters, and
  relevant docs/tests.

## 2026-07-03

- Completeness review clarified the final phase as a user-facing and
  development-doc support-boundary update gate.

## 2026-07-07

- Updated the feature plan after issue #1218 (E2-F8-P1) landed.
- Marked P1 complete as an audit-only baseline phase.
- Recorded that the implementation added three focused regressions only:
  condensation public multi-box rejection, coagulation helper-backed box-0
  reads, and particle-resolved coagulation box-0-only mutation.
- Corrected plan sections that previously implied CPU coagulation already
  rejects multi-box `ParticleData` or that user-facing docs changed in P1.

## 2026-07-07

- Updated the feature plan after issue #1219 (E2-F8-P2) landed.
- Marked P2 complete and recorded the shipped production change in
  `particula/dynamics/coagulation/coagulation_strategy/coagulation_strategy_abc.py`.
- Replaced stale box-0 fallback wording across feature sections with the shipped
  explicit single-box CPU coagulation rejection contract.
- Recorded that condensation coverage expanded through representative public
  multi-box rejection tests while user-facing docs still remain deferred to P3.
