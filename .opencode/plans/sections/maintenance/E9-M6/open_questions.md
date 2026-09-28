# Open Questions and Review Decisions

Authority: [E9 review decisions](../../epics/E9/appendix.md#review-decisions-2026-09-27).

- [x] **Q1 entry packet:** require actual completed-and-validated M1–M5 revisions,
  D1–D4 implementation evidence and Kyle's acceptance. A draft or checked
  planning decision is not removal authorization.
- [x] **Q2 useful conversions:** preserve native scientific binning, radius/mass
  and representation-conversion capabilities needed by supported consumers;
  delete facade machinery only after a traced native successor is verified.
- [x] **Q3 example policy:** retain all supported M5 content; no historical-only
  exceptions approved by this review. Freeze the completed path ledger before
  deletion and rerun affected published workloads afterwards.
- [x] **Q4 readiness owner/location:** Kyle/Gorkowski signs final readiness in
  docs/Features/Roadmap/data-native-v030-closeout.md with revision-specific PR
  and artifact links. Release publication remains a separate action.
- [x] **Q5 execution policy:** require a declared installed-Warp CPU environment
  and all supported CPU/notebook dependencies. Record execution operator/device
  availability before P5/P6. Optional CUDA may skip cleanly; unavailable required
  runs block readiness. No environment is represented as provisioned here.

## Evidence still required in M6

Close the legacy/seam inventory, validate distribution metadata propagation and
CPU/GPU non-unit-volume scientific corrections after removal, and publish actual
test/lint/docs/example results. Old ambiguous checkpoint handling must be
documented and tested. No API removal or release readiness is claimed complete.
