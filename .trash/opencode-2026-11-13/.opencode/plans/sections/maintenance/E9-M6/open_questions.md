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
  runs block readiness. Execution used the repository Python environment with Warp 1.11.1 CPU;
  native CUDA was unavailable. The final fresh-wheel run independently resolved
  Warp 1.19.0 and passed with an isolated cache.

## Implementation evidence and remaining review

The removal inventory, assertion-retention ledgers, complete example execution
and notebook output review are recorded beside this file. Full package coverage,
installed-wheel tests, native/GPU boundaries, lint/type checks and both strict
publication builds pass. Tagged v5 checkpoints remain supported; untagged
records reject before setup/upload. See `implementation.md` and the published
`docs/Features/Roadmap/data-native-v030-closeout.md`.

The implementation evidence is complete. Kyle/Gorkowski's explicit scientific
and release readiness approval remains a separate review; no release is
published by this PR.
