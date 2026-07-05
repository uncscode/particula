# Change Log

## 2026-07-03

- Drafted feature plan E2-F5 for issue #1172 feature E2-F5.
- Added four issue-sized phases covering compatibility design, environment
  normalization helpers, condensation migration, and coagulation/documentation
  handoff.
- Captured dependencies on E2-F2 environment containers and E2-F3 gas/environment
  boundary decisions.
- Documented scalar compatibility, per-box environment validation, and mismatch
  test requirements.

## 2026-07-03

- Completeness review expanded `success_criteria` with measurable helper,
  condensation, coagulation, and documentation acceptance checks.
- Completeness review clarified the final phase as a development-doc handoff
  step for downstream GPU kernel tracks.

## 2026-07-05

- Updated plan sections after issue #1203 delivered E2-F5-P1.
- Marked P1 complete with the shipped keyword-only `environment` contract,
  early mixed-input and pure explicit-environment guards, and focused kernel
  regression tests.
- Narrowed section language that previously implied environment normalization,
  shape/device validation, or real per-box execution had already shipped.

## 2026-07-05

- Updated plan sections after issue #1204 delivered the shared GPU environment
  normalization helper and runtime consumption path.
- Recorded new private module `particula/gpu/kernels/environment.py` plus
  condensation/coagulation entry-point updates that now accept valid explicit
  environment and direct `(n_boxes,)` Warp-array inputs.
- Added shipped test coverage details for helper validation, explicit
  environment success, hybrid direct inputs, mismatch failures, and
  condensation/coagulation reuse regressions.
