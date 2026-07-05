# Change Log

## 2026-07-03

- Created first-pass feature plan for `E2-F4` / issue `#1172` feature `E2-F4`.
- Added four issue-sized phases covering schema audit, name and partitioning
  semantics, vapor-pressure ownership, and migration documentation.
- Drafted all 13 canonical feature sections using classifier context,
  `E2` epic-drafter context, sibling feature summaries, and codebase research.

## 2026-07-03

- Completeness review expanded `success_criteria` into functional,
  verification, documentation, and done-signal checks.
- Completeness review clarified the final phase as an explicit development-doc
  update gate.

## 2026-07-04

- Updated `E2-F4` plan sections after issue `#1197` landed as
  `E2-F4-P1`.
- Recorded that the shipped implementation was a focused test-only change in
  `particula/gpu/tests/conversion_test.py`.
- Clarified that current contract coverage now locks shapes/dtypes,
  bool↔`int32` `partitioning`, explicit/implicit `vapor_pressure` behavior,
  placeholder names, name-length mismatch failures, and GPU-only
  `vapor_pressure` loss on restore.

## 2026-07-04

- Updated `E2-F4` plan sections after issue `#1198` landed as `E2-F4-P2`.
- Recorded that `particula/gpu/conversion.py` now makes the restore contract
  explicit for caller-supplied names, placeholder fallback, invalid name
  counts, and binary-only `partitioning` restore.
- Recorded that focused conversion tests now cover placeholder fallback,
  `name=None`, empty-name failures, invalid non-binary `partitioning`,
  retry-safe correction paths, vapor-pressure loss on restore, and multi-box
  shape preservation.

## 2026-07-04

- Updated `E2-F4` plan sections after issue `#1199` landed as `E2-F4-P3`.
- Recorded that shipped production edits stayed narrow to
  `particula/gpu/conversion.py` and `particula/gpu/warp_types.py`, with test
  updates in `particula/gpu/tests/conversion_test.py`.
- Recorded the explicit vapor-pressure contract: caller-supplied
  `(n_boxes, n_species)` values transfer as-is, omitted input allocates a
  zero-filled GPU buffer, invalid shapes raise `ValueError`, and CPU restore
  intentionally drops GPU-only `vapor_pressure` unless callers preserve a
  sidecar.
