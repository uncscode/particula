# Change Log

## 2026-07-08

- Created first-pass feature plan for `E3-F4`.
- Added four issue-sized phases covering import-path decision, export tests,
  direct-kernel quick-start, and troubleshooting/validation docs.
- Incorporated parent epic `E3` context and sibling dependency on `E3-F1`.
- Captured codebase research findings for current GPU exports, kernel modules,
  transfer helpers, examples, and test targets.

## 2026-07-08 — Completeness Review

- Replaced generic success bullets with measurable pass/fail criteria,
  evidence metrics, and a definition of done tied to import stability,
  explicit transfer boundaries, and runnable quick-start validation.

## 2026-07-10

- Updated plan sections to reflect shipped phase `E3-F4-P1` implementation.
- Recorded the finalized public import path
  `particula.gpu.kernels.{condensation_step_gpu, coagulation_step_gpu}`.
- Noted that `particula.gpu` remains intentionally non-reexporting and that
  `particula.gpu.kernels.__all__` is narrowed to the two supported step
  functions.
- Added plan references to focused regression coverage in
  `particula/gpu/tests/kernel_exports_test.py` and the roadmap wording update.
