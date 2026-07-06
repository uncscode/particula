# E2-F7 Change Log

## 2026-07-03 — Initial Draft

- Created first-pass feature plan content for issue #1213 feature E2-F7.
- Added four issue-sized phases covering stress-case definition, explicit GPU
  timestep measurement, fixed-shape integration candidate evaluation, and final
  documentation/recommendation.
- Incorporated context from parent epic E2, sibling feature drafter messages,
  and codebase research on GPU condensation, CPU mass-transfer references,
  staggered prior art, graph-capture constraints, and Warp autodiff limitations.
- Noted challenges: current GPU condensation is particle-only and explicit, while
  environment containers and the precision envelope remain upstream dependencies.

## 2026-07-06 — P1 implementation reflected

- Updated plan sections to reflect issue #1213 implementation for phase
  `E2-F7-P1`.
- Recorded that `particula/gpu/kernels/tests/condensation_test.py` now ships
  `CondensationStiffnessCase`, `CondensationStiffnessClassification`, and the
  reusable baseline helper checks.
- Recorded the delivered named baseline regimes: `nanometer`,
  `accumulation_mode`, and `droplet_like`.
- Recorded new roadmap documentation at
  `docs/Features/Roadmap/condensation-stiffness-study.md` plus the link added to
  `docs/Features/Roadmap/index.md`.
- Clarified that P1 is foundational only: no timestep sweep tables, measured
  bounds, or integrator comparisons were shipped.

## 2026-07-06 — P2 implementation reflected

- Updated plan sections to reflect issue #1214 implementation for phase
  `E2-F7-P2`.
- Recorded that `particula/gpu/kernels/tests/condensation_test.py` now ships a
  recorded timestep grid, a test-local trial record helper, stable/unstable
  assertions per named case, caller-owned `mass_transfer` buffer reuse, and
  unchanged-gas checks for the particle-only path.
- Recorded that single-box cases keep scalar environment inputs while
  `droplet_like` keeps direct Warp `(n_boxes,)` environment-array coverage, with
  optional guarded CUDA contract parity.
- Recorded the measured-results table added to
  `docs/Features/Roadmap/condensation-stiffness-study.md`.
