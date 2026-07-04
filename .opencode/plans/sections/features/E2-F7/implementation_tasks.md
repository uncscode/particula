# E2-F7 Implementation Tasks

## E2-F7-P1 Tasks

- Define `CondensationStiffnessCase` and metric helpers directly in
  `particula/gpu/kernels/tests/condensation_test.py` unless they push the file
  past a reviewable size; only then split them into
  `particula/gpu/kernels/tests/condensation_stiffness_helpers.py`.
- Add fixed-shape stress cases for nanometer/high-supersaturation,
  accumulation-mode, and droplet-like regimes with explicit `n_boxes`,
  `n_particles`, and `n_species` dimensions.
- Implement metric helpers for non-negativity, fractional mass change,
  boundedness, and particle-only parity caveats with names that can be reused by
  later CPU/GPU comparisons.
- Add fast tests validating case shapes, dtype assumptions, and metric threshold
  behavior before any timestep sweep logic lands.

## E2-F7-P2 Tasks

- Extend `particula/gpu/kernels/tests/condensation_test.py` to run each stress
  case through `condensation_step_gpu` with an explicitly preallocated
  `mass_transfer` buffer.
- Scan a fixed timestep grid that is recorded in code and mirrored into
  `docs/Features/Roadmap/condensation-stiffness-study.md`.
- Capture the resulting stable/unstable bounds in a compact markdown table
  rather than free-form prose.
- Document in both tests and the report that the current GPU condensation path
  updates particles only and does not mutate gas concentration.

## E2-F7-P3 Tasks

- Prototype fixed-count sub-step evaluation behind a narrowly scoped helper so
  production API churn stays under one file.
- Compare fixed sub-step counts against the explicit stiffness map already
  published in `docs/Features/Roadmap/condensation-stiffness-study.md`.
- Evaluate a deterministic semi-implicit/asymptotic first-order update using
  fixed shapes and preallocated scratch buffers only.
- Compare candidates against CPU reference calculations and document rejected
  alternatives, including random staggered theta modes, in the study report.
- Evaluate the work needed for gas-coupled production condensation integration,
  including gas-depletion conservation checks; split that implementation into a
  follow-up feature if it cannot remain issue-sized here.
- Record graph-capture and autodiff compatibility for each candidate with clear
  pass/fail notes.

## E2-F7-P4 Tasks

- Update `docs/Features/Roadmap/data-oriented-gpu.md` with the stiffness map,
  recommended integration foundation, and follow-up implementation gates.
- Update `docs/Features/Roadmap/warp-autodiff-limitations.md` with any new
  clamp, gradient, or fixed-loop guidance.
- Treat `docs/Features/Roadmap/condensation-stiffness-study.md` as the detailed
  evidence file and keep roadmap pages to summary links plus decisions.
- Cross-reference E2-F2 and E2-F6 dependencies in the recommendation.
- Ensure all new helper functions from earlier phases have co-located tests and
  Google-style docstrings.
