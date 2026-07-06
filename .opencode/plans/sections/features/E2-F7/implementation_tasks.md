# E2-F7 Implementation Tasks

## E2-F7-P1 Tasks

- Implemented in issue #1213:
  - `CondensationStiffnessCase` and
    `CondensationStiffnessClassification` were added directly to
    `particula/gpu/kernels/tests/condensation_test.py`; no helper-file split was
    needed.
  - Fixed-shape named stress cases now cover `nanometer`,
    `accumulation_mode`, and `droplet_like` regimes with explicit metadata.
  - Reusable helpers now cover metadata validation, non-negativity,
    finite-value checks, fractional mass change, zero-mass stability, and
    stable/unstable classification for the current particle-only path.
  - Fast tests now cover scalar and direct `(n_boxes,)` environment inputs,
    threshold-boundary behavior, dtype/shape metadata failures, particle-only
    caveat handling, and pre-launch validation short-circuit behavior.

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
