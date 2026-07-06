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

- Implemented in issue #1214:
  - `particula/gpu/kernels/tests/condensation_test.py` now records
    `_RECORDED_TIMESTEP_GRID_BY_CASE` and
    `_RECORDED_STIFFNESS_THRESHOLD_BY_CASE` directly in code.
  - A test-local recorded-grid helper rebuilds fresh deterministic inputs for
    each trial while reusing one caller-owned `mass_transfer` buffer per
    case/device.
  - Fast Warp CPU tests now assert exact timestep count/order, at least one
    stable and one unstable result per named case, buffer overwrite behavior,
    unchanged gas concentration, and scalar-vs-direct-Warp environment-input
    mode coverage.
  - An optional guarded CUDA parity test checks the same recorded-grid result
    contract without making CUDA required.
  - `docs/Features/Roadmap/condensation-stiffness-study.md` now ships a compact
    measured-results table synchronized with the recorded grid.

## E2-F7-P3 Tasks

- Implemented in issue #1215:
  - `particula/gpu/kernels/tests/condensation_test.py` now ships two
    deterministic test-local prototype candidates:
    `fixed_count_substeps_4` and `asymptotic_relaxation`.
  - Candidate coverage stays fixed-shape and reuses caller-owned buffers plus
    reusable scratch/storage inside the test harness; no production helper file
    split was needed.
  - Fast Warp CPU tests now assert repeated-run determinism, finite and
    non-negative particle masses, reusable fixed-shape scratch/buffer behavior,
    CPU-reference agreement within documented tolerances, and explicit-baseline
    error-bound comparisons.
  - `docs/Features/Roadmap/condensation-stiffness-study.md` now records the P3
    evidence, including graph-capture/autodiff notes and the explicit deferred
    gas-coupling boundary.
  - No public API change landed in `condensation_step_gpu(...)`, no private
    production helper landed in `particula/gpu/kernels/condensation.py`, no
    production gas-state update hook shipped, and no integration regression was
    added to
    `particula/integration_tests/condensation_particle_resolved_test.py`.

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
