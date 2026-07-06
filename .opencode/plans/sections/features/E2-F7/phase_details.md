# E2-F7 Phase Details

- [x] **E2-F7-P1:** Define condensation stiffness stress cases and metrics with tests
  - Issue: #1213 | Size: S | Status: Implemented
  - Goal: Create a small, reproducible catalog of condensation regimes and
    stability metrics that can be reused by GPU and CPU comparisons.
  - Files delivered: `particula/gpu/kernels/tests/condensation_test.py`,
    `docs/Features/Roadmap/condensation-stiffness-study.md`, and
    `docs/Features/Roadmap/index.md`.
  - Implementation notes: P1 added `CondensationStiffnessCase`,
    `CondensationStiffnessClassification`, named `nanometer`,
    `accumulation_mode`, and `droplet_like` baseline regimes, plus helper checks
    for metadata validation, finite/non-negative results, fractional mass
    change, zero-mass stability, threshold-boundary semantics, particle-only
    caveat handling, and pre-launch validation short-circuit behavior.
  - Tests: Unit and Warp CPU tests for stress-case construction, scalar/direct
    environment-input coverage, metric calculation, metadata failures, and
    stable/unstable classification.

- [x] **E2-F7-P2:** Measure stable explicit timestep for current GPU condensation path
  - Issue: #1214 | Size: S | Status: Implemented
  - Goal: Run the stress catalog against `condensation_step_gpu` with fixed
    shapes and preallocated buffers, producing a measured stability map for the
    current explicit GPU implementation.
  - Files delivered: `particula/gpu/kernels/tests/condensation_test.py` and
    `docs/Features/Roadmap/condensation-stiffness-study.md`.
  - Implementation notes: P2 added `_RECORDED_TIMESTEP_GRID_BY_CASE`,
    `_RECORDED_STIFFNESS_THRESHOLD_BY_CASE`, and
    `CondensationStiffnessTrialRecord`, then used a test-local sweep helper to
    rebuild fresh deterministic inputs per trial while reusing one
    caller-owned `mass_transfer` buffer per case/device. The shipped tests
    assert exact timestep order, at least one stable and one unstable result per
    case, unchanged gas concentration for the particle-only path, scalar inputs
    for single-box cases, direct Warp `(n_boxes,)` arrays for `droplet_like`,
    and optional guarded CUDA contract parity.
  - Tests: Fast Warp CPU recorded-grid tests plus a guarded CUDA parity check
    when available.

- [ ] **E2-F7-P3:** Evaluate fixed-shape sub-stepping and semi-implicit candidates
  - Issue: TBD | Size: S | Status: Not Started
  - Goal: Compare graph-capture-compatible options against the explicit map and
    identify the smallest safe foundation for future implementation while also
    proving whether the chosen production path can update gas state in the same
    step.
  - Files: a narrowly scoped helper in `particula/gpu/kernels/condensation.py`
    or a test-only prototype under `particula/gpu/kernels/tests/`,
    `particula/integration_tests/condensation_particle_resolved_test.py` for a
    bounded gas-coupled regression if production hooks land here, and
    `docs/Features/Roadmap/condensation-stiffness-study.md` analysis notes.
  - Tests: Unit tests for deterministic fixed-count sub-step behavior,
    semi-implicit/asymptotic candidate invariants, no dynamic allocation in
    captured-step candidates where practical, and gas-coupled particle-plus-gas
    conservation checks if this phase lands the production hook.

- [ ] **E2-F7-P4:** Publish integration recommendation and development documentation
  - Issue: TBD | Size: XS | Status: Not Started
  - Goal: Record the recommended integration foundation, rejected alternatives,
    graph-capture/autodiff constraints, and either the shipped gas-coupled
    production path or the exact follow-up split boundary required to land it.
  - Files: `docs/Features/Roadmap/data-oriented-gpu.md`,
    `docs/Features/Roadmap/warp-autodiff-limitations.md`,
    `docs/Features/Roadmap/condensation-stiffness-study.md`, and plan sections
    as needed.
  - Tests: Documentation link checks, reruns of the focused condensation tests,
    and reruns of the bounded gas-coupled regression when P3 or an earlier phase
    ships the production hook.

## Phase Ordering Notes

- P1 may begin from the current scalar-compatible GPU path, but it should record
  any assumptions that must be revisited once `E2-F2` lands.
- P2 should consume the P1 metric catalog and align its environment-shape
  expectations with `E2-F2` before the stability table is treated as reusable.
- P3 and P4 should wait on `E2-F6` if they compare or recommend anything other
  than the current `fp64` baseline.
