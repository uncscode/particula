# E2-F7 Phase Details

- [ ] **E2-F7-P1:** Define condensation stiffness stress cases and metrics with tests
  - Issue: TBD | Size: S | Status: Not Started
  - Goal: Create a small, reproducible catalog of condensation regimes and
    stability metrics that can be reused by GPU and CPU comparisons.
  - Files: `particula/gpu/kernels/tests/condensation_test.py`,
    `particula/gpu/kernels/tests/condensation_stiffness_helpers.py` if the
    fixtures/metrics no longer fit cleanly in the test file, and
    `docs/Features/Roadmap/condensation-stiffness-study.md`.
  - Tests: Unit tests for stress-case construction, metric calculation, and
    explicit timestep pass/fail classification.

- [ ] **E2-F7-P2:** Measure stable explicit timestep for current GPU condensation path
  - Issue: TBD | Size: S | Status: Not Started
  - Goal: Run the stress catalog against `condensation_step_gpu` with fixed
    shapes and preallocated buffers, producing a stability map for the current
    explicit GPU implementation.
  - Files: `particula/gpu/kernels/condensation.py` only for small helper hooks,
    `particula/gpu/kernels/tests/condensation_test.py`, and
    `docs/Features/Roadmap/condensation-stiffness-study.md` for the resulting
    timestep table.
  - Tests: Fast Warp CPU tests for explicit timestep bounds and optional CUDA
    benchmark coverage when available.

- [ ] **E2-F7-P3:** Evaluate fixed-shape sub-stepping and semi-implicit candidates
  - Issue: TBD | Size: S | Status: Not Started
  - Goal: Compare graph-capture-compatible options against the explicit map and
    identify the smallest safe foundation for future implementation.
  - Files: a narrowly scoped helper in `particula/gpu/kernels/condensation.py`
    or a test-only prototype under `particula/gpu/kernels/tests/`, plus
    `docs/Features/Roadmap/condensation-stiffness-study.md` analysis notes.
  - Tests: Unit tests for deterministic fixed-count sub-step behavior,
    semi-implicit/asymptotic candidate invariants, and no dynamic allocation in
    captured-step candidates where practical.

- [ ] **E2-F7-P4:** Publish integration recommendation and development documentation
  - Issue: TBD | Size: XS | Status: Not Started
  - Goal: Record the recommended integration foundation, rejected alternatives,
    graph-capture/autodiff constraints, and follow-up implementation gates.
  - Files: `docs/Features/Roadmap/data-oriented-gpu.md`,
    `docs/Features/Roadmap/warp-autodiff-limitations.md`,
    `docs/Features/Roadmap/condensation-stiffness-study.md`, and plan sections
    as needed.
  - Tests: Documentation link checks and any updated fast tests from earlier
    phases.

## Phase Ordering Notes

- P1 may begin from the current scalar-compatible GPU path, but it should record
  any assumptions that must be revisited once `E2-F2` lands.
- P2 should consume the P1 metric catalog and align its environment-shape
  expectations with `E2-F2` before the stability table is treated as reusable.
- P3 and P4 should wait on `E2-F6` if they compare or recommend anything other
  than the current `fp64` baseline.
