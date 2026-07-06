# E2-F7 Phase Details

- [x] **E2-F7-P1:** Define condensation stiffness stress cases and metrics with tests
  - Issue: #1213 | Size: S | Status: Implemented
  - Goal: Create a small, reproducible catalog of condensation regimes and
    stability metrics that can be reused by GPU and CPU comparisons.
  - Files delivered: `particula/gpu/kernels/tests/condensation_test.py`,
    `particula/gpu/kernels/tests/_condensation_test_support.py`,
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
  - Files delivered: `particula/gpu/kernels/tests/condensation_stiffness_test.py`,
    `particula/gpu/kernels/tests/_condensation_test_support.py`, and
    `docs/Features/Roadmap/condensation-stiffness-study.md`.
  - Implementation notes: P2 added `_RECORDED_TIMESTEP_GRID_BY_CASE`, the
    shared `_RECORDED_STIFFNESS_THRESHOLD`, and
    `CondensationStiffnessTrialRecord`, then used a test-local sweep helper to
    rebuild fresh deterministic inputs per trial while reusing one
    caller-owned `mass_transfer` buffer per case/device. The recorded grid is
    executable evidence for the current particle-only path: every shipped row is
    `stable` under the shared inclusive threshold, gas concentration remains
    unchanged, single-box cases keep scalar inputs, and `droplet_like` covers
    direct Warp `(n_boxes,)` arrays. Separate regression tests still cover the
    unstable branch for larger fractional changes, zero-mass growth, and
    non-finite values, plus a guarded CUDA parity check for the nanometer
    recorded-grid contract.
  - Tests: Fast Warp CPU recorded-grid tests plus a guarded CUDA parity check
    when available.

- [x] **E2-F7-P3:** Evaluate fixed-shape sub-stepping and semi-implicit candidates
  - Issue: #1215 | Size: S | Status: Implemented
  - Goal: Compare graph-capture-compatible options against the explicit map and
    identify the smallest safe foundation for future implementation without
    broadening the current production particle-only contract.
  - Files delivered: `particula/gpu/kernels/tests/condensation_stiffness_test.py`,
    `particula/gpu/kernels/tests/_condensation_test_support.py`,
    `particula/gpu/kernels/tests/condensation_test.py`, and
    `docs/Features/Roadmap/condensation-stiffness-study.md`.
  - Implementation notes: P3 stayed test-local and evidence-driven. It added
    deterministic prototype candidates `fixed_count_substeps_4` and
    `asymptotic_relaxation`, reusable fixed-shape scratch/buffer coverage,
    repeated-run determinism assertions, finite/non-negative particle-mass
    checks, CPU-reference tolerance checks, and explicit-baseline error-bound
    comparisons. The roadmap study now records graph-capture/autodiff notes and
    the deferred gas-coupling boundary. No public API change landed in
    `condensation_step_gpu(...)`, no private production helper was added under
    `particula/gpu/kernels/condensation.py`, no production gas-state update hook
    shipped, and
    `particula/integration_tests/condensation_particle_resolved_test.py`
    remained unchanged.
  - Tests: Fast Warp CPU candidate tests for fixed-count and asymptotic
    determinism, boundedness, reusable fixed-shape scratch coverage,
    CPU-reference tolerances, and explicit-baseline error bounds.

- [ ] **E2-F7-P4:** Publish integration recommendation and development documentation
  - Issue: #1216 | Size: XS | Status: In Progress
  - Goal: Record the recommended integration foundation, rejected alternatives,
    graph-capture/autodiff constraints, and the exact follow-up split boundary
    required before any gas-coupled production path is claimed.
  - Files: `docs/Features/Roadmap/data-oriented-gpu.md`,
    `docs/Features/Roadmap/warp-autodiff-limitations.md`,
    `docs/Features/Roadmap/condensation-stiffness-study.md`, and plan sections
    as needed.
  - Implementation notes: The published decision record now promotes the
    stiffness study into the canonical recommendation source, recommends
    `fixed_count_substeps_4` as the preferred fixed-shape implementation
    foundation, keeps production scope explicitly particle-only, names
    same-issue conservation coverage in
    `particula/integration_tests/condensation_particle_resolved_test.py` as the
    gate for any future gas-coupled production claim, and keeps the E2-F2
    environment-shape plus E2-F6 `float64` evidence boundaries explicit across
    the roadmap pages and plan metadata.
  - Tests: Documentation link checks, reruns of the executable focused
    condensation selectors in discoverable wrappers
    `particula/gpu/kernels/tests/condensation_test.py` and
    `particula/gpu/kernels/tests/condensation_stiffness_test.py`, reruns of
    `particula/dynamics/condensation/tests/staggered_stability_test.py -m slow
    -v`, and reruns of the bounded gas-coupled regression only when the final
    wording still depends on that gate.

## Phase Ordering Notes

- P1 may begin from the current scalar-compatible GPU path, but it should record
  any assumptions that must be revisited once `E2-F2` lands.
- P2 should consume the P1 metric catalog and align its environment-shape
  expectations with `E2-F2` before the stability table is treated as reusable.
- P3 and P4 should wait on `E2-F6` if they compare or recommend anything other
  than the current `fp64` baseline; P3 now recorded evidence only and left the
  production recommendation plus any gas-coupled hook to later work.
