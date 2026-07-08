# E3-F2 Phase Details

- [ ] **E3-F2-P1:** Add mixed NPF/droplet fixture and acceptance-rate diagnostics with unit tests
  - Issue: TBD | Size: S | Status: Not Started
  - Goal: Build reproducible mixed-scale coverage and expose measured sampler
    acceptance for test/debug analysis.
  - Files: `particula/gpu/kernels/tests/coagulation_test.py`, optional
    test-only helpers near `particula/gpu/kernels/coagulation.py` if needed.
  - Tests: Mixed NPF/droplet fixture construction, acceptance metric sanity,
    Warp CPU and CUDA-if-available execution.

- [ ] **E3-F2-P2:** Prototype bounded mixed-scale sampling hardening with conservation tests
  - Issue: TBD | Size: S | Status: Not Started
  - Goal: Evaluate a minimal fixed-bin majorant or stratified pair-selection
    path that improves or bounds acceptance without changing physics.
  - Files: `particula/gpu/kernels/coagulation.py`,
    `particula/gpu/kernels/tests/coagulation_test.py`.
  - Tests: Unit coverage for no self-pairs/duplicate invalid pairs, collision
    buffer bounds, and mass conservation on the mixed-scale fixture.

- [ ] **E3-F2-P3:** Compare statistical correctness against current Brownian behavior
  - Issue: TBD | Size: S | Status: Not Started
  - Goal: Demonstrate that selected behavior preserves expected Brownian rates
    within stochastic tolerances or document the bounded limitation.
  - Files: `particula/gpu/kernels/tests/coagulation_test.py`, optional
    benchmark or metric helper in `particula/gpu/tests/benchmark_test.py`.
  - Tests: Aggregate stochastic collision-rate checks, conservation checks,
    deterministic seeded/reused-RNG behavior aligned with E3-F1.

- [ ] **E3-F2-P4:** Document selected design or accepted mixed-scale limitation
  - Issue: TBD | Size: XS | Status: Not Started
  - Goal: Update roadmap or feature documentation with reproduction commands,
    observed acceptance bounds, and the chosen implementation decision.
  - Files: `docs/Features/Roadmap/data-oriented-gpu.md`, optional focused
    feature note under `docs/Features/`.
  - Tests: Documentation link/format validation plus any focused reproduction
    commands recorded in the document.
