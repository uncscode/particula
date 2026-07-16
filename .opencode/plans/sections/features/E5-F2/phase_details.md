# Phase Details

- [ ] **E5-F2-P1:** Port scalar Coulomb and reduced-property helpers with unit tests
  - Issue: TBD | Size: S | Status: Not Started
  - Goal: Implement stable fp64 pair primitives for Coulomb potential,
    reduced mass/friction, enhancement limits, and diffusive properties.
  - Files: `particula/gpu/dynamics/coagulation_funcs.py`,
    `particula/gpu/dynamics/tests/coagulation_funcs_test.py`
  - Tests: Neutral, attractive, repulsive/clipped, equal-particle, mixed-scale,
    and finite-limit comparisons against independently evaluated CPU formulas.

- [ ] **E5-F2-P2:** Port approved charged pair models with CPU parity tests
  - Issue: TBD | Size: S | Status: Not Started
  - Goal: Implement only the E5-approved charged model identifiers behind
    scalar pair helpers without wiring stochastic execution.
  - Files: `particula/gpu/dynamics/coagulation_funcs.py`,
    `particula/gpu/dynamics/tests/coagulation_funcs_test.py`
  - Tests: Deterministic CPU/Warp pair matrices for zero, same-sign, and
    opposite-sign charge across representative radius, mass, temperature, and
    pressure fixtures; unsupported model names remain unavailable.

- [ ] **E5-F2-P3:** Add charge preflight validation with state-preservation tests
  - Issue: TBD | Size: S | Status: Not Started
  - Goal: Reject malformed charge buffers before allocation, launch, particle
    mutation, or persistent RNG advancement.
  - Files: `particula/gpu/kernels/coagulation.py`,
    `particula/gpu/kernels/tests/coagulation_test.py`
  - Tests: Wrong shape, dtype, device, NaN, and infinity failures plus snapshots
    proving masses, concentration, charge, output buffers, and RNG are unchanged.

- [ ] **E5-F2-P4:** Transfer and clear charge during merges with conservation tests
  - Issue: TBD | Size: S | Status: Not Started
  - Goal: Add donor charge to the recipient and clear the donor atomically with
    existing mass/concentration updates for each accepted disjoint pair.
  - Files: `particula/gpu/kernels/coagulation.py`,
    `particula/gpu/kernels/tests/coagulation_test.py`
  - Tests: Direct merge-kernel fixtures, multi-species/multi-box accepted pairs,
    zero-collision and inactive slots, donor clearing, recipient sums, and
    separate per-box mass/charge conservation on supported devices.

- [ ] **E5-F2-P5:** Update development documentation
  - Issue: TBD | Size: XS | Status: Not Started
  - Goal: Record charged helper ownership, merge semantics, validated limits,
    and the E5-F3 handoff without claiming charged execution is available.
  - Files: `docs/Features/data-containers-and-gpu-foundations.md`,
    `docs/Features/coagulation_strategy_system.md`, `AGENTS.md`, E5 plan sections
  - Tests: Documentation link/reference validation and focused command checks.
