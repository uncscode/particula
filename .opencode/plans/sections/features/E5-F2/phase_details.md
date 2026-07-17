# Phase Details

- [x] **E5-F2-P1:** Port scalar Coulomb and reduced-property helpers with unit tests
  - Issue: #1336 | Size: S | Status: Completed
  - Delivered: Internal scalar fp64 `@wp.func` primitives for Coulomb potential,
    reduced values, kinetic/continuum limits, and diffusive Knudsen number.
    The helpers retain local invalid-domain fallbacks, the `-200.0` Coulomb
    lower clip, and the `< 1e-80` kinetic-limit zero result.
  - Files: `particula/gpu/dynamics/coagulation_funcs.py`,
    `particula/gpu/dynamics/tests/coagulation_funcs_test.py`
  - Tests: Independent co-located Warp probe/oracle parity coverage for neutral,
    attractive, repulsive/clipped, equal-particle, mixed-scale, invalid-domain,
    and extreme kinetic-threshold cases. Tests are deterministic `gpu_parity`
    coverage on Warp CPU with optional CUDA discovery.
  - Boundary: No public exports, data-container changes, Brownian dispatch,
    charged execution wiring, or module-boundary changes.

- [x] **E5-F2-P2:** Port approved charged pair models with CPU parity tests
  - Issue: #1337 | Size: S | Status: Completed
  - Delivered: Internal scalar fp64 `charged_hard_sphere_wp` composes the
    existing property, Coulomb, reduced-value, and diffusive-Knudsen helpers.
    It preserves finite/non-negative exact-safe-zero behavior, including the
    clipped extreme-repulsion path, without adding dispatch or execution.
  - Files: `particula/gpu/dynamics/coagulation_funcs.py`,
    `particula/gpu/dynamics/tests/coagulation_funcs_test.py`
  - Tests: Independent NumPy-oracle and fp64 Warp CPU/optional-CUDA parity for
    neutral, same-sign, opposite-sign, mixed-scale, temperature, and pressure
    cases; pair-order symmetry, exact extreme-repulsion zero, and exhaustive
    invalid state/charge/constant safe-zero cases.
  - Boundary: No public exports, model dispatch, kernel-entry integration,
    charged workflow, or changes to Brownian execution; unsupported models
    remain unavailable.

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
