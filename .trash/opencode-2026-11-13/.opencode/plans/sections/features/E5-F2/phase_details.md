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

- [x] **E5-F2-P3:** Add charge preflight validation with state-preservation tests
  - Issue: TBD | Size: S | Status: Completed
  - Delivered: Charge shape, `wp.float64` dtype, active-device ownership, and
    finite values are validated before downstream runtime work or mutation. The
    finite-value scan is read-only and uses only private status state.
  - Files: `particula/gpu/kernels/coagulation.py`,
    `particula/gpu/kernels/tests/coagulation_test.py`
  - Tests: Wrong shape, dtype, device, NaN, and infinity failures plus snapshots
    proving masses, concentration, charge, output buffers, and RNG are unchanged.

- [x] **E5-F2-P4:** Transfer and clear charge during merges with conservation tests
  - Issue: #1339 | Size: S | Status: Completed
  - Delivered: The private `apply_coagulation_kernel` now accepts the existing
    fp64 particle charge buffer, adds donor charge to the recipient, and clears
    donor charge with donor mass and concentration for each accepted disjoint
    pair. Production and all direct test launches pass that buffer.
  - Files: `particula/gpu/kernels/coagulation.py`,
    `particula/gpu/kernels/tests/coagulation_test.py`
  - Tests: Deterministic direct signed-charge, multi-box/multi-species fixtures
    compare complete mass, concentration, and charge arrays against independent
    expected state and separately conserve per-box species mass and total charge.
    Merge, zero-count, self-pair, and empty-pair coverage verifies donor clearing
    and no-op paths; step-level signed-charge coverage conserves charge while
    retaining supplied sidecar identity.
  - Boundary: Brownian selection remains charge-neutral. The public
    `coagulation_step_gpu` API and three-item return tuple, collision sidecars,
    and persistent RNG ownership are unchanged.

- [x] **E5-F2-P5:** Update development documentation
  - Issue: #1340 | Size: XS | Status: Completed
  - Delivered: Recorded charged-helper ownership, preflight and merge semantics,
    bounded evidence, and the E5-F3 handoff without claiming charged execution.
  - Files: `docs/Features/data-containers-and-gpu-foundations.md`,
    `docs/Features/coagulation_strategy_system.md`, `AGENTS.md`, E5 plan sections
  - Tests: Resolved published links and ran warning-clean focused evidence:
    `pytest particula/gpu/dynamics/tests/coagulation_funcs_test.py -q -Werror`
    and `pytest particula/gpu/kernels/tests/coagulation_test.py -q -Werror`.
