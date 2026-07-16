# Testing Strategy

This feature is release validation, but it remains split into reviewable phases
whose fixtures/helpers and assertions ship together. Coverage thresholds are
never lowered; changed test-support code maintains at least 80% coverage. Files
use the `*_test.py` convention and collect cleanly when Warp is absent.

## Per-Phase Testing Approach

- **P1 — Deterministic matrix:** Parameterize every shipped single and approved
  additive mechanism. Compare explicit fp64 Warp pair/property values to public
  CPU formulas or direct NumPy equations. Assert symmetry, finite non-negative
  values, and independent majorant coverage for every active unordered pair.
  Use `rtol=1e-7, atol=0` for Brownian; `rtol=1e-6, atol=0` for positive charged,
  SP2016, ST1956, and additive rates; `atol=1e-30` for the extreme repulsive
  charged fixture; and exact equality for exact-zero rates.
- **P2 — Conservation and edge matrix:** Exercise one-box and heterogeneous
  multi-box, one/multiple species, zero/one/two/many active particles, inactive
  gaps, mixed-sign charge, and mixed nanometer/droplet scales. Assert separate
  per-box/per-species mass and total-charge conservation, donor clearing,
  inactive preservation, sorted/in-range/disjoint pairs, capacity, caller-buffer
  identity, RNG reuse/reset, and fail-before-mutation snapshots.
- **P3 — Stochastic/device matrix:** Use repeated fresh seeded runs and
  independently derived expected aggregates over 100 independent seeds. Use a
  predeclared `3 * sqrt(expected_mean)` bound and uncapped fixtures with expected
  aggregate count of at least 100. Exact CPU/Warp pair replay is prohibited as
  a pass criterion. Apply deterministic invariants on every trial. Warp CPU is
  mandatory when Warp is installed; CUDA reuses cases and skips cleanly when
  unavailable.
- **P4 — Documentation:** Validate Markdown links, mechanism/support table rows,
  test marker names, tolerance descriptions, and executable reproduction
  commands.

## Test Locations and Markers

- Primary cross-mechanism coverage:
  `particula/gpu/kernels/tests/coagulation_validation_test.py`.
- Stochastic cross-mechanism coverage:
  `particula/gpu/kernels/tests/coagulation_stochastic_validation_test.py`.
- Shared non-discoverable case table:
  `particula/gpu/kernels/tests/_coagulation_validation_support.py`.
- Existing mechanism-specific/regression coverage:
  `particula/gpu/kernels/tests/coagulation_test.py` and
  `particula/gpu/dynamics/tests/coagulation_funcs_test.py`.
- Device helper: `particula/gpu/tests/cuda_availability.py`.
- Use `@pytest.mark.warp` plus `gpu_parity`, `stochastic`, or `cuda` for focused
  selection. Required correctness cases do not use slow/performance markers.

## Coverage and Pass Policy

1. Test coverage thresholds must never be lowered.
2. Each phase includes its own support-code tests and regression assertions.
3. Tests and any helper implementation are committed in the same phase.
4. Stochastic bounds may not weaken conservation or ownership criteria.
5. A supported row without deterministic and applicable end-to-end evidence is
   a failure, not an expected skip.
6. CUDA absence is the only expected device skip; Warp CPU remains the baseline
   whenever Warp is installed.
7. The executable case table must contain exactly four singleton rows, all six
   two-way rows, and the full four-way row; all four three-way masks must fail
   closed.

Focused runs should include deterministic parity, stochastic markers, optional
CUDA markers, then the complete coagulation suite to detect API and ownership
regressions.
