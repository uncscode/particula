# Testing Strategy

This feature is release validation, but it remains split into reviewable phases
whose fixtures/helpers and assertions ship together. Coverage thresholds are
never lowered; changed test-support code maintains at least 80% coverage. Files
use the `*_test.py` convention and collect cleanly when Warp is absent.

## Per-Phase Testing Approach

- **P1 — Deterministic matrix (implemented in #1362):** Parameterize every shipped single and approved
  additive mechanism. Compare explicit fp64 Warp pair/property values to public
  CPU formulas or direct NumPy equations. Assert symmetry, finite non-negative
  values, and independent majorant coverage for every active unordered pair.
  Use `rtol=1e-7, atol=0` for Brownian; `rtol=1e-6, atol=0` for positive charged,
   SP2016, ST1956, and additive rates; and exact equality for exact-zero rates,
   including the extreme repulsive charged fixture. Literal executable
  rows are `1`, `2`, `3`, `4`, `5`, `6`, `8`, `9`, `10`, `12`, and `15`;
   three-way rows `7`, `11`, `13`, and `14` have Warp-free deferred-error tests.
   Configuration resolution and deferred-error tests use the host-only resolver;
   pair/property/majorant observations are lazy Warp-CPU probes marked `warp`
   and `gpu_parity`.
- **P2 — Conservation and edge matrix (implemented in #1363):** The public-step
  matrix runs every executable mask for `normal` and heterogeneous `two_box`
  fixtures with one and two species, on each available Warp device. It asserts
  per-box/per-species inventory at `rtol=1e-12, atol=1e-30`, charge conservation
  for charge-enabled rows, donor clearing/recipient transfer, inactive-sentinel
  preservation, sorted/in-range/disjoint accepted prefixes, capacity bounds,
  collision/count sidecar identity, and persistent RNG initialization/advance.
  Focused public cases cover zero/one/two active slots, charged and
  sedimentation zero-rate no-ops, zero-capacity rejection, scalar/device-array
  turbulent inputs (masks 8 and 10), and exact snapshots for deferred or
  selected invalid-input preflight failures. It makes no repeated-seed or exact
  replay claim.
- **P3 — Stochastic/device matrix (implemented in #1364):** The dedicated
  stochastic module runs 100 fresh unique seeds for every executable mask and
  selected device. A host-only initial-state oracle sums enabled unordered-pair
  rates and includes time step, volume, and SP2016 scheduling concentration;
  it checks aggregate observations against `3 * sqrt(expected_mean)`. Every
  trial retains P2 return/RNG identity and physical/ownership invariants, with
  one-proposal capacity and counts constrained to `[0, 1]`. Exact CPU/Warp pair
  replay is not a criterion. Warp CPU is required when installed; CUDA uses the
  same matrix and cleanly skips when unavailable.
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
