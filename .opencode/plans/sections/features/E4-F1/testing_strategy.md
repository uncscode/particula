# Testing Strategy

Every implementation phase ships with co-located tests; coverage thresholds are
not lowered. Test files retain the `*_test.py` suffix, primarily under
`particula/gpu/kernels/tests/`.

## Per-Phase Coverage

- **P1:** Validate accepted constant/Buck configurations and reject missing,
  unsupported, wrong-shape/dtype/device, species-count, negative, NaN, and
  infinite inputs before mutation.
- **P2:** Compare Warp CPU results with `ConstantVaporPressureStrategy` and
  `get_buck_vapor_pressure()` below, at, and above freezing. Cover one/multiple
  boxes, one/multiple species, and mixed model ordering.
- **P3:** Exercise scalar, direct Warp-array, and `WarpEnvironmentData`
  temperature paths. Mutate temperature between calls and prove the next pressure
  and condensation result changes without a CPU refresh/transfer.
- **P4:** Test reusable configuration/output arrays, repeated calls, active-device
  mismatch, absent configuration behavior, positional API compatibility, and
  snapshots proving gas/particle arrays remain unchanged after early errors.
- **P5:** Validate documentation links, mode/units tables, and cross-references.

## Device and Numerical Policy

- Warp CPU parity is required when Warp is installed; CUDA is optional and skips
  cleanly when unavailable.
- Use explicit `rtol`/`atol`, `np.float64` reference fixtures, and exact shape
  assertions. Include the CPU Buck reference value near 298.15 K and branch-edge
  cases around 273.15 K.
- Run focused tests first, then normal fast tests. Slow/performance tests are not
  required because issue #1272 specifies no diagnostics or benchmark target.
- Maintain at least 80% changed-code coverage and never lower repository gates.
