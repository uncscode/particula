# Testing Strategy

- **P1 (implemented, issue #1287):** In
  `particula/gpu/dynamics/tests/condensation_funcs_test.py`, collection-safe
  Warp imports and parametrized wrappers compare the helpers with independent
  NumPy references. Ideal cases cover pure/mixed, zero-total, water-free, and
  nonzero-water-index compositions. Kappa cases cover wet, pure-water,
  dry/no-water, multi-solute, zero-kappa, and nonzero-water-index compositions;
  the multi-solute fixture verifies water is excluded from kappa weighting.
  The references explicitly mirror zero branches and do not call CPU activity
  functions or the Warp helpers.
- **P2:** In
  `particula/gpu/dynamics/tests/condensation_funcs_test.py`, parametrized
  tests cover static compatibility, selected composition weighting, zero
  weights, and effective surface input to Kelvin formulas.
- **P3:** `particula/gpu/kernels/tests/condensation_test.py` and
  `_condensation_test_support.py` cover one/multi-box coupling, refreshed E4-F1
  pressure, direct imports, and invalid mode/index/shape/dtype/device/value
  failures. State snapshots prove failure occurs before mutation.
- **P4:** End-to-end fixtures compare GPU results with an independent CPU
  sequence using recorded `rtol`/`atol`; Warp CPU runs whenever Warp is
  installed, while CUDA runs when available and otherwise skips cleanly.

Every phase ships implementation and `*_test.py` tests together. Existing test
coverage thresholds are never lowered and changed code must retain at least
80% coverage. Conservation, finiteness, and nonnegative-state invariants remain
tight rather than being hidden by aggregate tolerances.
