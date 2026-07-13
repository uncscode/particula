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
- **P2 (implemented, issue #1288):** In
  `particula/gpu/dynamics/tests/condensation_funcs_test.py`, Warp test kernels
  and independent NumPy fp64 references cover static requested-species
  selection with zero/mixed composition, one-species/pure/mixed
  composition-volume weighting (`rtol=1e-10`, `atol=0`), zero-volume arithmetic
  mean fallback under `-Werror`, weighted-mode index independence, and Kelvin
  radius/term parity using the effective scalar.
- **P3 (implemented, issue #1289):**
  `particula/gpu/kernels/tests/_condensation_test_support.py` and
  `particula/gpu/kernels/tests/condensation_test.py` provide deterministic fp64
  sidecars and independent NumPy pressure/transfer references. One-box tests
  cover all ideal/kappa × static/weighted combinations and prove non-water unit
  activity; multi-box tests cover temperature refresh and current composition.
  Frozen-config, legacy positional, inactive-composition, and invalid aggregate
  preflight cases are co-located. Invalid cases snapshot caller state and
  monkeypatch `wp.launch` to prove no launch or mutation.
- **P4 (completed, issue #1290):**
  `particula/gpu/kernels/tests/_condensation_test_support.py` and
  `particula/gpu/kernels/tests/condensation_test.py` provide deterministic fp64
  independent NumPy end-to-end references for each ideal/kappa ×
  static/composition-weighted pair over named one-box and multi-box fixtures.
  The cases collectively exercise constant and Buck vapor pressure (including
  ice and liquid Buck branches), pure and mixed composition, nonuniform static
  tension, and a designated evaporation clamp. Named parity tolerances compare
  raw transfer, final mass, refreshed vapor pressure, and unchanged gas;
  tighter invariants separately verify clamp, finiteness, nonnegativity, and
  ownership. Warp CPU is required whenever Warp is installed; CUDA is a
  separately marked availability-guarded test.

Every phase ships implementation and `*_test.py` tests together. Existing test
coverage thresholds are never lowered and changed code must retain at least
80% coverage. Conservation, finiteness, and nonnegative-state invariants remain
tight rather than being hidden by aggregate tolerances.
