# Testing Strategy

Every implementation phase ships co-located fast tests in the same change. The
configured minimum 80% coverage remains unchanged; no coverage threshold,
scientific tolerance, or E6-F3 invariant may be weakened. Test modules use the
`*_test.py` suffix.

## Per-Phase Approach

- **P1 configuration/preflight (shipped):** In
  `particula/gpu/kernels/tests/wall_loss_test.py`, cover spherical and
  rectangular charged forms, appended defaults, finite signed potential, and
  scalar/three-component field schemas. Tests reject malformed mode, scalar,
  field, particle, environment, time, and RNG inputs; assert field-before-charge
  ordering; and snapshot particle, charge, field, and supplied RNG state for
  preflight atomicity. Valid charged rectangular and zero-time cases preserve
  field ownership. Matched zero-charge neutral/charged cases assert exact
  particle and RNG equality because P1 leaves execution neutral.
- **P2 image charge (shipped):** In
  `particula/gpu/dynamics/tests/wall_loss_funcs_test.py`, independent test-local
  NumPy oracles compare fp64 Warp ratio and enhancement kernels at
  `rtol=1e-12, atol=0`. Coverage separately observes the `-200` ratio floor;
  covers signed, zero, ordinary, and strongly charged lanes; asserts exact
  zero-charge identity and sign symmetry; and covers exponent-only and
  ratio-floor-plus-exponent clipping. Warp CPU is required when installed and
  CUDA rows remain optional. This is primitive evidence only, not direct-step
  or CPU-strategy integration parity.
- **P3 field/composition:** In the same module, compare scalar spherical and
  vector rectangular field magnitudes, potential-derived fields, geometry
  scales, charge signs, zero-field drift, and final nonnegative clipping. Verify
  `neutral * factor + drift` against CPU fp64 outputs, including cancellation
  and extreme-but-valid cases.
- **P4 step integration:** In
  `particula/gpu/kernels/tests/wall_loss_test.py`, exercise mixed zero/nonzero
  charges, sparse inactive gaps, one/multi-box, one/multi-species, exact zero
  time, controlled all-survive/all-remove behavior, and repeated steps. Removed
  slots clear every species mass, concentration, and charge; survivors preserve
  values and all arrays preserve shape, device, dtype, and identity. Re-run the
  E6-F3 omitted/supplied/reset RNG lifecycle matrix and invalid-call
  non-advancement checks.
- **P5 parity/stochastic validation (deferred):** In
  `particula/gpu/kernels/tests/wall_loss_parity_test.py`, require Warp CPU for a
  deterministic geometry/charge/field coefficient matrix. Compare the charged
  mode with zero charge against the E6-F3 neutral device coefficient, survivor
  state, and final identically initialized RNG state by exact same-device
  equality, and compare both modes to CPU at recorded fp64 tolerances. Validate
  survival counts against `exp(-k*dt)` with the frozen eight-stratum exact
  binomial contract over independent particles/seeds.
  Exact NumPy/Warp draws are never expected. Repeat on CUDA when available and
  skip cleanly otherwise.
- **P6 documentation:** Validate links, imports, SI units, semantic examples,
  support/deferred tables, and focused commands.

## Regression and Focused Commands

- Keep all CPU charged and neutral tests under
  `particula/dynamics/wall_loss/**/` green; E6-F4 does not change CPU behavior.
- Keep E6-F3 neutral coefficient, fixed-slot, preflight, and persistent-RNG
  tests green, plus direct coagulation charge/RNG regressions.
- Focused commands:
  - `pytest particula/gpu/dynamics/tests/wall_loss_funcs_test.py -q -Werror`
  - `pytest particula/gpu/kernels/tests/wall_loss_test.py particula/gpu/kernels/tests/wall_loss_parity_test.py -q -Werror`
  - `pytest particula/dynamics/wall_loss/tests/wall_loss_strategies_test.py -q -Werror`

Deterministic acceptance records explicit float64 tolerances beside each matrix.
Stochastic acceptance records its sample size and confidence rule before
observing results. Zero-time, inactive-slot, and invalid-call no-ops are exact.
