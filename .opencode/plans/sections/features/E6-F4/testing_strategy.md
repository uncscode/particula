# Testing Strategy

Every implementation phase ships co-located fast tests in the same change. The
configured minimum 80% coverage remains unchanged; no coverage threshold,
scientific tolerance, or E6-F3 invariant may be weakened. Test modules use the
`*_test.py` suffix.

## Per-Phase Approach

- **P1 configuration/preflight:** In
  `particula/gpu/kernels/tests/wall_loss_test.py`, cover spherical and
  rectangular charged forms, zero defaults, finite signed potential, scalar and
  three-component fields, and unsupported combinations. Reject malformed
  fields, nonfinite charge/configuration, invalid particles/environment/time,
  wrong shapes/ranks/dtypes/devices, and inconsistent boxes before allocation,
  RNG initialization/advance, or any caller-array write. Snapshot identities
  and all state.
- **P2 image charge:** In
  `particula/gpu/dynamics/tests/wall_loss_funcs_test.py`, compare the fp64 device
  factor to the independent CPU `ChargedWallLossStrategy` array oracle for zero,
  positive, and negative charges over particle radii and temperatures. Require
  exact factor one for zero charge, nonzero enhancement at zero potential,
  finite behavior near clipping limits, and documented `rtol`/`atol`.
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
- **P5 parity/stochastic validation:** In
  `particula/gpu/kernels/tests/wall_loss_parity_test.py`, require Warp CPU for a
  deterministic geometry/charge/field coefficient matrix. Compare the charged
  mode with zero charge against the E6-F3 neutral device coefficient exactly or
  at the documented zero-roundoff contract, and compare both to CPU at recorded
  fp64 tolerances. Validate survival counts against `exp(-k*dt)` with predeclared
  binomial confidence/sigma bounds over enough independent particles/seeds.
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
