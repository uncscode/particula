# Scope

## In Scope

- Implement a shared private normalization helper for GPU environment inputs in
  `particula/gpu/kernels/environment.py`.
- Accept scalar, direct `(n_boxes,)` Warp-array, and explicit
  `WarpEnvironmentData` inputs on the two low-level GPU entry points while
  preserving current positional scalar callers.
- Add early mixed-input, missing-input, shape, and device validation ahead of
  helper calls, buffer preparation, and Warp launch setup.
- Update condensation and coagulation runtime paths to consume normalized
  per-box environment arrays instead of scalar-only launch inputs.
- Precompute condensation box-level environment-derived properties once per call
  and reuse them in the hot path.
- Preserve and extend scalar GPU API tests in
  `particula/gpu/kernels/tests/condensation_test.py` and
  `particula/gpu/kernels/tests/coagulation_test.py`.
- Add helper-focused regression tests in
  `particula/gpu/kernels/tests/environment_test.py` plus entry-point regression
  tests for mixed-input rejection, explicit-environment success, direct array
  success, mismatch failures, and pre-launch short-circuit behavior.

## Out of Scope

- Redesigning `GasData` to own temperature, pressure, or humidity; E2 keeps
  environment state separate from gas composition.
- Implementing full latent-heat, parcel-expansion, or wall-loss physics.
- Replacing existing scalar public APIs with mandatory environment arguments.
- Expanding the public `particula.gpu` conversion API beyond existing transfer
  helpers.
- Changing `WarpEnvironmentData` schema ownership or consuming
  `saturation_ratio` in these kernel entry points.
- Introducing a standalone testing-only phase; tests are co-located with each
  phase that changes behavior.
- Broad performance optimization beyond avoiding duplicate environment
  preparation inside normal kernel execution.

## Compatibility Requirements

- Existing scalar calls remain valid and are covered by tests.
- Explicit `environment` remains keyword-only.
- If both scalar values and an explicit environment are supplied, behavior must
  be an early `ValueError`.
- If only explicit `environment` is supplied with valid `(n_boxes,)` arrays on
  the caller device, execution should succeed.
