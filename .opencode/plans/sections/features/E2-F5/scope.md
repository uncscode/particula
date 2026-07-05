# Scope

## In Scope

- Define the compatibility contract for GPU kernel APIs that currently accept
  scalar `temperature` and `pressure`.
- Reserve a keyword-only explicit `environment` path on the two low-level GPU
  entry points while preserving current positional scalar callers.
- Add early mixed-input and pure explicit-environment guards ahead of helper
  calls and Warp launch setup.
- Preserve and extend scalar GPU API tests in
  `particula/gpu/kernels/tests/condensation_test.py` and
  `particula/gpu/kernels/tests/coagulation_test.py`.
- Add contract-focused regression tests for mixed-input rejection,
  explicit-environment P1 rejection, and pre-launch short-circuit behavior.
- Document downstream handoff points for later physics kernels.

## Out of Scope

- Redesigning `GasData` to own temperature, pressure, or humidity; E2 keeps
  environment state separate from gas composition.
- Implementing full latent-heat, parcel-expansion, or wall-loss physics.
- Replacing existing scalar public APIs with mandatory environment arguments.
- Implementing environment normalization helpers, scalar broadcast to
  `(n_boxes,)`, or environment shape/device validation.
- Executing condensation or coagulation through explicit per-box environment
  inputs in this phase.
- Introducing a standalone testing-only phase; tests are co-located with each
  phase that changes behavior.
- Broad performance optimization beyond avoiding unnecessary compatibility
  overhead in normal kernel execution.

## Compatibility Requirements

- Existing scalar calls remain valid and are covered by tests.
- Explicit `environment` remains keyword-only.
- If both scalar values and an explicit environment are supplied, behavior must
  be an early `ValueError`.
- If only explicit `environment` is supplied in P1, behavior must be an early
  phase-scoped `ValueError`.
