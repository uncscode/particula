# Scope

## In Scope

- Define the compatibility contract for GPU kernel APIs that currently accept
  scalar `temperature` and `pressure`.
- Add environment-aware normalization and validation helpers that can broadcast
  scalar values to `(n_boxes,)` arrays or accept validated per-box environment
  state.
- Migrate condensation and coagulation GPU launch paths so kernels can consume
  per-box environment values without breaking scalar callers.
- Preserve and extend scalar GPU API tests in
  `particula/gpu/kernels/tests/condensation_test.py` and
  `particula/gpu/kernels/tests/coagulation_test.py`.
- Add mismatch/error tests for `n_boxes`, shape, and device incompatibilities.
- Document downstream handoff points for later physics kernels.

## Out of Scope

- Redesigning `GasData` to own temperature, pressure, or humidity; E2 keeps
  environment state separate from gas composition.
- Implementing full latent-heat, parcel-expansion, or wall-loss physics.
- Replacing existing scalar public APIs with mandatory environment arguments.
- Introducing a standalone testing-only phase; tests are co-located with each
  phase that changes behavior.
- Broad performance optimization beyond avoiding unnecessary compatibility
  overhead in normal kernel execution.

## Compatibility Requirements

- Existing scalar calls remain valid and are covered by tests.
- Per-box environment arrays must match `particles.n_boxes` exactly.
- If both scalar values and an explicit environment are supplied, behavior must
  be documented as either explicit precedence or an early `ValueError`.
