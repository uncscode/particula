# Scope

This feature delivers seed-once, persisted RNG semantics for repeated
`coagulation_step_gpu` calls. It defines a backwards-compatible initialization
contract for `rng_seed` and optional `rng_states`, adds regression tests proving
caller-provided RNG buffers are not overwritten every timestep, implements the
host-side orchestration change, and documents the graph-capture caveat.

## In Scope

- Define the API compatibility plan for `rng_seed`, `rng_states`, and any
  explicit initialization mode or helper required to avoid ambiguous zeroed
  buffer behavior.
- Update `particula/gpu/kernels/coagulation.py` so preallocated `rng_states` can
  persist and advance across repeated calls without unconditional
  `_initialize_rng_states` launches.
- Preserve legacy convenience behavior for callers that do not provide
  `rng_states`: internal state is still allocated and seeded for that call.
- Preserve validation-before-mutation behavior for invalid environment, volume,
  shape, and device inputs.
- Add co-located tests in `particula/gpu/kernels/tests/coagulation_test.py` for
  Warp CPU and CUDA-if-available device parametrization.
- Update benchmark or documentation examples that currently increment
  `rng_seed` manually when a persistent `rng_states` buffer is available.
- Document graph-capture guidance: initialize and allocate RNG buffers before
  capture, then capture only repeated calls that advance existing state.

## Out of Scope

- Replacing Warp RNG primitives or changing the stochastic coagulation algorithm.
- Guaranteeing identical random sequences across CPU, Warp CPU, and CUDA devices.
- Changing particle data container schemas or CPU/GPU transfer helpers.
- Adding a standalone testing-only phase; all implementation phases include
  their own co-located tests.
- Broader RNG policy for non-coagulation GPU kernels unless needed for shared
  documentation consistency.
