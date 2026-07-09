# Scope

This feature delivers seed-once GPU RNG semantics in phases. The shipped work
for issues #1236, #1237, #1238, and #1239 now covers the compatibility
contract, repeated-call regressions, runtime semantics, and final documentation
handoff. The last phase was docs-only: it clarified the shipped behavior in the
public docstring and GPU guides without changing runtime logic.

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
  Warp CPU and CUDA-if-available device parametrization, including repeated
  valid-call persisted-buffer regressions and valid-then-invalid preservation
  checks.
- Document the local `coagulation_step_gpu` API contract in the function
  docstring.

## Out of Scope

- Replacing Warp RNG primitives or changing the stochastic coagulation algorithm.
- Guaranteeing identical random sequences across CPU, Warp CPU, and CUDA devices.
- Changing particle data container schemas or CPU/GPU transfer helpers.
- Production semantic changes in the P2 regression phase; that issue shipped as
  tests only.
- Broader RNG policy for non-coagulation GPU kernels unless needed for shared
  documentation consistency.
- Further runtime changes beyond the shipped docs-only clarification phase.
