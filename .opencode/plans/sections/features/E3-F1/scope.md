# Scope

This feature delivers seed-once GPU RNG semantics in phases. The shipped work
for issues #1236 and #1237 now covers the compatibility contract layer plus
targeted regression coverage: P1 defined the backwards-compatible interaction
between `rng_seed`, optional `rng_states`, and keyword-only `initialize_rng`,
and P2 added test-only coverage proving repeated valid calls advance a
caller-owned buffer while invalid follow-up calls preserve the already-advanced
state. Broader runtime expansion, docs, and benchmark guidance remain follow-up
work.

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
- Broader GPU docs, graph-capture guidance, and benchmark comment refreshes in
  this issue.
