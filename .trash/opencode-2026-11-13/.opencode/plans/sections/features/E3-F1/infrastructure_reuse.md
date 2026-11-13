# Infrastructure Reuse

## Existing Code to Reuse or Extend

- `particula/gpu/kernels/coagulation.py:63` -- `_initialize_rng_states(seed,
  rng_states)` initializes one Warp RNG state per box. Keep direct initializer
  coverage and reuse it for seed-once setup rather than replacing the RNG type.
- `particula/gpu/kernels/coagulation.py:260` -- `brownian_coagulation_kernel`
  already loads `state = rng_states[box_idx]`, consumes random draws, and writes
  `rng_states[box_idx] = state`; implementation should preserve this kernel
  persistence behavior.
- `particula/gpu/kernels/coagulation.py:474` -- `_validate_rng_states` already
  enforces shape and device compatibility for caller-provided RNG buffers.
- `particula/gpu/kernels/coagulation.py:544` -- `coagulation_step_gpu` is the
  public low-level API surface. Any new initialization control must remain
  keyword-only and source-compatible with existing positional callers.
- `particula/gpu/kernels/coagulation.py:603` --
  `initialize_coagulation_rng_states(...)` is the public seed-once helper used
  by benchmarks and callers that own persistent RNG state buffers.
- `particula/gpu/kernels/coagulation.py:691` -- `coagulation_step_gpu` now
  validates caller buffers before internal allocation, preserves advanced RNG
  state on fractional-trial early returns, and reuses caller-owned state unless
  explicit reset is requested.
- `particula/gpu/kernels/tests/coagulation_test.py:78` -- `device` fixture uses
  `warp_devices(wp)` so new tests automatically cover CPU and CUDA when Warp can
  access CUDA.
- `particula/gpu/kernels/tests/coagulation_test.py:750` -- existing invalid input
  tests assert no mutation and no launch work on validation failures. Follow this
  fail-before-mutation pattern.
- `particula/gpu/kernels/tests/coagulation_test.py:927` -- repeated stochastic
  timestep test currently increments `rng_seed`; adapt this pattern to prove
  persistent state removes the manual increment requirement.
- `particula/gpu/kernels/tests/coagulation_test.py:1127` -- preallocated buffer
  reuse test can be extended or mirrored for `rng_states` persistence.
- `particula/gpu/tests/benchmark_test.py:1434` -- benchmark code preallocates
  `rng_states_buf` and now seeds it once via the public helper before repeated
  calls run with `initialize_rng=False`.
- `docs/Features/Roadmap/data-oriented-gpu.md:756` -- roadmap text already names
  the graph-capture motivation that the shipped implementation now supports.

## Patterns to Follow

- Keep all device buffer validation before launch-time mutation.
- Avoid hidden host reads of GPU buffer contents to infer initialization state.
- Prefer explicit keyword-only API choices over implicit inspection of zeroed
  `rng_states` buffers.
- Keep tests property-based around state advancement and non-reset behavior;
  avoid exact RNG sequence equality across devices.
