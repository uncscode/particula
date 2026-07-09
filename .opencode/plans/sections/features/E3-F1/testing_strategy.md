# Testing Strategy

Every phase includes co-located tests with the implementation it changes. Tests
use the existing `*_test.py` convention and must not lower coverage thresholds.

## Per-Phase Testing Approach

- **P1:** Shipped in `particula/gpu/kernels/tests/coagulation_test.py`. Covers
  omitted `rng_states`, caller-provided `rng_states` reuse without implicit
  reset, explicit initialization via `initialize_rng=True`, wrong shape/device
  validation, and invalid inputs that must not mutate RNG buffers.
- **P2:** Add regression tests proving persistent RNG state advances across
  repeated `coagulation_step_gpu` calls. Tests should compare state snapshots
  after first and second valid calls and verify the second call does not reset to
  the same seed-derived state.
- **P3:** Run the P1/P2 tests against the implementation and extend coverage for
  any edge cases introduced by the chosen initialization mode. Preserve existing
  stochastic collision-rate tests and update them only if needed to remove manual
  seed increments.
- **P4:** Validate documentation links and any updated executable examples. If
  benchmark helper behavior changes, keep benchmark tests passing without
  requiring CUDA in normal CI.

## Test Locations

- Primary: `particula/gpu/kernels/tests/coagulation_test.py`
- Supporting device fixture: `particula/gpu/tests/cuda_availability.py`
- Deferred follow-up only: `particula/gpu/tests/benchmark_test.py`

## Coverage and Device Matrix

- Use the existing `device` fixture so tests run on Warp CPU and CUDA when
  available.
- Avoid exact stochastic sequence equality across devices; assert properties such
  as state changes, non-reset behavior, buffer identity, shape/device validation,
  and absence of mutation on invalid inputs.
- Run focused validation with:
  `pytest particula/gpu/kernels/tests/coagulation_test.py -q`
- Full validation remains the repository standard `pytest` and linting commands.
