# Testing Strategy

Every phase includes co-located tests with the implementation it changes. Tests
use the existing `*_test.py` convention and must not lower coverage thresholds.

## Per-Phase Testing Approach

- **P1:** Shipped in `particula/gpu/kernels/tests/coagulation_test.py`. Covers
  omitted `rng_states`, caller-provided `rng_states` reuse without implicit
  reset, explicit initialization via `initialize_rng=True`, wrong shape/device
  validation, and invalid inputs that must not mutate RNG buffers.
- **P2:** Shipped in `particula/gpu/kernels/tests/coagulation_test.py`. Covers
  the renamed repeated-valid-call regression
  `test_coagulation_step_gpu_persisted_rng_states_advance_across_repeated_valid_calls`,
  which snapshots initialized → post-first-call → post-second-call RNG state,
  plus the valid-then-invalid regression
  `test_coagulation_step_gpu_invalid_followup_preserves_advanced_rng_states`,
  which proves early `time_step` validation preserves an already-advanced
  caller-owned buffer.
- **P3:** Shipped in `particula/gpu/kernels/coagulation.py`,
  `particula/gpu/tests/benchmark_test.py`, and
  `particula/gpu/tests/benchmark_helpers_test.py`. Runtime validation continues
  to rely on the existing P1/P2 kernel regressions for omitted `rng_states`,
  caller-owned reuse, explicit reset, and invalid-follow-up preservation.
  Benchmark helper coverage now also includes
  `test_coagulation_scaling_reuses_persistent_rng_states_without_seed_drift`,
  which asserts repeated benchmark GPU steps keep a constant `rng_seed` while
  reusing the same persistent `rng_states` buffer.
- **P4:** Shipped as a docs-only validation phase. Readback and terminology
  checks now verify that `particula/gpu/kernels/coagulation.py`,
  `docs/Features/Roadmap/data-oriented-gpu.md`, and
  `docs/Features/data-containers-and-gpu-foundations.md` all describe the same
  seed-once contract: omitted `rng_states` allocate and seed per call,
  caller-owned persistent buffers are reused as-is, and explicit reset occurs
  only through `initialize_rng=True`. Focused runtime smoke validation remains
  `pytest particula/gpu/kernels/tests/coagulation_test.py -q -Werror` with no
  new production behavior tests required.

## Test Locations

- Primary: `particula/gpu/kernels/tests/coagulation_test.py`
- Supporting device fixture: `particula/gpu/tests/cuda_availability.py`
- Supporting benchmark coverage: `particula/gpu/tests/benchmark_helpers_test.py`
- Runtime benchmark path: `particula/gpu/tests/benchmark_test.py`

## Coverage and Device Matrix

- Use the existing `device` fixture so tests run on Warp CPU and CUDA when
  available.
- Avoid exact stochastic sequence equality across devices; assert properties such
  as state changes, non-reset behavior, buffer identity, shape/device validation,
  and absence of mutation on invalid inputs.
- Run focused validation with:
  `pytest particula/gpu/kernels/tests/coagulation_test.py -q -Werror`
- Full validation remains the repository standard `pytest` and linting commands.
