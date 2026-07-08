# Implementation Tasks

## Backend / GPU Kernel API

- [ ] In `particula/gpu/kernels/coagulation.py`, document the current
  `rng_seed` and `rng_states` behavior in the `coagulation_step_gpu` docstring
  before changing it.
- [ ] Choose the compatibility mechanism for explicit initialization, such as a
  keyword-only initialization mode or helper, without adding positional
  arguments.
- [ ] Refactor RNG setup so `_initialize_rng_states` is launched only when the
  contract says this call initializes the state.
- [ ] Keep `_validate_rng_states` and `_validate_device_match` checks before any
  RNG initialization launch.
- [ ] Preserve no-`rng_states` convenience calls by allocating a temporary state
  buffer and initializing it from `rng_seed`.
- [ ] Ensure caller-provided `rng_states` are advanced only by the coagulation
  kernel during valid repeated calls.
- [ ] Review `particula/gpu/tests/benchmark_test.py` for manual `rng_seed`
  increments and update to seed-once semantics if the benchmark passes a
  persistent `rng_states_buf`.

## Tooling / Tests

- [ ] Add `test_coagulation_step_gpu_advances_preallocated_rng_states` in
  `particula/gpu/kernels/tests/coagulation_test.py`.
- [ ] Add `test_coagulation_step_gpu_does_not_reinitialize_persisted_rng_states`
  covering two repeated calls with the same seed and persistent buffer.
- [ ] Add or update compatibility tests proving omitted `rng_states` still works
  for existing callers.
- [ ] Add invalid-input assertions that wrong environment, volume, shape, or
  device combinations leave `rng_states` unchanged.
- [ ] Use existing `device` fixture so tests run on Warp CPU and CUDA when
  available.

## Documentation

- [ ] Update `docs/Features/Roadmap/data-oriented-gpu.md` to replace the defect
  note with shipped seed-once behavior and graph-capture guidance.
- [ ] Update `docs/Features/data-containers-and-gpu-foundations.md` if the RNG
  state buffer ownership contract belongs in the transfer-boundary docs.
- [ ] Add or update any repeated timestep examples so they no longer imply
  callers must increment `rng_seed` manually when preserving `rng_states`.
