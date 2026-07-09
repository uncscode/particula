# Implementation Tasks

## Backend / GPU Kernel API

- [x] In `particula/gpu/kernels/coagulation.py`, document the current
  `rng_seed` and `rng_states` behavior in the `coagulation_step_gpu` docstring
  before changing it.
- [x] Choose the compatibility mechanism for explicit initialization as the
  keyword-only `initialize_rng` mode, without adding positional arguments.
- [x] Refactor RNG setup so `_initialize_rng_states` is launched only when the
  contract says this call initializes the state.
- [x] Keep `_validate_rng_states` and `_validate_device_match` checks before any
  RNG initialization launch.
- [x] Preserve no-`rng_states` convenience calls by allocating a temporary state
  buffer and initializing it from `rng_seed`.
- [x] Lock the caller-visible contract that provided `rng_states` are reused
  without implicit reset unless `initialize_rng=True` is passed.
- [x] Treat caller-provided `rng_states` as the bypass for automatic
  initialization unless the caller explicitly invokes the initializer/reset path.
- [x] Review `particula/gpu/tests/benchmark_test.py` for manual `rng_seed`
  increments and update to seed-once semantics when the benchmark passes a
  persistent `rng_states_buf`.

## Tooling / Tests

- [x] Add or update compatibility tests proving omitted `rng_states` still works
  for existing callers.
- [x] Rename the repeated valid-call regression to
  `test_coagulation_step_gpu_persisted_rng_states_advance_across_repeated_valid_calls`
  so the persisted caller-owned buffer contract is explicit.
- [x] Add a valid-then-invalid regression proving an already-advanced
  caller-owned `rng_states` buffer is preserved when a follow-up call fails
  early on `time_step` validation.
- [x] Add invalid-input assertions that wrong environment, volume, shape, or
  device combinations leave `rng_states` unchanged.
- [x] Cover explicit reset via `initialize_rng=True` and wrong-shape /
  wrong-device `rng_states` validation in
  `particula/gpu/kernels/tests/coagulation_test.py`.
- [x] Use existing `device` fixture so tests run on Warp CPU and CUDA when
  available.

## Documentation

- [ ] Update `docs/Features/Roadmap/data-oriented-gpu.md` to replace the defect
  note with shipped seed-once behavior and graph-capture guidance.
- [ ] Update `docs/Features/data-containers-and-gpu-foundations.md` if the RNG
  state buffer ownership contract belongs in the transfer-boundary docs.
- [ ] Add or update any repeated timestep examples so they no longer imply
  callers must increment `rng_seed` manually when preserving `rng_states`.
