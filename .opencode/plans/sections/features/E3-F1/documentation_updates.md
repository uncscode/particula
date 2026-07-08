# Documentation Updates

- `docs/Features/Roadmap/data-oriented-gpu.md` -- update the RNG
  re-initialization section to describe the shipped seed-once behavior, persistent
  `rng_states` ownership, and graph-capture caveat.
- `docs/Features/data-containers-and-gpu-foundations.md` -- add a concise note if
  RNG state buffer ownership belongs with the explicit CPU/GPU transfer boundary
  contract.
- `particula/gpu/kernels/coagulation.py` docstring for `coagulation_step_gpu` --
  document how `rng_seed`, omitted `rng_states`, provided `rng_states`, and any
  explicit initialization option interact.
- `particula/gpu/tests/benchmark_test.py` comments or benchmark helper docs --
  remove or explain manual `rng_seed` increments when persistent `rng_states_buf`
  is used.
- `.opencode/plans/sections/features/E3-F1/` -- update phase status and lessons
  learned as implementation issues ship.

No README change is expected unless the implementation exposes a user-facing
example outside the low-level GPU kernel API.
