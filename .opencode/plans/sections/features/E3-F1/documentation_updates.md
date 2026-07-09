# Documentation Updates

- `particula/gpu/kernels/coagulation.py` docstring for `coagulation_step_gpu` --
  updated to document how `rng_seed`, omitted `rng_states`, caller-provided
  `rng_states`, and keyword-only `initialize_rng` interact.
- Broader GPU docs remain intentionally deferred in this phase:
  `docs/Features/Roadmap/data-oriented-gpu.md`,
  `docs/Features/data-containers-and-gpu-foundations.md`, and benchmark helper
  comments were not changed.
- `.opencode/plans/sections/features/E3-F1/` -- update phase status and lessons
  learned as implementation issues ship.

No README change is expected unless a later phase exposes a user-facing example
outside the low-level GPU kernel API.
