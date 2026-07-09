# Documentation Updates

- `particula/gpu/kernels/coagulation.py` docstring for `coagulation_step_gpu` --
  updated to document how `rng_seed`, omitted `rng_states`, caller-provided
  `rng_states`, and keyword-only `initialize_rng` interact.
- `particula/gpu/kernels/tests/coagulation_test.py` -- test names and docstrings
  now serve as executable contract documentation for persisted caller-owned
  `rng_states`, including repeated valid calls and valid-then-invalid
  preservation.
- Broader GPU docs remain intentionally deferred in this phase:
  `docs/Features/Roadmap/data-oriented-gpu.md`,
  `docs/Features/data-containers-and-gpu-foundations.md`, and benchmark helper
  comments were not changed by issue #1237.
- `.opencode/plans/sections/features/E3-F1/` -- update phase status and lessons
  learned as implementation issues ship.

No README change is expected unless a later phase exposes a user-facing example
outside the low-level GPU kernel API.
