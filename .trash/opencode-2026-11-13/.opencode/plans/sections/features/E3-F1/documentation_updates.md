# Documentation Updates

- `particula/gpu/kernels/coagulation.py` docstring for `coagulation_step_gpu` --
  shipped as the authoritative public contract for `rng_seed`, omitted
  `rng_states` convenience allocation, caller-owned persistent `rng_states`
  reuse, and `initialize_rng=True` reset semantics.
- `docs/Features/Roadmap/data-oriented-gpu.md` -- now documents the shipped
  seed-once repeated-call contract, caller ownership of persistent
  `rng_states`, and graph-capture guidance to initialize or reset before the
  repeated loop or capture step.
- `docs/Features/data-containers-and-gpu-foundations.md` -- now states that
  coagulation `rng_states` are caller-owned Warp-resident sidecar state and are
  not part of `ParticleData`, `GasData`, `EnvironmentData`, or Warp container
  schemas.
- `particula/gpu/tests/benchmark_test.py` and
  `particula/gpu/tests/benchmark_helpers_test.py` -- verified to remain aligned
  with the shipped constant-seed persistent-buffer guidance; no new runtime or
  benchmark-behavior change was required in this docs-only phase.
- `.opencode/plans/sections/features/E3-F1/` -- updated to mark P4 shipped and
  capture the final docs-only handoff for issue #1239.

No README change is expected unless a later phase exposes a user-facing example
outside the low-level GPU kernel API.
