# Open Questions

Status: reviewed and answered on 2026-07-08.

## Resolved Decisions

- Add a documented initializer/reset helper for `rng_states` instead of a new
  positional argument or ambiguous boolean flag on `coagulation_step_gpu`. This
  preserves existing positional compatibility and makes repeatable-sequence
  behavior explicit.
- Do not add `rng_states` to the return tuple. Callers should retain the buffer
  they pass, matching the current Warp ownership pattern and avoiding a broader
  API shape change.
- Treat legacy zeroed `rng_states` as invalid unless callers explicitly run the
  initializer/reset helper. Add a migration note explaining that caller-owned
  state buffers are no longer implicitly reseeded on every step.
- Benchmark code should switch to seed-once semantics for correctness-focused
  runs. Historical comparability can be preserved by recording `rng_seed` and
  reset behavior explicitly in benchmark metadata.
