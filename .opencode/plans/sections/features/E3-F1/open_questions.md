# Open Questions

Status: reviewed and answered on 2026-07-08; updated after P1 shipped.

## Resolved Decisions

- Add a keyword-only `initialize_rng: bool = False` flag on
  `coagulation_step_gpu` rather than a new positional argument or standalone
  public helper. This preserves positional compatibility while making reset
  behavior explicit at the entrypoint that owns the launch.
- Do not add `rng_states` to the return tuple. Callers should retain the buffer
  they pass, matching the current Warp ownership pattern and avoiding a broader
  API shape change.
- Treat caller-provided `rng_states` as caller-owned state: validation happens
  first, no implicit reset occurs when `initialize_rng=False`, and explicit
  reset uses `initialize_rng=True` with `rng_seed`.
- Keep P2 as a test-only regression phase. The shipped follow-up for issue
  #1237 clarified the persisted-buffer contract through test names, docstrings,
  and assertions rather than changing runtime semantics again.
- Benchmark code should switch to seed-once semantics for correctness-focused
  runs in a later phase. Historical comparability can be preserved by recording
  `rng_seed` and reset behavior explicitly in benchmark metadata.
