# Open Questions

Status: reviewed and answered on 2026-07-09; updated through P4 shipment.

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
- Benchmark code now reuses a persistent `rng_states_buf` with a constant
  `rng_seed` across repeated coagulation steps. Helper regression coverage locks
  this path so per-step seed incrementation is not silently reintroduced while
  the same buffer is reused.
- Documentation now explicitly places coagulation `rng_states` outside the
  CPU/GPU container schemas. Persistent RNG state is caller-owned Warp-resident
  sidecar state, not a `ParticleData`, `GasData`, or `EnvironmentData` field.
