# Success Criteria

## Pass / Fail Criteria

- [ ] `coagulation_step_gpu` no longer overwrites caller-provided persisted
  `rng_states` on every repeated timestep.
- [ ] Repeated calls with the same `rng_seed` and the same initialized
  `rng_states` buffer advance the state and do not require manual seed
  increments to avoid correlated draws.
- [ ] Calls that omit `rng_states` remain source-compatible and initialize usable
  RNG state for that call.
- [ ] Invalid input combinations still fail before mutating particles,
  `rng_states`, collision buffers, or launching downstream kernels.
- [ ] Tests pass on Warp CPU and CUDA-if-available using the existing device
  fixture.
- [ ] Documentation explains seed-once usage and graph-capture setup caveats.

## Metrics

| Metric | Baseline | Target | Source |
|--------|----------|--------|--------|
| Preallocated `rng_states` overwritten per call | Yes | No | Coagulation regression tests |
| Manual seed increments needed for repeated persisted-state calls | Yes | No | Repeated-call RNG tests |
| Invalid-input RNG mutation | Must be none | None preserved | Existing and new validation tests |
| Warp device coverage | CPU plus optional CUDA fixture exists | New tests use same fixture | `coagulation_test.py` |
| Documentation of graph-capture caveat | Roadmap defect note only | Usage guidance updated | Docs diff |
