# Success Criteria

## Pass / Fail Criteria

- [x] `coagulation_step_gpu` documents and ships the keyword-only
  `initialize_rng` contract for omitted vs caller-provided `rng_states`.
- [x] Calls that omit `rng_states` remain source-compatible and initialize usable
  RNG state for that call.
- [x] Caller-provided `rng_states` are not implicitly reset when
  `initialize_rng=False`, even if the same `rng_seed` is repeated.
- [x] Invalid input combinations still fail before mutating particles,
  `rng_states`, collision buffers, or launching downstream kernels.
- [x] Explicit reset through `initialize_rng=True` is covered at the public API.
- [x] Repeated valid-call coverage proves the same caller-owned `rng_states`
  buffer advances across successive valid calls and is not restored to the
  original seed-derived state when the same `rng_seed` is reused.
- [x] A valid-then-invalid regression proves an already-advanced caller-owned
  `rng_states` buffer is preserved when a follow-up call fails early on
  `time_step` validation.
- [x] The shipped regression coverage uses the existing Warp CPU / CUDA-if-
  available device fixture in `coagulation_test.py`.
- [ ] Documentation explains broader seed-once usage and graph-capture setup
  caveats in follow-up docs work.

## Metrics

| Metric | Baseline | Target | Source |
|--------|----------|--------|--------|
| Omitted vs provided `rng_states` contract ambiguity | Present | Resolved in public API tests | Coagulation compatibility tests |
| Implicit reset of caller-provided `rng_states` with repeated seed | Present | No implicit reset when `initialize_rng=False`, including repeated valid-call regression coverage | Repeated-call RNG tests |
| Invalid-input RNG mutation | Must be none | None preserved after valid-then-invalid follow-up | Existing and new validation tests |
| Warp device coverage | CPU plus optional CUDA fixture exists | Shipped repeated-call regressions use same fixture | `coagulation_test.py` |
| Documentation of graph-capture caveat | Roadmap defect note only | Deferred to P4 | Docs diff |
