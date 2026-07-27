# Success Criteria

- [ ] Explicit CPU and Warp Brownian requests resolve through the E7-F1
  execution context under E7-F6 policy and invoke exactly one selected adapter.
- [ ] The CPU adapter delegates exact `time_step`/`sub_steps` and preserves the
  existing returned `Aerosol` behavior without changing Brownian physics.
- [ ] The Warp adapter delegates to `coagulation_step_gpu`, preserves particle
  and supplied output identities, and reports collision diagnostics without
  hidden host readback.
- [ ] A caller-owned `(n_boxes,)` `wp.uint32` RNG buffer is initialized once,
  advances across repeated steps, and resets only on an explicit request.
- [ ] Unsupported mechanisms/distributions/devices and malformed state fail
  clearly before adapter invocation or mutation where selection owns preflight.
- [ ] No selected path performs hidden conversion, restore, synchronization,
  silent fallback, runtime retry on CPU, or per-step reseeding.
- [ ] Warp CPU fixtures satisfy documented deterministic tolerances, aggregate
  stochastic bounds, accepted-pair validity, and mass/charge conservation;
  optional CUDA rows skip cleanly.
- [ ] Validation failures preserve caller state before launch, while runtime
  failure documentation makes the post-launch no-rollback boundary explicit.
- [ ] Public exports are deliberate and contract-tested; direct mechanism and
  scratch internals are not accidentally promoted.
- [ ] Every phase includes tests, changed code has at least 80% coverage, Ruff
  and mypy pass, existing kernel/runnable regressions pass, and no threshold is
  lowered.
- [ ] User documentation covers usage, ownership, support, errors, limitations,
  and E7-F5/E7-F8 handoffs; `mkdocs build --strict` passes.

## Metrics

| Metric | Baseline | Target | Source |
|--------|----------|--------|--------|
| Backend-selected Brownian workflows | 0 | CPU and Warp paths | Adapter contract tests |
| Hidden transfer/synchronization/fallback calls per step | Caller-managed/ad hoc | 0 | Spies and negative tests |
| Persistent RNG reseeds during normal repeated calls | Easy to misuse per call | 0 | RNG progression tests |
| Supplied particle/output/RNG identity retained | Direct API only | 100% | Identity assertions |
| Declared capability cases exercised | 0 selected cases | 100% | Parameterized matrix tests |
| Conservation failures in supported fixtures | N/A | 0 | Warp CPU invariant tests |
| Exact CPU/Warp stochastic trajectory requirement | Not supported | 0 required | Statistical test design |
| Changed-module statement coverage | N/A | >=80% | pytest-cov |
