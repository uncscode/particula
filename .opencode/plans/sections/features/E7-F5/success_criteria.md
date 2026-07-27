# Success Criteria

- [ ] A typed graph accepts exactly supported process/update declarations and
  rejects duplicates, cycles, unavailable capabilities, and malformed state
  before any process launch.
- [ ] Equivalent declarations resolve to the same canonical order regardless of
  registration or mapping iteration order.
- [ ] Condensation, Brownian coagulation, dilution, wall loss, and nucleation
  execute through their owning adapters/direct boundaries without physics rewrites.
- [ ] Environment changes precede vapor-pressure and saturation refresh; every
  consumer observes current derived thermodynamic and gas state.
- [ ] Repeated normal timesteps perform no bulk conversion, host readback,
  checkpoint/finalize call, implicit synchronization, or silent fallback.
- [ ] Resident container, array, sidecar, output, and RNG identities and fixed
  shapes remain stable across successful timesteps.
- [ ] Prelaunch failures preserve state; post-launch uncertainty faults the
  session and retains the original exception without rollback claims.
- [ ] Warp CPU complete-loop tests pass with explicit deterministic tolerances,
  conservation checks, and independent-box equivalence; optional CUDA skips cleanly.
- [ ] Documentation states scope and limitations from issue #1451 and does not
  absorb T7/T8/T9, Epic H, or Epic I work.

## Metrics

| Metric | Baseline | Target | Source |
|--------|----------|--------|--------|
| Canonical supported processes in one step | Illustrative direct calls only | 5 | Scheduler integration tests |
| Intermediate bulk transfers/syncs per normal GPU timestep | Caller-dependent | 0 | Conversion/sync spies |
| Stale thermodynamic consumer paths | Possible in ad hoc order | 0 | Graph/order tests |
| Registration-order variants yielding identical order | Not defined | 100% | Process-graph parametrization |
| Changed-module coverage | N/A | >=80% | pytest-cov |
| Required routine GPU backend | None integrated | Warp CPU | CI/focused tests |
