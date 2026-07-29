# Success Criteria

- [x] P1 typed declarations accept exactly the supported closed-catalogue nodes
  and dependency pairs and reject duplicates, cycles, unavailable requirements,
  and malformed declarations before any runtime behavior (#1492).
- [x] P1 equivalent declarations normalize to identical node-ID and edge-pair
  ordering, independent of input declaration order; this is not a scheduler or
  topological execution order (#1492).
- [x] P1 remains unexported, pure, and backend-neutral: it does not import
  Warp/GPU modules, access resources, schedule, execute, or mutate state
  (#1492).
- [x] P2 resolves registration-independent lexical topological order and a
  declaration-only selected schedule with explicit/derived closure, reviewed
  direction policy, and no disabled-endpoint edges (#1493).
- [x] P2 remains direct-import-only and prelaunch-only: no package export,
  lifecycle/resource/backend import, launch, transfer, synchronization, or
  mutation occurs during schedule resolution (#1493).
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
