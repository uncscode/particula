# Success Criteria

- [ ] Backend selection is located in a documented, typed, separate execution
  context rather than embedded in strategies, builders, or `RunnableABC`.
- [ ] The capability matrix deterministically accepts every declared supported
  combination and rejects every unsupported combination before mutation.
- [ ] Backend, device, state ownership, mutation, identity, and return behavior
  are represented in types and documented for downstream adapters.
- [ ] The CPU adapter preserves existing process physics, delegates exact time
  and substep inputs, and retains current `Aerosol` return semantics.
- [ ] No selection or failure path performs hidden CPU/Warp transfer, catches a
  runtime error to retry another backend, or requires Warp for CPU-only import.
- [ ] Public exports are deliberate and contract-tested; direct kernels and
  concrete sidecars are not promoted through the new surface.
- [ ] E7-F6 can extend policy and E7-F2/F3/F4 can register implementations
  without changing the T1 request, state, or result vocabulary.
- [ ] Existing runnable and GPU export regressions pass unchanged.
- [ ] Every phase includes tests, changed execution code has at least 80%
  coverage, Ruff/mypy pass, and no repository threshold is lowered.
- [ ] Backend/device selection behavior and limitations are published and
  `mkdocs build --strict` succeeds.

## Metrics

| Metric | Baseline | Target | Source |
|--------|----------|--------|--------|
| User-facing typed selection boundaries | 0 | 1 separate context API | Export tests |
| Implicit fallback/transfer paths | Not centrally constrained | 0 | Negative tests and transfer spies |
| Declared capability matrix cases tested | 0 | 100% of T1 declarations | Parameterized unit tests |
| CPU adapter dispatches per request | N/A | Exactly 1 | Adapter spy tests |
| CPU-only import success without Warp | Ad hoc APIs only | 100% | Fresh-process test |
| Changed-module statement coverage | N/A | >=80% | pytest-cov |
| Existing process physics changed | 0 desired | 0 | Diff and runnable regressions |
