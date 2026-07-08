# Risk Register

| Risk | Impact | Likelihood | Mitigation | Owner |
|------|--------|------------|------------|-------|
| Ambiguous behavior for zeroed caller-provided `rng_states` after reinit is removed | Existing callers may have relied on implicit per-call seeding | Medium | Define an explicit compatibility contract in P1 and provide a clear initialization mode or helper | Implementor |
| Hidden host inspection of GPU buffers to infer initialization | Adds synchronization overhead and violates explicit transfer-boundary patterns | Low | Do not inspect buffer contents on host; make initialization caller/API controlled | Implementor |
| Exact random sequence assertions are flaky across Warp CPU/CUDA | Tests could fail on valid device differences | Medium | Assert state advancement/non-reset properties rather than exact sequence equality | Tester |
| Validation order regression mutates RNG before raising | Invalid calls could corrupt long-running simulations | Medium | Preserve existing validation-before-mutation tests and add RNG-specific invalid-input checks | Implementor/Tester |
| Graph-capture guidance remains unclear | Users may still capture initialization and freeze repeated seeds | Medium | Document setup outside capture and repeated calls inside capture in P4 | Documentation owner |
| Benchmark changes obscure performance comparisons | Seed behavior changes may alter stochastic workload distribution | Low | Keep benchmark intent documented and avoid using performance numbers as correctness tests | Benchmark maintainer |
