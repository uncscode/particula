# Risk Register

| Risk | Likelihood | Impact | Mitigation / Done Signal | Owner |
|------|------------|--------|--------------------------|-------|
| The abstraction is too narrow for condensation, coagulation, or resident state | Medium | High | Review P1-P3 protocols against direct-kernel signatures and E7-F2/F3/F4 needs; prove extension with fake adapters before exports | E7-F1 |
| The abstraction becomes a second scheduler or rewrites process physics | Medium | High | Limit T1 to one validated dispatch; retain `RunnableSequence` as CPU reference and defer ordering to E7-F5 | E7-F1 |
| Importing public selection types eagerly imports Warp | Medium | High | Keep `particula.execution` dependency-neutral and add fresh-process tests with Warp blocked | E7-F1 |
| "Automatic" selection introduces silent fallback or hidden transfer | Medium | Critical | Require an explicit backend/device, fail closed, spy on conversion calls, and leave transition policy to E7-F6 | E7-F1 / E7-F6 |
| State/result types obscure in-place mutation or identity | Medium | High | Encode ownership and mutation in typed results; add identity tests for CPU and fake future-GPU adapters | E7-F1 |
| Public names are frozen before downstream adapters validate them | Medium | Medium | Export the minimum vocabulary, mark extension internals private, and make E7-F6 the stability-policy gate | E7-F1 / E7-F6 |
| Registry mutation permits collisions or order-dependent behavior | Low | High | Use typed keys, reject duplicates atomically, and test deterministic lookup independent of registration order | E7-F1 |
| Validation changes existing runnable behavior | Low | High | Keep direct APIs untouched; validation applies only at the new boundary and existing runnable regressions must pass | E7-F1 |
| Optional CUDA availability makes routine tests flaky | Low | Medium | Use CPU and fake adapters for T1; CUDA is optional and cleanly skipped | E7-F1 |
| Later tracks bypass the root contract | Medium | High | Document E7's dependency chain and require E7-F2/F3/F4/F6 plans to consume the published capability/state seams | E7 maintainers |
