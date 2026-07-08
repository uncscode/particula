# Risk Register

| Risk | Impact | Likelihood | Mitigation |
|------|--------|------------|------------|
| RNG persistence changes expected stochastic sequences in existing tests | Medium | Medium | Update tests to assert semantic properties, not incidental exact sequences; document default seed behavior. |
| Mixed-scale rejection sampling requires more than characterization | High | Medium | Keep Epic C limited to correctness/hardening within existing physics; defer major algorithm redesign to a follow-up if needed. |
| Exporting kernels at top-level expands public API too broadly | Medium | Medium | Prefer documentation of `particula.gpu.kernels` unless promotion has clear user value and compatibility tests. |
| CUDA availability varies across contributors and CI | Medium | High | Use CPU Warp as required target and CUDA-if-available helpers with clean skips. |
| Statistical coagulation tests become flaky | High | Medium | Use seeded ensembles, tolerance bands, and aggregate assertions based on existing patterns. |
| Benchmark evidence is slow or environment-sensitive | Medium | Medium | Keep benchmarks opt-in and record qualitative limits with bounded reproduction commands. |
| Latent-heat conservation baseline couples to fragile integration state | Medium | Low | Use deterministic CPU fixtures, public APIs, and focused conservation assertions. |
| Epic plan tooling rejects phases for epics | Low | High | Capture work breakdown in child tracks and milestones; mention tooling limitation in completion summary. |

## Highest-Priority Mitigations

1. Land E3-F1 first so downstream stochastic tests and docs depend on stable RNG
   semantics.
2. Keep E3-F2 explicit about whether behavior is hardened or merely bounded.
3. Treat CUDA as additional evidence, not as a mandatory gate.
4. Ensure every public-facing example preserves explicit transfer ownership.
