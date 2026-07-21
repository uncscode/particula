# Risk Register

| Risk | Likelihood | Impact | Mitigation | Owner |
|---|---|---|---|---|
| Dilution is incorrectly represented by reducing per-particle mass rather than concentration. | Medium | High | Snapshot mass/distribution fields; permit concentration-only writes; document the Cloud Chamber anti-pattern. | E6-F1 implementer |
| Particle representation volume/count storage is confused with physical concentration. | Medium | High | Test multiple representation types and non-unit volume; route updates through reviewed container semantics. | Particle API reviewer |
| Gas and particle paths use different finite-step formulas or substep rules. | Medium | High | Freeze one P1 contract, share one concentration factor/calculation path, and assert strategy/runnable agreement. | Physics reviewer |
| Large `alpha * dt` creates negative concentration or unstable output. | Medium | High | Select and document a nonnegative finite-step rule in P1; validate computed outputs before commit. | Numerical reviewer |
| Updating particle state succeeds before a gas validation/write failure. | Low | High | Precompute and validate both outputs before either write; snapshot complete state in failure tests. | E6-F1 implementer |
| Validation changes break existing free-function scalar/array callers. | Medium | Medium | Preserve supported behavior, add regression tests, and separate process validation when helper compatibility requires it. | API reviewer |
| Runnable substeps double-split time inside `RunnableSequence`. | Medium | Medium | Follow existing process conventions and test direct execution versus sequence execution explicitly. | Runnable maintainer |
| Public API expands unnecessarily into builders/factories or inlet-source modeling. | Low | Medium | Keep the issue-bounded strategy/runnable surface; resolve additions only through an open-question decision. | E6 owner |
| CPU semantics drift after E6-F2 parity is written. | Low | High | Treat this plan as the parity contract and require coordinated downstream test updates for semantic changes. | E6-F1/E6-F2 owners |
| Documentation implies GPU, scheduling, or performance support. | Low | Medium | State E6-F2 and Epic G boundaries in API docs and example review. | Documentation reviewer |
