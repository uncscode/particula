# Risk Register

| Risk | Impact | Likelihood | Mitigation | Owner |
| --- | --- | --- | --- | --- |
| Scalar compatibility is accidentally broken by signature changes. | Existing GPU users fail at import or call time. | Medium | Keep positional scalar parameters stable or add keyword-only environment support; preserve current scalar tests. | E2-F5 implementer |
| Environment container details from E2-F2 change during implementation. | Helpers target the wrong fields or names. | Medium | Confirm E2-F2 shipped schema before coding; isolate field access in one helper. | E2-F5 implementer |
| Gas/environment ownership boundary is blurred. | Future APIs duplicate state or mutate gas data incorrectly. | Low | Follow E2-F3 decisions; do not add T/P to `GasData`. | Reviewer |
| Per-box arrays introduce avoidable GPU allocation overhead for scalar callers. | Scalar workflows regress in performance. | Medium | Use minimal broadcast allocation, consider caching only if profiling later proves necessary, and avoid hidden CPU/GPU round trips. | Implementer |
| Coagulation stochastic tests become flaky when comparing scalar and env paths. | CI failures despite correct behavior. | Medium | Reuse deterministic seeds and existing statistical tolerances; compare invariants where exact equality is inappropriate. | Test author |
| Device mismatch errors surface inside Warp kernels instead of validation. | Failures are hard to debug. | Low | Validate devices before launch using existing helper style. | Implementer |
