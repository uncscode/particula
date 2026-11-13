# E2-F7 Risk Register

| Risk | Impact | Likelihood | Mitigation | Owner |
| --- | --- | --- | --- | --- |
| Explicit GPU path lacks gas concentration updates, making conservation metrics incomplete. | Stiffness results could be misread as full gas-particle coupling evidence. | High | Label particle-only metrics clearly and compare with CPU conservation references. | E2-F7 implementer |
| E2-F2/E2-F3 environment containers are not available before this feature starts. | Per-box T/P contracts may remain provisional. | Medium | Use scalar compatibility path and document migration hooks for `WarpEnvironmentData`. | E2-F7 implementer with E2-F2/E2-F3 owners |
| E2-F6 precision study changes dtype assumptions. | Stiffness bounds may differ under fp32 or mixed precision. | Medium | Treat fp64 as reference and mark non-fp64 results as follow-up unless E2-F6 is complete. | E2-F7 implementer with E2-F6 owner |
| Semi-implicit/asymptotic candidate requires more production code than planned. | Feature could exceed issue-sized phases. | Medium | Keep candidate evaluation as prototype/test-only if needed and document implementation follow-up. | E2-F7 implementer |
| Hard clamps and in-place writes produce poor gradients. | Recommended foundation may not support future optimization. | Medium | Document clamp gradient strategy and prefer deterministic guarded updates. | Autodiff reviewer |
| Random staggered or dynamic adaptive loops appear attractive for stability. | Recommendation could violate graph-capture/autodiff constraints. | Medium | Reject stochastic/dynamic-loop paths for gradient/capture use; allow only fixed deterministic variants. | E2-F7 implementer |
| Slow sweeps become too expensive for CI. | Tests may be flaky or skipped, reducing confidence. | Low | Keep fast representative tests in CI and move broad sweeps to slow benchmarks or docs scripts. | Test owner |
