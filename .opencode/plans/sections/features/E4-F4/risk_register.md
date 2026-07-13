# Risk Register

| Risk | Impact | Likelihood | Mitigation | Owner |
|---|---|---|---|---|
| Latent and pressure-delta paths use different surface pressure | High | Medium | Compute once and parity-test | P2 owner |
| Energy uses proposed rather than clamped transfer | High | Medium | Bound once before application/accounting | P3 owner |
| Aggregation semantics remain ambiguous | High | Medium | Specify/test whole-call box/species totals | Feature owner |
| Return change breaks existing callers | High | Low | Keyword-only opt-in output; preserve default return | P2/P3 owner |
| Stale temperature or pressure enters correction | High | Medium | Refresh each substep | P2 owner |
| Scratch allocation harms graph readiness | Medium | Medium | Caller-owned fixed shapes and reuse tests | P1 owner |
| Loose physics tolerance masks energy defects | High | Medium | Separate tight identity assertions | Test owner |
| Feature overclaims gas conservation | Medium | Medium | Gate system claims on E4-F5 | Feature owner |
| CUDA is unavailable in CI | Medium | Medium | Require Warp CPU; clean optional skip | Test owner |
