# Risk Register

| Risk | Likelihood | Impact | Mitigation | Owner |
|------|------------|--------|------------|-------|
| Integrated assertions confuse conservative transfers with physical removal by dilution or wall loss | Medium | High | Use per-process snapshots and independent budgets; never apply one aggregate conservation assertion to every process | Physics/test maintainer |
| Stochastic wall-loss or coagulation tests become flaky | Medium | High | Use deterministic coefficient checks plus predeclared statistical bounds and deterministic seeds; do not compare exact CPU/Warp streams | GPU test maintainer |
| Example introduces hidden orchestration or implies production scheduling | Medium | High | Call direct functions visibly, label order illustrative, and assert no intermediate conversion; reserve scheduler APIs for Epic G | GPU API maintainer |
| Optional Warp/CUDA environments make CI behavior ambiguous | Medium | Medium | Require Warp CPU when Warp is installed and use explicit clean skips for unavailable CUDA | CI maintainer |
| Upstream API names or sidecars change before E6-F9 starts | Medium | Medium | Begin only after E6-F1-F8 ship; bind fixtures to their documented public/concrete-module contracts and update cross-links during P1 | E6 owner |
| Closeout is published while an upstream feature or exit criterion is incomplete | Low | High | Gate roadmap status and parent lifecycle changes on all child statuses, focused tests, documentation checks, and plan validation | Epic owner |
| Roadmap language accidentally activates Epic G implementation | Medium | High | Keep backend selection, high-level runnables, scheduling, resident loops, and transport explicitly excluded in every public closeout artifact | Documentation owner |
| Large integrated fixtures obscure failure diagnosis | Medium | Medium | Retain feature-local tests and use small deterministic scenarios with assertions after each direct step | Test maintainer |
