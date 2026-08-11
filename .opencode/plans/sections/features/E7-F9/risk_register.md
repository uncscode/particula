# Risk Register

| Risk | Likelihood | Impact | Mitigation | Owner | Status |
|------|------------|--------|------------|-------|--------|
| Closeout fixtures encode implementation details instead of public contracts | Medium | High | Invoke public `particula.execution` APIs; keep fixture helpers private | E7-F9 owner | Open |
| Diagnostics trigger host readback or synchronization each step | Medium | High | Same-device registered buffers, transfer spies, explicit observation boundaries | P1 owner | Open |
| Total-mass diagnostics mix concentration and extensive units | Medium | High | Freeze units and include per-box volume in independent NumPy oracle | P1/domain reviewer | Open |
| Checkpoint omits metadata or stream state | Medium | High | Versioned manifest audit against E7-F4/F8; round-trip and restart tests | P2 owner | Open |
| Full-loop parity masks process-level errors | Medium | High | Compare particle/gas/environment fields separately; preserve upstream contract tests | P3 owner | Open |
| Stochastic tests are flaky or overclaim cross-backend replay | Medium | High | Same-backend stream checks plus aggregate bounds; no CPU/CUDA exact claim | P4/P5 owner | Open |
| Transport conservation ignores open boundaries or volume changes | Medium | High | Extensive-amount oracle and explicit source/sink ledgers | P5 owner | Open |
| Optional CUDA behavior escapes validation | Low | High | Warp CPU required baseline; named CUDA rows with clean skip and recorded local evidence | E7-F9 owner | Open |
| Example reintroduces direct kernels or hidden convenience transfers | Medium | High | Concrete resident-scheduler-only example and upload/identity regression coverage | P6 owner | Mitigated by #1533 |
| Evidence marks Epic G complete before upstream tracks ship | Medium | High | Dependency gate and checklist require E7-F1 through E7-F8 shipped artifacts | Epic owner | Open |
| Work expands into graph capture/performance or autodiff | Medium | Medium | Reject against issue #1451 guardrails; defer to Epics H/I | Epic owner | Open |
