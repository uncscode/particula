# Risk Register

| Risk | Likelihood | Impact | Mitigation | Owner |
|------|------------|--------|------------|-------|
| CPU, uncaptured, and captured fixtures drift apart | Medium | High | Construct all paths from one immutable scenario and assert configuration signatures | P1 owner |
| Production helpers contaminate the independent oracle | Medium | High | Restrict oracle code to NumPy/CPU equations and review imports explicitly | P1 reviewer |
| CUDA absence masks capture regressions | High on CI | Medium | Keep hardware-free lifecycle tests, require clean skip text, and record optional qualified-device runs | P3 owner |
| Stochastic checks become flaky or relax conservation | Medium | High | Separate aggregate sigma bounds from deterministic parity and tight inventory checks | P4 owner |
| Replay tests synchronize or allocate inside the measured boundary | Medium | High | Spy on allocation, transfer, readback, and synchronization; synchronize only at assertion boundaries | P2/P3 owner |
| Rejection tests mutate state before failing | Low | High | Snapshot accessible state and launch count before every preflight rejection row | P4 owner |
| Parent E8 child table conflicts with the E8-F5 orchestrator assignment | High | Medium | Preserve this handoff, flag the mismatch, and resolve IDs before issue generation | Plan owner |
| Validation scope expands into benchmarking or API redesign | Medium | Medium | Defer scaling, memory, profiling, examples, and contract changes to sibling tracks | E8 owner |
