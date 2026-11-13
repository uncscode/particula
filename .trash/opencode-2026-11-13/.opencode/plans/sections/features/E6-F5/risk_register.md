# Risk Register

| Risk | Impact | Likelihood | Mitigation | Owner |
|------|--------|------------|------------|-------|
| Processes use different active predicates | High: inactive slots enter physics or capacity is misreported | Medium | Freeze one truth table; exact CPU/Warp parity and downstream contract tests | E6-F5 implementer |
| Contradictory partial state is treated as free | High: activation silently overwrites mass or charge | Medium | Classify only all-zero records as free; reject all mixed states before writes | E6-F5 implementer |
| Parallel enumeration is nondeterministic | High: CPU/GPU source-to-slot mapping diverges | Medium | Use deterministic box-local ascending scan and exact index tests | GPU implementer |
| Diagnostics mutate before later validation fails | High: callers observe a partial operation | Medium | Complete state/request/capacity/sidecar preflight before clearing outputs | API owner |
| Host reads are introduced for GPU convenience | Medium: hidden synchronization breaks resident workflows | Medium | Keep successful diagnostics device-resident and caller-owned | GPU reviewer |
| E6-F6 exhaustion policy leaks into activation | Medium: sequencing and defaults become ambiguous | Medium | E6-F5 reports/fails only; E6-F6 owns policy and recovery | E6 leads |
| Box-local scan scales poorly at extreme slot counts | Medium | Low | Keep bounded fixed-shape implementation; benchmark only in later performance work | Performance reviewer |
