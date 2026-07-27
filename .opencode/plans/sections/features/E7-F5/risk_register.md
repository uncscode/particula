# Risk Register

| Risk | Likelihood | Impact | Mitigation | Owner |
|------|------------|--------|------------|-------|
| Canonical order encodes a scientifically wrong gas or thermodynamic dependency | Medium | High | Model invalidation/consumption edges explicitly; compare against CPU/NumPy references and assert call order | E7-F5 |
| Scheduler duplicates process physics or diverges from direct contracts | Medium | High | Thin adapters only; exact-call spies and owning-module regression tests | E7-F5 + process owners |
| A validation failure occurs after mutation begins | Medium | High | Validate complete graph/state/resources before `begin_step()`; fault on uncertain launched work | E7-F4/E7-F5 |
| Hidden transfer, host readback, or synchronization enters the normal loop | Medium | High | Conversion/sync/`.numpy()` spies across repeated timesteps | E7-F5 |
| RNG is silently reinitialized or process reordering changes ownership | Medium | High | Reuse E7-F3/E7-F4 persistent resources; defer final stream policy to E7-F8 | E7-F3/E7-F8 |
| Update APIs permit invalid values, aliases, or cross-device arrays | Low | High | Fixed-schema preflight and rejected-call immutability tests | E7-F5 |
| Scope expands into transport, optimization, or final diagnostics product | Medium | Medium | Enforce issue #1451 track boundaries and explicit E7-F7/F8/F9, Epic H/I handoffs | E7 owner |
| Optional CUDA behavior masks Warp CPU regressions | Low | Medium | Warp CPU is mandatory baseline; CUDA rows only supplement and skip cleanly | Test owner |
