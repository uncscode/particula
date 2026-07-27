# Risk Register

| Risk | Likelihood | Impact | Mitigation | Owner |
|---|---|---|---|---|
| Positional box index leaks into stream identity and breaks reorder invariance | High | High | Require unique logical IDs, key registry state by ID, and run permutation/addition regressions | E7-F8-P1/P5 owner |
| Coagulation and wall loss accidentally share or correlate a namespace | Medium | High | Include frozen process ID in derivation and require separate arrays/known-answer vectors | E7-F8-P1/P3 owner |
| Disabled boxes still consume RNG inside a full-box kernel launch | Medium | High | Resolve enablement before launch or add a validated device mask that skips reads/writes; test state unchanged | E7-F8-P3/P5 owner |
| Checkpoint omits current stream state or logical-ID mapping | Medium | High | Version a complete stream manifest and compare split versus uninterrupted runs exactly | E7-F8-P6 owner |
| Reset occurs after partial validation or on a faulted session | Low | High | Validate complete target set/lifecycle first; snapshot all state in negative tests | E7-F8-P4 owner |
| Seed mixer changes across Python/platform versions | Medium | High | Use specified fixed-width integer operations, never `hash()`, and freeze known-answer vectors | E7-F8-P1 owner |
| New abstraction adds hidden synchronization or per-step readback | Medium | High | Keep state device-resident and enforce conversion/sync/`.numpy()` spies | E7-F8 reviewer |
| Exact reproducibility is overclaimed across CPU, Warp CPU, and CUDA | Medium | Medium | Limit exact guarantee to compatible same-backend/device-class restarts; retain statistical cross-backend evidence | Documentation owner |
| Kernel interface changes regress direct callers | Low | High | Preserve existing defaults/signatures/returns and run direct coagulation/wall-loss suites | Adapter owners |
| Post-launch partial RNG advancement is mistaken for rollback-safe state | Medium | High | Fault session, prohibit checkpoint/continuation, and document discard semantics | Session owner |
