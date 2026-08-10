# Risk Register

| Risk | Likelihood | Impact | Mitigation | Owner |
|---|---|---|---|---|
| Positional box index leaks into stream identity and breaks reorder invariance | High | High | Require unique logical IDs, key registry state by ID, and run permutation/addition regressions | E7-F8-P1/P5 owner |
| Coagulation and wall loss accidentally share or correlate a namespace | Closed | High | P3 publishes independently derived canonical-manifest sidecars and rejects cross-family identity/aliasing; regressions cover both arrays | E7-F8-P1/P3 owner |
| Disabled boxes still consume RNG inside a full-box kernel launch | Closed for P3 selection | High | P3 validates scheduler selection and invokes the unchanged kernel on selected one-box aliases only; disabled/skipped/no-work lane preservation is regression-covered | E7-F8-P3/P5 owner |
| Checkpoint omits or corrupts current stream state or logical-ID mapping | Closed for P6 | High | Schema-v3 owns complete canonical metadata and immutable words; fail-closed preflight and exact split-run regressions cover valid and malformed records | E7-F8-P6 owner |
| Reset occurs after partial validation or on a faulted session | Closed for P4 | High | P4 validates exact ACTIVE closed bindings, complete selectors, publication scope, and retained schemas before writers; negative tests preserve observable state | E7-F8-P4 owner |
| Seed mixer changes across Python/platform versions | Medium | High | Use specified fixed-width integer operations, never `hash()`, and freeze known-answer vectors | E7-F8-P1 owner |
| New abstraction adds hidden synchronization or per-step readback | Closed for P6 boundary | High | Capture synchronizes once and bulk-reads at most two streams; normal dispatch/reacquisition spies reject conversion, readback, synchronization, allocation, and reseeding | E7-F8 reviewer |
| Exact reproducibility is overclaimed across CPU, Warp CPU, and CUDA | Medium | Medium | Limit exact guarantee to compatible same-backend/device-class restarts; retain statistical cross-backend evidence | Documentation owner |
| Kernel interface changes regress direct callers | Low | High | Preserve existing defaults/signatures/returns and run direct coagulation/wall-loss suites | Adapter owners |
| Post-launch partial RNG advancement is mistaken for rollback-safe state | Medium | High | Fault session, prohibit checkpoint/continuation, and document discard semantics | Session owner |
