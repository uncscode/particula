# Risk Register

| Risk | Likelihood | Impact | Mitigation | Owner |
|------|------------|--------|------------|-------|
| A host validation/readback or lazy allocation remains in prepared enqueue and invalidates CUDA capture | Medium | Critical | E8-F2 separation plus forbidden-operation spies; fail capture without publishing a handle | E8-F4 P2 |
| Signature omits a nested resource or process identity and a stale graph launches | Medium | Critical | Consume E8-F1 canonical signature and E8-F3 closed inventory; parametrically replace every category and assert zero launches | E8-F4 P3 |
| Capture cleanup is attempted twice or masks the original enqueue failure | Medium | High | Track active-capture ownership explicitly; consume once; retain dual failures in an exception group | E8-F4 P1/P2 |
| Replay failure leaves graph and resident lifecycle inconsistent | Medium | Critical | Reuse resident writer-may-have-launched handling and test graph/session faulting together | E8-F4 P4 |
| Mutable scientific state is mistaken for structural drift, forcing recapture each step | Medium | High | Signature metadata/identities only; tests advance concentrations, slots, diagnostics, and RNG words under stable arrays | E8-F4 P3 |
| Changed scalar controls are captured by Python value rather than pinned device storage | Medium | High | E8-F2 preparation explicitly classifies structural scalars versus device controls; reject unsupported control drift | E8-F2/E8-F4 |
| CUDA-only tests are skipped everywhere and native behavior remains unverified | Medium | High | Hardware-independent fake-runtime contract tests are required; record CUDA availability and treat required native evidence as pass-or-clean-skip, not CPU fallback | E8-F4 P2/P5 |
| Stochastic differences are incorrectly judged by exact seed replay across devices | Medium | Medium | Separate deterministic parity, tight conservation, and aggregate stochastic bounds per testing guide | E8-F4 P5 |
| Graph handle is serialized or reused after checkpoint restart/device change | Low | Critical | Keep handle concrete-only and out of checkpoint schema; terminal/restart identity tests reject before launch | E8-F4 P4 |
| E8-F4 duplicates scheduler logic and captured/uncaptured physics diverge | Medium | Critical | One E8-F2 prepared enqueue is authoritative for both paths; compare launch traces and three-way outputs | E8-F4 P2/P5 |
| Scope expands into benchmarks, examples, memory, or profiling before runtime correctness is stable | Medium | Medium | Preserve handoffs to E8-F5 through E8-F8 and limit P5 to correctness/documentation contracts | Feature owner |
