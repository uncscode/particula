# Risk Register

| Risk | Likelihood | Impact | Mitigation / Done Signal | Owner |
|------|------------|--------|--------------------------|-------|
| A size-extrema heuristic underestimates a charged rate for mixed signs or magnitudes | Medium | Critical | Start with the exhaustive unique-active-pair maximum; assert every deterministic pair is bounded; require proof before optimization | E5-F3-P1 |
| Combined execution launches separate Brownian and charged samplers, duplicating opportunities and RNG advancement | Medium | High | Keep one resolved mask, one total rate, one acceptance draw, one active-set removal, and explicit RNG-pass tests | E5-F3-P3 |
| Charged helper/model semantics drift from E5-F2's approved subset | Medium | High | Consume E5-F2 identifiers/helpers directly; reject unregistered variants; cross-reference independent CPU fixtures | E5-F3-P1/P2 |
| Non-finite rates or majorants produce invalid acceptance ratios | Low | High | Host preflight caller inputs; device-side finite/non-negative guards; deterministic extreme fixtures and zero-rate tests | E5-F3-P1 |
| Accepted execution loses charge or leaves donor state active | Low | Critical | Mitigated in P2: shared E5-F2 merge is covered by per-box mass/charge conservation and donor clearing tests; retain for P3 combined coverage | E5-F3-P2/P3 |
| Capability validation or allocations mutate output/RNG state before failure | Low | High | Mitigated in P2: charged-only forced finite-charge preflight and invalid-input snapshots occur before output/RNG work; retain for P3 | E5-F3-P2/P3 |
| Exhaustive charged-majorant scan adds O(n²) work | High | Medium | Accept correctness-first cost in bounded T3 scope, document it, reuse fixed buffers, and defer optimization until a safe tighter proof exists | E5-F3-P1 / future performance owner |
| Stochastic tests are flaky or require exact CPU/Warp replay | Medium | Medium | Mitigated for P2 with fixed fresh seeds and independent aggregate bounds; retain the same approach for P3 | E5-F3-P2/P3 |
| CUDA-only behavior blocks release or hides Warp CPU regression | Low | Medium | Require Warp CPU when installed; treat CUDA as optional additive evidence with clean skips | E5-F3-P2/P3 |
| Documentation overstates support before E5-F6/F7/F9 | Medium | Medium | Publish only charged and Brownian-plus-charged direct low-level support and list deferred owners explicitly | E5-F3-P4 |
