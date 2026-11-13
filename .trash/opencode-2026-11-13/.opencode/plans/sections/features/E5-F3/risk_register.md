# Risk Register

| Risk | Likelihood | Impact | Mitigation / Done Signal | Owner |
|------|------------|--------|--------------------------|-------|
| A size-extrema heuristic underestimates a charged rate for mixed signs or magnitudes | Medium | Critical | Start with the exhaustive unique-active-pair maximum; assert every deterministic pair is bounded; require proof before optimization | E5-F3-P1 |
| Combined execution launches separate Brownian and charged samplers, duplicating opportunities and RNG advancement | Low | High | Mitigated in P3: one normalized mask, additive rate, acceptance draw, active-set removal, selector/apply launch, and RNG stream are regression-tested | E5-F3-P3 |
| Charged helper/model semantics drift from E5-F2's approved subset | Medium | High | Consume E5-F2 identifiers/helpers directly; reject unregistered variants; cross-reference independent CPU fixtures | E5-F3-P1/P2 |
| Non-finite rates or majorants produce invalid acceptance ratios | Low | High | Host preflight caller inputs; device-side finite/non-negative guards; deterministic extreme fixtures and zero-rate tests | E5-F3-P1 |
| Accepted execution loses charge or leaves donor state active | Low | Critical | Mitigated in P2/P3: shared E5-F2 merge has per-box mass/charge conservation and donor-clearing coverage for charged-only and combined calls | E5-F3-P2/P3 |
| Capability validation or allocations mutate output/RNG state before failure | Low | High | Mitigated in P2/P3: charged-containing finite-charge preflight and invalid-input snapshots occur before output/RNG work | E5-F3-P2/P3 |
| Exhaustive charged-majorant scan adds O(n²) work | High | Medium | Accept correctness-first cost in bounded T3 scope, document it, reuse fixed buffers, and defer optimization until a safe tighter proof exists | E5-F3-P1 / future performance owner |
| Stochastic tests are flaky or require exact CPU/Warp replay | Low | Medium | Mitigated in P2/P3 with fixed fresh seeds and independent aggregate bounds | E5-F3-P2/P3 |
| CUDA-only behavior blocks release or hides Warp CPU regression | Low | Medium | Require Warp CPU when installed; treat CUDA as optional additive evidence with clean skips | E5-F3-P2/P3 |
| Documentation overstates support before E5-F6/F7/F9 | Medium | Medium | Publish only charged and Brownian-plus-charged direct low-level support and list deferred owners explicitly | E5-F3-P4 |
