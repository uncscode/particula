# Risk Register

| Risk | Likelihood | Impact | Mitigation | Owner | Status |
|------|------------|--------|------------|-------|--------|
| Selection API leaks heterogeneous kernel details | Medium | High | Freeze typed protocol and capability matrix in E7-F1 before adapters | E7-F1 owner | Open |
| Runtime failures trigger implicit CPU movement | Medium | High | E7-F6 error taxonomy, explicit fallback request, transfer spies, negative tests | E7-F6 owner | Open |
| Scheduler consumes stale environment or vapor-pressure state | Medium | High | Canonical dependency graph and call-order/derived-state tests | E7-F5 owner | Open |
| Repeated steps allocate scratch or reset RNG | Medium | High | Session-owned registries, identity assertions, persistent-stream tests | E7-F4/E7-F8 owners | Open |
| CPU and GPU adapter semantics diverge | Medium | High | Independent CPU oracles, explicit tolerances, conservation checks | E7-F2/E7-F3 owners | Open |
| Multi-box transport violates conservation or box isolation | Medium | High | CPU transport oracle, sparse-map fixtures, independent-box metamorphic tests | E7-F7 owner | Open |
| Checkpoint omits lossy or CPU-owned metadata | Medium | High | Versioned checkpoint schema and round-trip/restart tests | E7-F4/E7-F9 owners | Open |
| Public exports accidentally promote scratch internals | Medium | Medium | Narrow export review and kernel export regression tests | E7-F6 owner | Open |
| CUDA-only behavior escapes routine validation | Low | High | Warp CPU baseline and optional CUDA matrix with clear support limits | E7-F9 owner | Open |
| Epic absorbs H/I scope and stalls | Medium | Medium | Enforce non-goals in feature review and redirect optimization/autodiff work | Epic owner | Open |
