# Risk Register

| Risk | Likelihood | Impact | Mitigation | Owner |
|------|------------|--------|------------|-------|
| Session abstraction hides transfers or synchronization | Medium | High | Allow transfer only in named setup/checkpoint/finalize methods; enforce conversion and sync spies | E7-F4 implementer |
| Sidecar registry broadly exports concrete kernel internals | Medium | Medium | Use typed process views and owner-module imports; add export regression tests under E7-F6 policy | E7-F4/E7-F6 owners |
| Cross-device, wrong-dtype, wrong-shape, or aliased arrays corrupt state | Medium | High | Validate exact metadata and prohibited aliases before registration or launch | E7-F4 implementer |
| Checkpoint silently loses GPU-only gas or process state | Medium | High | Version metadata; record vapor-pressure/resource payload authority; test restart equivalence | E7-F4 implementer |
| Ordered gas names are lost on restore | Medium | Medium | Store immutable names outside Warp and always pass them to gas restore | E7-F4 implementer |
| Partial setup exposes leaked or usable incomplete state | Low | High | P2 validates before imports, converts in fixed order, and publishes only after final `ResidentSession` validation; do not promise rollback of private conversion allocations | E7-F4 implementer |
| Post-launch failure is mistaken for an atomic rollback | Medium | High | Mark session faulted, propagate original error, prohibit further steps/checkpoint, document discard semantics | E7-F4 implementer |
| T4 expands into scheduling, transport, or RNG policy | Medium | Medium | Keep explicit seams and defer behavior to E7-F5, E7-F7, and E7-F8 | E7 owner |
| CUDA-only assumptions break routine CI | Medium | Medium | Use Warp CPU as baseline, lazy imports, and optional CUDA skips | Test owner |
| Resource allocation scales excessively with B/N/S | Medium | Medium | Publish shape/cost manifest, allocate once, fail clearly, and defer optimization/graph capture to Epic H | E7-F4 owner |
| Checkpoint schema is treated as arbitrary-code serialization | Low | High | Use typed in-memory records and validated primitive/array fields; keep disk deserialization out of scope | Security reviewer |
