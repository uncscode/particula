# Risk Register

| Risk | Likelihood | Impact | Mitigation | Owner | Status |
|------|------------|--------|------------|-------|--------|
| Host allocation, validation, or synchronization leaks into replay | Medium | High | Instrument capture/replay boundaries; reject incompatible calls before graph launch | E8-F2 | Open |
| Captured and uncaptured scheduler semantics diverge | Medium | High | Reuse one resolved order and require three-way full-loop tests | E8-F4/E8-F5 | Open |
| Mutable buffer identity or shape invalidates a graph silently | Medium | High | Bind exact device, schema, shape, configuration, and object identities; expose explicit recapture triggers | E8-F1/E8-F3 | Open |
| Persistent RNG is accidentally reseeded or aliased | Medium | High | Reuse registry-owned nonaliasing streams and test initialize/reset/checkpoint/replay lifecycle | E8-F3/E8-F5 | Open |
| Writer failure leaves replay state ambiguous | Medium | High | Preserve resident fault semantics; make no rollback or retry promise after launch | E8-F1 | Open |
| CUDA-only evidence is unavailable in standard CI | High | Medium | Keep Warp CPU compatibility tests mandatory and CUDA tests pass-or-clean-skip; require qualified hardware before closeout | E8-F5/E8-F8 | Open |
| Benchmarks overfit one GPU or hide single-box limits | Medium | High | Publish hardware-qualified raw artifacts and vary boxes, particles, species, active fraction, and process combinations | E8-F6/E8-F7 | Open |
| Memory model omits hidden or peak allocations | Medium | High | Reconcile analytical accounting with observed peak device memory and list exclusions explicitly | E8-F6 | Open |
| Capture optimization weakens scientific correctness | Low | High | Preserve tolerance and conservation gates; performance regressions never waive correctness failures | E8-F5 | Open |
| Documentation implies automatic capture or universal speedup | Medium | Medium | Add limitations, lifecycle, recapture matrix, and bounded claims to the runnable example | E8-F8 | Open |
