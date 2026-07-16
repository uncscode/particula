# Risk Register

| Risk | Impact | Mitigation | Owner |
|---|---|---|---|
| Docs overstate low-level support as high-level production GPU support | High | Exact negative claims and regression tests | P1 owner |
| Example diverges from final E5 API | High | Freeze after E5-F6; execute against public lazy import | P2 owner |
| Stochastic output makes example flaky | Medium | Assert invariants/bounds and deterministic metadata, not exact pair replay | P2 owner |
| Roadmap IDs or links drift | Medium | One-to-one ID matrix and local-link tests | P3 owner |
| E5 closes before children/evidence pass | Critical | Fail-closed P4 predicate plus negative transition tests | E5 owner |
| Optional CUDA absence blocks release | Medium | Require Warp CPU when installed; keep CUDA optional/skip-safe | Validation owner |
| E5-F8 deferred scope is implied shipped | High | Preserve explicit downstream-owner table and bounded claims | Docs owner |
| Nested research unavailable | Low | Use authoritative parent/sibling plans and direct source inspection; record challenge | Plan drafter |
