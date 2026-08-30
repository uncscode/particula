# Risk Register

| Risk | Likelihood | Impact | Mitigation | Owner |
|------|------------|--------|------------|-------|
| Validation-free enqueue becomes reachable with forged or stale resources | Medium | Critical | Keep records concrete-only, exact-type checked, signature-bound, and invalidated by E8-F1 before launch | E8-F2 P1/P7 |
| Public direct APIs and prepared paths diverge in physics or launch order | Medium | High | Extract one private launch primitive used by both paths; retain existing direct suites and add launch-trace parity tests | E8-F2 P2-P6 |
| Hidden allocation or `.numpy()` readback remains in a nested process helper | High | High | Instrument every prepared full-loop test with allocator/readback/sync spies and inventory nested nucleation/exhaustion paths | E8-F2 P2-P7 |
| E8-F2 duplicates E8-F3 buffer ownership or allocates placeholders | Medium | High | Define required resource descriptors only; fail setup when missing and hand the inventory to E8-F3 | E8-F2 / E8-F3 |
| Python scalar changes are mistaken for replay-safe dynamic inputs | Medium | High | Freeze normalized scalar controls in the compatibility record; permit dynamic values only through prebound device arrays | E8-F1 / E8-F2 |
| Persistent RNG streams are reset during preparation or replay | Low | High | Require explicit pre-capture initialization through shipped lifecycle APIs; assert enqueue always uses reset false | E8-F2 P5/P7 |
| Empty/no-op dimensions alter launch traces and invalidate capture unexpectedly | Medium | Medium | Resolve empty behavior during setup and test canonical `(0,S)`, `(B,0)`, and zero-work cases | E8-F2 P2-P7 |
| CUDA-only capture failures hide behind broad skips | Medium | High | Reuse narrow capability skip matching; fail unexpected exceptions and keep Warp CPU as uncaptured evidence only | E8-F2 P7 |
| Refactor exceeds issue-sized review scope across large kernel modules | Medium | Medium | Ship one bounded process family per phase, preserve public wrappers, and avoid unrelated physics/performance changes | Feature owner |
