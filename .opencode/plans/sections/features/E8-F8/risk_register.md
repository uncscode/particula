# Risk Register

| Risk | Likelihood | Impact | Mitigation | Owner |
|------|------------|--------|------------|-------|
| Example drifts from concrete-only runtime contracts | Medium | High | Import production seams directly, assert sequence and exports in hardware-free tests, update example with upstream API changes | E8-F8-P1 owner |
| Stale graph is replayed after structural change | Medium | Critical | Publish exhaustive trigger table, demonstrate rejection, require explicit teardown and fresh capture, never auto-recapture | E8-F8-P2 owner |
| CUDA absence is reported as successful capture evidence | Medium | High | Separate required and optional rows, record qualified device, permit only recognized clean skips, prohibit CPU fallback | E8-F8-P3 owner |
| Machine-specific speedups become portable promises | Medium | High | Preserve workload/device/version metadata and descriptive-only language; link raw artifacts and bounded conclusions | E8-F7/E8-F8 owners |
| Closeout omits a changed executable coverage target | Medium | High | Derive targets from implementation records and final diff before commands; retain per-target term-missing rows | E8-F8-P3 owner |
| Required command cannot run but epic is promoted | Low | Critical | Fail closed: record unavailable output and blocker; P4 cannot mark Shipped until all required rows pass | E8-F8-P4 owner |
| Parent E8 labels conflict with orchestrator T7/T8 assignment | High | Medium | Reconcile child plan, dependency, milestone, roadmap, and sibling cross-references in P4 | E8-F8-P4 owner |
| Raw evidence leaks pointers or opaque native handles | Low | High | Normalize only nonsecret environment/measurement metadata; never serialize handles, pointers, payloads, or RNG words | E8-F8-P3 owner |
