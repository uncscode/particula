# Risk Register

| Risk | Likelihood | Impact | Mitigation | Owner |
|------|------------|--------|------------|-------|
| E8-F2 requirement inventory changes after manifests freeze | Medium | High | Version/fingerprint requirements; require exact completeness and explicit recapture rather than silently omitting roles | E8-F2/E8-F3 |
| Candidate allocation fails after earlier arrays were created | Medium | Medium | Stage locally and publish atomically; injected early/middle/late failure tests; permit clean retry | E8-F3-P4 |
| New sidecars overlap primaries or other families through views | Low | High | Validate exact identity and half-open byte ranges across the complete candidate plus established storage before publication | E8-F3-P2/P3 |
| Byte totals overflow or diverge from allocation formulas | Low | High | Share checked shape/item-size helpers with allocation; test boundaries and zero dimensions; reject unsupported dtypes/capacities | E8-F3-P1 |
| Logical-byte totals are mistaken for allocator-reserved memory | Medium | Medium | Name and document logical bytes; exclude pool reservation, fragmentation, checkpoint copies, and future tape; hand broader model to E8-F6 | E8-F3-P1/P5 |
| RNG sidecars are reseeded on repeated setup | Low | High | Reuse shipped first-publication initialization and exact views; assert words/identity are preserved on compatible reacquisition | E8-F3-P2/P4 |
| Diagnostics remain caller-owned and unstable during capture | Medium | High | Close selected registrations during setup and pin all output/accounting identities in the capture set | E8-F3-P3 |
| New concrete records leak into public exports | Low | Medium | Keep direct-module-only and retain dependency-neutral export tests | E8-F3-P1/P5 |
| CUDA-only capture assumptions weaken CPU validation | Medium | Medium | Test metadata, accounting, atomicity, and uncaptured reuse on Warp CPU; CUDA rows pass or cleanly skip without fallback | E8-F3-P5 |
