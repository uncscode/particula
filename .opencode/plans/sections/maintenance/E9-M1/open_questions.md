# Open Questions

- [ ] **Q1 — P1 blocking: Which explicit ordered species mapping is approved?**
  Shared full gas-width lanes or explicit process-owned mapping to existing
  particle/environment layouts? Recommend the latter only where needed to
  preserve current schemas; prohibit implicit partitioning-mask compaction.
  Record nonpartitioning and particle-only cases, name policy and configuration
  alignment. Evidence: GasData owns names/mask; particle/environment lanes have
  no names. Maintainer and scientific reviewer must select the representation.
- [ ] **Q2 — P1 blocking: What exact raw concentration convention applies to
  each supported distribution/native consumer?** ParticleData documents counts
  versus density, while the facade getter unconditionally divides by volume.
  Recommend a consumer-backed normalization ledger and explicit interpretation
  where needed, without changing storage schemas. Resolve with independent
  non-unit-volume characterization before adding a universal helper.
- [ ] **Q3 — P2 blocking: Which minimal API spelling and field-level copy/view
  rules are approved?** Whole-container identity and rejection invariants are
  fixed, but helper names, writable views versus copies, and coordinated setter
  spelling need approval. Recommend explicit unit-bearing names for ambiguous
  concentration quantities and reuse existing derived properties. P2 publishes
  the exact M2 acceptance contract; M2 implements it.
- [ ] **Q4 — P1/P2 blocking: Which helper gaps are genuinely needed, and where
  should shared read-only alignment validation live?** Inventory actual M2–M4
  callers before choosing additions. Recommend concrete data-native functions
  or existing properties, never a new state facade or coercing reconstruction.
  Explicitly distinguish generic schema checks from process physical admission.
- [ ] **Q5 — Handoff governance: Who approves scientific semantics and M2
  admission?** Maintainer must assign reviewers and implementation owners; no
  fixed date is supplied. Record named approval with final-revision evidence.

Already resolved by issue #1602: three containers, all gas categories,
process-owned physics, identity-preserving whole-container access, all-or-none
replacement, single-box CPU processes, retained transfers, no automatic GPU
rebinding, strict serial execution and deletion last. These are not options.
The above five questions are first-pass review items; no unsupported decision
is represented as already approved.
