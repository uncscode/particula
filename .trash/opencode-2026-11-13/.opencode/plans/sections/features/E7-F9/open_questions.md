# Open Questions

- [x] What diagnostic result surface is public versus checkpoint-only?
  - Corrected 2026-08-10 by issue #1528: P1 ships a concrete-only closed
    registration/dispatch protocol in `particula.execution.diagnostics`; it adds
    no public descriptors, observation results, package exports, or user docs.
  - Rationale: The resident diagnostic boundary is an internal direct-import
    seam, so caller-owned same-device bindings can be validated without
    expanding the public execution API.
  - Evidence:
    - `docs/Features/Roadmap/data-oriented-gpu.md:1568` - diagnostics are
      optional GPU-side reductions, without a prescribed result API.
  - Resolved by: issue #1528 implementation

- [x] Which checkpoint encoding is durable beyond in-memory round trips?
  - Resolved 2026-07-27: None in Epic G. The supported checkpoint remains a
    versioned synchronized in-memory CPU snapshot.
  - Rationale: No durable codec, migration policy, or compatibility lifetime is
    defined, and E7-F4 explicitly excludes file serialization.
  - Evidence:
    - `.opencode/plans/sections/features/E7-F4/scope.md:40` - durable file formats
      are outside the checkpoint feature.
    - `particula/gpu/conversion.py:422` - existing checkpoint operations restore
      runtime CPU containers rather than a durable encoding.
  - Resolved by: plan-question-resolver

- [x] What exact fixture sizes qualify as the issue's “larger multi-box and
  particle-resolved” regressions?
  - Resolved 2026-07-27: Use a 4-box, 16-particle-slot, 2-species fixture.
  - Rationale: This exceeds the existing full-process fixture in box and slot
    dimensions while retaining a small species matrix for the fast suite.
  - Evidence:
    - `particula/gpu/tests/process_sequence_test.py:117` - current largest
      resident full-process fixture is 2 boxes × 4 slots × 2 species.
  - Resolved by: PR #1452 decision

- [x] Which process-specific tolerances supersede the default tight conservation
  target in the final matrix?
  - Resolved 2026-07-27: None supersede conservation. Apply each owning process's
    recorded parity or stochastic tolerance only to its observable, and retain
    separate concentration-weighted inventory checks at `rtol=1e-12,
    atol=1e-30` where that contract applies.
  - Rationale: Deterministic parity, conservation, and stochastic acceptance are
    distinct evidence classes and broad tolerance substitution would hide drift.
  - Evidence:
    - `.opencode/guides/testing_guide.md:369` - device-aware policy requires
      separate parity, conservation, and stochastic pass criteria.
    - `docs/Features/Roadmap/coagulation-validation.md:43` - rate tolerances are
      looser than the separately recorded tight inventory tolerance.
  - Resolved by: plan-question-resolver

- [x] Where should dated closeout evidence live?
  - Resolved 2026-07-27: Record Epic G status and exit-bar disposition in
    `docs/Features/Roadmap/data-oriented-gpu.md`, add a dated summary to
    `docs/Features/Roadmap/index.md`, and use a dedicated validation record when
    the command/tolerance matrix is too large for the roadmap.
  - Rationale: Repository documentation is durable and reproducible, unlike
    transient CI URLs or workflow-only state.
  - Evidence:
    - `docs/Features/Roadmap/index.md:131` - prior epic shipments have dated,
      discoverable roadmap summaries.
    - `docs/Features/Roadmap/coagulation-validation.md:41` - detailed validation
      matrices already use dedicated repository records.
  - Resolved by: plan-question-resolver
