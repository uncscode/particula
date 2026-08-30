# Open Questions

- [x] Which assignment governs the stale E8-F7/E8-F8 parent labels?
  - Resolved 2026-08-30: Follow the orchestrator's explicit T8 assignment:
    E8-F7 owns profiling and E8-F8 owns the graph example, runbook, recapture
    triggers, limitations, roadmap updates, and epic closeout.
  - Rationale: The current workflow handoff and canonical feature plans are the
    accepted decision; older parent labels are stale.
  - Evidence:
    - `.opencode/plans/sections/features/E8-F7/overview.md:14` - E8-F7 owns the
      profiling workloads and recommendations.
    - `.opencode/plans/sections/features/E8-F8/overview.md:5` - E8-F8 records the
      explicit T8 example and closeout assignment.
  - Resolved by: plan-question-resolver

- [x] What exact executable modules require per-target closeout coverage?
  - Resolved 2026-08-30: At E8-F8 P3 preflight, derive and freeze every changed
    production executable module from shipped E8-F1 through E8-F7 implementation
    records plus the final diff. Exclude documentation-only and test files as
    coverage targets, and retain a literal per-target result for every frozen
    module alongside the untargeted repository suite.
  - Rationale: Final implementation evidence is authoritative and avoids both a
    stale hand-written module list and dilution by unrelated execution modules.
  - Options:
    - [x] A. Freeze every changed executable module at P3 preflight and exclude documentation-only files (Selected)
    - [ ] B. Use only the five modules in the existing resident closeout guide
    - [ ] C. Cover all `particula.execution` modules regardless of the final diff
  - Evidence:
    - `.opencode/guides/testing_guide.md:275` - Existing closeout targets are
      conditional guidance for changed resident modules, not a frozen E8 list.
  - Resolved by: user decision

- [x] Which normalized E8-F6/E8-F7 artifacts and checksums are final inputs?
  - Resolved 2026-08-30: Accept only committed normalized artifacts produced by
    shipped E8-F6/E8-F7 revisions whose schema version, source commit, workload
    IDs, device/tool provenance, and SHA-256 match the committed closeout
    manifest. Local-only raw E8-F7 reports may contribute recorded checksums but
    are not closeout inputs that E8-F8 can retrieve.
  - Rationale: Provisional, stale, schema-incompatible, or provenance-mismatched
    measurements cannot support an auditable closeout, and values must not be
    copied manually around artifact validation.
  - Options:
    - [x] A. Accept only immutable shipped artifacts with matching schema, commit provenance, and checksums (Selected)
    - [ ] B. Accept the latest artifact path without provenance verification
    - [ ] C. Copy summary values manually and omit artifact checksums
  - Evidence:
    - `.opencode/plans/sections/features/E8-F8/implementation_tasks.md:20` - P3
      requires final paths, schema versions, provenance, and checksums.
    - `.opencode/plans/sections/features/E8-F7/open_questions.md` - E8-F7 commits
      bounded normalized summaries while retaining raw reports locally only.
  - Resolved by: user decision

- [x] Does a qualified CUDA device exist for all required measured exit rows?
  - Resolved 2026-08-30: At closeout, designate one currently qualified CUDA
    device and require it to complete every required capture, correctness,
    benchmark, memory, and profiler exit row at the final source revision. The
    local NVIDIA GeForce RTX 5060 is the current candidate, not a permanent
    hardware restriction. Probe additional Warp-visible devices independently
    as supplemental evidence; they block only if explicitly promoted to required.
  - Rationale: One complete device prevents clean skips or historical results
    from substituting for measured CUDA evidence without making future optional
    or unsupported devices block the epic.
  - Options:
    - [x] A. Probe the closeout environment and keep missing required rows unshipped (Selected; one designated device must complete all rows)
    - [ ] B. Treat clean CUDA skips as satisfying required measured rows
    - [ ] C. Accept historical device measurements from an earlier commit
  - Evidence:
    - `particula/gpu/tests/cuda_availability.py:17` - CUDA availability is probed
      from the current Warp runtime.
    - `particula/gpu/kernels/tests/condensation_graph_capture_test.py:186` -
      Capture qualification requires the public capture API set on a CUDA device.
    - `.opencode/plans/sections/features/E8-F5/open_questions.md:31` - Every
      Warp-visible device is probed independently, while current local evidence
      begins with the RTX 5060.
  - Resolved by: user decision
