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

- [ ] What exact executable modules require per-target closeout coverage?
  - Open: Only merged E8-F1 through E8-F7 implementation records and the final
    executable diff can establish the authoritative changed-module list.
  - Recommendation: **A - Freeze every changed executable module at P3 preflight and exclude documentation-only files**
  - Suggested answer: Choose **A** because it makes coverage targets complete,
    reproducible, and resistant to stale candidate lists.
  - Options:
    - [ ] A. Freeze every changed executable module at P3 preflight and exclude documentation-only files (Recommended)
    - [ ] B. Use only the five modules in the existing resident closeout guide
    - [ ] C. Cover all `particula.execution` modules regardless of the final diff
  - Evidence considered:
    - `.opencode/guides/testing_guide.md:275` - Existing closeout targets are
      conditional guidance for changed resident modules, not a frozen E8 list.

- [ ] Which normalized E8-F6/E8-F7 artifacts and checksums are final inputs?
  - Open: Artifact paths, schema versions, commit provenance, and checksums are
    produced only after E8-F6 and E8-F7 ship.
  - Recommendation: **A - Accept only immutable shipped artifacts with matching schema, commit provenance, and checksums**
  - Suggested answer: Choose **A** because provisional or stale measurements
    cannot support an auditable epic closeout.
  - Options:
    - [ ] A. Accept only immutable shipped artifacts with matching schema, commit provenance, and checksums (Recommended)
    - [ ] B. Accept the latest artifact path without provenance verification
    - [ ] C. Copy summary values manually and omit artifact checksums
  - Evidence considered:
    - `.opencode/plans/sections/features/E8-F8/implementation_tasks.md:20` - P3
      requires final paths, schema versions, provenance, and checksums.

- [ ] Does a qualified CUDA device exist for all required measured exit rows?
  - Open: Qualification is environment-specific and requires successful capture,
    memory, benchmark, and profiler probes at closeout time.
  - Recommendation: **A - Probe the closeout environment and keep missing required rows unshipped**
  - Suggested answer: Choose **A** because a historical device or clean skip is
    availability evidence, not a passing measured exit row.
  - Options:
    - [ ] A. Probe the closeout environment and keep missing required rows unshipped (Recommended)
    - [ ] B. Treat clean CUDA skips as satisfying required measured rows
    - [ ] C. Accept historical device measurements from an earlier commit
  - Evidence considered:
    - `particula/gpu/tests/cuda_availability.py:17` - CUDA availability is probed
      from the current Warp runtime.
    - `particula/gpu/kernels/tests/condensation_graph_capture_test.py:186` -
      Capture qualification requires the public capture API set on a CUDA device.
