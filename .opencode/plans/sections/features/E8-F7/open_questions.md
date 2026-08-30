# Open Questions

- [ ] Which profiler export versions and minimum metric set are required for a
  profiling row to count as complete?
  - Open: The repository delegates detailed profiling to Nsight but defines no
    validated export version or stable parser schema.
  - Recommendation: **A - Pin one validated Nsight Systems/Compute version pair and semantic metric floor**
  - Suggested answer: Choose **A** because versioned semantic fields can fail
    closed while preserving duration, invocation count, occupancy, memory-access,
    command, and provenance evidence.
  - Options:
    - [ ] A. Pin one validated Nsight Systems/Compute version pair and semantic metric floor (Recommended)
    - [ ] B. Accept arbitrary Nsight exports with best-effort parsing
    - [ ] C. Require Warp profiler output only and omit hardware counters
  - Evidence considered:
    - `particula/gpu/tests/benchmark_test.py:1` - Current benchmark guidance names
      Warp and Nsight but defines no export schema.

- [ ] What exact small and medium dimensions should be canonical after E8-F6
  produces its feasible scaling rows?
  - Open: Exact dimensions depend on E8-F6 feasibility and memory evidence that
    does not yet exist.
  - Recommendation: **A - Freeze the smallest launch-sensitive and largest repeatably feasible E8-F6 rows**
  - Suggested answer: Choose **A** because it derives representative workloads
    from measured feasibility while preventing dimensions from drifting by run.
  - Options:
    - [ ] A. Freeze the smallest launch-sensitive and largest repeatably feasible E8-F6 rows (Recommended)
    - [ ] B. Hard-code dimensions before E8-F6 evidence exists
    - [ ] C. Select new dimensions dynamically for every profiling run
  - Evidence considered:
    - `.opencode/plans/sections/features/E8-F6/success_criteria.md:3` - E8-F6 owns
      the feasible box-first matrix and raw timing evidence.

- [ ] Should raw Nsight exports be committed, attached externally, or retained
  only as local checksummed artifacts?
  - Open: Repository guidance supports bounded committed JSON but does not define
    an attachment service, retention period, or release-artifact policy.
  - Recommendation: **A - Commit normalized fixtures and summaries; attach checksummed raw exports externally**
  - Suggested answer: Choose **A** because it keeps reviewable evidence in Git
    while preserving auditable full reports without committing bulky binaries.
  - Options:
    - [ ] A. Commit normalized fixtures and summaries; attach checksummed raw exports externally (Recommended)
    - [ ] B. Commit every raw profiler export to Git
    - [ ] C. Retain checksummed raw exports locally without a shared attachment
  - Evidence considered:
    - `particula/gpu/tests/benchmark_test.py:228` - Existing artifacts are bounded
      to a controlled repository artifact root.
    - `.opencode/plans/sections/features/E8-F7/architecture_design.md:43` - The
      plan separates compact normalized evidence from full raw exports.

- [x] Is E8-F7 or E8-F8 the authoritative profiling/closeout track?
  - Resolved 2026-08-30: E8-F7 owns T7 profiling and machine-bounded
    recommendations; E8-F8 owns the T8 example, runbook, reconciliation, and epic
    closeout.
  - Rationale: The explicit orchestrator assignment and canonical feature plans
    agree, while the parent child table contains stale labels.
  - Evidence:
    - `.opencode/plans/sections/features/E8-F7/overview.md:14` - E8-F7 defines
      representative profiling workloads and recommendations.
    - `.opencode/plans/sections/features/E8-F8/overview.md:5` - E8-F8 follows the
      explicit T8 example and closeout assignment.
  - Resolved by: plan-question-resolver
