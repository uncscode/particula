# Open Questions

- [x] Which profiler export versions and minimum metric set are required for a
  profiling row to count as complete?
  - Resolved 2026-08-30: Choose **A** and use the official Arch Linux
    `nsight-systems 2026.1.3.425-1` and `nsight-compute 2026.2.1.5-1` package
    pair as the selected local baseline. Its executable identities are Nsight
    Systems `2026.1.3.425-261338342291v0` and Nsight Compute `2026.2.1.0`
    (build `38283040`, public release). Persist the literal `nsys --version` and
    `ncu --version` output as the parser identity; any other export version
    remains unsupported until a checked-in fixture and parser test qualify it.
  - Metric floor: Every complete row records kernel duration, invocation count,
    exact command, workload ID, CUDA device/runtime/driver identity, commit, and
    profiler provenance. Each dominant kernel also requires one documented
    occupancy metric and one documented memory-access or achieved-bandwidth
    metric with units. Missing required fields produce an unavailable row and
    are never inferred.
  - GPU-only boundary: Real Nsight collection requires a qualified NVIDIA CUDA
    GPU. Missing CUDA, permissions, counters, or either profiler records
    unavailable evidence without Warp CPU or other CPU fallback. Default tests
    parse bounded fixtures and do not require a GPU or installed NVIDIA tools.
  - Packaging: Nsight Systems and Nsight Compute are external system tools, not
    Python packages, and are not added to `pyproject.toml`.
  - Rationale: Versioned semantic fields can fail closed while preserving the
    minimum performance and provenance evidence.
  - Options:
    - [x] A. Pin one validated Nsight Systems/Compute version pair and semantic metric floor (Selected)
    - [ ] B. Accept arbitrary Nsight exports with best-effort parsing
    - [ ] C. Require Warp profiler output only and omit hardware counters
  - Evidence:
    - `particula/gpu/tests/benchmark_test.py:1` - Current benchmark guidance names
      Warp and Nsight but defines no export schema.
    - Arch Linux package index, queried 2026-08-30 - Both selected profiler
      packages are available from the official `Extra` repository.
    - Local development probe, 2026-08-30 - `nsys --version`, `ncu --version`,
      and `nvidia-smi` confirmed the selected executables on an NVIDIA GeForce
      RTX 5060 with KMD/driver `610.57.04` and CUDA UMD `13.3`. This establishes
      local availability; P3 must still validate export parsing, profiler exit
      status, permissions, and required hardware counters.
  - Resolved by: user decision

- [x] What exact small and medium dimensions should be canonical after E8-F6
  produces its feasible scaling rows?
  - Resolved 2026-08-30: After E8-F6 publishes its fixed feasibility matrix,
    freeze the smallest executed row that exposes launch sensitivity as the
    canonical small workload and the largest repeatably feasible executed row as
    the canonical medium workload. Preserve their exact dimensions and IDs for
    later devices; do not reselect them per run.
  - Rationale: This derives representative workloads from measured feasibility
    on the initial development device while preserving stable cross-run and
    cross-device comparison. A future device may report either fixed row
    unavailable rather than substituting dimensions.
  - Options:
    - [x] A. Freeze the smallest launch-sensitive and largest repeatably feasible E8-F6 rows (Selected)
    - [ ] B. Hard-code dimensions before E8-F6 evidence exists
    - [ ] C. Select new dimensions dynamically for every profiling run
  - Evidence:
    - `.opencode/plans/sections/features/E8-F6/success_criteria.md:3` - E8-F6 owns
      the feasible box-first matrix and raw timing evidence.
  - Resolved by: user decision

- [x] Should raw Nsight exports be committed, attached externally, or retained
  only as local checksummed artifacts?
  - Resolved 2026-08-30: Keep raw `.nsys-rep`, `.ncu-rep`, and machine-readable
    exports only in the gitignored local
    `.artifacts/benchmarks/profiling/raw/` subtree. Commit bounded normalized
    summaries containing commands, versions, workload/device provenance,
    metrics, statuses, raw filenames, byte sizes, and SHA-256 checksums. Do not
    upload or commit the raw reports.
  - Rationale: This keeps Git and release storage small and preserves locally
    verifiable provenance. It deliberately does not provide collaborators a
    shared full raw report; published conclusions must therefore be supported by
    the committed normalized summary and this limitation must remain explicit.
  - Options:
    - [ ] A. Commit normalized fixtures and summaries; attach checksummed raw exports externally
    - [ ] B. Commit every raw profiler export to Git
    - [x] C. Retain ignored local raw exports and commit bounded summaries/checksums (Selected)
  - Evidence:
    - `particula/gpu/tests/benchmark_test.py:228` - Existing artifacts are bounded
      to a controlled repository artifact root.
    - `.opencode/plans/sections/features/E8-F7/architecture_design.md:43` - The
      plan separates compact normalized evidence from full raw exports.
    - `.gitignore` - The raw profiler staging subtree is narrowly ignored while
      its directory-local `.gitignore` keeps the path available in every clone.
  - Resolved by: user decision

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
