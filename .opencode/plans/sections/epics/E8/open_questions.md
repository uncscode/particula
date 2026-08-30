# Open Questions

- [x] Which scalar timestep inputs can change between replays without graph
  recapture?
  - Resolved 2026-08-30: No Python-hosted timestep input may change in the
    first replay contract; changing the common duration requires preparation
    and graph recapture.
  - Rationale: The current scheduler requires every process timestep to equal
    one host duration, and no pinned device timestep-control buffer exists.
  - Evidence:
    - `particula/execution/resident_scheduler.py:443` - Duration validation
      compares every process timestep with the supplied execution duration.
    - `.opencode/plans/sections/features/E8-F2/architecture_design.md:63` -
      Python scalar changes require a new prepared record.
  - Resolved by: plan-question-resolver

- [x] Should checkpoint/finalize operations be legal while a captured graph
  object exists but is not executing?
  - Resolved 2026-08-30: Require successful captured-graph teardown before
    checkpoint or finalize. A later continuation or restart must establish a
    fresh capture against its active identities.
  - Rationale: This preserves the existing fail-closed checkpoint lifecycle and
    avoids inventing dormant native-handle synchronization, pointer-lifetime, or
    post-checkpoint replay semantics.
  - Options:
    - [x] A. Require capture teardown before checkpoint or finalize (Selected)
    - [ ] B. Permit checkpoint between replays after explicit synchronization
  - Evidence:
    - `particula/execution/checkpoint.py:386` - Checkpoint validates a closed
      identity-bound lifecycle and then synchronizes resident state.
    - `.opencode/plans/sections/features/E8-F1/open_questions.md:15` - Native
      graph handles are excluded from checkpoint continuation state.
  - Resolved by: user decision

- [x] What minimum repeated-timestep count defines the launch-overhead
  benchmark matrix?
  - Resolved 2026-08-30: Measure exactly `1`, `10`, `100`, and `1000` repeated
    timesteps for each canonical workload and replay mode. Keep any row that
    exceeds a documented runtime or memory budget explicitly unavailable rather
    than changing its count.
  - Rationale: The fixed matrix separates one-launch, short, medium, and strongly
    amortized behavior while preserving cross-run and cross-device comparison.
  - Options:
    - [x] A. Measure 1, 10, 100, and 1000 timesteps (Selected)
    - [ ] B. Measure 1, 10, and 100 timesteps only to bound runtime
    - [ ] C. Select counts dynamically per device and forgo a cross-run matrix
  - Evidence:
    - `docs/Features/Roadmap/data-oriented-gpu.md:1779` - The roadmap requires
      small and medium repeated-timestep launch-overhead evidence without counts.
  - Resolved by: user decision

- [x] Which peak-memory measurement method is portable enough for the closeout
  artifact?
  - Resolved 2026-08-30: Use deterministic analytical and registry logical-byte
    accounting as the portable baseline, augment it with a version-qualified
    allocator peak when available, and mark unsupported observed peaks unavailable.
  - Rationale: Shape-and-dtype accounting is device-independent, while the
    repository exposes total device memory but no stable allocator high-water mark.
  - Evidence:
    - `particula/gpu/tests/benchmark_test.py:312` - Existing helpers compute
      deterministic array bytes from shapes and dtypes.
    - `particula/gpu/tests/benchmark_test.py:246` - Device metadata records total
      memory but not allocator peak usage.
  - Resolved by: plan-question-resolver

- [x] Should the parent child-plan table and dependency map be corrected to
  assign graph replay, correctness validation, profiling, and documentation to
  the same E8-F4 through E8-F8 IDs as the feature plans? (reviewer:
  plan-review-architecture)
  - Resolved 2026-08-30: Yes. Correct the parent mappings so E8-F4 owns replay,
    E8-F5 correctness, E8-F6 scaling and memory, E8-F7 profiling, and E8-F8
    example, runbook, and closeout work.
  - Rationale: The explicit orchestrator handoff and canonical feature plans
    agree; the parent table is stale and would misroute dependency gates.
  - Evidence:
    - `.opencode/plans/sections/features/E8-F5/overview.md:13` - E8-F5 supplies
      the three-way correctness validation matrix.
    - `.opencode/plans/sections/features/E8-F7/overview.md:14` - E8-F7 owns
      representative CUDA profiling and recommendations.
  - Resolved by: plan-question-resolver
