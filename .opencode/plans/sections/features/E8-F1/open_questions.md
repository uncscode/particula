# Open Questions

- [x] Should Warp CPU be considered graph-capture capable for contract tests?
  - Resolved 2026-08-30: No. Warp CPU remains the uncaptured parity baseline;
    graph capture is CUDA-gated and must cleanly skip when unavailable.
  - Rationale: The current public capture test explicitly requires CUDA while
    repository policy retains Warp CPU for uncaptured parity.
  - Evidence:
    - `particula/gpu/kernels/tests/condensation_graph_capture_test.py:186` - The
      capture API gate skips CPU because graph capture requires CUDA.
    - `.opencode/plans/sections/epics/E8/scope_constraints.md:37` - Epic scope
      requires Warp CPU parity and CUDA-gated capture evidence.
  - Resolved by: plan-question-resolver

- [x] Should a native captured graph handle be included in schema-v3 resident
  checkpoints?
  - Resolved 2026-08-30: No. Checkpoints retain canonical state and continuation
    metadata; restart creates fresh exact-device identities and requires a fresh
    capture.
  - Rationale: Native graph ownership is not canonical recovery state, and
    restart deliberately reconstructs fresh resident identities.
  - Evidence:
    - `particula/execution/checkpoint.py:550` - Restart creates a fresh compatible
      resident session without reusing source identities.
    - `.opencode/plans/sections/epics/E8/implementation_strategy.md:29` - Graph
      handles belong to capture lifecycle rather than checkpoint continuation.
  - Resolved by: plan-question-resolver

- [x] Should invalidation mutate the existing capture record to `INVALIDATED`,
  or return an immutable invalidation result while a controller owns mutable
  state?
  - Resolved 2026-08-30: Keep capture signatures and invalidation results
    immutable; bind mutable lifecycle state and graph ownership to one narrow
    controller.
  - Rationale: This preserves transition history and follows the existing
    immutable-checkpoint plus identity-bound-controller design.
  - Evidence:
    - `particula/execution/checkpoint.py:73` - Checkpoint payload records are
      frozen dataclasses.
    - `particula/execution/checkpoint.py:324` - A controller owns lifecycle state
      for one exact session, registry, and guard binding.
  - Resolved by: plan-question-resolver

- [x] Should a scalar timestep duration be captured as a Python value in the
  compatibility signature or represented by a pinned device-side control?
  - Resolved 2026-08-30: Capture the normalized Python duration in the
    compatibility record and require recapture when it changes; a dynamic device
    control requires separate future scope.
  - Rationale: Current execution compares host scalar timesteps, and no canonical
    pinned duration-control schema exists.
  - Evidence:
    - `particula/execution/resident_scheduler.py:443` - All process timestep
      values must equal the supplied host duration.
    - `.opencode/plans/sections/features/E8-F2/architecture_design.md:63` - Only
      values already stored in bound device arrays can vary without preparation.
  - Resolved by: plan-question-resolver

- [ ] Does Warp provide a stable public API to explicitly destroy a graph
  handle across all supported versions?
  - Open: The repository uses an unbounded Warp dependency and verifies only
    capture begin, end, and launch; it establishes no cross-version destroy API.
  - Recommendation: **A - Retire the owner and use native destroy only when a qualified public API exists**
  - Suggested answer: Choose **A** because Particula can guarantee replay
    rejection without calling private or undocumented Warp internals.
  - Options:
    - [ ] A. Retire the owner and use native destroy only when a qualified public API exists (Recommended)
    - [ ] B. Pin a Warp version with a verified public destroy API and require it
    - [ ] C. Require reference retirement only and never call native teardown
  - Evidence considered:
    - `particula/gpu/kernels/tests/condensation_graph_capture_test.py:186` - The
      tested public surface contains only capture begin, end, and launch.
    - `pyproject.toml:24` - The Warp dependency has no repository version bound.
