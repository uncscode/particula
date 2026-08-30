# Open Questions

- [x] Which Warp graph-handle teardown operation is available and required for
  the repository's supported Warp version, if any?
  - Resolved 2026-08-30: Always retire the Python owner. Call a native release
    only when the active Warp version exposes a documented public operation that
    has passed version-qualified lifecycle tests; otherwise owner retirement is
    the complete supported teardown.
  - Rationale: Lifecycle rejection remains deterministic when no public native
    release exists, while qualified runtimes may use documented cleanup without
    private or guessed APIs.
  - Options:
    - [x] A. Always retire the Python owner and call only a verified public native release (Selected)
    - [ ] B. Pin and require a Warp version with an explicit public release API
    - [ ] C. Retire the Python owner only and prohibit native release calls
  - Evidence:
    - `particula/gpu/kernels/tests/condensation_graph_capture_test.py:186` - The
      current required API set contains no graph-destroy operation.
    - `.opencode/plans/sections/features/E8-F1/open_questions.md:58` - E8-F1 uses
      the same owner-retirement and qualified-public-release policy.
  - Resolved by: user decision

- [x] Should replay duration be represented by a fixed captured Python scalar or
  an E8-F3 pinned device control array?
  - Resolved 2026-08-30: Use a fixed captured Python scalar in the first contract
    and treat duration changes as structural recapture triggers.
  - Rationale: E8-F2 freezes normalized scalar controls, and E8-F3 defines no
    duration-specific control array.
  - Evidence:
    - `particula/execution/resident_scheduler.py:443` - Current execution requires
      all process scalar timesteps to equal the supplied duration.
    - `.opencode/plans/sections/features/E8-F2/architecture_design.md:63` - Host
      scalar changes require new preparation and recapture.
  - Resolved by: plan-question-resolver

- [x] What minimum supported process matrix belongs in the first three-way
  captured full-loop fixture when some prepared process paths are unavailable?
  - Resolved 2026-08-30: Require the complete prepared twelve-node resident
    schedule; block E8-F4 when any required path is unavailable rather than
    relabeling a reduced subset as full-loop.
  - Rationale: The shipped scheduler accepts exactly the complete loop, and the
    parent outcome requires fixed-order full-loop equivalence.
  - Evidence:
    - `particula/execution/resident_scheduler.py:256` - Scheduler validation
      rejects schedules that are not exactly the complete resident loop.
    - `.opencode/plans/sections/features/E8-F2/phase_details.md:73` - E8-F2 owns
      preparation of the complete twelve-node sequence.
  - Resolved by: plan-question-resolver

- [x] Should E8-F4-P5 split three-way full-loop evidence from its documentation
  handoff before implementation? (reviewer: plan-review-sizing)
  - Resolved 2026-08-30: Yes. Keep three-way integration evidence in P5 and move
    documentation handoff plus strict documentation validation into a final P6.
  - Rationale: Numerical, conservation, stochastic, and lifecycle validation is
    independently reviewable from documentation publication and validation.
  - Evidence:
    - `.opencode/plans/sections/features/E8-F4/phase_details.md:47` - Current P5
      combines three execution paths, multiple evidence classes, and docs work.
    - `.opencode/plans/templates/feature/phase_details.md:7` - The final feature
      phase should update development documentation.
  - Resolved by: plan-question-resolver
