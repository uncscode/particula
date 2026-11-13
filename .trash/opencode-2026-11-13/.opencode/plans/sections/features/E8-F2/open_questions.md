# Open Questions

- [x] Should prepared process records live in one
  `particula.execution.resident_enqueue` module or beside each existing
  executor/adapter?
  - Resolved 2026-08-30: Keep process-specific records and enqueue helpers beside
    their owning modules; keep only the aggregate prepared timestep and sequence
    composition in `resident_enqueue`.
  - Rationale: Existing architecture centralizes ordering while delegating
    process behavior, avoiding a duplicate implementation layer.
  - Evidence:
    - `particula/execution/resident_scheduler.py:465` - The scheduler composes the
      complete order and delegates each process.
    - `particula/execution/process_adapters.py:242` - Process-specific delegation
      remains in concrete local adapters.
  - Resolved by: plan-question-resolver

- [x] How should per-timestep scalar controls be represented if users need to
  change them between graph replays?
  - Resolved 2026-08-30: Freeze Python scalar controls in the prepared record and
    require preparation and recapture when they change; defer replay-variable
    pinned controls until explicitly scoped.
  - Rationale: Existing kernels and scheduler consume host scalars, and E8-F3
    has not defined a stable control-array schema.
  - Evidence:
    - `particula.execution.resident_scheduler._validate_resident_durations()`
      (called by `_validate_complete_resident_timestep_metadata`) enforces
      duration agreement across host scalar request values.
    - `architecture_design.md` — **Data / API / Workflow Changes**, “Dynamic
      controls” - Python scalar changes require a new prepared record.
  - Resolved by: plan-question-resolver

- [x] Can selected wall-loss logical lanes be frozen as one reusable device
  array before E8-F3 publishes the final buffer inventory?
  - Resolved 2026-08-30: E8-F2 must freeze the logical selection and required
    schema, while E8-F3 allocates, publishes, and pins the reusable device array.
  - Rationale: The current adapter allocates the selected-lane array at dispatch,
    but reusable capture-lifetime storage belongs to the registry track.
  - Evidence:
    - `particula.execution.process_adapters.ResidentWallLossAdapter._enqueue_selected()`
      - Partial wall-loss dispatch creates a private Warp selected-box array.
    - `.opencode/plans/sections/features/E8-F3/architecture_design.md:5` - E8-F3
      establishes the registry as capture storage authority.
  - Resolved by: plan-question-resolver

- [x] Should E8-F2-P5 through E8-F2-P7 be split by independently deliverable
  prepared-process boundaries before implementation? (reviewer:
  plan-review-sizing)
  - Resolved 2026-08-30: Split P5 into independently tested coagulation,
    dilution, and wall-loss prepared boundaries; retain P6 as the cohesive
    nucleation/exhaustion boundary and P7 as integration unless concrete diffs
    exceed the review limit.
  - Rationale: P5 spans three independent process families, while P6 and P7 each
    represent one deliberate semantic or integration boundary.
  - Evidence:
    - `phase_details.md` — **E8-F2-P5** - P5 combines coagulation, dilution,
      wall loss, selected lanes, and persistent RNG.
    - `.opencode/plans/templates/feature/phase_details.md:3` - Feature phases
      target one reviewable approximately 100-LOC production change.
  - Resolved by: plan-question-resolver
