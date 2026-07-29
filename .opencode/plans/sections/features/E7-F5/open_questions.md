# Open Questions

- [x] Is caller registration order authoritative?
  - Resolved 2026-07-26: No. The validated dependency graph and stable process-ID
    tie breaker define canonical order.
  - Rationale: Issue #1451 requires deterministic scheduling and prevention of
    stale thermodynamic state.
  - Evidence: `.opencode/plans/sections/epics/E7/implementation_strategy.md:13-18`.

- [x] May the scheduler restore CPU state or retry on CPU after a GPU failure?
  - Resolved 2026-07-26: No. Normal steps neither synchronize nor restore, and
    runtime failures never imply fallback.
  - Evidence: `.opencode/plans/sections/features/E7-F4/architecture_design.md:71-82`.

- [x] What is the precise canonical placement of nucleation relative to
  condensation when both consume the same gas species?
  - Resolved 2026-07-27: Each reviewed scheduling profile must declare one fixed
    nucleation/condensation edge for its configured workflow. No universal
    scientific order is imposed.
  - Rationale: Both processes mutate current gas inventory, and repository
    evidence establishes sequential visibility without justifying one universal
    scientific ordering.
  - Evidence:
    - `docs/Features/Roadmap/data-oriented-gpu.md:1527` - the roadmap gives only
      an example process ordering, not a nucleation/condensation rule.
  - Resolved by: PR #1452 decision

- [x] Does saturation refresh require a new narrow GPU primitive or can it be
  composed safely from existing thermodynamic state without host readback?
  - Resolved 2026-07-29 in #1496: The concrete coordinator owns a private
    device saturation writer that writes `environment.saturation_ratio` from
    current gas concentration, molar mass, temperature, and refreshed vapor
    pressure. It composes the existing authoritative vapor-pressure primitive
    without host payload readback.
  - Rationale: The necessary state is resident, but the existing refresh writes
    only vapor pressure and cannot update saturation ratio.
  - Evidence:
    - `particula/gpu/kernels/thermodynamics.py:318` - the shipped primitive
      mutates only `gas.vapor_pressure`.
    - `particula/gpu/warp_types.py:164` - the destination saturation-ratio array
      is already device resident.
    - `particula/gas/properties/pressure_function.py:19` - CPU reference
      equations define partial pressure and saturation ratio from those fields.
  - Resolved by: #1496

- [x] Who owns derived-state freshness and what happens when one virtual writer
  fails after another succeeds?
  - Resolved 2026-07-29 in #1496: The concrete coordinator exclusively owns
    stale markers and its local schedule cursor. A failed vapor writer leaves
    both fields stale; if vapor succeeds and saturation fails, vapor remains
    fresh and saturation remains stale. In either case the cursor does not
    advance and the consumer callback is not invoked.
  - Rationale: Freshness must reflect successful writes without claiming a
    rollback that cannot be guaranteed after a device writer launches.
  - Evidence: `particula/execution/thermodynamic_updates.py` and
    `particula/execution/tests/thermodynamic_updates_test.py`.

- [x] Does a prescribed resident environment or gas update execute scheduling or
  derived-state refresh?
  - Resolved 2026-07-29: No. P4 is only a graph-node-bound, direct-import
    in-place update seam. Scheduling and P5 vapor-pressure/saturation refresh
    remain separate work.
  - Evidence: `particula/execution/state_updates.py` and
    `particula/execution/tests/state_updates_test.py`.
