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

- [ ] What is the precise canonical placement of nucleation relative to
  condensation when both consume the same gas species?
  - Open: Both processes mutate current gas inventory, but the only shipped
    sequence is explicitly illustrative and cannot establish scientific intent.
  - Recommendation: **A - Require a reviewed scheduling profile to declare the order**
  - Suggested answer: Choose **A** because repository evidence establishes
    sequential gas visibility but does not justify one universal scientific
    ordering.
  - Options:
    - [ ] A. Require a reviewed scheduling profile to declare one fixed edge for
      the configured workflow (Recommended)
    - [ ] B. Make condensation before nucleation universal
    - [ ] C. Make nucleation before condensation universal
  - Evidence considered:
    - `docs/Features/Roadmap/data-oriented-gpu.md:1527` - the roadmap gives only
      an example process ordering, not a nucleation/condensation rule.

- [x] Does saturation refresh require a new narrow GPU primitive or can it be
  composed safely from existing thermodynamic state without host readback?
  - Resolved 2026-07-27: Add a narrow setup-validated device primitive that
    writes `environment.saturation_ratio` from current gas concentration, molar
    mass, temperature, and refreshed vapor pressure.
  - Rationale: The necessary state is resident, but the existing refresh writes
    only vapor pressure and cannot update saturation ratio.
  - Evidence:
    - `particula/gpu/kernels/thermodynamics.py:318` - the shipped primitive
      mutates only `gas.vapor_pressure`.
    - `particula/gpu/warp_types.py:164` - the destination saturation-ratio array
      is already device resident.
    - `particula/gas/properties/pressure_function.py:19` - CPU reference
      equations define partial pressure and saturation ratio from those fields.
  - Resolved by: plan-question-resolver
