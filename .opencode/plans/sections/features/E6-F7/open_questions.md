# Open Questions

- [ ] What coefficient-unit forms and default validity intervals should builders
  accept?
  - Open for P1: require explicit units/domain; do not invent a universal
    coefficient or extrapolate. Store normalized SI values.
- [ ] Should an unsatisfied optional saturation gate return zero or raise?
  - Proposed for P1: valid `S<=1` returns zero when configured; malformed or
    out-of-domain environmental state raises before mutation.
- [ ] Which source-record/diagnostics names exactly align with E6-F5/F6?
  - Open for P2/P3: retain potential/admitted events, limiting species,
    requested/activated slots, policy, gas removed, and residual demand.
- [ ] How are continuous physical events packaged into integer computational
  slots?
  - Open for P2: use a deterministic configured target weight; preserve
    represented number and each species mass without rounding demand away.
- [ ] Does the runnable accept only data containers or legacy facades too?
  - Open for P5: prefer data containers; facades require identical behavior and
    consistency with current migration policy.
- [x] Is a full Vehkamäki binary parameterization part of E6-F7?
  - Resolved 2026-07-21: No. Initially support bounded activation and kinetic
    empirical strategies; Vehkamäki et al. (2002) is cited context.
- [x] Are slot activation and exhaustion private nucleation implementations?
  - Resolved 2026-07-21: No. Consume E6-F5 and E6-F6 contracts.
