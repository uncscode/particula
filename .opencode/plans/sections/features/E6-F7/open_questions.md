# Open Questions

All E6-F7 planning questions were resolved on 2026-07-21. The selected CPU
reference is a bounded empirical particle source, not a predictive
critical-cluster model.

- [x] Which coefficient units and validity intervals do builders accept?
  - Decision: activation coefficients accept `s^-1`; kinetic coefficients accept
    `m^3/s` and `cm^3/s`; concentration bounds accept `1/m^3` and `1/cm^3`;
    temperature uses K. Normalize to SI and require coefficient provenance plus
    explicit closed concentration, temperature, and configured gate bounds. No
    universal scientific interval or coefficient is supplied.
- [x] What does an unsatisfied optional saturation gate do?
  - Decision: a valid value below its configured threshold returns exactly zero.
    A disabled gate is not evaluated. Nonfinite, negative, or declared-domain
    violations raise before mutation; activation/kinetic laws do not implicitly
    require `S>1`.
- [x] Which source and diagnostic names align with E6-F5/F6?
  - Decision: use `potential_event_count`, `gas_admitted_event_count`,
    `represented_event_count`, `gas_limited_event_count`,
    `representation_reduction_event_count`, `residual_event_count`,
    `limiting_species_index`, `gas_mass_removed`, `requested_slot_count`,
    `activated_slot_count`, `released_slot_count`, `exhaustion_policy_code`,
    `representative_volume_scale`, and `conservation_residual`.
    `residual_event_count` is unrepresented demand in the final scaled domain
    and must be zero on success. Do not duplicate E6-F5 free-index fields.
- [x] How are continuous events packaged into integer slots?
  - Decision: after representation scaling, for `represented_event_count=E` and
    maximum slot weight `W`, request zero slots when `E=0`; otherwise request
    `ceil(E/W)` and assign equal weight `E/n_slots` to every source slot. This
    preserves all represented events, avoids a small remainder slot, and stays
    within the configured maximum.
- [x] Does the runnable accept containers or legacy facades?
  - Decision: strategy, finalizer, and slot/exhaustion APIs accept `GasData`,
    `ParticleData`, and `EnvironmentData`. The single-box high-level
    `Nucleation` runnable accepts `Aerosol` and a construction-time
    `EnvironmentData` provider, then unwraps backing particle/gas data without
    copy. Multi-box behavior remains data-level. Mixed overloaded
    facade/container mutation APIs are not added.
- [x] Is a full Vehkamaki binary or CNT parameterization part of E6-F7?
  - Decision: no. Ship caller-calibrated `J=A*C` and `J=K*C^2` source laws with
    caller-specified formation composition, size, and survival. Documentation
    must not call these supplied formation particles predicted critical clusters.
- [x] Are slot activation and exhaustion private nucleation implementations?
  - Decision: no. Consume the E6-F5 and E6-F6 contracts.
